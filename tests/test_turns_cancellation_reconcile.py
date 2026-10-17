"""MCH-002: cancellation during claim acquisition must not leak or steal claims.

The traced ordering that must hold: acquire the local mutex -> construct the
owner token -> await the durable claim -> register local ownership -> return
the token. A cancellation delivered while awaiting the durable claim (or any
later acquisition stage) previously bypassed the ``except Exception`` cleanup
entirely: the durable claim could stay committed under a token nobody
received, and the local mutex stayed held.

Now cancellation runs a shielded, bounded, token-matched reconciliation:
release the claim ONLY if it is still provably ours, leave it for stale-claim
recovery when the outcome is ambiguous, record the failure, and always
re-raise ``CancelledError``. A successor's claim is never released.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from machina.core import turns


@pytest.fixture(autouse=True)
def _isolate_local_claims():
    """The coordinator registries are process-global; keep tests isolated."""
    turns._local_reset_for_tests()
    yield
    turns._local_reset_for_tests()


class _FakeClaimDb:
    """Minimal claim-table substitute for the coordinator's SQL seams."""

    connected = True

    def __init__(self) -> None:
        self.claims: dict[str, dict] = {}

    async def fetchrow(self, sql: str, *args):
        head = " ".join(sql.split())
        if head.startswith("SELECT claim_owner"):
            record = self.claims.get(str(args[0]))
            if record is None:
                return None
            return {
                "claim_owner": record["owner"],
                "claim_kind": record["kind"],
                "claim_expires_at": record["expires"],
            }
        # Durable claim: UPDATE sessions SET claim_owner = $2 ...
        sid, owner, kind, ttl = args[0], args[1], args[2], args[3]
        record = self.claims.get(str(sid))
        if record is not None and record["expires"] > datetime.now(UTC):
            return None  # a live claim blocks the UPDATE
        self.claims[str(sid)] = {
            "owner": owner,
            "kind": kind,
            "expires": datetime.now(UTC) + timedelta(seconds=ttl),
        }
        return {"id": str(sid)}

    async def execute(self, sql: str, *args):
        head = " ".join(sql.split())
        if head.startswith("UPDATE sessions SET claim_owner = NULL"):
            sid, owner = args[0], args[1]
            record = self.claims.get(str(sid))
            if record is not None and record["owner"] == owner:
                del self.claims[str(sid)]
                return "UPDATE 1"
            return "UPDATE 0"
        raise AssertionError(f"unexpected SQL in fixture: {head[:80]}")

    def owner(self, sid: uuid.UUID) -> str | None:
        record = self.claims.get(str(sid))
        return None if record is None else record["owner"]

    def expire(self, sid: uuid.UUID) -> None:
        """Simulate TTL lapse (stale-claim recovery precondition)."""
        self.claims[str(sid)]["expires"] = datetime.now(UTC) - timedelta(seconds=1)


_ACQUISITIONS = [
    pytest.param(turns.try_begin_turn, turns.end_turn, id="turn"),
    pytest.param(turns.begin_mutation, turns.end_mutation, id="mutation"),
]


def _hanging_claim(fake: _FakeClaimDb, committed: asyncio.Event, hook: asyncio.Event):
    """Patched ``_db_claim`` that commits the durable claim, then hangs.

    Cancelling the acquisition task at this await point reproduces the exact
    MCH-002 ordering: the claim is durable under a token the caller never
    received, no local ownership was registered yet, and the call never
    returned.
    """

    async def claim(session_id, owner, kind, ttl_s):
        assert await fake.fetchrow("UPDATE claim", session_id, owner, kind, ttl_s) is not None
        committed.set()
        await hook.wait()  # THE await point the test cancels at
        return True

    return claim


@pytest.mark.parametrize("kind_fn,end_fn", _ACQUISITIONS)
async def test_cancelled_acquisition_releases_only_own_claim_then_propagates(
    monkeypatch, kind_fn, end_fn
):
    fake = _FakeClaimDb()
    monkeypatch.setattr("machina.db.connection.db", fake)
    real_claim = turns._db_claim
    committed, hook = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(turns, "_db_claim", _hanging_claim(fake, committed, hook))
    sid = uuid.uuid4()

    task = asyncio.create_task(kind_fn(sid))
    await committed.wait()
    orphaned_token = fake.owner(sid)
    assert orphaned_token, "precondition: the durable claim committed before the cancel"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(turns, "_db_claim", real_claim)

    # Reconciliation released the claim that is still provably ours.
    assert fake.owner(sid) is None, "the orphaned durable claim must be reconciled"
    assert turns._local_owner(sid) is None and not turns.turn_locked(sid)

    # A successor acquires; the cancelled task's stale token can never
    # release the successor's claim (token-matched release proves ownership).
    successor = await kind_fn(sid)
    assert successor and successor != orphaned_token
    assert fake.owner(sid) == successor
    assert await end_fn(sid, orphaned_token) is False
    assert fake.owner(sid) == successor, "a stale cancelled token cannot steal the claim"
    assert await end_fn(sid, successor) is True


@pytest.mark.parametrize("kind_fn,end_fn", _ACQUISITIONS)
async def test_ambiguous_release_leaves_claim_for_stale_recovery(monkeypatch, kind_fn, end_fn):
    fake = _FakeClaimDb()
    monkeypatch.setattr("machina.db.connection.db", fake)
    real_claim, real_release = turns._db_claim, turns._db_release
    committed, hook = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(turns, "_db_claim", _hanging_claim(fake, committed, hook))

    async def raising_release(session_id, owner):
        raise ConnectionError("database offline mid-reconciliation")

    monkeypatch.setattr(turns, "_db_release", raising_release)
    failures: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(turns, "record_failure", lambda site, error: failures.append((site, error)))
    sid = uuid.uuid4()

    task = asyncio.create_task(kind_fn(sid))
    await committed.wait()
    orphaned_token = fake.owner(sid)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(turns, "_db_claim", real_claim)
    monkeypatch.setattr(turns, "_db_release", real_release)

    assert failures and failures[0][0].startswith("turns."), "ambiguity must be recorded"
    assert isinstance(failures[0][1], ConnectionError)
    # Ambiguous ownership: the durable claim is LEFT for stale-claim recovery
    # (never released blind) — a successor stays blocked until expiry.
    assert fake.owner(sid) == orphaned_token
    assert await kind_fn(sid) is None
    # ... and stale-claim recovery (TTL lapse) restores availability.
    fake.expire(sid)
    recovered = await kind_fn(sid)
    assert recovered and recovered != orphaned_token
    assert await end_fn(sid, recovered) is True


async def test_turn_release_budget_is_bounded_and_never_masks_cancellation(monkeypatch):
    fake = _FakeClaimDb()
    monkeypatch.setattr("machina.db.connection.db", fake)
    committed, hook = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(turns, "_db_claim", _hanging_claim(fake, committed, hook))
    monkeypatch.setattr(turns, "CLAIM_RECONCILE_TIMEOUT_SECONDS", 0.05)
    detached: list[asyncio.Task] = []

    async def hanging_release(session_id, owner):
        detached.append(asyncio.current_task())
        await asyncio.Event().wait()

    monkeypatch.setattr(turns, "_db_release", hanging_release)
    failures: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(turns, "record_failure", lambda site, error: failures.append((site, error)))
    sid = uuid.uuid4()

    task = asyncio.create_task(turns.try_begin_turn(sid))
    await committed.wait()
    orphaned_token = fake.owner(sid)
    import time

    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    bounded = time.monotonic() - started

    assert bounded < 2.0, "a hung release must be bounded, never stall cancellation"
    assert failures and isinstance(failures[0][1], TimeoutError), "the budget overrun is recorded"
    assert fake.owner(sid) == orphaned_token, "unresolved ownership stays for stale recovery"
    for pending in detached:
        pending.cancel()
    await asyncio.gather(*detached, return_exceptions=True)


async def test_second_cancellation_during_reconciliation_still_propagates(monkeypatch):
    """A re-cancelled acquisition keeps raising CancelledError, never an error.

    The delivered-again cancellation lands inside the reconciliation window
    (the acquisition is awaiting its token-matched release); the stop must
    propagate through reconcile, local cleanup, and the re-raise cleanly.
    """
    fake = _FakeClaimDb()
    monkeypatch.setattr("machina.db.connection.db", fake)
    real_claim = turns._db_claim
    committed, hook = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(turns, "_db_claim", _hanging_claim(fake, committed, hook))
    sid = uuid.uuid4()

    async def release_then_reacancel(session_id, owner):
        mine = fake.owner(sid) == owner
        assert await fake.execute(
            "UPDATE sessions SET claim_owner = NULL", session_id, owner
        )  # token-matched claim release -> "UPDATE 1"
        task.cancel()  # second cancellation delivered mid-reconciliation
        return mine

    monkeypatch.setattr(turns, "_db_release", release_then_reacancel)
    task = asyncio.create_task(turns.try_begin_turn(sid))
    await committed.wait()
    task.cancel()  # first cancellation surfaces inside the hanging claim
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(turns, "_db_claim", real_claim)

    assert fake.owner(sid) is None, "the token-matched release still cleared our own claim"
    assert turns._local_owner(sid) is None and not turns.turn_locked(sid)
    successor = await turns.try_begin_turn(sid)
    assert successor, "a re-cancelled acquisition must not block the next owner"
    assert await turns.end_turn(sid, successor) is True
