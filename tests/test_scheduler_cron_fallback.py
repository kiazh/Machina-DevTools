"""Evidence tests for the unparseable-cron reschedule defect (core-loop review
P0-2, assigned as P0-5 in this campaign).

``JobStore.finish`` reschedules cron jobs via ``_finish_next_run``, which
historically caught every ``next_cron_time`` exception and returned ``None``.
The UPDATE then wrote ``next_run_at = CASE WHEN kind = 'cron' THEN $4 ...`` —
for a broken expression $4 was SQL NULL, so ``claim_due``'s
``next_run_at <= now()`` predicate could never match and the job silently
stopped firing forever (no error surfaced anywhere).

The fix: an unparseable expression (or a missing future fire time) is recorded
via ``record_failure`` and the job is parked on a bounded retry from the
database clock. NULL stays reserved for non-cron kinds (the SQL ELSE branch
ignores the parameter there).

Tests are RED on the pre-fix code and GREEN after the fallback.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core import scheduler as scheduler_module
from machina.core.scheduler import JobStore

_BAD_CRON = "not a cron expression"
_CLOCK = datetime(2035, 1, 1, 12, 0, tzinfo=UTC)


def _expected_retry() -> timedelta | None:
    """Bounded retry window; None while the constant is absent (RED state)."""
    retry = getattr(scheduler_module, "CRON_RESCHEDULE_RETRY_SECONDS", None)
    return None if retry is None else timedelta(seconds=retry)


def _assert_not_numberless(next_run) -> None:
    """next_run_at must be a reachable future time, never the NULL reschedule."""
    expected = _expected_retry()
    if expected is not None:
        assert next_run == _CLOCK + expected
    assert next_run is not None, "cron reschedule wrote NULL (job disabled forever)"
    assert next_run > _CLOCK


async def test_unparseable_cron_falls_back_to_bounded_retry_not_null(monkeypatch):
    """A cron finish with a broken expression must never reschedule to NULL."""
    from machina.core import scheduler

    failures = []

    def fake_record_failure(source, exc):
        failures.append((source, exc))

    monkeypatch.setattr(scheduler, "record_failure", fake_record_failure)
    db = SimpleNamespace(fetchval=AsyncMock(return_value=_CLOCK), execute=AsyncMock())
    monkeypatch.setattr(scheduler, "db", db)
    store = JobStore()
    store.get = AsyncMock(
        return_value=SimpleNamespace(kind="cron", cron_expression=_BAD_CRON)
    )
    job_id = uuid.uuid4()

    assert await store.finish(job_id) is True

    # SQL $4 (execute arg index 4) is the cron next_run_at: it must be a
    # reachable future time, not the silently-disabling NULL.
    _assert_not_numberless(db.execute.await_args.args[4])
    # The reschedule failure is surfaced, not swallowed.
    assert any(source == "scheduler._finish_next_run" for source, _ in failures)


async def test_cron_parser_exception_falls_back_and_records_failure(monkeypatch):
    """A raise from next_cron_time parks the job on the bounded retry."""
    from machina.core import scheduler

    failures = []

    def fake_record_failure(source, exc):
        failures.append((source, exc))

    monkeypatch.setattr(scheduler, "record_failure", fake_record_failure)

    def boom(_expr, _now):
        raise ValueError("seconds field unsupported")

    monkeypatch.setattr(scheduler, "next_cron_time", boom)
    db = SimpleNamespace(fetchval=AsyncMock(return_value=_CLOCK), execute=AsyncMock())
    monkeypatch.setattr(scheduler, "db", db)
    store = JobStore()
    store.get = AsyncMock(
        return_value=SimpleNamespace(kind="cron", cron_expression="5 4 * * 9")
    )

    assert await store.finish(uuid.uuid4()) is True
    _assert_not_numberless(db.execute.await_args.args[4])
    assert any(
        isinstance(exc, ValueError) for _, exc in failures
    ), "the parser error context must be recorded"


async def test_cron_no_future_fire_time_falls_back_not_null(monkeypatch):
    """next_cron_time returning None (no fire time) must not write NULL either."""
    from machina.core import scheduler

    monkeypatch.setattr(scheduler, "record_failure", lambda *a: None)

    def no_fire(_expr, _now):
        return None

    monkeypatch.setattr(scheduler, "next_cron_time", no_fire)
    db = SimpleNamespace(fetchval=AsyncMock(return_value=_CLOCK), execute=AsyncMock())
    monkeypatch.setattr(scheduler, "db", db)
    store = JobStore()
    store.get = AsyncMock(
        return_value=SimpleNamespace(kind="cron", cron_expression="* * * * *")
    )

    assert await store.finish(uuid.uuid4()) is True
    _assert_not_numberless(db.execute.await_args.args[4])


_JOB_COLUMNS = (
    "id",
    "name",
    "kind",
    "session_id",
    "agent_name",
    "prompt",
    "interval_seconds",
    "enabled",
    "status",
    "last_run_at",
    "next_run_at",
    "last_error",
    "run_count",
    "cron_expression",
    "model",
    "provider",
    "no_agent",
    "script_path",
    "claim_token",
    "paused_for_approval",
)


def _job_row(job_id: uuid.UUID, kind: str, cron_expr: str) -> dict:
    row = {name: None for name in _JOB_COLUMNS}
    row.update(
        id=job_id,
        name="j",
        kind=kind,
        interval_seconds=60,
        enabled=True,
        status="idle",
        next_run_at=_CLOCK - timedelta(minutes=5),
        cron_expression=cron_expr,
    )
    return row


class _FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


class _FakeConn:
    def __init__(self, row: dict, clock: datetime) -> None:
        self._row = row
        self._clock = clock
        self.executions: list[tuple[str, tuple]] = []

    def transaction(self):
        return _FakeTransaction()

    async def fetchrow(self, *_args, **_kwargs):
        return self._row

    async def fetchval(self, *_args, **_kwargs):
        return self._clock

    async def execute(self, sql, *params):
        self.executions.append((sql, params))
        return "UPDATE 1"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


async def test_conn_path_unparseable_cron_falls_back_not_null(monkeypatch):
    """The production (acquire/transaction) branch inherits the same fallback."""
    from machina.core import scheduler

    failures = []

    def fake_record_failure(source, exc):
        failures.append((source, exc))

    monkeypatch.setattr(scheduler, "record_failure", fake_record_failure)
    job_id = uuid.uuid4()
    conn = _FakeConn(_job_row(job_id, "cron", _BAD_CRON), _CLOCK)
    monkeypatch.setattr(scheduler, "db", SimpleNamespace(acquire=lambda: conn))
    store = JobStore()

    assert await store.finish(job_id) is True

    assert conn.executions, "the UPDATE must run on the connection"
    sql, params = conn.executions[0]
    assert "UPDATE jobs" in sql
    # next_run is bound at $4: params[3] must be the bounded retry, never NULL.
    _assert_not_numberless(params[3])
    assert any(source == "scheduler._finish_next_run" for source, _ in failures)


async def test_non_cron_finish_keeps_sql_interval_branch(monkeypatch):
    """Non-cron kinds are rescheduled by SQL (ELSE now()+interval); param stays None."""
    from machina.core import scheduler

    monkeypatch.setattr(scheduler, "record_failure", lambda *a: None)
    job_id = uuid.uuid4()
    conn = _FakeConn(_job_row(job_id, "interval", None), _CLOCK)
    monkeypatch.setattr(scheduler, "db", SimpleNamespace(acquire=lambda: conn))
    store = JobStore()

    assert await store.finish(job_id) is True

    assert conn.executions[0][1][3] is None


@pytest.mark.parametrize("bad_expr", ["", "   ", "99 25 * * *"])
async def test_empty_and_invalid_cron_falls_back_never_null(monkeypatch, bad_expr):
    """Empty, whitespace, and out-of-range expressions take the fallback."""
    from machina.core import scheduler

    failures = []

    def fake_record_failure(source, exc):
        failures.append((source, exc))

    monkeypatch.setattr(scheduler, "record_failure", fake_record_failure)
    db = SimpleNamespace(fetchval=AsyncMock(return_value=_CLOCK), execute=AsyncMock())
    monkeypatch.setattr(scheduler, "db", db)
    store = JobStore()
    store.get = AsyncMock(return_value=SimpleNamespace(kind="cron", cron_expression=bad_expr))

    assert await store.finish(uuid.uuid4()) is True
    _assert_not_numberless(db.execute.await_args.args[4])
    assert failures, "the unusable expression must be surfaced"
