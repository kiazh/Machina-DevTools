"""Honest post-turn compaction status counters (spec §14.7, §21).

The B3 evidence recorded this as an unmet gap: background compaction had no
counters distinguishing scheduled / skipped / started / completed / failed.
This suite pins the counters added to ``services.maybe_auto_compact`` and
``TurnEngine.schedule_auto_compact``:

* ``compaction.scheduled`` — emitted at schedule time (task created), before
  any claim is attempted, so a scheduled-but-never-started task is visible.
* ``compaction.started`` — emitted only once the mutation claim is acquired.
* ``compaction.completed`` — emitted when compression actually committed,
  plus a bounded ``compaction_status`` audit event carrying size before/after.
* ``compaction.failed`` — emitted with the error class (never the message) in
  a bounded ``compaction_status`` audit event.
* ``compaction.skipped.{disabled,busy,below_threshold,no_progress,session_missing,no_change}``
  — exactly one counter per skip decision.

Busy/deferred/threshold etc. run against ``MACHINA_TEST_DATABASE_URL`` and skip
without it. No live provider calls: the one real compaction uses truncation.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from machina.core.config import config
from machina.core.metrics import metrics

# ─── counters under test ─────────────────────────────────────────────────────

#: The exact counter keys this task introduces. A decision must move exactly
#: one of these (``started``/``completed`` are lifecycle stages, not mutually
#: exclusive with a later skip/fail).
COUNTERS = (
    "compaction.scheduled",
    "compaction.started",
    "compaction.completed",
    "compaction.failed",
    "compaction.skipped.disabled",
    "compaction.skipped.busy",
    "compaction.skipped.below_threshold",
    "compaction.skipped.no_progress",
    "compaction.skipped.session_missing",
    "compaction.skipped.no_change",
)


def _snapshot() -> dict[str, int]:
    return {name: metrics.get_counter(name) for name in COUNTERS}


def _delta(before: dict[str, int]) -> dict[str, int]:
    """Only the counters that moved since *before* (zero deltas dropped)."""
    return {name: metrics.get_counter(name) - value for name, value in before.items()
            if metrics.get_counter(name) != value}


def _capture_status_audits(monkeypatch) -> list[tuple[tuple, dict]]:
    """Capture the bounded ``compaction_status`` audit events as they emit."""

    events: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "machina.core.provider.audit_log",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    return events


def _status_events(events: list[tuple[tuple, dict]]) -> list[dict]:
    return [kwargs for args, kwargs in events if args == ("compaction_status",)]


# ─── DB helpers (skip without MACHINA_TEST_DATABASE_URL) ─────────────────────


@pytest.fixture
async def live_db():
    import os

    if not os.environ.get("MACHINA_TEST_DATABASE_URL"):
        pytest.skip("MACHINA_TEST_DATABASE_URL not set")
    from machina.db.connection import db

    await db.connect()
    await db.initialize_schema()
    yield db
    await db.close()


async def _mk_session(db, *, agent_id: str = "harness", state: dict | None = None) -> uuid.UUID:
    session_id = uuid.uuid4()
    await db.execute(
        "INSERT INTO sessions (id, agent_id, state_msgpack) VALUES ($1, $2, $3)",
        session_id,
        agent_id,
        None,
    )
    if state:
        from machina.core.session import session_manager

        await session_manager.update_state_fields(session_id, state)
    return session_id


async def _drop_sessions(db, ids) -> None:
    if ids:
        await db.execute("DELETE FROM sessions WHERE id = ANY($1::uuid[])", list(ids))


async def _add_chunks(session_id: uuid.UUID, count: int, token_count: int = 50) -> None:
    from machina.core.context import context_manager

    for i in range(count):
        await context_manager.add_chunk(
            session_id,
            "harness",
            "user_message",
            {"content": f"deterministic compaction fixture line {i:03d} " * 4},
            token_count=token_count,
        )


# ─── config-disabled skips (DB-free) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_disabled_records_skipped_disabled(monkeypatch):
    from machina import services as services_module

    before = _snapshot()
    monkeypatch.setattr(config, "auto_compaction_enabled", False)
    assert await services_module.maybe_auto_compact(uuid.uuid4()) is None
    assert _delta(before) == {"compaction.skipped.disabled": 1}

    before = _snapshot()
    monkeypatch.setattr(config, "auto_compaction_enabled", True)
    monkeypatch.setattr(config, "compression_enabled", False)
    assert await services_module.maybe_auto_compact(uuid.uuid4()) is None
    assert _delta(before) == {"compaction.skipped.disabled": 1}


# ─── busy / no-claim deferral (DB-backed) ────────────────────────────────────


@pytest.mark.asyncio
async def test_busy_records_skipped_busy_without_starting(live_db):
    from machina import services as services_module
    from machina.core import turns

    session_id = await _mk_session(live_db)
    try:
        token = await turns.try_begin_turn(session_id)
        assert token
        before = _snapshot()
        assert await services_module.maybe_auto_compact(session_id) is None
        # Exactly one counter: skipped(busy). A deferred attempt never starts.
        assert _delta(before) == {"compaction.skipped.busy": 1}
        assert await turns.end_turn(session_id, token) is True
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_below_threshold_records_skipped_below_threshold(live_db, monkeypatch):
    from machina import services as services_module

    session_id = await _mk_session(live_db)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=1000))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10))
    try:
        before = _snapshot()
        assert await services_module.maybe_auto_compact(session_id) is None
        # The claim was acquired (started) then the threshold decision skipped.
        assert _delta(before) == {
            "compaction.started": 1,
            "compaction.skipped.below_threshold": 1,
        }
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_no_progress_records_skipped_no_progress(live_db, monkeypatch):
    from machina import services as services_module
    from machina.core.context import context_manager

    session_id = await _mk_session(
        live_db, state={"last_auto_compact_tokens": 10**6}
    )
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))
    monkeypatch.setattr(context_manager, "get_compressible_token_usage", AsyncMock(return_value=5))
    try:
        before = _snapshot()
        assert await services_module.maybe_auto_compact(session_id) is None
        assert _delta(before) == {
            "compaction.started": 1,
            "compaction.skipped.no_progress": 1,
        }
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_no_change_records_skipped_no_change(live_db, monkeypatch):
    from machina import services as services_module

    session_id = await _mk_session(live_db)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))
    # compress_session returns None (no-op/stale/busy collapse); busy is
    # impossible here because we hold the mutation claim, so it is a no-change.
    monkeypatch.setattr(services_module, "compress_session", AsyncMock(return_value=None))
    try:
        before = _snapshot()
        assert await services_module.maybe_auto_compact(session_id) is None
        assert _delta(before) == {
            "compaction.started": 1,
            "compaction.skipped.no_change": 1,
        }
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_missing_session_records_skipped_session_missing(monkeypatch):
    from machina import services as services_module
    from machina.core import turns
    from machina.core.session import session_manager

    # A claim acquired on a row that disappears before the read: the missing
    # session is one honest skip reason, not a silent None. (A genuinely
    # absent row makes begin_mutation itself return None -> skipped.busy, so
    # that path is pinned by test_busy_*; this pins the post-claim read.)
    session_id = uuid.uuid4()
    monkeypatch.setattr(turns, "begin_mutation", AsyncMock(return_value="mutation:tok"))
    monkeypatch.setattr(turns, "end_mutation", AsyncMock(return_value=True))
    monkeypatch.setattr(session_manager, "get", AsyncMock(return_value=None))
    before = _snapshot()
    assert await services_module.maybe_auto_compact(session_id) is None
    assert _delta(before) == {
        "compaction.started": 1,
        "compaction.skipped.session_missing": 1,
    }


# ─── real compaction: started + completed with before/after ──────────────────


@pytest.mark.asyncio
async def test_real_compaction_records_started_and_completed_with_sizes(live_db, monkeypatch):
    from machina import services as services_module

    session_id = await _mk_session(live_db)
    await _add_chunks(session_id, count=6, token_count=50)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))
    # Truncation path: no provider call, deterministic.
    monkeypatch.setattr(config, "compression_llm_summarize", False)
    monkeypatch.setattr(config, "compression_preserve_recent", 2)
    events = _capture_status_audits(monkeypatch)
    try:
        before = _snapshot()
        result = await services_module.maybe_auto_compact(session_id)
        assert result is not None and result["compressed"] is True
        assert _delta(before) == {
            "compaction.started": 1,
            "compaction.completed": 1,
        }
        assert result["originalTokens"] > 0
        completed = [e for e in _status_events(events) if e["status"] == "completed"]
        assert len(completed) == 1
        assert completed[0]["tokens_before"] == result["originalTokens"]
        assert completed[0]["tokens_after"] == result["compressedTokens"]
        assert completed[0]["error_class"] is None
        # Bounded: no chunk content in the event.
        assert "deterministic compaction fixture line" not in json.dumps(completed[0])
    finally:
        await _drop_sessions(live_db, [session_id])


# ─── induced failure: failed with the error class only ───────────────────────


@pytest.mark.asyncio
async def test_induced_failure_records_failed_with_error_class(live_db, monkeypatch):
    from machina import services as services_module

    session_id = await _mk_session(live_db)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))
    monkeypatch.setattr(
        services_module,
        "compress_session",
        AsyncMock(side_effect=RuntimeError("SECRET-COMPACTION-PAYLOAD")),
    )
    events = _capture_status_audits(monkeypatch)
    try:
        before = _snapshot()
        assert await services_module.maybe_auto_compact(session_id) is None
        assert _delta(before) == {
            "compaction.started": 1,
            "compaction.failed": 1,
        }
        failed = [e for e in _status_events(events) if e["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["error_class"] == "RuntimeError"
        # Bounded: the class, never the raw message payload.
        assert "SECRET-COMPACTION-PAYLOAD" not in json.dumps(failed[0])
    finally:
        await _drop_sessions(live_db, [session_id])


# ─── scheduled then busy: one skipped, zero started ──────────────────────────


async def _drive_schedule_auto_compact(monkeypatch, session_id) -> int:
    """Execute the real ``TurnEngine.schedule_auto_compact`` and await its task."""
    import machina.core.turn_engine as engine_module

    created: list[asyncio.Task] = []
    real_create_task = asyncio.create_task

    def _capture(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        created.append(task)
        return task

    tracked: list[asyncio.Task] = []

    def _track(task):
        tracked.append(task)
        return task

    monkeypatch.setattr(engine_module, "runtime_services", SimpleNamespace(track=_track))
    fake_engine = SimpleNamespace(
        spec=SimpleNamespace(session_id=session_id),
        _record=lambda site, exc: None,
    )
    with mock.patch.object(asyncio, "create_task", _capture):
        engine_module.TurnEngine.schedule_auto_compact(fake_engine)
    assert len(created) == 1
    for task in created:
        await task
    assert tracked == created
    return len(created)


@pytest.mark.asyncio
async def test_scheduled_then_busy_records_one_skip_and_zero_started(live_db, monkeypatch):
    from machina.core import turns

    session_id = await _mk_session(live_db)
    try:
        token = await turns.try_begin_turn(session_id)
        assert token
        before = _snapshot()
        assert await _drive_schedule_auto_compact(monkeypatch, session_id) == 1
        # Scheduled at schedule time; the claim is held so it never started.
        assert _delta(before) == {
            "compaction.scheduled": 1,
            "compaction.skipped.busy": 1,
        }
        assert await turns.end_turn(session_id, token) is True
    finally:
        await _drop_sessions(live_db, [session_id])


# ─── surfacing: the existing metric snapshot accessor ────────────────────────


@pytest.mark.asyncio
async def test_counters_are_surfaced_in_prometheus_snapshot(monkeypatch):
    from machina import services as services_module
    from machina.observability.metrics import prometheus_text

    monkeypatch.setattr(config, "auto_compaction_enabled", False)
    assert await services_module.maybe_auto_compact(uuid.uuid4()) is None

    body = prometheus_text()
    assert 'operation="compaction.skipped.disabled"' in body
    assert metrics.get_all_metrics()["counters"]["compaction.skipped.disabled"] >= 1
