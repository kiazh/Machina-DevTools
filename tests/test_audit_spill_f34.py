"""Evidence tests for F-34: dropped audit events must survive a restart.

``AuditPersistence`` is best effort: a writer that has not started, a writer
during/after shutdown, a full queue, and a failed SQL write all drop the event
with only an in-memory counter. If the process then exits, no record of the
dropped security event survives.

Fix: every drop path best-effort appends one sanitized JSON line to
``<machina_home>/logs/audit-spill.jsonl`` (rotated once at the documented size
cap). Spilling never raises and never waits on the caller's DB path; a spill
failure keeps the old drop behavior (metrics + log only).

All tests here are DB-free: the SQL branch is replaced at the ``db.execute``
seam, and every assertion reads the spill file.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from machina.observability import audit as audit_module
from machina.observability.audit import AuditPersistence


@pytest.fixture
def spill_path(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "audit-spill.jsonl"
    monkeypatch.setattr(audit_module, "_spill_path", lambda: path)
    return path


def _records(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_before_start_drop_spills_json_line(spill_path):
    writer = AuditPersistence()
    writer.submit({"event": "approval.resolve", "decision": "allow"})

    records = _records(spill_path)
    assert len(records) == 1
    assert records[0]["reason"] == "before-start"
    assert records[0]["entry"]["event"] == "approval.resolve"
    assert records[0]["spilled_at"], "each spilled line must be timestamped"


def test_shutdown_drop_spills_json_line(spill_path):
    writer = AuditPersistence()
    writer._ever_started = True  # writer existed and has since stopped
    writer.submit({"event": "mode.switch", "mode": "full"})

    records = _records(spill_path)
    assert [record["reason"] for record in records] == ["shutdown"]
    assert records[0]["entry"]["event"] == "mode.switch"


async def test_queue_full_drop_spills_json_line(spill_path):
    writer = AuditPersistence(max_pending=1)
    writer._ever_started = True
    # No running writer: the queue stays full and this task is still alive.
    writer._task = asyncio.current_task()
    writer._queue = asyncio.Queue(maxsize=1)
    writer._queue.put_nowait({"event": "first"})

    writer.submit({"event": "approval.request", "capability": "terminal"})

    records = _records(spill_path)
    assert [record["reason"] for record in records] == ["queue-full"]
    assert records[0]["entry"]["event"] == "approval.request"


async def test_persistence_failure_spills_json_line(spill_path, monkeypatch):
    """A failed SQL write is also a durable-record drop (DB unavailability)."""
    monkeypatch.setattr(
        audit_module.db, "execute", AsyncMock(side_effect=RuntimeError("db down"))
    )
    writer = AuditPersistence()
    writer.start()
    try:
        writer.submit({"event": "secret.rotate", "name": "provider"})
        await asyncio.wait_for(writer._queue.join(), timeout=1)
    finally:
        await writer.stop()

    records = _records(spill_path)
    assert [record["reason"] for record in records] == ["persistence-failed"]
    assert records[0]["entry"]["event"] == "secret.rotate"


def test_spilled_entry_is_sanitized(spill_path):
    writer = AuditPersistence()
    writer.submit(
        {
            "event": "auth.login",
            "api_key": "sk-live-must-not-hit-disk",
            "authorization": "Bearer must-not-hit-disk",
        }
    )

    raw = spill_path.read_text()
    assert "must-not-hit-disk" not in raw
    assert _records(spill_path)[0]["entry"]["event"] == "auth.login"


def test_spill_failure_never_raises_into_the_caller(monkeypatch):
    """On spill failure the documented drop behavior stays: metric + log only."""

    def unavailable():
        raise OSError("read-only audit home")

    monkeypatch.setattr(audit_module, "_spill_path", unavailable)
    writer = AuditPersistence()

    # Must not raise.
    assert writer.submit({"event": "approval.resolve"}) is None


def test_spill_rotates_at_the_documented_size_cap(spill_path, monkeypatch):
    monkeypatch.setattr(audit_module, "AUDIT_SPILL_MAX_BYTES", 200)
    writer = AuditPersistence()

    for index in range(10):
        writer._spill({"event": f"event-{index}"}, "before-start")

    rotated = spill_path.with_name(spill_path.name + ".1")
    assert rotated.exists(), "the cap must rotate rather than grow without bound"
    # Current generation is bounded by the cap plus at most the last record.
    assert spill_path.stat().st_size < 1024
    assert len(_records(rotated)) > 0


def test_spill_path_honors_configured_machina_home(tmp_path, monkeypatch):
    from machina.core.config import config

    monkeypatch.setattr(
        config,
        "get",
        lambda key: str(tmp_path) if key == "machina_home" else None,
    )
    assert audit_module._spill_path() == tmp_path / "logs" / "audit-spill.jsonl"
