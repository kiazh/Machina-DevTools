"""Combined #17: a scheduled job's agent override is run-scoped, not permanent."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from machina.core.scheduler import JobRunner


async def test_reconcile_returns_the_owner_and_restore_puts_it_back(monkeypatch):
    session_id = uuid.uuid4()
    job = SimpleNamespace(
        id=uuid.uuid4(),
        session_id=session_id,
        agent_name="researcher",
        no_agent=False,
    )
    session = SimpleNamespace(agent_id="harness")
    calls: list[tuple[str, tuple]] = []

    class _DB:
        connected = True

        async def execute(self, query, *args):
            calls.append((query, args))
            return "UPDATE 1"

    monkeypatch.setattr("machina.db.connection.db", _DB())
    monkeypatch.setattr(
        "machina.core.session.session_manager.get_fresh", AsyncMock(return_value=session)
    )
    runner = JobRunner(agent_factory=None)

    previous = await runner._reconcile_session_owner(job)
    assert previous == "harness"
    assert calls[0][1] == (session_id, "researcher")

    await runner._restore_session_owner(job, previous)
    assert len(calls) == 2
    query, args = calls[1]
    assert "agent_id = $2" in query and "agent_id = $3" in query
    assert args == (session_id, "harness", "researcher")
