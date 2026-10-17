"""Combined #23: cancelling an open approval card resolves its durable row."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from machina.core.session import SessionManager
from machina.permissions import broker as broker_module
from machina.permissions import store as approval_store
from machina.permissions.policy import build_request


async def test_cancelled_interactive_wait_marks_the_approval_cancelled(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.session.db", db_pool)
    session = await SessionManager().create(title="approval cancel", agent_id="harness")
    entered = asyncio.Event()

    async def handler(_view):
        entered.set()
        await asyncio.Event().wait()

    handler_token = broker_module._approval_handler.set(handler)
    principal_token = broker_module._principal.set("tui")
    request = build_request(
        operation="process.exec",
        argv=["rm", "-rf", "/tmp/does-not-exist"],
        mode="ask",
        agent_id="harness",
        session_id=str(session.id),
        turn_id="cancel-turn",
    )
    task = asyncio.create_task(broker_module.permission_broker.guard(request))
    try:
        await asyncio.wait_for(entered.wait(), 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = await approval_store.get_approval(request.request_id)
        assert record is not None
        assert record["status"] == "cancelled"
    finally:
        broker_module._approval_handler.reset(handler_token)
        broker_module._principal.reset(principal_token)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", session.id)
