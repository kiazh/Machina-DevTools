"""Limits on batch and recursive agent delegation."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent_def import AgentDef
from machina.core.orchestrator import Orchestrator
from machina.gateway.errors import INVALID_PARAMS, RpcError
from machina.gateway.features.agents import _steps


def test_agents_run_rejects_excessive_steps():
    with pytest.raises(RpcError) as error:
        _steps({"steps": [{"agent": "coder", "task": "work"}] * 17})
    assert error.value.code == INVALID_PARAMS


@pytest.mark.parametrize("nested", [False, True])
async def test_delegation_has_no_implicit_deadline_and_cancels_cleanly(monkeypatch, nested):
    from machina.core import orchestrator as module
    from machina.core import turns

    entered = asyncio.Event()

    class HangingAgent:
        async def run(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(
        module.agent_registry, "get", AsyncMock(return_value=AgentDef(name="coder"))
    )
    monkeypatch.setattr(
        module.session_manager, "create", AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
    )
    orch = Orchestrator(agent_factory=lambda _definition: HangingAgent())
    orch._record_start = AsyncMock(return_value=uuid.uuid4())
    orch._record_status = AsyncMock()
    orch._record_end = AsyncMock()
    monkeypatch.setattr(orch, "_charged_child_tokens", AsyncMock(return_value=0))
    monkeypatch.setattr(turns, "try_begin_turn", AsyncMock(return_value="test-claim"))
    monkeypatch.setattr(turns, "end_turn", AsyncMock(return_value=True))
    token = module.delegation_depth.set(1 if nested else 0)
    task = asyncio.create_task(orch.delegate("coder", "wait", _hop_count=2 if nested else 0))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        module.delegation_depth.reset(token)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert orch._record_end.await_args.kwargs["status"] == "cancelled"


async def test_delegate_tool_passes_nested_depth(monkeypatch):
    from machina.core import orchestrator as module
    from machina.tools.agents import delegate

    fake_result = SimpleNamespace(
        status="complete", response="ok", session_id=uuid.uuid4(), tokens=0, needs_approval=[]
    )
    call = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(module.orchestrator, "delegate", call)
    token = module.delegation_depth.set(3)
    try:
        assert "ok" in await delegate("coder", "next")
    finally:
        module.delegation_depth.reset(token)
    assert call.await_args.kwargs["_hop_count"] == 4
