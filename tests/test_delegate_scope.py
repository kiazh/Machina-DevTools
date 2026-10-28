"""Delegation through an agent tool must stay within its owned session."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


def test_delegate_inline_wait_setting_is_registered_and_bounded():
    from machina.core.config import DEFAULTS, Config, validate_value

    assert DEFAULTS["delegate_inline_wait_seconds"] == 120
    assert Config().delegate_inline_wait_seconds == 120
    assert validate_value("delegate_inline_wait_seconds", 0) == 0
    assert validate_value("delegate_inline_wait_seconds", 3600) == 3600
    with pytest.raises(ValueError):
        validate_value("delegate_inline_wait_seconds", 3601)


@pytest.mark.parametrize(
    ("status", "failed", "outcome"),
    [
        ("running", False, "running"),
        ("interrupted", True, "interrupted"),
        ("waiting_approval", False, "paused"),
        ("error", True, "failed"),
    ],
)
def test_worker_status_is_reflected_in_tool_outcome(status, failed, outcome):
    from machina.core.agent import _tool_outcome

    assert _tool_outcome({"_delegation_status": status, "_failed": failed}) == outcome


@pytest.mark.asyncio
async def test_delegate_refuses_a_parent_session_owned_by_another_agent(monkeypatch):
    from machina.core import orchestrator as module
    from machina.core.exceptions import ToolError
    from machina.tools.agents import current_agent_id, current_session_id, delegate

    calls = []

    async def fake_owner(*args):
        return "another-agent"

    async def fake_delegate(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(status="complete", response="leaked")

    monkeypatch.setattr("machina.tools.agents.db.fetchval", fake_owner)
    monkeypatch.setattr(module.orchestrator, "delegate", fake_delegate)
    session_token = current_session_id.set(uuid4())
    agent_token = current_agent_id.set("researcher")
    try:
        with pytest.raises(ToolError, match="does not belong"):
            await delegate("coder", "summarize parent context")
    finally:
        current_agent_id.reset(agent_token)
        current_session_id.reset(session_token)
    assert calls == []


@pytest.mark.asyncio
async def test_delegate_records_the_actual_parent_agent(monkeypatch):
    from machina.core import orchestrator as module
    from machina.tools.agents import current_agent_id, current_session_id, delegate

    calls = []

    async def fake_owner(*args):
        return "researcher"

    async def fake_delegate(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="complete", response="done", session_id=uuid4())

    monkeypatch.setattr("machina.tools.agents.db.fetchval", fake_owner)
    monkeypatch.setattr(module.orchestrator, "delegate", fake_delegate)
    session_id = uuid4()
    session_token = current_session_id.set(session_id)
    agent_token = current_agent_id.set("researcher")
    try:
        await delegate("coder", "check code")
    finally:
        current_agent_id.reset(agent_token)
        current_session_id.reset(session_token)
    assert calls == [
        {
            "from_agent": "researcher",
            "parent_session_id": session_id,
            "_hop_count": 1,
            # LP-08: the executing parent's authority propagates (None when
            # unrestricted) so the child cannot exceed it.
            "authority": None,
            "on_started": calls[0]["on_started"],
        }
    ]


@pytest.mark.asyncio
async def test_long_delegate_returns_a_trackable_child_and_keeps_running(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from machina.core import orchestrator as module
    from machina.tools import agents

    session_id, child_id = uuid4(), uuid4()
    release = asyncio.Event()

    async def owner(*args):
        return "researcher"

    async def fake_delegate(*args, **kwargs):
        kwargs["on_started"](child_id)
        await release.wait()
        return SimpleNamespace(
            status="complete",
            response="finished",
            session_id=child_id,
            tokens=4,
            needs_approval=[],
        )

    monkeypatch.setattr(agents.db, "fetchval", owner)
    monkeypatch.setattr(module.orchestrator, "delegate", fake_delegate)
    monkeypatch.setattr(
        module.orchestrator,
        "latest_child_delegation",
        AsyncMock(return_value={"status": "running"}),
    )
    session_token = agents.current_session_id.set(session_id)
    agent_token = agents.current_agent_id.set("researcher")
    try:
        result = await agents.delegate("coder", "long task", wait_seconds=1)
    finally:
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)

    assert "[coder running]" in result
    assert str(child_id) in result
    task = next(task for task in agents._background_delegations if not task.done())
    release.set()
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_headless_delegate_waits_for_worker_before_process_can_exit(monkeypatch):
    import asyncio

    from machina.core import orchestrator as module
    from machina.permissions import broker
    from machina.tools import agents

    child_id = uuid4()
    started, release = asyncio.Event(), asyncio.Event()

    async def fake_delegate(*args, **kwargs):
        kwargs["on_started"](child_id)
        started.set()
        await release.wait()
        return SimpleNamespace(
            status="complete", response="done", session_id=child_id, tokens=1, needs_approval=[]
        )

    monkeypatch.setattr(agents.db, "fetchval", AsyncMock(return_value="harness"))
    monkeypatch.setattr(module.orchestrator, "delegate", fake_delegate)
    monkeypatch.setattr(
        module.orchestrator, "latest_child_delegation", AsyncMock(return_value={"status": "running"})
    )
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("harness")
    principal_token = broker._principal.set("headless")
    task = None
    try:
        task = asyncio.create_task(agents.delegate("researcher", "research", wait_seconds=1))
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.sleep(1.1)
        assert not task.done()
        release.set()
        assert (await asyncio.wait_for(task, timeout=2)).delegation_status == "complete"
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        broker._principal.reset(principal_token)
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)


@pytest.mark.asyncio
async def test_background_delegate_is_cancelled_by_runtime_shutdown_tracking(monkeypatch):
    import asyncio

    from machina.core import orchestrator as module
    from machina.core import runtime
    from machina.core.cleanup import cancel_and_join
    from machina.tools import agents

    child_id = uuid4()
    stopped = asyncio.Event()
    owner = runtime.RuntimeServices()

    async def fake_delegate(*args, **kwargs):
        kwargs["on_started"](child_id)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(agents.db, "fetchval", AsyncMock(return_value="harness"))
    monkeypatch.setattr(module.orchestrator, "delegate", fake_delegate)
    monkeypatch.setattr(module.orchestrator, "latest_child_delegation", AsyncMock(return_value=None))
    monkeypatch.setattr(runtime, "runtime_services", owner)
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("harness")
    try:
        await agents.delegate("researcher", "long task", wait_seconds=1)
        tasks = list(agents._background_delegations)
        assert len(tasks) == 1 and tasks[0] in owner._background
        assert await cancel_and_join(list(owner._background), 2, "test shutdown") == []
        assert stopped.is_set()
    finally:
        for task in list(agents._background_delegations):
            task.cancel()
        await asyncio.gather(*agents._background_delegations, return_exceptions=True)
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)


@pytest.mark.asyncio
async def test_delegate_status_only_reads_children_of_the_current_parent(monkeypatch):
    from machina.core import orchestrator as module
    from machina.core.exceptions import ToolError
    from machina.tools import agents

    async def owner(*args):
        return "researcher"

    monkeypatch.setattr(agents.db, "fetchval", owner)
    monkeypatch.setattr(module.orchestrator, "latest_child_delegation", AsyncMock(return_value=None))
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("researcher")
    try:
        with pytest.raises(ToolError, match="belongs to this session"):
            await agents.delegate_status(str(uuid4()))
    finally:
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)


@pytest.mark.asyncio
async def test_delegate_status_marks_a_dead_expired_worker_interrupted(monkeypatch):
    from machina.core import orchestrator as module
    from machina.tools import agents

    child_id, entry_id = uuid4(), uuid4()
    fetches = iter(["researcher", False])
    execute = AsyncMock(return_value="UPDATE 1")

    async def fetchval(*args):
        return next(fetches)

    monkeypatch.setattr(agents.db, "fetchval", fetchval)
    monkeypatch.setattr(agents.db, "execute", execute)
    monkeypatch.setattr(
        module.orchestrator,
        "latest_child_delegation",
        AsyncMock(
            return_value={
                "id": str(entry_id),
                "childSessionId": str(child_id),
                "status": "running",
                "response": None,
            }
        ),
    )
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("researcher")
    try:
        result = await agents.delegate_status(str(child_id))
    finally:
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)

    assert "Worker interrupted" in result
    assert result.delegation_status == "interrupted"
    assert result.is_error is True
    assert execute.await_args.args[1] == entry_id


@pytest.mark.asyncio
async def test_delegate_status_returns_completed_result_when_worker_finishes_during_check(
    monkeypatch,
):
    from machina.core import orchestrator as module
    from machina.tools import agents

    child_id, entry_id = uuid4(), uuid4()
    monkeypatch.setattr(agents.db, "fetchval", AsyncMock(side_effect=["researcher", False]))
    monkeypatch.setattr(agents.db, "execute", AsyncMock(return_value="UPDATE 0"))
    monkeypatch.setattr(
        module.orchestrator,
        "latest_child_delegation",
        AsyncMock(
            side_effect=[
                {"id": str(entry_id), "status": "running", "response": None},
                {"id": str(entry_id), "status": "complete", "response": "finished"},
            ]
        ),
    )
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("researcher")
    try:
        result = await agents.delegate_status(str(child_id))
    finally:
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)

    assert result.delegation_status == "complete"
    assert "finished" in result


@pytest.mark.asyncio
async def test_delegate_status_keeps_a_live_worker_running(monkeypatch):
    import asyncio

    from machina.core import orchestrator as module
    from machina.tools import agents

    child_id = uuid4()
    hold = asyncio.Event()
    worker = asyncio.create_task(hold.wait())
    module._live_delegations[child_id] = worker

    async def owner(*args):
        return "researcher"

    monkeypatch.setattr(agents.db, "fetchval", owner)
    monkeypatch.setattr(
        module.orchestrator,
        "latest_child_delegation",
        AsyncMock(
            return_value={"childSessionId": str(child_id), "status": "running", "response": None}
        ),
    )
    execute = AsyncMock()
    monkeypatch.setattr(agents.db, "execute", execute)
    session_token = agents.current_session_id.set(uuid4())
    agent_token = agents.current_agent_id.set("researcher")
    try:
        result = await agents.delegate_status(str(child_id))
    finally:
        agents.current_agent_id.reset(agent_token)
        agents.current_session_id.reset(session_token)
        module._live_delegations.pop(child_id, None)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert "Worker running" in result
    assert result.delegation_status == "running"
    assert result.is_error is False
    execute.assert_not_awaited()


def test_system_prompt_teaches_parallel_fan_out():
    from machina.core.agent import SYSTEM_PROMPT

    assert "list_agents" in SYSTEM_PROMPT
    assert "configured concurrency limit queues excess work" in SYSTEM_PROMPT
    assert "Workers inherit your restrictions and cannot approve actions" in SYSTEM_PROMPT
    assert "never delegate to evade an approval" in SYSTEM_PROMPT


def test_delegate_description_advertises_parallel_fan_out():
    import machina.tools.agents  # noqa: F401 — registers delegate on the global registry
    from machina.tools.base import registry

    desc = next(d.description for d in registry.get_tool_definitions() if d.name == "delegate")
    assert "single block" in desc
    assert "additional calls queue" in desc


@pytest.mark.asyncio
async def test_parallel_delegate_fan_out_runs_every_subagent(tmp_path, monkeypatch):
    """One assistant message with three delegate calls runs all three agents."""
    import asyncio
    import copy
    import json
    import uuid
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from machina.core.agent import BaseReActAgent, ReActAgent
    from machina.core.config import config
    from machina.core.models import LLMResponse, Session, StreamEvent
    from machina.permissions.broker import set_approval_handler

    monkeypatch.setenv("MACHINA_SKILLS_DIR", str(tmp_path / "skills"))
    for field in ("memory_enabled", "rag_enabled", "personality_growth_enabled"):
        monkeypatch.setattr(config, field, False)
    session = Session(id=uuid.uuid4(), context_budget=8000)
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    monkeypatch.setattr("machina.core.agent.session_manager.update_activity", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("machina.core.agent.context_manager.add_chunks_batch", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(BaseReActAgent, "_reserve_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(BaseReActAgent, "_finish_usage", AsyncMock())

    async def unexpected_approval(card):
        pytest.fail("fan-out requested approval")

    set_approval_handler(unexpected_approval)

    from machina.core import orchestrator as orchestrator_module
    from machina.tools import agents as agents_module

    ran = []
    all_started = asyncio.Event()

    async def fake_owner(*args):
        return "harness"

    async def fake_delegate(agent, task, **kwargs):
        ran.append(agent)
        if len(ran) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        return SimpleNamespace(status="complete", response=f"{agent} done", session_id=uuid.uuid4())

    monkeypatch.setattr(agents_module.db, "fetchval", fake_owner)
    monkeypatch.setattr(orchestrator_module.orchestrator, "delegate", fake_delegate)

    wanted = [("researcher", "gather facts"), ("coder", "fix the bug"), ("coder", "add tests")]

    class Provider:
        def __init__(self):
            self.calls = []

        async def complete(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    model="fake",
                    usage={"total_tokens": 1},
                    tool_calls=[
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {
                                "name": "delegate",
                                "arguments": json.dumps({"agent": a, "task": t}),
                            },
                        }
                        for i, (a, t) in enumerate(wanted)
                    ],
                )
            return LLMResponse(content="synthesized", model="fake", usage={"total_tokens": 1})

        async def stream_complete(self, **kwargs):
            yield StreamEvent(type="done", response=await self.complete(**kwargs))

    provider = Provider()
    agent = ReActAgent(provider=provider, allowed_tools=["delegate"])
    try:
        response = await agent.run(session.id, "Research, fix, and test", verbose=False)
        assert response.content == "synthesized"
        assert sorted(ran) == ["coder", "coder", "researcher"]
        results = [
            message.get("content", "")
            for message in provider.calls[1]["messages"]
            if message.get("role") == "tool"
        ]
        assert len(results) == 3
        assert any("researcher done" in r for r in results)
    finally:
        set_approval_handler(None)
