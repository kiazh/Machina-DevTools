"""Phase 5 multi-agent: definitions, registry, orchestrator, tools, gateway methods."""

from __future__ import annotations

import os
import uuid

import pytest

from machina.core.agent import ReActAgent
from machina.core.agent_def import BUILTIN_AGENTS, AgentDef, agent_registry
from machina.core.models import AgentResponse
from machina.gateway.server import INVALID_PARAMS, NOT_FOUND
from tests.test_gateway import Harness

TEST_DSN = os.environ.get("MACHINA_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not TEST_DSN, reason="MACHINA_TEST_DATABASE_URL not set")


@pytest.fixture
async def connected_db():
    """Connect the shared db for tests that call the registry/orchestrator directly
    (the gateway Harness connects via initialize, so its tests don't need this)."""
    from machina.db.connection import db

    await db.connect()
    yield
    await db.close()


# ─── definitions / tool allow-list (no DB) ────────────────────────────────────


def test_builtin_agents_have_prompts_and_tool_scopes():
    assert {"harness", "researcher", "coder"} <= set(BUILTIN_AGENTS)
    assert len([agent for agent in BUILTIN_AGENTS.values() if agent.source == "ecc"]) == 71
    assert "write_file" not in BUILTIN_AGENTS["researcher"].tools
    assert "write_file" in BUILTIN_AGENTS["coder"].tools
    assert BUILTIN_AGENTS["harness"].tools == []  # unrestricted


def test_agent_tool_allowlist_filters_definitions():
    import machina.tools  # noqa: F401 — ensure tools are registered

    unrestricted = ReActAgent.__new__(ReActAgent)
    unrestricted.allowed_tools = None
    unrestricted.delegated = False
    restricted = ReActAgent.__new__(ReActAgent)
    restricted.allowed_tools = ["read_file", "web_search"]
    restricted.delegated = False

    all_names = {d.name for d in unrestricted._tool_defs()}
    scoped = {d.name for d in restricted._tool_defs()}
    assert {"read_file", "web_search"} <= all_names
    assert scoped == {"read_file", "web_search"}


def test_delegated_agents_cannot_delegate_or_inspect_parent_workers():
    import machina.tools  # noqa: F401 — registers delegation tools

    child = ReActAgent.__new__(ReActAgent)
    child.agent_id = "researcher"
    child.delegated = True
    child.allowed_tools = None
    names = {definition.name for definition in child._tool_defs()}
    assert "delegate" not in names
    assert "delegate_status" not in names
    assert "list_agents" not in names


# ─── registry / orchestrator / tools (real DB) ────────────────────────────────


class FakeAgent:
    """Deterministic stand-in: echoes the task back as the response."""

    def __init__(self, definition: AgentDef) -> None:
        self.definition = definition

    async def run(self, session_id, user_message, verbose=True) -> AgentResponse:
        return AgentResponse(
            content=f"[{self.definition.name}] handled: {user_message[:60]}",
            tool_calls=[],
            tokens_used=11,
            iterations=1,
        )


async def test_delegate_capacity_is_shared_across_callers(monkeypatch):
    import asyncio

    from machina.core import orchestrator as module

    active = 0
    peak = 0
    capacity = module._DelegateCapacity()
    monkeypatch.setattr(
        module.config, "get", lambda key, *args: 2 if key == "max_delegate_concurrency" else 4
    )

    async def worker():
        nonlocal active, peak
        async with capacity.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(7)))
    assert peak == 2


@needs_db
@pytest.mark.usefixtures("connected_db")
class TestRegistry:
    async def test_save_list_get_delete(self):
        name = f"tester-{uuid.uuid4().hex[:8]}"
        saved = await agent_registry.save(
            AgentDef(name=name, description="a test agent", tools=["read_file"])
        )
        assert saved.source == "db" and saved.tools == ["read_file"]

        fetched = await agent_registry.get(name)
        assert fetched is not None and fetched.description == "a test agent"

        names = {a.name for a in await agent_registry.list()}
        assert name in names and {"harness", "researcher", "coder"} <= names

        assert await agent_registry.delete(name) is True
        assert await agent_registry.get(name) is None

    async def test_stored_agent_overrides_builtin(self):
        try:
            await agent_registry.save(AgentDef(name="researcher", description="overridden"))
            fetched = await agent_registry.get("researcher")
            assert fetched.description == "overridden" and fetched.source == "db"
        finally:
            await agent_registry.delete("researcher")
        # Falls back to the built-in after deletion.
        assert (await agent_registry.get("researcher")).source == "builtin"


@needs_db
@pytest.mark.usefixtures("connected_db")
class TestOrchestrator:
    @pytest.fixture
    def orch(self):
        from machina.core.orchestrator import Orchestrator

        return Orchestrator(agent_factory=FakeAgent)

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("failure", ["database_error", "not_queued"])
    async def test_worker_never_runs_without_durable_running_transition(
        self, monkeypatch, nested, failure
    ):
        from machina.core import orchestrator as module
        from machina.db.connection import db

        ran = False

        class Worker(FakeAgent):
            async def run(self, session_id, user_message, verbose=True):
                nonlocal ran
                ran = True
                return await super().run(session_id, user_message, verbose)

        execute = db.execute

        async def fail_transition(query, *args):
            if "UPDATE agent_messages SET status = $2 WHERE id = $1 AND status = 'queued'" in query:
                if failure == "database_error":
                    raise RuntimeError("transition unavailable")
                return "UPDATE 0"
            return await execute(query, *args)

        monkeypatch.setattr(db, "execute", fail_transition)
        orch = module.Orchestrator(agent_factory=Worker)
        depth_token = module.delegation_depth.set(1 if nested else 0)
        try:
            result = await orch.delegate(
                "researcher", "research", parent_session_id=await _parent_session()
            )
        finally:
            module.delegation_depth.reset(depth_token)

        assert not ran
        assert result.status == "error"

    async def test_delegate_records_message_and_child_session(self, orch):
        parent = await _parent_session()
        result = await orch.delegate(
            "researcher", "find the latest release", parent_session_id=parent
        )
        assert result.status == "complete"
        assert "handled:" in result.response
        assert "researcher" in result.response
        assert result.tokens == 11

        history = await orch.history(parent)
        assert len(history) == 1
        assert history[0]["toAgent"] == "researcher" and history[0]["status"] == "complete"
        assert history[0]["tokens"] == 11

    async def test_unknown_agent_raises(self, orch):
        from machina.core.orchestrator import AgentNotFoundError

        with pytest.raises(AgentNotFoundError):
            await orch.delegate("nobody", "task")

    async def test_cancelled_status_is_recorded(self, orch):
        from machina.db.connection import db

        # Apply the 'cancelled' CHECK migration in case the database was
        # initialized from an older schema revision.
        await db.execute(
            "ALTER TABLE agent_messages DROP CONSTRAINT IF EXISTS agent_messages_status_check"
        )
        await db.execute(
            "ALTER TABLE agent_messages ADD CONSTRAINT agent_messages_status_check "
            "CHECK (status IN ('queued', 'running', 'pending', 'complete', 'error', 'cancelled', 'waiting_approval', 'interrupted'))"
        )
        message_id = await orch._record_start(None, "orchestrator", "coder", "task")
        try:
            await orch._record_end(message_id, status="cancelled", response="cancelled", tokens=0)
            status = await db.fetchval(
                "SELECT status FROM agent_messages WHERE id = $1", message_id
            )
            assert status == "cancelled"
        finally:
            await db.execute("DELETE FROM agent_messages WHERE id = $1", message_id)

    async def test_sequential_passes_prior_results_forward(self, orch):
        captured: list[str] = []

        class Recorder(FakeAgent):
            async def run(self, session_id, user_message, verbose=True):
                captured.append(user_message)
                return await super().run(session_id, user_message, verbose)

        from machina.core.orchestrator import Orchestrator

        orch = Orchestrator(agent_factory=Recorder)
        results = await orch.run_sequential(
            [("researcher", "gather facts"), ("coder", "write code")]
        )
        assert len(results) == 2
        assert "## Results so far" in captured[1]  # second step sees the first's output
        assert "## Results so far" not in captured[0]

    async def test_parallel_runs_all(self, orch):
        results = await orch.run_parallel([("researcher", "a"), ("coder", "b"), ("harness", "c")])
        assert {r.agent for r in results} == {"researcher", "coder", "harness"}
        assert all(r.status == "complete" for r in results)

    async def test_parallel_is_bounded_into_waves(self, orch, monkeypatch):
        import asyncio
        from types import SimpleNamespace

        active = 0
        peak = 0

        async def fake_delegate(agent, task, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(
                agent=agent,
                task=task,
                response="ok",
                session_id=uuid.uuid4(),
                tokens=0,
                iterations=0,
                status="complete",
                needs_approval=[],
            )

        monkeypatch.setattr(orch, "delegate", fake_delegate)
        results = await orch.run_parallel(
            [("researcher", str(index)) for index in range(6)],
            max_concurrency=2,
        )
        assert len(results) == 6
        assert peak == 2

    async def test_delegate_limit_is_shared_across_parallel_batches(self, monkeypatch, orch):
        import asyncio

        from machina.core.config import config
        from machina.core.orchestrator import Orchestrator

        parent = await _parent_session()
        monkeypatch.setattr(config, "max_delegate_concurrency", 2)
        active = 0
        peak = 0
        both_running = asyncio.Event()
        release = asyncio.Event()

        class SlowAgent:
            def __init__(self, definition):
                self.definition = definition

            async def run(self, session_id, user_message, verbose=True):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_running.set()
                try:
                    await release.wait()
                finally:
                    active -= 1
                return AgentResponse(content="done", tokens_used=1, iterations=1)

        first = Orchestrator(agent_factory=SlowAgent)
        second = Orchestrator(agent_factory=SlowAgent)
        batches = asyncio.ensure_future(
            asyncio.gather(
                first.run_parallel(
                    [("researcher", str(i)) for i in range(3)], parent_session_id=parent
                ),
                second.run_parallel(
                    [("researcher", str(i)) for i in range(3)], parent_session_id=parent
                ),
            )
        )
        await asyncio.wait_for(both_running.wait(), timeout=5)

        async def queued_batch_visible():
            while True:
                history = await first.history(parent)
                if len(history) >= 3:
                    return history
                await asyncio.sleep(0.01)

        queued_history = await asyncio.wait_for(queued_batch_visible(), timeout=5)
        assert sum(entry["status"] == "running" for entry in queued_history) == 2
        assert sum(entry["status"] == "queued" for entry in queued_history) >= 1
        release.set()
        results = await batches
        assert all(result.status == "complete" for batch in results for result in batch)
        assert peak == 2
        history = await first.history(parent)
        assert len(history) == 6
        assert all(entry["childSessionId"] for entry in history)


@needs_db
@pytest.mark.usefixtures("connected_db")
class TestDelegateTool:
    async def test_remote_worker_survives_status_check_and_crash_becomes_interrupted(self):
        import asyncio
        import sys

        from machina.core.exceptions import ToolError
        from machina.db.connection import db
        from machina.tools import agents

        parent = await _parent_session()
        child_script = '''
import asyncio, logging, os, re, sys, uuid
from machina.core import orchestrator as module
from machina.db.connection import db
from machina.tools import agents
class Worker:
    def __init__(self, definition): pass
    async def run(self, session_id, user_message, verbose=True):
        await asyncio.Event().wait()
async def main():
    db.dsn = os.environ["MACHINA_TEST_DATABASE_URL"]
    await db.connect()
    module.orchestrator._agent_factory = Worker
    agents.current_session_id.set(uuid.UUID(sys.argv[1]))
    agents.current_agent_id.set("harness")
    receipt = await agents.delegate("researcher", "remote task", wait_seconds=1)
    child = re.search(r"Child session: ([0-9a-f-]{36})", str(receipt)).group(1)
    print("READY " + child, flush=True)
    await asyncio.Event().wait()
logging.disable(logging.CRITICAL)
asyncio.run(main())
'''
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            child_script,
            str(parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        try:
            line = (await asyncio.wait_for(process.stdout.readline(), timeout=15)).decode().strip()
            assert line.startswith("READY "), line
            child_id = uuid.UUID(line.removeprefix("READY "))
            assert "Worker running" in await agents.delegate_status(str(child_id))
            with pytest.raises(ToolError, match="already running"):
                await agents.delegate(
                    "researcher", "overlapping task", session_id=str(child_id), wait_seconds=1
                )
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)
            # The actual lease expires naturally; move it into the past in the
            # isolated test DB to avoid a multi-minute sleep in this test.
            await db.execute(
                "UPDATE sessions SET claim_expires_at = now() - interval '1 second' WHERE id = $1",
                child_id,
            )
            status = await agents.delegate_status(str(child_id))
            assert status.delegation_status == "interrupted"
            assert "resume" in status.lower()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_cancellation_during_cleanup_releases_child_claim(self, monkeypatch):
        import asyncio
        import re

        from machina.core import orchestrator as orch_mod
        from machina.core import runtime, turns
        from machina.tools import agents

        cleanup_started = asyncio.Event()
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        owner = runtime.RuntimeServices()
        end_turn = turns.end_turn
        cleanup_calls = 0

        async def blocked_cleanup(agent):
            nonlocal cleanup_calls
            cleanup_calls += 1
            cleanup_started.set()
            if cleanup_calls == 1:
                await asyncio.Event().wait()

        async def slow_release(session_id, token):
            release_started.set()
            await allow_release.wait()
            return await end_turn(session_id, token)

        monkeypatch.setattr(runtime, "runtime_services", owner)
        monkeypatch.setattr(turns, "end_turn", slow_release)
        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", FakeAgent)
        monkeypatch.setattr(orch_mod, "join_agent_background_tasks", blocked_cleanup)
        parent = await _parent_session()
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        worker = None
        try:
            receipt = await agents.delegate("researcher", "research", wait_seconds=1)
            child = uuid.UUID(re.search(r"Child session: ([0-9a-f-]+)", receipt).group(1))
            assert cleanup_started.is_set()
            worker = orch_mod._live_delegations[child]
            worker.cancel()
            await asyncio.wait_for(release_started.wait(), timeout=5)
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            assert await turns.try_begin_turn(child) is None
            allow_release.set()
            await asyncio.wait_for(asyncio.gather(*list(owner._essential_background)), timeout=5)
            token = await turns.try_begin_turn(child)
            assert token is not None
            assert await end_turn(child, token)
        finally:
            allow_release.set()
            if worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await asyncio.gather(*list(owner._essential_background), return_exceptions=True)
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_finished_worker_remains_running_until_session_claim_released(self, monkeypatch):
        import asyncio
        import re

        from machina.core import orchestrator as orch_mod
        from machina.core import turns
        from machina.tools import agents

        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        end_turn = turns.end_turn

        async def slow_release(session_id, token):
            release_started.set()
            await allow_release.wait()
            return await end_turn(session_id, token)

        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", FakeAgent)
        monkeypatch.setattr(turns, "end_turn", slow_release)
        parent = await _parent_session()
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        try:
            receipt = await agents.delegate("researcher", "research", wait_seconds=1)
            child = uuid.UUID(re.search(r"Child session: ([0-9a-f-]+)", receipt).group(1))
            assert release_started.is_set()
            assert receipt.delegation_status == "running"
            assert "Worker running" in await agents.delegate_status(str(child))
            allow_release.set()
            tasks = list(agents._background_delegations)
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            assert "Worker complete" in await agents.delegate_status(str(child))
        finally:
            allow_release.set()
            await asyncio.gather(*list(agents._background_delegations), return_exceptions=True)
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_lost_child_claim_interrupts_worker_without_cancelling_parent(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from machina.core import orchestrator as orch_mod
        from machina.core import turns
        from machina.tools import agents

        stopped = asyncio.Event()

        class Worker(FakeAgent):
            async def run(self, session_id, user_message, verbose=True):
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        original = turns.TurnOwnership.start_renewal

        def start_fast(self, interval_s=60.0, ttl_s=turns.TURN_TTL_SECONDS, on_lost=None):
            return original(self, interval_s=0.01, ttl_s=ttl_s, on_lost=on_lost)

        monkeypatch.setattr(turns.TurnOwnership, "start_renewal", start_fast)
        monkeypatch.setattr(turns, "renew_turn", AsyncMock(return_value=False))
        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", Worker)
        parent = await _parent_session()
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        try:
            result = await asyncio.wait_for(
                agents.delegate("researcher", "research", wait_seconds=5), timeout=5
            )
            assert result.delegation_status == "interrupted"
            assert result.is_error is True
            assert stopped.is_set()
        finally:
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_shutdown_marks_background_worker_interrupted_and_resumable(self, monkeypatch):
        import asyncio
        import re

        from machina.core import orchestrator as orch_mod
        from machina.core import runtime
        from machina.core.cleanup import cancel_and_join
        from machina.tools import agents

        class Worker(FakeAgent):
            async def run(self, session_id, user_message, verbose=True):
                await asyncio.Event().wait()

        owner = runtime.RuntimeServices()
        monkeypatch.setattr(runtime, "runtime_services", owner)
        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", Worker)
        parent = await _parent_session()
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        try:
            receipt = await agents.delegate("researcher", "research", wait_seconds=1)
            child = uuid.UUID(re.search(r"Child session: ([0-9a-f-]+)", receipt).group(1))
            owner._shutting_down = True
            assert await cancel_and_join(list(owner._background), 5, "test shutdown") == []
            status = await agents.delegate_status(str(child))
            assert "Worker interrupted" in status
            assert "resume" in status.lower()
            owner._shutting_down = False
            monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", FakeAgent)
            followup = await agents.delegate(
                "researcher", "resume", session_id=str(child), wait_seconds=5
            )
            assert followup.delegation_status == "complete"
        finally:
            for task in list(owner._background):
                task.cancel()
            await asyncio.gather(*owner._background, return_exceptions=True)
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_background_worker_status_and_followup_use_the_same_child(self, monkeypatch):
        import asyncio
        import re

        from machina.core import orchestrator as orch_mod
        from machina.core.exceptions import ToolError
        from machina.tools import agents

        release = asyncio.Event()
        release_followup = asyncio.Event()
        calls = []

        class Worker(FakeAgent):
            async def run(self, session_id, user_message, verbose=True):
                calls.append(session_id)
                if len(calls) == 1:
                    await release.wait()
                else:
                    await release_followup.wait()
                return AgentResponse(content="worker result", tokens_used=1, iterations=1)

        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", Worker)
        parent = await _parent_session()
        session_token = agents.current_session_id.set(parent)
        agent_token = agents.current_agent_id.set("harness")
        try:
            receipt = await agents.delegate("researcher", "research", wait_seconds=1)
            child = uuid.UUID(re.search(r"Child session: ([0-9a-f-]+)", receipt).group(1))
            assert receipt.delegation_status == "running"
            assert "Worker running" in await agents.delegate_status(str(child))
            # Simulate a status request served by another process, which has
            # no access to this process's live-task registry.
            live_task = orch_mod._live_delegations.pop(child)
            try:
                assert "Worker running" in await agents.delegate_status(str(child))
                with pytest.raises(ToolError, match="already running"):
                    await agents.delegate(
                        "researcher", "overlapping follow-up", session_id=str(child), wait_seconds=1
                    )
            finally:
                orch_mod._live_delegations[child] = live_task
            release.set()

            async def completed():
                while True:
                    status = await agents.delegate_status(str(child))
                    if "Worker complete" in status:
                        return status
                    await asyncio.sleep(0.01)

            assert "worker result" in await asyncio.wait_for(completed(), timeout=5)
            followup = await agents.delegate(
                "researcher", "follow up", session_id=str(child), wait_seconds=1
            )
            assert followup.delegation_status == "running"
            assert "Worker running" in await agents.delegate_status(str(child))
            release_followup.set()
            assert "Worker complete" in await asyncio.wait_for(completed(), timeout=5)
            assert calls == [child, child]
        finally:
            release.set()
            release_followup.set()
            await asyncio.gather(*list(agents._background_delegations), return_exceptions=True)
            agents.current_agent_id.reset(agent_token)
            agents.current_session_id.reset(session_token)

    async def test_delegate_tool_uses_orchestrator(self, monkeypatch):
        import machina.tools  # noqa: F401
        from machina.core import orchestrator as orch_mod
        from machina.tools.base import registry

        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", FakeAgent)
        out = await registry.execute("delegate", agent="researcher", task="summarize the readme")
        assert "[researcher responded]" in out
        assert "handled:" in out
        assert "researcher" in out

    async def test_delegate_tool_unknown_agent(self):
        import machina.tools  # noqa: F401
        from machina.core.exceptions import ToolError
        from machina.tools.base import registry

        with pytest.raises(ToolError, match="no agent named"):
            await registry.execute("delegate", agent="ghost", task="x")

    async def test_list_agents_tool(self):
        import machina.tools  # noqa: F401
        from machina.tools.base import registry

        out = await registry.execute("list_agents")
        assert "researcher:" in out and "coder:" in out


@needs_db
class TestGatewayAgents:
    @pytest.fixture
    async def h(self, monkeypatch):
        from machina.core import orchestrator as orch_mod

        monkeypatch.setattr(orch_mod.orchestrator, "_agent_factory", FakeAgent)
        harness = Harness()
        assert "result" in await harness.call("initialize")
        yield harness
        await harness.gateway.close()

    async def test_list_save_delete(self, h):
        name = f"gw-agent-{uuid.uuid4().hex[:8]}"
        listed = (await h.call("agents.list"))["result"]["agents"]
        assert {"harness", "researcher", "coder"} <= {a["name"] for a in listed}

        saved = (
            await h.call("agents.save", {"name": name, "description": "x", "tools": ["read_file"]})
        )["result"]
        assert saved["agent"]["name"] == name and saved["agent"]["tools"] == ["read_file"]

        assert (await h.call("agents.delete", {"name": name}))["result"] == {"deleted": True}
        assert (await h.call("agents.delete", {"name": name}))["error"]["code"] == NOT_FOUND

    async def test_cannot_delete_builtin_or_save_builtin_name(self, h):
        assert (await h.call("agents.delete", {"name": "harness"}))["error"][
            "code"
        ] == INVALID_PARAMS
        assert (await h.call("agents.save", {"name": "harness"}))["error"]["code"] == INVALID_PARAMS
        assert (await h.call("agents.save", {"name": "coder"}))["error"]["code"] == INVALID_PARAMS

    async def test_run_sequential_and_history(self, h):
        parent = (await h.call("session.create", {"title": "orchestration"}))["result"]["session"][
            "id"
        ]
        run = await h.call(
            "agents.run",
            {
                "sessionId": parent,
                "mode": "sequential",
                "steps": [
                    {"agent": "researcher", "task": "research"},
                    {"agent": "coder", "task": "implement"},
                ],
            },
        )
        results = run["result"]["results"]
        assert [r["agent"] for r in results] == ["researcher", "coder"]
        assert all(r["status"] == "complete" for r in results)

        history = (await h.call("agents.history", {"sessionId": parent}))["result"]["messages"]
        assert len(history) == 2

    async def test_run_validates_steps_and_mode(self, h):
        assert (await h.call("agents.run", {"steps": []}))["error"]["code"] == INVALID_PARAMS
        bad = await h.call("agents.run", {"steps": [{"agent": "x"}]})
        assert bad["error"]["code"] == INVALID_PARAMS
        bad_mode = await h.call(
            "agents.run", {"mode": "race", "steps": [{"agent": "harness", "task": "t"}]}
        )
        assert bad_mode["error"]["code"] == INVALID_PARAMS

    async def test_run_unknown_agent(self, h):
        response = await h.call("agents.run", {"steps": [{"agent": "ghost", "task": "t"}]})
        assert response["error"]["code"] == NOT_FOUND


async def _parent_session() -> uuid.UUID:
    from machina.core.session import session_manager

    return (await session_manager.create(title="parent")).id
