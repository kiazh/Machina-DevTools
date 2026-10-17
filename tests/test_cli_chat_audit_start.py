"""Evidence tests for P0-18: headless ``machina chat`` audit ownership.

The one-shot CLI chat path never called ``audit_persistence.start()``, so every
audit event this transport emitted (including the run's start record) hit the
documented "dropped before the writer started" path and was lost. The fix owns
the audit writer for the duration of the run and emits a ``chat.start`` record
after the turn claim is acquired but BEFORE the agent executes — a crash during
the run still leaves forensic evidence of the attempted turn.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import machina.cli as cli
from machina.core import turns
from machina.core.models import AgentResponse, StreamEvent

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_local_claims():
    turns._local_reset_for_tests()
    yield
    turns._local_reset_for_tests()


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        provider="openrouter",
        model="mock",
        title="audit",
        execution_mode="ask",
    )


class _Agent:
    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.provider = SimpleNamespace(model="mock")

    async def run_stream(self, session_id, message, verbose=False):
        self.order.append("run")
        if self.fail:
            raise RuntimeError("agent crashed before producing evidence")
        yield StreamEvent(type="text", content="Hello")
        yield StreamEvent(
            type="done",
            response=AgentResponse(content="Hello", tokens_used=1, iterations=1),
        )


def _wire(monkeypatch, session, agent, order, submitted):
    def fake_start():
        order.append("audit-start")

    async def fake_stop():
        order.append("audit-stop")

    def fake_submit(entry):
        order.append(f"audit-submit:{entry.get('event')}")
        submitted.append(entry)

    from machina.observability.audit import audit_persistence

    monkeypatch.setattr(audit_persistence, "start", fake_start)
    monkeypatch.setattr(audit_persistence, "stop", fake_stop)
    monkeypatch.setattr(audit_persistence, "submit", fake_submit)
    monkeypatch.setattr(cli.db, "connect", AsyncMock())
    monkeypatch.setattr(cli.db, "close", AsyncMock())
    monkeypatch.setattr(cli.db, "initialize_runtime_schema", AsyncMock())
    monkeypatch.setattr(cli.session_manager, "get", AsyncMock(return_value=session))
    monkeypatch.setattr(
        "machina.core.agent_factory.build_agent_for_session",
        AsyncMock(return_value=agent),
    )
    monkeypatch.setattr("machina.core.agent_factory.close_agent_provider", AsyncMock())
    monkeypatch.setattr(cli, "drain_background_tasks", AsyncMock())


def _invoke_chat(session):
    return runner.invoke(
        cli.app, ["chat", "hello", "--session", str(session.id), "--quiet"]
    )


def test_chat_starts_audit_writer_and_records_start_before_run(monkeypatch):
    session = _session()
    order: list[str] = []
    submitted: list[dict] = []
    agent = _Agent(order)
    _wire(monkeypatch, session, agent, order, submitted)

    result = _invoke_chat(session)

    assert result.exit_code == cli.EXIT_OK, result.output
    starts = [entry for entry in submitted if entry.get("event") == "chat.start"]
    assert len(starts) == 1, "exactly one start record per one-shot run"
    record = starts[0]
    assert record["session_id"] == str(session.id)
    assert record["transport"] == "cli"
    assert record["mode"] == "ask"
    assert record["turn_id"]
    # Writer ownership spans the run; the record is submitted BEFORE executing.
    assert (
        order.index("audit-start")
        < order.index("audit-submit:chat.start")
        < order.index("run")
    )
    assert order.index("run") < order.index("audit-stop")


def test_chat_start_record_survives_a_crash_during_the_run(monkeypatch):
    session = _session()
    order: list[str] = []
    submitted: list[dict] = []
    agent = _Agent(order, fail=True)
    _wire(monkeypatch, session, agent, order, submitted)

    result = _invoke_chat(session)

    assert result.exit_code != cli.EXIT_OK
    starts = [entry for entry in submitted if entry.get("event") == "chat.start"]
    assert starts, "a crash after the attempt starts must still leave a start record"
    assert order.index("audit-submit:chat.start") < order.index("run")
    assert order.index("run") < order.index("audit-stop"), "audit ownership still drains"
