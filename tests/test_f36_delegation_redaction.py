"""F36: delegated-child exception text is redacted before it is persisted or
returned. A child failure can echo a bearer token or a DSN password; the
durable agent_messages row, the parent-facing result, and legacy history reads
must all pass through the shared redaction registry.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent_def import AgentDef
from machina.core.models import AgentResponse, Session

SECRET = "sk-or-v1-abcdef1234567890"
DSN = "postgresql://user:password@host:5432/db"


class _ExplodingAgent:
    """Agent stand-in whose child run raises a secret-bearing exception."""

    def __init__(self) -> None:
        async def _close() -> None:
            return None

        self.provider = SimpleNamespace(close=_close)
        self._owns_provider = True

    async def run(self, *_args, **_kwargs):
        raise RuntimeError(f"child failed: bearer {SECRET} dsn {DSN}")


class _FailedOutcomeAgent(_ExplodingAgent):
    """Typed child failure: returns content instead of raising."""

    async def run(self, *_args, **_kwargs):
        return AgentResponse(
            content=f"child failed Authorization: Bearer {SECRET}", outcome="failed"
        )


@pytest.fixture
def _delegate_env(monkeypatch):
    from machina.core import orchestrator as module

    child = Session(id=uuid.uuid4(), agent_id="coder")
    parent = SimpleNamespace(id=uuid.uuid4(), agent_id="coder")
    monkeypatch.setattr(
        module.agent_registry, "get", AsyncMock(return_value=AgentDef(name="coder"))
    )
    monkeypatch.setattr(module.session_manager, "create", AsyncMock(return_value=child))
    monkeypatch.setattr(module.session_manager, "archive", AsyncMock())
    monkeypatch.setattr(module.session_manager, "get_fresh", AsyncMock(return_value=parent))
    monkeypatch.setattr(
        "machina.core.session_mode.resolve_effective_mode", AsyncMock(return_value="ask")
    )
    monkeypatch.setattr(module.context_manager, "add_chunk", AsyncMock())
    return module, child, parent


@pytest.mark.asyncio
async def test_delegate_error_redacts_persisted_and_returned(monkeypatch, _delegate_env):
    module, _child, parent = _delegate_env
    orch = module.Orchestrator()
    recorded: dict[str, str] = {}

    async def fake_record_end(message_id, *, status, response, tokens):
        recorded["status"] = status
        recorded["response"] = response

    monkeypatch.setattr(orch, "_default_agent", AsyncMock(return_value=_ExplodingAgent()))
    monkeypatch.setattr(orch, "_record_start", AsyncMock(return_value=uuid.uuid4()))
    monkeypatch.setattr(orch, "_record_end", fake_record_end)
    monkeypatch.setattr(orch, "_charged_child_tokens", AsyncMock(return_value=0))
    monkeypatch.setattr(orch, "_parent_context", AsyncMock(return_value=""))

    result = await orch.delegate(
        "coder", "work", parent_session_id=parent.id, from_agent="parent"
    )

    assert result.status == "error"
    for text in (result.response, recorded["response"]):
        assert SECRET not in text
        assert "user:password" not in text
        assert "[REDACTED_TOKEN]" in text


@pytest.mark.asyncio
async def test_typed_child_failure_redacts_parent_chunk_and_result(monkeypatch, _delegate_env):
    """F36: a child that returns outcome='failed' (no exception) is still a
    diagnostic response and must be redacted in the durable row, the parent
    context chunk, and the returned result."""
    module, _child, parent = _delegate_env
    orch = module.Orchestrator()
    recorded: dict[str, str] = {}
    chunks: list[dict] = []

    async def fake_record_end(message_id, *, status, response, tokens):
        recorded["status"] = status
        recorded["response"] = response

    async def fake_add_chunk(**kwargs):
        chunks.append(kwargs)

    monkeypatch.setattr(orch, "_default_agent", AsyncMock(return_value=_FailedOutcomeAgent()))
    monkeypatch.setattr(orch, "_record_start", AsyncMock(return_value=uuid.uuid4()))
    monkeypatch.setattr(orch, "_record_end", fake_record_end)
    monkeypatch.setattr(orch, "_charged_child_tokens", AsyncMock(return_value=0))
    monkeypatch.setattr(orch, "_parent_context", AsyncMock(return_value=""))
    monkeypatch.setattr(module.context_manager, "add_chunk", fake_add_chunk)

    result = await orch.delegate(
        "coder", "work", parent_session_id=parent.id, from_agent="parent"
    )

    assert result.status == "error"
    assert SECRET not in result.response
    assert SECRET not in recorded["response"]
    parent_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "result")
    assert SECRET not in parent_chunk["payload"]["content"]


@pytest.mark.asyncio
async def test_history_redacts_legacy_rows(monkeypatch, _delegate_env):
    module, _child, _parent = _delegate_env
    orch = module.Orchestrator()

    class FakeDb:
        async def fetch(self, *_args, **_kwargs):
            return [
                {
                    "id": uuid.uuid4(),
                    "from_agent": "a",
                    "to_agent": "b",
                    "task": "t",
                    "response": f"old leak Authorization: Bearer {SECRET} {DSN}",
                    "status": "error",
                    "tokens_used": 0,
                    "created_at": None,
                    "completed_at": None,
                }
            ]

    monkeypatch.setattr(module, "db", FakeDb())
    rows = await orch.history(uuid.uuid4())
    assert SECRET not in rows[0]["response"]
    assert "user:password" not in rows[0]["response"]


def test_redaction_failure_is_fail_closed(monkeypatch, _delegate_env):
    module, _child, _parent = _delegate_env

    def _boom(_text):
        raise RuntimeError("redactor down")

    monkeypatch.setattr("machina.memory.redaction.redact_secrets", _boom)
    from machina.memory.redaction import redact_diagnostic

    assert redact_diagnostic(SECRET) == "[redaction unavailable]"
    assert module._redacted(SECRET) == "[redaction unavailable]"
