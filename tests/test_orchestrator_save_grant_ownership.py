"""Evidence tests for the inherited-grant save failure (core-loop review P0-1,
assigned as the delegated-orphan defect in this campaign).

``Orchestrator.delegate`` creates the fresh child session, then (for a full-mode
parent) copies the inherited full-mode grant via ``permission_store.save_grant``.
That write historically ran *outside* the ownership-cleanup scope that only
opened later at the ``agent = None ... try:`` block, so a DB failure in the
grant insert raised past the ``finally`` that archives the child — leaking a
live child session.

These tests fail on the pre-fix code (RED) and pass once the grant write lives
inside the cleanup scope (GREEN).
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent_def import AgentDef
from machina.core.models import AgentResponse, Session


class _OwnedAgent:
    """Agent stand-in on the orchestrator-owned provider path."""

    def __init__(self) -> None:
        self.closes = 0

        async def _close() -> None:
            self.closes += 1

        self.provider = SimpleNamespace(close=_close)
        self._owns_provider = True

    async def run(self, session_id, *_args, **_kwargs):
        return AgentResponse(content="done", tokens_used=3)


_INHERITED_GRANT = {
    "id": uuid.uuid4(),
    "agent_id": "coder",
    "mode": "full",
    "capability": "session",
    "digest": "full-mode-session",
    "expires_at": None,
}


@pytest.fixture
def _stub_delegate_env(monkeypatch):
    from machina.core import orchestrator as module
    from machina.core.usage import usage_store
    from machina.permissions import store as permission_store

    child = Session(id=uuid.uuid4(), agent_id="coder")
    parent = SimpleNamespace(id=uuid.uuid4(), agent_id="coder")
    archived: list[uuid.UUID] = []
    saved_grants: list[dict] = []

    async def fake_archive(session_id: uuid.UUID) -> None:
        archived.append(session_id)

    async def fake_save_grant(grant: dict) -> None:
        saved_grants.append(grant)

    async def fake_summary(session_id, agent_id):
        return {"session": {"chargedTokens": 3}}

    monkeypatch.setattr(
        module.agent_registry, "get", AsyncMock(return_value=AgentDef(name="coder"))
    )
    monkeypatch.setattr(module.session_manager, "create", AsyncMock(return_value=child))
    monkeypatch.setattr(module.session_manager, "archive", AsyncMock(side_effect=fake_archive))
    monkeypatch.setattr(module.session_manager, "get_fresh", AsyncMock(return_value=parent))
    monkeypatch.setattr(
        "machina.core.session_mode.resolve_effective_mode", AsyncMock(return_value="full")
    )
    monkeypatch.setattr(
        permission_store, "list_grants", AsyncMock(return_value=[_INHERITED_GRANT])
    )
    monkeypatch.setattr(permission_store, "save_grant", fake_save_grant)
    monkeypatch.setattr(module.context_manager, "add_chunk", AsyncMock())
    monkeypatch.setattr(usage_store, "summary", fake_summary)
    return SimpleNamespace(
        module=module,
        child=child,
        parent=parent,
        archived=archived,
        saved_grants=saved_grants,
    )


def _build_orch(monkeypatch, stub, agent):
    """Orchestrator on the default (owned-provider) path with stub accounting."""
    orch = stub.module.Orchestrator()

    async def fake_default_agent(*_args, **_kwargs):
        return agent

    monkeypatch.setattr(orch, "_default_agent", fake_default_agent)
    monkeypatch.setattr(orch, "_record_start", AsyncMock(return_value=uuid.uuid4()))
    monkeypatch.setattr(orch, "_record_end", AsyncMock())
    monkeypatch.setattr(orch, "_parent_context", AsyncMock(return_value=""))
    return orch


async def test_save_grant_failure_archives_child_and_surfaces_error(
    monkeypatch, _stub_delegate_env
):
    """A grant-insert DB error must archive the child and propagate, not orphan it."""
    from machina.permissions import store as permission_store

    stub = _stub_delegate_env
    agent = _OwnedAgent()

    async def failing_save_grant(_grant: dict) -> None:
        raise RuntimeError("grant db down")

    monkeypatch.setattr(permission_store, "save_grant", failing_save_grant)
    orch = _build_orch(monkeypatch, stub, agent)

    with pytest.raises(RuntimeError, match="grant db down"):
        await orch.delegate(
            "coder", "work", parent_session_id=stub.parent.id, from_agent="parent"
        )

    assert stub.archived == [stub.child.id]


async def test_save_grant_cancellation_propagates_and_archives_child(
    monkeypatch, _stub_delegate_env
):
    """Cancellation during the grant write must stay a CancelledError."""
    from machina.permissions import store as permission_store

    stub = _stub_delegate_env
    agent = _OwnedAgent()

    async def cancelled_save_grant(_grant: dict) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(permission_store, "save_grant", cancelled_save_grant)
    orch = _build_orch(monkeypatch, stub, agent)

    with pytest.raises(asyncio.CancelledError):
        await orch.delegate(
            "coder", "work", parent_session_id=stub.parent.id, from_agent="parent"
        )

    assert stub.archived == [stub.child.id]


async def test_save_grant_success_path_unchanged(_stub_delegate_env):
    """Green path: inherited grant is copied onto the child, delegation completes."""
    stub = _stub_delegate_env
    agent = _OwnedAgent()
    orch = stub.module.Orchestrator()

    async def fake_default_agent(*_args, **_kwargs):
        return agent

    orch._default_agent = fake_default_agent
    orch._record_start = AsyncMock(return_value=uuid.uuid4())
    orch._record_end = AsyncMock()
    orch._parent_context = AsyncMock(return_value="")

    result = await orch.delegate(
        "coder", "work", parent_session_id=stub.parent.id, from_agent="parent"
    )

    assert result.status == "complete"
    assert result.session_id == stub.child.id
    saved = [grant for grant in stub.saved_grants if grant.get("session_id")]
    assert saved and saved[0]["session_id"] == str(stub.child.id)
