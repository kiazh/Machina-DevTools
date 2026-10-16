"""Token-efficiency integration: oversized tool results become bounded views.

End-to-end through the ReAct loop: a >1000-char tool result is replaced with a
reducer view that discloses omissions, carries an artifact handle when the
store retained the original, and survives later condensation passes.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent import BaseReActAgent, ReActAgent
from machina.core.config import config
from machina.core.models import LLMResponse, Session, StreamEvent
from machina.core.reducers import VIEW_MARKER
from machina.permissions.broker import set_approval_handler


def _wire_session(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    monkeypatch.setattr(config, "workspace_root", str(tmp_path / "work"))
    monkeypatch.setattr(config, "execution_mode", "ask")
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
    return session


def _read_file_provider(calls_holder):
    class Provider:
        async def complete(self, **kwargs):
            calls_holder.append(copy.deepcopy(kwargs))
            if len(calls_holder) == 1:
                return LLMResponse(
                    content="",
                    model="fake",
                    usage={"total_tokens": 1},
                    tool_calls=[
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "huge.txt"}),
                            },
                        }
                    ],
                )
            return LLMResponse(content="done", model="fake", usage={"total_tokens": 1})

        async def stream_complete(self, **kwargs):
            yield StreamEvent(type="done", response=await self.complete(**kwargs))

    return Provider()


@pytest.mark.asyncio
async def test_no_store_keeps_legacy_blind_cap(tmp_path, monkeypatch):
    """Without a durable original, nothing lossy is published (no claim)."""
    session = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "huge.txt").write_text("Z" * 5000)
    calls: list = []

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=_read_file_provider(calls), allowed_tools=["read_file"])
    try:
        response = await agent.run(session.id, "Dump the file", verbose=False)
        assert response.content == "done"
        tool_messages = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        content = tool_messages[0]["content"]
        # Blind cap kept, but the lost bytes are explicit and unrecoverable.
        assert content.startswith("Z")
        assert len(content) <= 1200  # cap + bounded no_artifact marker
        assert "no_artifact" in content
        assert VIEW_MARKER not in content
        assert "artifact_read" not in content
    finally:
        set_approval_handler(None)


@pytest.mark.asyncio
async def test_view_carries_artifact_handle_when_store_retains(tmp_path, monkeypatch):
    session = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "huge.txt").write_text("Q" * 5000)
    calls: list = []

    class FakeStore:
        async def store(self, kind, content, *, session_id, agent_id, status=None, **kwargs):
            assert kind == "tool:read_file"
            assert content == "Q" * 5000  # verbatim original retained
            # B0/G3: the agent call surface stamps the build version.
            assert str(kwargs.get("source_version", "")).startswith("machina:")
            return {"artifact_id": "hand-1", "kind": kind, "content_bytes": len(content)}

    monkeypatch.setattr("machina.core.tool_views.artifact_store", FakeStore())

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=_read_file_provider(calls), allowed_tools=["read_file"])
    try:
        await agent.run(session.id, "Dump the file", verbose=False)
        content = [m for m in calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
        assert "hand-1" in content
        assert "artifact_read" in content
        assert VIEW_MARKER in content
        assert len(content) <= 1000
    finally:
        set_approval_handler(None)


@pytest.mark.asyncio
async def test_reducers_disabled_keeps_cap_plus_handle(tmp_path, monkeypatch):
    """With the store available, truncated results still carry recovery."""
    session = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "huge.txt").write_text("W" * 5000)
    calls: list = []
    config.set("reducers_enabled", False)

    class FakeStore:
        async def store(self, kind, content, *, session_id, agent_id, status=None, **kwargs):
            return {"artifact_id": "hand-w", "kind": kind, "content_bytes": len(content)}

    monkeypatch.setattr("machina.core.tool_views.artifact_store", FakeStore())

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=_read_file_provider(calls), allowed_tools=["read_file"])
    try:
        await agent.run(session.id, "Dump the file", verbose=False)
        content = [m for m in calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
        assert content.startswith("W" * 1000)
        assert VIEW_MARKER not in content
        assert "hand-w" in content and "artifact_read" in content
    finally:
        set_approval_handler(None)


@pytest.mark.asyncio
async def test_reducer_failure_falls_back_to_blind_cap(tmp_path, monkeypatch):
    """A broken reducer must never break the turn or fabricate a handle."""
    session = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "huge.txt").write_text("E" * 5000)
    calls: list = []

    def broken_reducer(text, *, budget_chars, artifact_id=None, source_version=None):
        raise RuntimeError("reducer exploded")

    monkeypatch.setattr("machina.core.tool_views.reduce_tool_result", broken_reducer)

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=_read_file_provider(calls), allowed_tools=["read_file"])
    try:
        response = await agent.run(session.id, "Dump the file", verbose=False)
        assert response.content == "done"
        content = [m for m in calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
        # No retained original in this environment: cap + explicit unrecoverable
        # marker, never a fabricated handle.
        assert content.startswith("E")
        assert "no_artifact" in content
        assert "artifact_read" not in content
    finally:
        set_approval_handler(None)


@pytest.mark.asyncio
async def test_store_failure_keeps_legacy_cap(tmp_path, monkeypatch):
    """Store refusal/failure marks the loss unrecoverable (C06)."""
    session = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "huge.txt").write_text("B" * 5000)
    calls: list = []

    class FailingStore:
        async def store(self, *args, **kwargs):
            return None  # store refused/unavailable

    monkeypatch.setattr("machina.core.tool_views.artifact_store", FailingStore())

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=_read_file_provider(calls), allowed_tools=["read_file"])
    try:
        await agent.run(session.id, "Dump the file", verbose=False)
        content = [m for m in calls[1]["messages"] if m.get("role") == "tool"][0]["content"]
        assert content.startswith("B")
        assert len(content) <= 1200  # cap + bounded no_artifact marker
        assert "no_artifact" in content
        assert VIEW_MARKER not in content
        assert "artifact_read" not in content
    finally:
        set_approval_handler(None)
