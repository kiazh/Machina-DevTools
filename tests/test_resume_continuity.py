"""Resume continuity and large-input token-count hardening."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent import ReActAgent
from machina.core.assembler import PromptAssembler, get_token_count
from machina.core.config import config
from machina.core.context import ContinuityWindow, context_manager
from machina.core.models import ContextChunk, Session
from machina.core.prompt import build_native_turn_messages


def test_native_history_keeps_chronological_roles_and_avoids_duplication():
    history = [
        {"role": "user", "content": "the original question"},
        {"role": "assistant", "content": "the latest assistant answer"},
    ]
    messages = build_native_turn_messages(
        "system",
        "summary of earlier turns",
        history,
        "## Current Query\ncurrent question",
        [],
    )
    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1]["role"] == "user"
    assert "summary of earlier turns" in messages[2]["content"]
    assert messages[3] == {"role": "user", "content": "the original question"}
    assert messages[4] == {
        "role": "assistant",
        "content": "the latest assistant answer",
    }
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].count("current question") == 1


def test_native_history_merges_adjacent_same_role_messages():
    messages = build_native_turn_messages(
        "",
        None,
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer"},
        ],
        "## Current Query\ncontinue",
        [],
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "first\n\nsecond"
    assert messages[1] == {"role": "assistant", "content": "answer"}
    assert messages[-1]["role"] == "user"


def test_native_tail_merges_into_interrupted_prior_user_turn():
    messages = build_native_turn_messages(
        "",
        None,
        [{"role": "user", "content": "unanswered prior question"}],
        "## Current Query\ncurrent question",
        [],
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "unanswered prior question" in messages[0]["content"]
    assert "current question" in messages[0]["content"]


def test_recent_activity_prunes_cold_tool_output_and_keeps_hot_output():
    recent = [
        {"type": "user_message", "payload": {"content": "newest user"}},
        {
            "type": "result",
            "payload": {"status": "ok", "result": "h" * 3000},
        },
        {"type": "user_message", "payload": {"content": "second user"}},
        {
            "type": "result",
            "payload": {"status": "ok", "result": "c" * 9000},
        },
        {"type": "user_message", "payload": {"content": "third user"}},
        {
            "type": "result",
            "payload": {"status": "ok", "result": "o" * 9000},
        },
    ]
    prompt = PromptAssembler(session_budget=100_000).assemble(
        system_prompt="",
        goal=None,
        recent_chunks=recent,
        retrieved_chunks=[],
        query="current",
        skip_conversation=True,
    )
    assert "h" * 2000 in prompt
    assert "h" * 2001 not in prompt
    assert "o" * 2000 not in prompt
    assert "9,000 chars pruned" in prompt


def test_large_token_count_is_chunked_and_does_not_crash():
    count = get_token_count("x" * 300_000)
    assert count > 0
    # The chunked path must still be in the right order of magnitude.
    assert 20_000 < count < 120_000


def test_large_prompt_section_uses_chunked_path():
    huge = ContextChunk(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        agent_id="harness",
        chunk_type="document",
        payload={"text": "y" * 300_000, "source": "huge.md"},
        token_count=75_000,
    )
    prompt = PromptAssembler(session_budget=2000).assemble(
        system_prompt="system",
        goal=None,
        recent_chunks=[],
        retrieved_chunks=[(huge, 1.0)],
        query="summarize",
    )
    assert "## Relevant Context" in prompt
    assert "summarize" in prompt
    assert len(prompt) < 500_000


@pytest.mark.asyncio
async def test_prepare_context_sends_prior_conversation_to_the_model(monkeypatch):
    session_id = uuid.uuid4()
    session = Session(id=session_id, agent_id="harness", context_budget=8000)
    agent = ReActAgent(provider=SimpleNamespace(model="test-model"))

    monkeypatch.setattr(config, "memory_enabled", False)
    monkeypatch.setattr(config, "personality_growth_enabled", False)
    monkeypatch.setattr(config, "rag_enabled", False)
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    monkeypatch.setattr("machina.core.agent.plugin_registry.dispatch", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_continuity_history",
        AsyncMock(
            return_value=ContinuityWindow(
                messages=[
                    {"role": "user", "content": "EARLIER-USER-CONTEXT"},
                    {"role": "assistant", "content": "EARLIER-ASSISTANT-CONTEXT"},
                ],
                summary=None,
                boundary_id=None,
                compacted=False,
            )
        ),
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr("machina.core.agent.context_manager.add_chunk", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text",
        AsyncMock(return_value=[]),
    )

    _, messages = await agent._prepare_context(session_id, "current question")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "EARLIER-USER-CONTEXT"}
    assert messages[2] == {"role": "assistant", "content": "EARLIER-ASSISTANT-CONTEXT"}
    assert messages[-1]["role"] == "user"
    assert "## Current Query" in messages[-1]["content"]
    assert messages[-1]["content"].count("current question") == 1
    assert "EARLIER-USER-CONTEXT" not in messages[-1]["content"]


@pytest.mark.asyncio
async def test_prepare_context_persists_compacted_boundary_and_anchor(monkeypatch):
    session_id = uuid.uuid4()
    session = Session(id=session_id, agent_id="harness", context_budget=8000)
    agent = ReActAgent(provider=SimpleNamespace(model="test-model"))

    monkeypatch.setattr(config, "memory_enabled", False)
    monkeypatch.setattr(config, "personality_growth_enabled", False)
    monkeypatch.setattr(config, "rag_enabled", False)
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    update_state = AsyncMock(return_value={})
    monkeypatch.setattr("machina.core.agent.session_manager.update_state_fields", update_state)
    monkeypatch.setattr("machina.core.agent.plugin_registry.dispatch", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_continuity_history",
        AsyncMock(
            return_value=ContinuityWindow(
                messages=[
                    {"role": "user", "content": "kept user"},
                    {"role": "assistant", "content": "kept assistant"},
                ],
                summary="folded summary",
                boundary_id="boundary-id",
                compacted=True,
            )
        ),
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr("machina.core.agent.context_manager.add_chunk", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text",
        AsyncMock(return_value=[]),
    )

    _, messages = await agent._prepare_context(session_id, "current question")

    update_state.assert_awaited_once_with(
        session_id,
        {
            "compaction_boundary_msg_id": "boundary-id",
            "compaction_summary": "folded summary",
        },
    )
    assert session.state["compaction_boundary_msg_id"] == "boundary-id"
    assert messages[1]["role"] == "user"
    assert messages[2] == {"role": "assistant", "content": "folded summary"}
    assert messages[3] == {"role": "user", "content": "kept user"}
    assert messages[4] == {"role": "assistant", "content": "kept assistant"}


@pytest.mark.db
async def test_context_manager_returns_only_conversation_messages(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.context.db", db_pool)
    session_id = uuid.uuid4()
    try:
        await db_pool.execute(
            "INSERT INTO sessions(id, title) VALUES($1, 'resume continuity test')",
            session_id,
        )
        await db_pool.execute("DELETE FROM context_chunks WHERE session_id = $1", session_id)
        for chunk_type, content in (
            ("user_message", "first"),
            ("tool_call", "tool row"),
            ("assistant_message", "second"),
        ):
            await context_manager.add_chunk(
                session_id=session_id,
                agent_id="harness",
                chunk_type=chunk_type,
                payload={"content": content},
                token_count=1,
            )
        window = await context_manager.get_continuity_history(session_id, context_budget=8000)
        assert [row["role"] for row in window.messages] == ["user", "assistant"]
        assert [row["content"] for row in window.messages] == ["first", "second"]
        assert window.compacted is False
        limited = await context_manager.get_transcript_chunks(
            session_id, limit=2, include_embedding=False
        )
        assert len(limited) == 2
        assert all(chunk.embedding is None for chunk in limited)
    finally:
        await db_pool.execute("DELETE FROM context_chunks WHERE session_id = $1", session_id)
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", session_id)


@pytest.mark.db
async def test_continuity_sawtooth_boundary_is_inclusive(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.context.db", db_pool)
    session_id = uuid.uuid4()
    try:
        await db_pool.execute(
            "INSERT INTO sessions(id, title) VALUES($1, 'sawtooth continuity test')",
            session_id,
        )
        await db_pool.execute("DELETE FROM context_chunks WHERE session_id = $1", session_id)
        for index in range(3):
            for role in ("user", "assistant"):
                await context_manager.add_chunk(
                    session_id=session_id,
                    agent_id="harness",
                    chunk_type=f"{role}_message",
                    payload={"content": f"{role}-{index}"},
                    token_count=1000,
                )
        window = await context_manager.get_continuity_history(session_id, context_budget=9000)
        assert window.compacted is True
        assert [row["content"] for row in window.messages] == ["user-2", "assistant-2"]
        assert "user-0" in (window.summary or "")
        assert window.boundary_id is not None

        resumed = await context_manager.get_continuity_history(
            session_id,
            context_budget=9000,
            boundary_id=window.boundary_id,
            summary=window.summary,
        )
        assert resumed.compacted is False
        assert [row["content"] for row in resumed.messages] == ["user-2", "assistant-2"]
    finally:
        await db_pool.execute("DELETE FROM context_chunks WHERE session_id = $1", session_id)
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
