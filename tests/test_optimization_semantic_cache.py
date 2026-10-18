"""Semantic-cache repository and response-safety contracts (no database)."""

from __future__ import annotations

import uuid

import pytest

from machina.core.models import LLMResponse
from machina.core.optimization.cache_context import CacheContext
from machina.core.optimization.canonicalizer import canonicalize
from machina.core.optimization.semantic_cache import (
    SemanticCache,
    normalize_embedding,
    valid_response,
)


def _context(session_id=None):
    return CacheContext.for_auxiliary(
        session_id=session_id or uuid.uuid4(),
        agent_id="harness",
        provider="deepseek",
        model="deepseek-chat",
        purpose="compression_summary",
        messages=[{"role": "user", "content": "Summarize this"}],
        tools=[],
    )


class FakeDb:
    connected = True

    def __init__(self, rows=None):
        self.rows = rows or []
        self.fetch_calls = []
        self.execute_calls = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return self.rows

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"


def test_normalize_embedding_pads_and_rejects_zero():
    assert normalize_embedding([1.0, 2.0])[:2] == [1.0, 2.0]
    assert len(normalize_embedding([1.0, 2.0])) == 1536
    assert normalize_embedding([0.0] * 1536) is None
    assert normalize_embedding([float("nan")]) is None
    assert normalize_embedding([1.0] * 1600) is not None
    assert len(normalize_embedding([1.0] * 1600)) == 1536


def test_valid_response_rejects_tools_empty_truncated_and_secrets():
    assert valid_response(LLMResponse(content="ok", model="m")) is True
    assert valid_response(LLMResponse(content="", model="m")) is False
    assert (
        valid_response(LLMResponse(content="ok", model="m", tool_calls=[{"id": "1"}]))
        is False
    )
    assert (
        valid_response(
            LLMResponse(content="ok", model="m", raw={"finish_reason": "length"})
        )
        is False
    )
    assert (
        valid_response(
            LLMResponse(content="ok", model="m", raw={"stop_reason": "max_tokens"})
        )
        is False
    )
    assert (
        valid_response(
            LLMResponse(
                content="token sk-abcdefghijklmnopqrstuvwxyz12345",
                model="m",
            )
        )
        is False
    )
    assert (
        valid_response(LLMResponse(content="contact someone@example.com", model="m"))
        is False
    )


@pytest.mark.asyncio
async def test_lookup_returns_semantic_hit_and_is_scope_filtered():
    session = uuid.uuid4()
    context = _context(session)
    canonical = canonicalize([{"role": "user", "content": "Summarize this"}])
    entry_id = uuid.uuid4()
    db = FakeDb(
        rows=[
            {
                "id": entry_id,
                "response_content": "short answer",
                "response_model": "deepseek-chat",
                "similarity": 0.99,
            }
        ]
    )

    async def embed(text):
        return [1.0] * 1536

    cache = SemanticCache(
        database=db,
        embedder=embed,
        embedding_space=lambda: "local:test:1536",
    )
    hit = await cache.lookup(context, canonical, threshold=0.975)
    assert hit is not None
    assert hit.response.content == "short answer"
    assert hit.response.cache_hit is True
    assert hit.response.cache_source == "semantic"
    assert hit.response.usage["total_tokens"] == 0
    assert hit.similarity == 0.99
    # Scope is an exact SQL parameter; no vector comparison can cross it.
    assert db.fetch_calls
    assert db.fetch_calls[0][1][1]
    assert db.fetch_calls[0][1][2] == "local:test:1536"


@pytest.mark.asyncio
async def test_lookup_below_threshold_is_a_miss():
    db = FakeDb(
        rows=[
            {
                "id": uuid.uuid4(),
                "response_content": "answer",
                "response_model": "m",
                "similarity": 0.5,
            }
        ]
    )

    async def embed(text):
        return [1.0] * 1536

    cache = SemanticCache(database=db, embedder=embed, embedding_space=lambda: "space")
    assert (
        await cache.lookup(
            _context(),
            canonicalize([{"role": "user", "content": "x"}]),
            threshold=0.975,
        )
        is None
    )


@pytest.mark.asyncio
async def test_store_uses_parameterized_scope_and_never_stores_prompt_text():
    db = FakeDb()
    context = _context()
    canonical = canonicalize(
        [{"role": "user", "content": "a private prompt that must not be stored"}]
    )
    response = LLMResponse(content="public answer", model="deepseek-chat")

    async def embed(text):
        return [0.25] * 1536

    cache = SemanticCache(database=db, embedder=embed, embedding_space=lambda: "space")
    assert await cache.store(
        context, canonical, response, ttl_seconds=60, max_entries=10
    )
    insert_call = next(call for call in db.execute_calls if "INSERT INTO" in call[0])
    args = insert_call[1]
    assert response.content in args
    assert canonical.text not in args
    assert "private prompt" not in " ".join(str(arg) for arg in args)
    assert args[1] == context.session_id
    assert args[5] == "auxiliary"


@pytest.mark.asyncio
async def test_store_rejects_invalid_or_secret_response():
    db = FakeDb()

    async def embed(text):
        return [0.25] * 1536

    cache = SemanticCache(database=db, embedder=embed, embedding_space=lambda: "space")
    context = _context()
    canonical = canonicalize([{"role": "user", "content": "x"}])
    assert not await cache.store(
        context,
        canonical,
        LLMResponse(content="sk-abcdefghijklmnopqrstuvwxyz12345", model="m"),
        ttl_seconds=60,
        max_entries=10,
    )
    assert not await cache.store(
        context,
        canonical,
        LLMResponse(content="ok", model="m", tool_calls=[{"id": "1"}]),
        ttl_seconds=60,
        max_entries=10,
    )


@pytest.mark.asyncio
async def test_invalidate_builds_bounded_where_clause():
    db = FakeDb()
    cache = SemanticCache(
        database=db, embedder=lambda text: None, embedding_space=lambda: "s"
    )
    session = uuid.uuid4()
    assert await cache.invalidate(session_id=session) == 1
    query, args = db.execute_calls[0]
    assert "session_id = $1" in query
    assert args == (session,)
