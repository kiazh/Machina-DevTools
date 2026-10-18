"""Database-backed semantic-cache and migration contracts."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from machina.core.config import config
from machina.core.models import LLMResponse
from machina.core.optimization.cache_context import CacheContext
from machina.core.optimization.canonicalizer import canonicalize
from machina.core.optimization.semantic_cache import SemanticCache
from machina.core.optimization.shadow import (
    ShadowEvent,
    promotion_report,
    prune_shadow_events,
    record_shadow,
)
from machina.core.usage import UsageStore

pytestmark = pytest.mark.db


def _context(session_id):
    return CacheContext.for_auxiliary(
        session_id=session_id,
        agent_id="harness",
        provider="deepseek",
        model="deepseek-chat",
        purpose="compression_summary",
        messages=[{"role": "user", "content": "Summarize this"}],
        tools=[],
        workspace_root="/workspace/test",
    )


def _embedder(vector):
    async def embed(text):
        return vector

    return embed


@pytest.mark.asyncio
async def test_semantic_cache_round_trip_and_session_isolation(db_pool):
    space = f"test:{uuid.uuid4()}"
    session = uuid.uuid4()
    other_session = uuid.uuid4()
    cache = SemanticCache(
        database=db_pool,
        embedder=_embedder([0.01] * 1536),
        embedding_space=lambda: space,
    )
    canonical = canonicalize([{"role": "user", "content": "Summarize this"}])
    try:
        assert await cache.store(
            _context(session),
            canonical,
            LLMResponse(content="cached answer", model="deepseek-chat"),
            ttl_seconds=60,
            max_entries=10,
        )
        hit = await cache.lookup(_context(session), canonical, threshold=0.9)
        assert hit is not None
        assert hit.response.content == "cached answer"
        assert hit.response.cache_source == "semantic"

        miss = await cache.lookup(_context(other_session), canonical, threshold=0.9)
        assert miss is None
    finally:
        await db_pool.execute(
            "DELETE FROM semantic_cache_entries WHERE session_id = ANY($1::uuid[])",
            [session, other_session],
        )


@pytest.mark.asyncio
async def test_usage_store_second_auxiliary_call_is_a_zero_billing_hit(
    db_pool, monkeypatch
):
    space = f"test:{uuid.uuid4()}"
    session = uuid.uuid4()
    provider = type(
        "Provider",
        (),
        {
            "model": "deepseek-chat",
            "complete": AsyncMock(
                return_value=LLMResponse(
                    content="stable answer",
                    model="deepseek-chat",
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 2,
                        "total_tokens": 14,
                    },
                )
            ),
        },
    )()
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    monkeypatch.setattr(
        "machina.core.usage.SemanticCache",
        lambda: SemanticCache(
            database=db_pool,
            embedder=_embedder([0.03] * 1536),
            embedding_space=lambda: space,
        ),
    )
    store = UsageStore()
    messages = [{"role": "user", "content": "Summarize this stable request"}]
    try:
        first = await store.complete_call(
            provider,
            session,
            "harness",
            messages,
            purpose="compression_summary",
            temperature=0.0,
        )
        second = await store.complete_call(
            provider,
            session,
            "harness",
            messages,
            purpose="compression_summary",
            temperature=0.0,
        )
        assert first.content == "stable answer"
        assert second.content == "stable answer"
        assert second.cache_source == "semantic"
        assert second.usage["total_tokens"] == 0
        assert provider.complete.await_count == 1
    finally:
        await db_pool.execute(
            "DELETE FROM semantic_cache_entries WHERE session_id = $1", session
        )


@pytest.mark.asyncio
async def test_shadow_telemetry_round_trip_and_report(db_pool):
    session = uuid.uuid4()
    await record_shadow(
        ShadowEvent(
            subsystem="semantic_cache",
            decision="miss",
            reason_codes=("eligible",),
            mode="shadow",
            session_id=session,
            agent_id="harness",
            provider="deepseek",
            model="deepseek-chat",
            task_class="auxiliary",
            request_hash="abc123",
            tokens_before=100,
            tokens_after=100,
            similarity=0.8,
            latency_ms=2.5,
        ),
        database=db_pool,
    )
    try:
        row = await db_pool.fetchrow(
            "SELECT * FROM optimization_shadow_events WHERE session_id = $1", session
        )
        assert row["decision"] == "miss"
        assert row["reason_codes"] == ["eligible"]
        assert row["tokens_before"] == 100
        report = await promotion_report(days=1, database=db_pool)
        assert "semantic_cache" in report
        assert report["semantic_cache"]["decisions"]["miss"]["events"] >= 1
        # No prompt or response column exists on the table.
        columns = await db_pool.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'optimization_shadow_events'"
        )
        names = {item["column_name"] for item in columns}
        assert "prompt" not in names
        assert "response" not in names
        await db_pool.execute(
            "UPDATE optimization_shadow_events SET created_at = now() - interval '10 days' "
            "WHERE session_id = $1",
            session,
        )
        assert await prune_shadow_events(days=1, database=db_pool) >= 1
    finally:
        await db_pool.execute(
            "DELETE FROM optimization_shadow_events WHERE session_id = $1", session
        )


@pytest.mark.asyncio
async def test_semantic_cache_vacuum_removes_expired_rows(db_pool):
    space = f"test:{uuid.uuid4()}"
    session = uuid.uuid4()
    cache = SemanticCache(
        database=db_pool,
        embedder=_embedder([0.02] * 1536),
        embedding_space=lambda: space,
    )
    canonical = canonicalize([{"role": "user", "content": "Expire this"}])
    try:
        await cache.store(
            _context(session),
            canonical,
            LLMResponse(content="temporary", model="deepseek-chat"),
            ttl_seconds=60,
            max_entries=10,
        )
        await db_pool.execute(
            "UPDATE semantic_cache_entries SET expires_at = now() - interval '1 second' "
            "WHERE session_id = $1",
            session,
        )
        assert await cache.vacuum() >= 1
        assert await cache.lookup(_context(session), canonical, threshold=0.9) is None
    finally:
        await db_pool.execute(
            "DELETE FROM semantic_cache_entries WHERE session_id = $1", session
        )
