"""Combined #38/#40/#42: RAG generation leases.

The shared pipeline used ``id(pipeline)`` as its lifecycle key and incremented
the user count after selection. These tests pin the replacement contract:
acquisition retains the generation inside the selection lock, every acquire
gets an idempotent token, and retirement/shutdown closes only after the last
lease is released.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

IDENTITY = (("reranker",), ("embedding",), (0.0, 0.0))
NEW_IDENTITY = (("reranker", "new"), ("embedding",), (0.0, 0.0))


@pytest.fixture
def shared_pipeline(monkeypatch):
    from machina.tools import rag

    embedder = SimpleNamespace(close=AsyncMock())
    reranker = SimpleNamespace(close=AsyncMock())
    pipeline = SimpleNamespace(_embedder=embedder, _reranker=reranker)
    monkeypatch.setattr(rag, "_rag_pipeline", pipeline)
    monkeypatch.setattr(rag, "_rag_pipeline_identity", IDENTITY)
    monkeypatch.setattr(rag, "_current_pipeline_identity", lambda: IDENTITY)
    monkeypatch.setattr(rag, "_pipeline_users", {})
    monkeypatch.setattr(rag, "_pipeline_leases", {}, raising=False)
    monkeypatch.setattr(rag, "_retired_pipelines", {})
    return rag, pipeline, embedder, reranker, monkeypatch


async def test_acquire_increments_the_lease_while_holding_the_lock(shared_pipeline):
    rag, pipeline, _embedder, _reranker, monkeypatch = shared_pipeline
    observed: list[bool] = []

    class TrackingUsers(dict):
        def __setitem__(self, key, value):
            observed.append(rag._rag_pipeline_lock.locked())
            super().__setitem__(key, value)

    monkeypatch.setattr(rag, "_pipeline_users", TrackingUsers())
    lease = await rag.acquire_rag_pipeline()
    try:
        assert observed == [True], "lease count was published outside the selection lock"
        assert isinstance(lease, rag.RAGLease)
        assert lease.pipeline is pipeline
        assert id(pipeline) not in rag._pipeline_users, "lifecycle still keyed by object id"
        assert lease.token in rag._pipeline_leases
    finally:
        await rag.release_rag_pipeline(lease)


async def test_duplicate_release_cannot_close_a_retired_generation(shared_pipeline):
    rag, _pipeline, embedder, reranker, monkeypatch = shared_pipeline
    first = await rag.acquire_rag_pipeline()
    second = await rag.acquire_rag_pipeline()

    newer = SimpleNamespace(_embedder=SimpleNamespace(close=AsyncMock()), _reranker=None)
    monkeypatch.setattr(rag, "_current_pipeline_identity", lambda: NEW_IDENTITY)
    monkeypatch.setattr(rag, "_build_shared_pipeline", lambda: newer)
    monkeypatch.setattr(rag, "_built_reranker_identity", lambda _pipeline: None)
    assert await rag.get_rag_pipeline() is newer

    await rag.release_rag_pipeline(first)
    embedder.close.assert_not_awaited()
    await rag.release_rag_pipeline(first)  # duplicate token: must be a no-op
    embedder.close.assert_not_awaited()
    reranker.close.assert_not_awaited()

    await rag.release_rag_pipeline(second)
    embedder.close.assert_awaited_once()
    reranker.close.assert_awaited_once()


async def test_close_active_defers_until_the_last_lease_is_released(shared_pipeline):
    rag, _pipeline, embedder, reranker, _monkeypatch = shared_pipeline
    lease = await rag.acquire_rag_pipeline()

    await rag.close_active_rag_pipeline()
    assert rag._rag_pipeline is None
    embedder.close.assert_not_awaited()
    reranker.close.assert_not_awaited()

    await rag.release_rag_pipeline(lease)
    embedder.close.assert_awaited_once()
    reranker.close.assert_awaited_once()
