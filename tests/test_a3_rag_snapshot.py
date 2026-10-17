"""A3 campaign: snapshot-consistent retrieval verification (MCH-009 / F-28).

The embedding-space verdict and the dense candidate query must observe the
same database snapshot, and the verdict query must be a single DISTINCT scan,
not a Python keyset page walk.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import msgpack
import pytest

from machina.rag.embedder import Embedder
from machina.rag.pipeline import RAGConfig, RAGPipeline

DEFAULT_MODEL = "text-embedding-3-small"
OTHER_MODEL = "text-embedding-ada-002"
VECTOR = [1.0] + [0.0] * 1535


class QueryEmbedder(Embedder):
    def __init__(self, model=DEFAULT_MODEL):
        self._model = model

    async def embed(self, text):
        return list(VECTOR)

    async def embed_batch(self, texts):
        return [await self.embed(text) for text in texts]

    @property
    def dimensions(self):
        return len(VECTOR)

    @property
    def model_name(self):
        return self._model


def _row(session_id, model, chunk_type="document"):
    return {
        "id": uuid.uuid4(),
        "session_id": session_id,
        "agent_id": "owner",
        "chunk_type": chunk_type,
        "payload_msgpack": msgpack.packb(
            {"text": "shared query", "metadata": {"embedding_model": model, "embedding_dims": 1536}},
            use_bin_type=True,
        ),
        "token_count": 2,
        "embedding": str(VECTOR),
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "accessed_at": None,
        "similarity": 0.9,
        "rank": 0.9,
    }


class TransactionBoundary:
    """One connection, one repeatable-read transaction, ordered query log."""

    def __init__(self, rows):
        self.session_id = uuid.uuid4()
        self.rows = rows
        self.queries = []
        self.transaction_kwargs = []
        self.in_transaction = False

        boundary = self

        class Connection:
            async def fetch(self, query, *args):
                if "DISTINCT" in query:
                    assert boundary.in_transaction, "verification ran outside the snapshot"
                boundary.queries.append(query)
                if "DISTINCT" in query:
                    scoped = [
                        r
                        for r in boundary.rows
                        if r["session_id"] == args[0] and r["chunk_type"] in args[1]
                    ]
                    unique = {r["payload_msgpack"]: r for r in scoped}
                    return [{"payload_msgpack": p} for p in unique]
                if "embedding <=>" in query:
                    return [
                        r
                        for r in boundary.rows
                        if r["chunk_type"] in args[3] and r["embedding"] is not None
                    ][: args[2]]
                if "ts_rank" in query:
                    return []
                raise AssertionError(f"unexpected query: {query}")

            def transaction(self, **kwargs):
                boundary.transaction_kwargs.append(kwargs)

                class Tx:
                    async def __aenter__(self_inner):
                        boundary.in_transaction = True
                        return self_inner

                    async def __aexit__(self_inner, *exc):
                        boundary.in_transaction = False
                        return False

                return Tx()

        conn = Connection()
        self.conn = conn

        class Acquire:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        self.acquire = lambda *a, **k: Acquire()


def _pipeline():
    return RAGPipeline(
        embedder=QueryEmbedder(), config=RAGConfig(enable_reranking=False)
    )


async def test_verification_and_dense_query_share_one_repeatable_read_snapshot(monkeypatch):
    boundary = TransactionBoundary([_row(None, DEFAULT_MODEL)])
    boundary.rows[0]["session_id"] = boundary.session_id
    monkeypatch.setattr("machina.rag.pipeline.db", boundary)

    results = await _pipeline().search("shared query", boundary.session_id, rerank=False)

    assert results
    assert boundary.transaction_kwargs == [{"isolation": "repeatable_read"}]
    assert any("DISTINCT" in q for q in boundary.queries), "single DISTINCT verdict query"
    assert not any("id >" in q for q in boundary.queries), "keyset paging must be gone"
    distinct_at = next(i for i, q in enumerate(boundary.queries) if "DISTINCT" in q)
    dense_at = next(i for i, q in enumerate(boundary.queries) if "embedding <=>" in q)
    assert distinct_at < dense_at


async def test_mismatch_refuses_before_the_dense_query(monkeypatch):
    boundary = TransactionBoundary(
        [
            {**_row(None, DEFAULT_MODEL), "session_id": None},
            {**_row(None, OTHER_MODEL), "session_id": None},
        ]
    )
    for row in boundary.rows:
        row["session_id"] = boundary.session_id
    monkeypatch.setattr("machina.rag.pipeline.db", boundary)

    with pytest.raises(ValueError, match="embedding space mismatch"):
        await _pipeline().search("shared query", boundary.session_id, rerank=False)
    assert not any("embedding <=>" in q for q in boundary.queries)


async def test_cached_verdict_still_queries_once_per_session(monkeypatch):
    boundary = TransactionBoundary([_row(None, DEFAULT_MODEL)])
    boundary.rows[0]["session_id"] = boundary.session_id
    monkeypatch.setattr("machina.rag.pipeline.db", boundary)

    pipeline = _pipeline()
    await pipeline.search("shared query", boundary.session_id, rerank=False)
    await pipeline.search("another query", boundary.session_id, rerank=False)

    distinct_queries = [q for q in boundary.queries if "DISTINCT" in q]
    assert len(distinct_queries) == 1
