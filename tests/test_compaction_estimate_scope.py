"""Combined #35: auto-compaction estimates the whole compressible context."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest


async def test_compaction_trigger_reads_all_compressible_chunks(monkeypatch):
    import machina.services as services

    all_chunks = AsyncMock(
        return_value=[
            type("Chunk", (), {"chunk_type": "user_message", "token_count": 50})(),
            type("Chunk", (), {"chunk_type": "document", "token_count": 999})(),
        ]
    )
    monkeypatch.setattr(services.context_manager, "get_all_chunks", all_chunks)

    assert await services._compressible_context_tokens(uuid.uuid4()) == 50
    all_chunks.assert_awaited_once()
