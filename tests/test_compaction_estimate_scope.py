"""Combined #35: auto-compaction estimates the whole compressible context."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest


async def test_compaction_trigger_reads_all_compressible_chunks(monkeypatch):
    import machina.services as services

    usage = AsyncMock(return_value=50)
    monkeypatch.setattr(
        services.context_manager, "get_compressible_token_usage", usage
    )

    assert await services._compressible_context_tokens(uuid.uuid4()) == 50
    usage.assert_awaited_once()
