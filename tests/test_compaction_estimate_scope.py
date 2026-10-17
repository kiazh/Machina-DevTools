"""Combined #35: auto-compaction estimates the whole compressible context."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest


async def test_estimate_reads_all_chunks_not_the_last_ten(monkeypatch):
    import machina.services as services

    all_chunks = AsyncMock(return_value=[])
    recent = AsyncMock(side_effect=AssertionError("old 10-chunk window used"))
    monkeypatch.setattr(services.context_manager, "get_all_chunks", all_chunks)
    monkeypatch.setattr(services.context_manager, "get_recent_context", recent)
    monkeypatch.setattr(
        "machina.core.session.session_manager.get", AsyncMock(return_value=None)
    )

    await services._estimate_next_request(uuid.uuid4())

    all_chunks.assert_awaited_once()
    recent.assert_not_awaited()
