"""Combined #37: cache hits are identity-scoped and not billed twice."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

from machina.core.models import LLMResponse
from machina.core.provider.caches import _cache_clear, _cache_get, _cache_put
from machina.core.usage import UsageStore


async def test_cache_hit_is_marked_and_records_zero_provider_usage(monkeypatch):
    _cache_clear()
    _cache_put(
        "hit-key",
        LLMResponse(
            content="cached",
            model="m",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    )
    hit = _cache_get("hit-key")
    assert hit is not None and hit.cache_hit is True

    class Provider:
        model = "m"

        def __init__(self):
            self.kwargs = None

        async def complete(self, **kwargs):
            self.kwargs = kwargs
            return hit

    provider = Provider()
    store = UsageStore()
    session_id = uuid.uuid4()
    reservation = uuid.uuid4()
    reserve = AsyncMock(return_value=reservation)
    finish = AsyncMock(return_value=True)
    monkeypatch.setattr(store, "reserve", reserve)
    monkeypatch.setattr(store, "finish", finish)

    await store.complete_call(
        provider,
        session_id,
        "harness",
        [{"role": "user", "content": "same prompt"}],
    )

    assert provider.kwargs["session_id"] == session_id
    assert provider.kwargs["agent_id"] == "harness"
    usage = finish.call_args.args[1]
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _cache_clear()
