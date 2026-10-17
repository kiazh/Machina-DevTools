"""Combined #19: streamed provider calls record LLM metrics like non-streamed."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import machina
from machina.core.provider import OpenRouterProvider
from machina.core.provider import openrouter as openrouter_module


class _Ctx:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return False


def _openrouter() -> tuple[OpenRouterProvider, MagicMock]:
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.api_key = "test-key"
    provider.model = "vendor/paid"
    provider._default_effort = ""
    provider._rate_limiter = AsyncMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    provider.client = MagicMock()
    return provider, response


async def test_openrouter_stream_records_success_metrics(monkeypatch):
    provider, response = _openrouter()

    async def lines():
        yield 'data: {"model": "vendor/paid", "choices": [{"delta": {"content": "Hi"}}]}'
        yield (
            'data: {"model": "vendor/paid", "choices": [], '
            '"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}'
        )
        yield "data: [DONE]"

    response.aiter_lines = lines
    provider.client.stream = MagicMock(return_value=_Ctx(response))
    recorded: list[dict] = []
    monkeypatch.setattr(
        openrouter_module.metrics, "record_llm_call", lambda **kwargs: recorded.append(kwargs)
    )

    events = [
        event
        async for event in provider.stream_complete([{"role": "user", "content": "hi"}])
    ]
    assert events[-1].type == "done"
    assert recorded, "streamed call recorded no LLM metric"
    success = recorded[-1]
    assert success.get("is_error", False) is False
    assert success["model"] == "vendor/paid"
    assert success["prompt_tokens"] == 5
    assert success["completion_tokens"] == 3


async def test_openrouter_stream_records_error_metrics(monkeypatch):
    provider, response = _openrouter()

    async def lines():
        yield 'data: {"error": {"message": "boom"}}'

    response.aiter_lines = lines
    provider.client.stream = MagicMock(return_value=_Ctx(response))
    recorded: list[dict] = []
    monkeypatch.setattr(
        openrouter_module.metrics, "record_llm_call", lambda **kwargs: recorded.append(kwargs)
    )

    with pytest.raises(Exception):
        _ = [
            event
            async for event in provider.stream_complete([{"role": "user", "content": "hi"}])
        ]
    assert recorded and recorded[-1]["is_error"] is True


@pytest.mark.parametrize("name", ["openrouter", "anthropic", "direct", "ollama"])
def test_every_stream_provider_calls_record_llm_call(name):
    path = Path(machina.__file__).resolve().parent / "core/provider" / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    stream = source[source.index("async def stream_complete") :]
    assert "metrics.record_llm_call" in stream
    assert "is_error=True" in stream
