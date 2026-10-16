"""Token-efficiency T1: provider cache capability profiles + breakpoint helpers.

The capability profile is the single place that decides whether a provider
caches implicitly, accepts explicit breakpoints, or must be left untouched
(unknown = unsupported until verified).
"""

from __future__ import annotations

import pytest

from machina.core.provider import PROVIDER_SPECS, DirectProvider, OllamaProvider, OpenRouterProvider
from machina.core.provider.anthropic import AnthropicProvider
from machina.core.provider.prompt_cache import CACHE_NONE, cache_profile
from machina.core.provider.transport import _mark_message_tail_breakpoint


def test_direct_specs_map_to_verified_modes():
    def provider(name):
        return DirectProvider(PROVIDER_SPECS[name], api_key="test-key")

    assert cache_profile(provider("openai")).mode == "implicit"
    assert cache_profile(provider("deepseek")).mode == "implicit"
    assert cache_profile(provider("google")).mode == "implicit"
    for spec in ("groq", "together", "mistral", "xai", "opencode", "kimi"):
        assert cache_profile(provider(spec)) is CACHE_NONE, spec


def test_anthropic_is_explicit_with_breakpoint_budget():
    profile = cache_profile(AnthropicProvider(api_key="test-key"))
    assert profile.mode == "explicit"
    assert profile.max_breakpoints == 4
    assert profile.min_prefix_tokens == 1024
    assert "cache_read_input_tokens" in profile.usage_fields


def test_openrouter_routes_by_model_family():
    openrouter = OpenRouterProvider(api_key="k", model="anthropic/claude-sonnet-4")
    assert cache_profile(openrouter).mode == "explicit"
    assert cache_profile(OpenRouterProvider(api_key="k", model="openai/gpt-4o")).mode == "implicit"
    assert cache_profile(OpenRouterProvider(api_key="k", model="deepseek/deepseek-v3")).mode == (
        "implicit"
    )
    assert cache_profile(OpenRouterProvider(api_key="k", model="meta/llama-3")).mode == "none"


def test_ollama_and_unknown_are_none():
    assert cache_profile(OllamaProvider(model="llama3.1")) is CACHE_NONE

    class Mystery:
        pass

    assert cache_profile(Mystery()) is CACHE_NONE


# ─── conversation-tail breakpoint helper ────────────────────────────────────


def test_tail_breakpoint_marks_final_block_without_mutation():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    marked = _mark_message_tail_breakpoint(messages)
    assert marked is not None
    assert marked[0] == messages[0]
    assert marked[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}}],
    }
    assert messages[-1]["content"] == "second"  # caller history untouched


def test_tail_breakpoint_marks_last_block_of_block_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    marked = _mark_message_tail_breakpoint(messages)
    assert marked is not None
    assert "cache_control" not in marked[-1]["content"][0]
    assert marked[-1]["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[0]["content"][1]


def test_tail_breakpoint_returns_none_when_unmarkable():
    assert _mark_message_tail_breakpoint([]) is None
    assert _mark_message_tail_breakpoint([{"role": "user", "content": ""}]) is None
    assert _mark_message_tail_breakpoint([{"role": "tool", "content": None}]) is None


# ─── cache usage reaches the response (CodeRabbit pass-1 finding) ───────────


@pytest.mark.asyncio
async def test_openrouter_nonstream_response_keeps_parsed_cache_categories():
    import httpx

    from machina.core.provider import AsyncTokenBucket

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "deepseek/deepseek-chat",
                "choices": [{"message": {"content": "OK", "tool_calls": []}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 10,
                    "total_tokens": 1010,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        )

    provider = OpenRouterProvider(api_key="k", model="deepseek/deepseek-chat")
    provider._rate_limiter = AsyncTokenBucket(rate=0, capacity=0)
    provider.client = httpx.AsyncClient(
        base_url=OpenRouterProvider.BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await provider.complete([{"role": "user", "content": "hi"}])
        assert response.usage["cache_read_tokens"] == 800
        assert response.usage["uncached_input_tokens"] == 200
        assert response.usage["usage_basis"] == "provider"
    finally:
        await provider.close()


# ─── breakpoint application gate (CodeRabbit pass-2 finding) ────────────────


def test_apply_annotations_gates_on_explicit_profiles():
    """None/implicit profiles must never receive cache fields."""
    import dataclasses

    from machina.core.provider.anthropic import AnthropicProvider
    from machina.core.provider.prompt_cache import (
        CACHE_NONE,
        apply_anthropic_breakpoints,
        cache_profile,
    )

    explicit = cache_profile(AnthropicProvider(api_key="test-key"))
    payload: dict = {}
    flags = apply_anthropic_breakpoints(
        payload, system="s", messages=[{"role": "user", "content": "hi"}], tools=[{"name": "t"}], profile=CACHE_NONE
    )
    assert flags == (False, False, False)
    assert payload == {}, "no cache_control for a non-caching profile"

    limited = dataclasses.replace(explicit, max_breakpoints=1)
    payload2: dict = {}
    flags2 = apply_anthropic_breakpoints(
        payload2, system="s", messages=[{"role": "user", "content": "hi"}], tools=[{"name": "t"}], profile=limited
    )
    assert flags2 == (True, False, False), "budget of 1 marks only the system block"
    assert "tools" not in payload2
    assert "messages" not in payload2, "the helper only writes messages when it marks the tail"

    payload3: dict = {}
    flags3 = apply_anthropic_breakpoints(
        payload3,
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "t"}],
        profile=explicit,
    )
    assert flags3 == (True, True, True)
    assert str(payload3).count("cache_control") == 3
