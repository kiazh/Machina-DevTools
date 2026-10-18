"""End-to-end seams: UsageStore semantic cache/compression and warmers."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from machina.cli import app as cli_app
from machina.core.agent import ReActAgent
from machina.core.config import config
from machina.core.models import LLMResponse, StreamEvent
from machina.core.optimization.compression.learned import CompressionResult
from machina.core.optimization.semantic_cache import SemanticHit
from machina.core.optimization.warmers.prefix import plan_prefix_warm
from machina.core.optimization.warmers.reuse import estimate_expected_turns
from machina.core.optimization.warmers.startup import (
    run_startup_warmers,
    schedule_startup_warmers,
)
from machina.core.pricing import PriceEntry
from machina.core.usage import UsageStore


class FakeProvider:
    def __init__(self, response):
        self.model = "fake-model"
        self.complete = AsyncMock(return_value=response)


class FakeSemanticCache:
    def __init__(self, hit=None, store_result=True):
        self.hit = hit
        self.store_result = store_result
        self.lookups = []
        self.stores = []

    async def lookup(self, context, canonical, *, threshold, record_hit=True):
        self.lookups.append((context, canonical, threshold, record_hit))
        return self.hit

    async def store(self, context, canonical, response, *, ttl_seconds, max_entries):
        self.stores.append((context, canonical, response, ttl_seconds, max_entries))
        return self.store_result


@pytest.mark.asyncio
async def test_wired_semantic_hit_returns_zero_usage_without_provider(monkeypatch):
    session = uuid.uuid4()
    hit = SemanticHit(
        response=LLMResponse(
            content="cached answer",
            model="fake-model",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cache_hit=True,
            cache_source="semantic",
        ),
        similarity=0.99,
        entry_id=uuid.uuid4(),
    )
    provider = FakeProvider(LLMResponse(content="live answer", model="fake-model"))
    cache = FakeSemanticCache(hit=hit)
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    monkeypatch.setattr("machina.core.usage.SemanticCache", lambda: cache)
    store = UsageStore()
    reserve = AsyncMock()
    monkeypatch.setattr(store, "reserve", reserve)

    result = await store.complete_call(
        provider,
        session,
        "harness",
        [{"role": "user", "content": "Summarize this"}],
        purpose="compression_summary",
        temperature=0.0,
    )

    assert result.cache_source == "semantic"
    assert result.content == "cached answer"
    assert provider.complete.await_count == 0
    reserve.assert_not_awaited()
    assert cache.lookups and cache.lookups[0][3] is True


@pytest.mark.asyncio
async def test_shadow_hit_never_serves_and_never_stores(monkeypatch):
    session = uuid.uuid4()
    hit = SemanticHit(
        response=LLMResponse(
            content="cached answer", model="fake-model", cache_hit=True
        ),
        similarity=0.99,
        entry_id=uuid.uuid4(),
    )
    provider = FakeProvider(LLMResponse(content="live answer", model="fake-model"))
    cache = FakeSemanticCache(hit=hit)
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "shadow")
    monkeypatch.setattr("machina.core.usage.SemanticCache", lambda: cache)
    store = UsageStore()

    result = await store.complete_call(
        provider,
        session,
        "harness",
        [{"role": "user", "content": "Summarize this"}],
        purpose="compression_summary",
        temperature=0.0,
    )

    assert result.content == "live answer"
    assert provider.complete.await_count == 1
    assert cache.stores == []
    assert cache.lookups[0][3] is False


@pytest.mark.asyncio
async def test_tool_request_bypasses_semantic_cache(monkeypatch):
    provider = FakeProvider(LLMResponse(content="live", model="fake-model"))
    cache = FakeSemanticCache()
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    monkeypatch.setattr("machina.core.usage.SemanticCache", lambda: cache)
    store = UsageStore()
    tool = SimpleNamespace(
        name="read_file", description="read", parameters={"type": "object"}
    )

    result = await store.complete_call(
        provider,
        uuid.uuid4(),
        "harness",
        [{"role": "user", "content": "Read a file"}],
        tools=[tool],
        purpose="compression_summary",
        temperature=0.0,
    )
    assert result.content == "live"
    assert cache.lookups == []


@pytest.mark.asyncio
async def test_wired_miss_stores_response_under_original_request(monkeypatch):
    provider = FakeProvider(LLMResponse(content="fresh answer", model="fake-model"))
    cache = FakeSemanticCache()
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    monkeypatch.setattr("machina.core.usage.SemanticCache", lambda: cache)
    store = UsageStore()
    messages = [{"role": "user", "content": "Summarize this"}]

    result = await store.complete_call(
        provider,
        uuid.uuid4(),
        "harness",
        messages,
        purpose="compression_summary",
        temperature=0.0,
    )

    assert result.content == "fresh answer"
    assert cache.stores
    stored_context, canonical, stored_response, _, _ = cache.stores[0]
    assert canonical.text == "user:\nSummarize this"
    assert stored_response.content == "fresh answer"
    assert stored_context.session_id is not None


@pytest.mark.asyncio
async def test_compression_wired_changes_only_provider_payload(monkeypatch):
    original = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "old " * 100},
        {"role": "user", "content": "current"},
    ]
    compressed = [original[0], {"role": "assistant", "content": "short"}, original[2]]
    provider = FakeProvider(LLMResponse(content="answer", model="fake-model"))
    monkeypatch.setattr(config, "optimization_compression_mode", "wired")
    monkeypatch.setattr(
        "machina.core.usage.maybe_compress_request",
        AsyncMock(
            return_value=CompressionResult(
                messages=compressed,
                applied=True,
                tokens_before=800,
                tokens_after=500,
                reason="compressed",
            )
        ),
    )
    store = UsageStore()

    await store.complete_call(provider, None, "harness", original, temperature=0.0)

    sent = provider.complete.await_args.kwargs["messages"]
    assert sent == compressed
    assert original[1]["content"] == "old " * 100


@pytest.mark.asyncio
async def test_stream_path_applies_compression(monkeypatch):
    original = [{"role": "user", "content": "stream this"}]
    compressed = [{"role": "user", "content": "short"}]

    class StreamProvider:
        model = "fake-model"

        def __init__(self):
            self.seen = None

        async def stream_complete(self, messages, **kwargs):
            self.seen = messages
            yield StreamEvent(type="text", content="ok")
            yield StreamEvent(
                type="done",
                response=LLMResponse(
                    content="ok",
                    model="fake-model",
                    usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                ),
            )

    provider = StreamProvider()
    agent = ReActAgent(provider=provider)
    monkeypatch.setattr(
        "machina.core.usage.maybe_compress_request",
        AsyncMock(
            return_value=CompressionResult(
                messages=compressed,
                applied=True,
                tokens_before=100,
                tokens_after=60,
                reason="compressed",
            )
        ),
    )

    events = [
        event
        async for event in agent._stream_llm_with_retry(
            original, tools=[], session_id=None
        )
    ]
    assert [event.type for event in events] == ["text", "done"]
    assert provider.seen == compressed
    assert original == [{"role": "user", "content": "stream this"}]


def test_reuse_estimate_is_conservative_without_history():
    assert estimate_expected_turns([]).expected_turns == 1.0
    assert estimate_expected_turns([4, 5, 6]).expected_turns > 1.0


def _verified_profile():
    return type(
        "Profile",
        (),
        {
            "supports_paid_warmers": True,
            "price": PriceEntry(
                input_per_mtok=3.0,
                output_per_mtok=15.0,
                cache_read_per_mtok=0.3,
                cache_write_per_mtok=3.75,
                verified_at="2026-10-17",
                source="test",
            ),
        },
    )()


def test_prefix_economics_requires_verified_metadata_and_positive_net():
    profile = _verified_profile()
    plan = plan_prefix_warm(profile, prefix_tokens=10_000, expected_turns=4.0)
    assert plan.allowed is True
    assert plan.estimated_savings_usd and plan.estimated_savings_usd > 0
    assert (
        plan_prefix_warm(profile, prefix_tokens=100, expected_turns=4.0).allowed
        is False
    )
    assert (
        plan_prefix_warm(profile, prefix_tokens=10_000, expected_turns=1.0).allowed
        is False
    )
    unknown = type("Unknown", (), {"supports_paid_warmers": False, "price": None})()
    assert (
        plan_prefix_warm(unknown, prefix_tokens=10_000, expected_turns=4.0).allowed
        is False
    )


def test_cli_status_and_prefix_plan_are_read_only():
    runner = CliRunner()
    status = runner.invoke(cli_app, ["optimization", "status"])
    assert status.exit_code == 0
    assert "semanticCache" in status.stdout
    plan = runner.invoke(
        cli_app,
        [
            "optimization",
            "prefix-plan",
            "deepseek",
            "--model",
            "deepseek-chat",
            "--prefix-tokens",
            "10000",
            "--history-turns",
            "3,4,5",
        ],
    )
    assert plan.exit_code == 0
    assert "positive_net_savings" in plan.stdout
    assert "estimated net USD" in plan.stdout


@pytest.mark.asyncio
async def test_startup_warmers_isolate_failures(monkeypatch):
    from machina.core.optimization.warmers import startup

    async def ok():
        return None

    async def bad():
        raise RuntimeError("boom")

    monkeypatch.delenv("MACHINA_GATEWAY_NO_SCHEDULER", raising=False)
    monkeypatch.setattr(startup, "_WARMERS", (("ok", ok), ("bad", bad)))
    monkeypatch.setattr(config, "optimization_warmers_startup_enabled", True)
    report = await run_startup_warmers()
    assert report.ran is True
    assert report.durations_ms["ok"] >= 0
    assert report.failures == {"bad": "RuntimeError"}


@pytest.mark.asyncio
async def test_startup_warmers_are_skipped_by_config_or_test_flag(monkeypatch):
    monkeypatch.setattr(config, "optimization_warmers_startup_enabled", False)
    report = await run_startup_warmers()
    assert report.ran is False
    assert report.skipped == "disabled"

    monkeypatch.setattr(config, "optimization_warmers_startup_enabled", True)
    monkeypatch.setenv("MACHINA_GATEWAY_NO_SCHEDULER", "1")
    report = await run_startup_warmers()
    assert report.ran is False
    assert report.skipped == "no_background_scheduler"
    assert schedule_startup_warmers() is None
