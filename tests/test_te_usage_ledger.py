"""Token-efficiency T0: normalized usage, cache accounting, cost, acceptance.

Provider usage blocks from different vendors partition cache tokens
differently; the ledger must normalize them without ever presenting a local
estimate as a bill and without turning an unreported cache category into a
fabricated zero.
"""

from __future__ import annotations

import json
import uuid

import pytest

from machina.core.config import config
from machina.core.pricing import PRICE_TABLE_VERSION, estimate_cost, lookup_price, parse_overrides
from machina.core.provider.usage_map import (
    parse_anthropic_usage,
    parse_chatgpt_usage,
    parse_ollama_usage,
    parse_openai_usage,
)
from machina.core.usage import UsageStore

# ─── provider usage normalization ───────────────────────────────────────────


def test_anthropic_cache_categories_are_additive_not_subset():
    usage = parse_anthropic_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
        }
    )
    assert usage["prompt_tokens"] == 1050  # logical input stays the sum
    assert usage["uncached_input_tokens"] == 100
    assert usage["cache_read_tokens"] == 900
    assert usage["cache_write_tokens"] == 50
    assert usage["usage_basis"] == "provider"
    assert usage["cache_reported"] is True


def test_anthropic_without_cache_fields_keeps_legacy_shape():
    usage = parse_anthropic_usage({"input_tokens": 5, "output_tokens": 3})
    assert usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert "cache_read_tokens" not in usage  # unknown is not zero


def test_openai_cached_tokens_are_a_subset_of_prompt_tokens():
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    )
    assert usage["prompt_tokens"] == 1000
    assert usage["uncached_input_tokens"] == 200
    assert usage["cache_read_tokens"] == 800
    assert usage["usage_basis"] == "provider"


def test_deepseek_hit_miss_split():
    usage = parse_openai_usage(
        {
            "prompt_tokens": 500,
            "completion_tokens": 5,
            "total_tokens": 505,
            "prompt_cache_hit_tokens": 480,
            "prompt_cache_miss_tokens": 20,
        }
    )
    assert usage["uncached_input_tokens"] == 20
    assert usage["cache_read_tokens"] == 480


def test_openai_without_details_counts_as_uncached_upper_bound():
    usage = parse_openai_usage({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_chatgpt_cached_details():
    usage = parse_chatgpt_usage(
        {
            "input_tokens": 300,
            "output_tokens": 3,
            "total_tokens": 303,
            "input_tokens_details": {"cached_tokens": 100},
        }
    )
    assert usage["uncached_input_tokens"] == 200
    assert usage["cache_read_tokens"] == 100


def test_ollama_normalization_labels_basis():
    usage = parse_ollama_usage(11, 7)
    assert usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_estimated_usage_never_labels_itself_provider_reported():
    """An estimate with cache details must keep the estimate basis."""
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
        estimated=True,
    )
    assert usage["estimated"] is True
    assert usage["usage_basis"] == "local_estimate"


def test_partial_partitions_keep_reported_values_with_partial_basis():
    # A reported cache reading without a prompt total must not become zero,
    # and the ledger must disclose the partial basis.
    openai_usage = parse_openai_usage(
        {"completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 800}}
    )
    assert openai_usage["cache_read_tokens"] == 800
    assert "uncached_input_tokens" not in openai_usage
    assert openai_usage["usage_basis"] == "provider_partial"

    chatgpt_usage = parse_chatgpt_usage(
        {"output_tokens": 3, "input_tokens_details": {"cached_tokens": 100}}
    )
    assert chatgpt_usage["cache_read_tokens"] == 100
    assert chatgpt_usage["usage_basis"] == "provider_partial"

    # Explicit null cache keys are missing, not zero; explicit zeros are real.
    null_keys = parse_anthropic_usage(
        {"input_tokens": 5, "output_tokens": 3, "cache_read_input_tokens": None}
    )
    assert "cache_read_tokens" not in null_keys
    zero_keys = parse_anthropic_usage(
        {"input_tokens": 5, "output_tokens": 3, "cache_read_input_tokens": 0}
    )
    assert zero_keys["cache_read_tokens"] == 0
    assert zero_keys["usage_basis"] == "provider"


def test_missing_usage_stays_zero_shape_for_legacy_consumers():
    for parser in (parse_anthropic_usage, parse_chatgpt_usage, parse_openai_usage):
        usage = parser({})
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ─── pricing ────────────────────────────────────────────────────────────────


def test_lookup_price_matches_provider_and_unknown_stays_unknown():
    assert lookup_price("deepseek", "deepseek-chat") is not None
    assert lookup_price("deepseek", "deepseek-chat").verified_at == "2026-10-14"
    assert lookup_price("openai", "gpt-4o") is None
    assert lookup_price(None, None) is None


def test_estimate_cost_splits_cache_rates():
    cost, version = estimate_cost(
        "deepseek",
        "deepseek-chat",
        uncached_input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=0,
        output_tokens=1_000_000,
    )
    assert abs(cost - (0.30 + 0.006 + 1.20)) < 1e-9
    assert version.startswith(PRICE_TABLE_VERSION)


def test_estimate_cost_unknown_provider_is_none_never_zero():
    assert estimate_cost(
        "openai",
        "gpt-4o",
        uncached_input_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=100,
    ) == (None, None)


def test_estimate_cost_requires_rate_for_used_categories():
    overrides = {"mockco": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
    # A used cache-read category without a configured rate makes the whole
    # estimate unknown — the ledger must not under-report.
    assert (
        estimate_cost(
            "mockco",
            "m",
            uncached_input_tokens=1,
            cache_read_tokens=10,
            cache_write_tokens=0,
            output_tokens=1,
            overrides=overrides,
        )[0]
        is None
    )


def test_estimate_cost_override_merge_and_validation():
    overrides = parse_overrides(
        json.dumps(
            {
                "anthropic": {
                    "input_per_mtok": 3.0,
                    "output_per_mtok": 15.0,
                    "cache_read_per_mtok": 0.3,
                    "cache_write_per_mtok": 3.75,
                    "verified_at": "2026-10-16",
                }
            }
        )
    )
    cost, version = estimate_cost(
        "anthropic",
        "claude-3-5-sonnet",
        uncached_input_tokens=1_000_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
        overrides=overrides,
    )
    assert abs(cost - (3.0 + 0.6 + 3.75 + 15.0)) < 1e-9
    assert version.endswith(":2026-10-16")
    # Malformed overrides degrade to {} — an optional enhancement must never
    # break the ledger.
    assert parse_overrides("{not json") == {}
    assert parse_overrides("[1,2]") == {}


# ─── ledger persistence (real test database) ────────────────────────────────


@pytest.mark.asyncio
async def test_finish_persists_cache_split_basis_and_cost(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.usage.db", db_pool)
    store = UsageStore()
    sid = uuid.uuid4()
    agent_id = f"te-ledger-{sid}"
    try:
        reservation = await store.reserve(
            sid,
            agent_id,
            "deepseek",
            "deepseek-chat",
            [{"role": "user", "content": "hi"}],
            [],
            16,
            purpose="compression_summary",
        )
        await store.finish(
            reservation,
            parse_openai_usage(
                {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "prompt_cache_hit_tokens": 400_000,
                    "prompt_cache_miss_tokens": 600_000,
                }
            ),
            model="deepseek-chat",
        )
        row = await db_pool.fetchrow("SELECT * FROM llm_usage WHERE id = $1", reservation)
        assert row["status"] == "complete"
        assert row["prompt_tokens"] == 1_000_000
        assert row["cache_read_tokens"] == 400_000
        # DeepSeek has no cache-write category: the column stays NULL
        # (not applicable), never a fabricated zero.
        assert row["cache_write_tokens"] is None
        assert row["usage_basis"] == "provider"
        assert row["purpose"] == "compression_summary"
        assert row["requested_model"] == "deepseek-chat"
        assert row["cost_usd"] is not None
        expected = (600_000 * 0.30 + 400_000 * 0.006 + 1_000_000 * 1.20) / 1_000_000
        assert abs(float(row["cost_usd"]) - expected) < 1e-9
        assert row["price_version"].startswith(PRICE_TABLE_VERSION)
        assert row["accepted"] is None  # never marked for auxiliary calls
    finally:
        await db_pool.execute("DELETE FROM llm_usage WHERE session_id = $1", sid)


@pytest.mark.asyncio
async def test_missing_usage_keeps_reservation_and_null_cost(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.usage.db", db_pool)
    store = UsageStore()
    sid = uuid.uuid4()
    agent_id = f"te-missing-{sid}"
    try:
        reservation = await store.reserve(
            sid, agent_id, "openrouter", "openrouter/free", [{"role": "user", "content": "x"}], [], 8
        )
        await store.finish(reservation, {"prompt_tokens": 0, "completion_tokens": 0})
        row = await db_pool.fetchrow("SELECT * FROM llm_usage WHERE id = $1", reservation)
        assert row["prompt_tokens"] is None
        assert row["cache_read_tokens"] is None
        assert row["usage_basis"] == "missing"
        assert row["cost_usd"] is None  # unknown is not zero
        assert row["accounted_tokens"] == row["reserved_tokens"]
    finally:
        await db_pool.execute("DELETE FROM llm_usage WHERE session_id = $1", sid)


@pytest.mark.asyncio
async def test_mark_accepted_marks_only_latest_complete_agent_call(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.usage.db", db_pool)
    store = UsageStore()
    sid = uuid.uuid4()
    agent_id = f"te-accept-{sid}"
    try:
        first = await store.reserve(
            sid, agent_id, "fake", "fake", [{"role": "user", "content": "a"}], [], 8
        )
        await store.finish(first, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})
        summary_call = await store.reserve(
            sid,
            agent_id,
            "fake",
            "fake",
            [{"role": "user", "content": "b"}],
            [],
            8,
            purpose="compression_summary",
        )
        await store.finish(
            summary_call, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        )

        assert await store.mark_accepted(sid, agent_id) is True
        rows = await db_pool.fetch(
            "SELECT id, purpose, accepted FROM llm_usage WHERE session_id = $1 ORDER BY created_at",
            sid,
        )
        accepted = {str(row["id"]): row["accepted"] for row in rows}
        assert accepted[str(first)] is True
        # Auxiliary calls are never marked accepted even though they are newer.
        assert accepted[str(summary_call)] is None

        # A second mark has nothing left to mark (idempotent-safe).
        assert await store.mark_accepted(sid, agent_id) is False
    finally:
        await db_pool.execute("DELETE FROM llm_usage WHERE session_id = $1", sid)


@pytest.mark.asyncio
async def test_summary_reports_cache_cost_and_cost_per_accepted(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.usage.db", db_pool)
    store = UsageStore()
    sid = uuid.uuid4()
    agent_id = f"te-summary-{sid}"
    try:
        before = await store.summary(sid, agent_id)
        assert before["session"]["costPerAcceptedCompletionUsd"] is None

        reservation = await store.reserve(
            sid,
            agent_id,
            "deepseek",
            "deepseek-chat",
            [{"role": "user", "content": "x"}],
            [],
            8,
        )
        await store.finish(
            reservation,
            parse_openai_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}),
            model="deepseek-chat",
        )
        await store.mark_accepted(sid, agent_id)

        summary = await store.summary(sid, agent_id)
        view = summary["session"]
        assert view["acceptedCompletions"] == 1
        assert view["cacheReadTokens"] == 0
        assert abs(view["knownCostUsd"] - 0.30) < 1e-9
        assert abs(view["costPerAcceptedCompletionUsd"] - 0.30) < 1e-6
        assert view["unknownCostCalls"] == 0
    finally:
        await db_pool.execute("DELETE FROM llm_usage WHERE session_id = $1", sid)


# ─── config surface ─────────────────────────────────────────────────────────


def test_output_policy_validation():
    from machina.core.config import validate_value

    assert validate_value("output_policy", "concise") == "concise"
    assert validate_value("output_policy", "LEGACY") == "legacy"
    with pytest.raises(ValueError):
        validate_value("output_policy", "telegraphic")


def test_pricing_table_json_validation():
    from machina.core.config import validate_value

    assert validate_value("pricing_table_json", "") == ""
    payload = json.dumps({"openai": {"input_per_mtok": 1.0}})
    assert validate_value("pricing_table_json", payload) == payload
    with pytest.raises(ValueError):
        validate_value("pricing_table_json", "{broken")
    with pytest.raises(ValueError):
        validate_value("pricing_table_json", "[1]")
    with pytest.raises(ValueError):
        validate_value("pricing_table_json", "x" * (33 * 1024))


def test_te_defaults_are_registered():
    from machina.core.config import DEFAULTS

    assert DEFAULTS["artifacts_enabled"] is True
    assert DEFAULTS["artifact_retention_days"] == 30
    assert DEFAULTS["reducers_enabled"] is True
    assert DEFAULTS["retrieval_selection_enabled"] is True
    assert DEFAULTS["retrieval_token_budget"] == 6000
    assert DEFAULTS["output_policy"] == "legacy"
    assert config.get("artifacts_enabled") in (True, False)
