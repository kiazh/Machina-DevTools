"""Token-efficiency review fixes (2026-10-16).

Regression pins for review findings on the token-efficiency work:

- ``reduce_tool_result`` emits a single-bracket ``[machina view:`` disclosure
  that ``prompt._REDUCED_VIEW_TAIL_RE`` matches, so curated views survive
  stale-round condensation.
- DeepSeek ``prompt_cache_hit_tokens``/``prompt_cache_miss_tokens`` beat a
  conflicting (zero) ``prompt_tokens_details.cached_tokens`` reading.
- Built-in prices are model-scoped: an unlisted model is NULL, and an
  override never inherits the base snapshot's date.
- An explicit macOS Automation denial (-1743 / "Not authorized") maps to the
  named, cached permission state instead of "nothing playing".
- Reducer-exempt tools are still append-bounded (artifact pages are
  page-aware), so a 20 000-char output cannot slip through verbatim.
"""

from __future__ import annotations

import json
import uuid

import pytest

from machina.core import nowplaying
from machina.core.pricing import (
    PRICE_TABLE_VERSION,
    estimate_cost,
    lookup_price,
    parse_overrides,
)
from machina.core.prompt import TOOL_RESULT_STALE_CAP, _REDUCED_VIEW_TAIL_RE, condense_history
from machina.core.provider.usage_map import parse_openai_usage
from machina.core.reducers import VIEW_MARKER, reduce_tool_result


@pytest.fixture(autouse=True)
def _reset_nowplaying_cache():
    nowplaying._cache = None
    nowplaying._cache_at = 0.0
    yield
    nowplaying._cache = None
    nowplaying._cache_at = 0.0


# ─── 1. reducer disclosure matches the prompt-side exemption ────────────────


def test_reducer_view_disclosure_is_single_bracketed():
    payload = "HEAD-" + "h" * 5000 + "-TAIL"
    reduction = reduce_tool_result(payload, budget_chars=1000, artifact_id="art-1")
    assert reduction.changed
    view = reduction.output
    assert VIEW_MARKER in view
    assert "\n[machina view:" in view
    assert "\n[[machina view:" not in view
    assert _REDUCED_VIEW_TAIL_RE.search(view)


def test_producer_view_survives_condensation_unchanged():
    payload = "HEAD-" + "h" * 5000 + "-TAIL"
    view = reduce_tool_result(payload, budget_chars=1000, artifact_id="art-2").output
    assert len(view) > TOOL_RESULT_STALE_CAP  # would be condensed without the exemption
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": view, "tool_call_id": "t1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "content": "latest", "tool_call_id": "t2"},
    ]
    condense_history(messages)
    assert messages[2]["content"] == view


# ─── 2. DeepSeek hit/miss precedence over cached details ────────────────────


def test_deepseek_hit_miss_wins_over_zero_cached_details():
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 0},
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
    )
    assert usage["cache_read_tokens"] == 800
    assert usage["uncached_input_tokens"] == 200
    assert usage["prompt_tokens"] == 1000


def test_plain_openai_cached_details_still_used_without_hit_miss():
    usage = parse_openai_usage(
        {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 600}}
    )
    assert usage["cache_read_tokens"] == 600
    assert usage["uncached_input_tokens"] == 400


# ─── 3. model-scoped built-in prices and override provenance ────────────────


def test_builtin_prices_are_model_scoped():
    chat = lookup_price("deepseek", "deepseek-chat")
    assert chat is not None and chat.verified_at == "2026-10-14"
    # A sibling model must never be priced with the chat snapshot.
    assert lookup_price("deepseek", "deepseek-reasoner") is None
    assert lookup_price("deepseek", None) is None
    assert estimate_cost(
        "deepseek",
        "deepseek-reasoner",
        uncached_input_tokens=1_000_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=1_000_000,
    ) == (None, None)


def test_override_without_date_never_claims_the_base_snapshot():
    overrides = parse_overrides(
        json.dumps(
            {"deepseek/deepseek-chat": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
        )
    )
    entry = lookup_price("deepseek", "deepseek-chat", overrides=overrides)
    assert entry is not None
    assert entry.verified_at == "unverified"
    assert entry.source == "override"
    cost, version = estimate_cost(
        "deepseek",
        "deepseek-chat",
        uncached_input_tokens=1_000_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=1_000_000,
        overrides=overrides,
    )
    assert abs(cost - 3.0) < 1e-9
    assert version == f"{PRICE_TABLE_VERSION}:unverified"
    assert "2026-10-14" not in version


def test_provider_level_override_also_stays_unverified():
    overrides = parse_overrides(
        json.dumps({"deepseek": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})
    )
    entry = lookup_price("deepseek", "deepseek-chat", overrides=overrides)
    assert entry is not None and entry.verified_at == "unverified"


# ─── 4. Automation denial is a permission state, not "nothing playing" ──────


def _no_other_layers(monkeypatch):
    monkeypatch.setattr(nowplaying, "_media_remote_probe", lambda: None)
    monkeypatch.setattr(nowplaying, "_cli_probe", lambda: None)


def test_automation_denial_maps_to_permission_state(monkeypatch):
    _no_other_layers(monkeypatch)
    monkeypatch.setattr(nowplaying, "_app_running", lambda name: name == "Spotify")
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (
            1,
            "",
            "execution error: Not authorized to send Apple events to Spotify. (-1743)",
        ),
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is False
    assert result["title"] is None
    assert "blocked automation for Spotify" in (result["error"] or "")
    # The denial path reuses the longer permission cache TTL.
    assert nowplaying._cache_ttl(result) > nowplaying._CACHE_TTL_S


def test_automation_timeout_still_maps_to_permission_state(monkeypatch):
    _no_other_layers(monkeypatch)
    monkeypatch.setattr(nowplaying, "_app_running", lambda name: name == "Spotify")
    monkeypatch.setattr(
        nowplaying, "_osascript", lambda script, timeout=3.0: (124, "", "timeout")
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is False
    assert "blocked automation for Spotify" in (result["error"] or "")


def test_unrelated_script_error_is_not_a_permission_denial(monkeypatch):
    _no_other_layers(monkeypatch)
    monkeypatch.setattr(nowplaying, "_app_running", lambda name: name == "Spotify")
    monkeypatch.setattr(
        nowplaying, "_osascript", lambda script, timeout=3.0: (1, "", "syntax error: unexpected token")
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is True
    assert result["error"] is None


def test_browser_probe_denial_maps_to_permission_state(monkeypatch):
    monkeypatch.setattr(nowplaying, "_app_running", lambda name: name == "Google Chrome")
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (
            1,
            "",
            "Not authorized to send Apple events to Google Chrome. (-1743)",
        ),
    )
    probe = nowplaying._browser_probe(
        "Google Chrome",
        "title of active tab of front window",
        "URL of active tab of front window",
    )
    assert probe == {"error": "permission", "source": "Google Chrome"}


# ─── 5. reducer-exempt tools keep an append bound ───────────────────────────


@pytest.mark.asyncio
async def test_reducer_exempt_tools_are_still_append_bounded():
    from machina.core.tool_views import bounded_tool_view

    oversized = "A" * 20_000
    for tool_name, cap in (("artifact_read", 4000), ("artifact_search", 4000), ("skill_read", 9000)):
        view, artifact_id = await bounded_tool_view(
            tool_name, oversized, session_id=uuid.uuid4(), agent_id="agent"
        )
        assert artifact_id is None
        assert len(view) <= cap, tool_name


@pytest.mark.asyncio
async def test_default_sized_artifact_page_passes_through():
    from machina.core.tool_views import bounded_tool_view

    page = "P" * 3900  # within artifact_read's default 4000-char page
    view, artifact_id = await bounded_tool_view(
        "artifact_read", page, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert view == page
    assert artifact_id is None


# ─── 6. DeepSeek hit/miss reconciliation against the prompt total ───────────


def test_openai_hit_only_reconciles_uncached_to_prompt_total():
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_cache_hit_tokens": 800,
        }
    )
    assert usage["uncached_input_tokens"] == 200  # prompt - hit
    assert usage["cache_read_tokens"] == 800
    assert usage["prompt_tokens"] == 1000
    assert usage["usage_basis"] == "provider"
    # Degenerate hit reading: the partition still never exceeds the total.
    degenerate = parse_openai_usage(
        {"prompt_tokens": 100, "completion_tokens": 1, "prompt_cache_hit_tokens": 200}
    )
    assert degenerate["uncached_input_tokens"] == 0
    assert degenerate["cache_read_tokens"] == 100
    assert degenerate["prompt_tokens"] == 100


def test_openai_miss_only_derives_cache_read_from_prompt_total():
    # Review finding: the miss-only branch used to keep miss as the whole
    # prompt (200 here) and discard the provider's 1000-token total, while the
    # hit-only branch derived its missing side from that total.
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_cache_miss_tokens": 200,
        }
    )
    assert usage["uncached_input_tokens"] == 200
    assert usage["cache_read_tokens"] == 800  # prompt - miss
    assert usage["prompt_tokens"] == 1000
    assert usage["usage_basis"] == "provider"
    # A derived split never upgrades an estimate to a provider bill.
    estimate = parse_openai_usage(
        {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_cache_miss_tokens": 200},
        estimated=True,
    )
    assert estimate["usage_basis"] == "local_estimate"
    # A miss reading larger than the total clamps to it instead of exceeding.
    degenerate = parse_openai_usage(
        {"prompt_tokens": 100, "completion_tokens": 1, "prompt_cache_miss_tokens": 200}
    )
    assert degenerate["uncached_input_tokens"] == 100
    assert degenerate["cache_read_tokens"] == 0
    assert degenerate["prompt_tokens"] == 100


def test_openai_consistent_hit_miss_split_is_copied_verbatim():
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
    )
    assert usage["uncached_input_tokens"] == 200
    assert usage["cache_read_tokens"] == 800
    assert usage["prompt_tokens"] == 1000
    assert usage["usage_basis"] == "provider"


def test_openai_inconsistent_hit_miss_reconciles_to_prompt_total():
    # Over-reported split (800 + 300 vs 1000): the reported miss bounds the
    # cache read, so the partition never exceeds the provider's total and the
    # conservative uncached side keeps the provider's own figure.
    over = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1110,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 300,
        }
    )
    assert over["uncached_input_tokens"] == 300
    assert over["cache_read_tokens"] == 700  # min(hit, prompt - miss)
    assert over["prompt_tokens"] == 1000
    assert over["total_tokens"] == 1110
    assert over["usage_basis"] == "provider"

    # Under-reported split (800 + 100 vs 1000): the explicit hit survives and
    # the unaccounted remainder of the total joins the uncached side.
    under = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 100,
        }
    )
    assert under["uncached_input_tokens"] == 200
    assert under["cache_read_tokens"] == 800
    assert under["prompt_tokens"] == 1000
    for usage in (over, under):
        assert (
            usage["uncached_input_tokens"] + usage["cache_read_tokens"] == usage["prompt_tokens"]
        )


def test_openai_cached_tokens_only_shape_stays_byte_identical():
    # The `prompt_tokens_details` path is untouched by the hit/miss
    # reconciliation: pin the complete dict (legacy keys + split + basis).
    usage = parse_openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "total_tokens": 1010,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    )
    assert usage == {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "total_tokens": 1010,
        "uncached_input_tokens": 200,
        "cache_read_tokens": 800,
        "cache_reported": True,
        "usage_basis": "provider",
    }


def test_openai_without_cache_fields_makes_no_basis_claim():
    usage = parse_openai_usage({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert "cache_reported" not in usage
    assert "usage_basis" not in usage  # no cache claim means no basis claim


def test_openai_hit_miss_without_prompt_total_keeps_verbatim_fields():
    # No prompt total: the reported fields are kept verbatim and the unknown
    # side stays missing (never a fabricated zero).
    both = parse_openai_usage(
        {
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
    )
    assert both["uncached_input_tokens"] == 200
    assert both["cache_read_tokens"] == 800
    assert both["prompt_tokens"] == 1000
    miss_only = parse_openai_usage({"completion_tokens": 10, "prompt_cache_miss_tokens": 200})
    assert miss_only["uncached_input_tokens"] == 200
    assert miss_only["prompt_tokens"] == 200
    assert "cache_read_tokens" not in miss_only
