"""B3 fix-campaign regressions: token basis, capacity table, pricing, reservations.

Every test here is derived from an audit finding and must fail before its fix:

1. current model ids resolve to their documented windows (not the 32k default);
2. the auto-compaction trigger uses the SAME tiktoken/JSON basis as the hard
   request gate, including CJK text;
3. reducer-produced views (built from the REAL ``reduce_tool_result`` output)
   survive stale-round condensation across multiple rounds;
4. ``harness_status`` keeps learning notes and the feature map under its cap;
5. provider reservations settle exactly once on generator close/cancel;
6. cost is unknown (None), never a fabricated zero, when only a total is
   reported; DeepSeek models never inherit a sibling's price snapshot; the
   DeepSeek hit/miss split wins over OpenAI-style details.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# ─── 1. context windows ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "window"),
    [
        ("gpt-4o", 128_000),
        ("gpt-4o-2024-11-20", 128_000),
        ("gpt-4.1", 1_047_576),
        ("gpt-4.1-mini", 1_047_576),
        ("o3", 200_000),
        ("o3-mini", 200_000),
        ("o4-mini", 200_000),
        ("gpt-5", 400_000),
        ("gpt-5.6-codex", 400_000),
        ("claude-sonnet-4-20250514", 200_000),
        ("claude-opus-4-1", 200_000),
        ("claude-3-7-sonnet-latest", 200_000),
        ("gemini-2.5-pro", 1_048_576),
        ("gemini-2.5-flash", 1_048_576),
        ("deepseek-v3", 64_000),
        ("deepseek-r1", 64_000),
        ("qwen3-235b", 131_072),
        ("qwen-max", 32_768),
    ],
)
def test_current_models_use_documented_windows(model, window):
    from machina.core.context_windows import context_window_for

    assert context_window_for(model) == window


def test_unknown_model_still_falls_back_conservatively():
    from machina.core.context_windows import DEFAULT_CONTEXT_WINDOW, context_window_for

    assert context_window_for("some-future-model") == DEFAULT_CONTEXT_WINDOW
    assert context_window_for(None) == DEFAULT_CONTEXT_WINDOW
    assert context_window_for("") == DEFAULT_CONTEXT_WINDOW


# ─── 2. compaction trigger vs request gate basis ───────────────────────────


def _install_estimate_fixture(monkeypatch, services_module, contents):
    """Wire recent-context chunks + one tool into ``_estimate_next_request``."""
    from machina.core.models import Session, ToolDefinition

    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    recent = [
        {"type": "user_message", "payload": {"content": text}, "tokens": 0}
        for text in contents
    ]
    context_stub = MagicMock()
    context_stub.get_recent_context = AsyncMock(return_value=recent)
    # The old implementation summed this heuristic instead of serializing.
    context_stub.get_token_usage = AsyncMock(return_value=sum(len(c) // 4 for c in contents))
    monkeypatch.setattr(services_module, "context_manager", context_stub)

    session = Session(
        id=uuid.uuid4(), agent_id="harness", provider="openai", model="gpt-4o"
    )

    class _Sessions:
        @staticmethod
        async def get(_session_id):
            return session

    class _Agents:
        @staticmethod
        async def get(_agent_id):
            return None

    import machina.core.agent_def as agent_def_module
    import machina.core.session as session_module
    import machina.tools.base as tools_base

    monkeypatch.setattr(session_module, "session_manager", _Sessions())
    monkeypatch.setattr(agent_def_module, "agent_registry", _Agents())
    monkeypatch.setattr(tools_base.registry, "get_tool_definitions", lambda: tools)
    monkeypatch.setattr("machina.core.config.config.max_tokens", 1024)
    return session, tools


def _gate_estimate(contents, tools, provider="openai"):
    from machina.core.agent import SYSTEM_PROMPT
    from machina.core.prompt import request_context_tokens

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": "user", "content": text} for text in contents]
    return request_context_tokens(messages, tools, provider)


@pytest.mark.parametrize(
    "contents",
    [
        ["Summarize this repository." * 200],
        ["请总结这个仓库的实现细节。" * 200, "第二段中文内容。" * 100],
    ],
    ids=["ascii", "cjk"],
)
async def test_compaction_trigger_matches_gate_basis(monkeypatch, contents):
    """The trigger must use the gate's tiktoken/JSON serialization (ratio ~1).

    Before the fix the CJK case counted ``len//4`` for stored chunks and a
    ``str(tool_defs)`` repr, landing far below the gate and skipping
    auto-compaction until the hard capacity guard refused the turn.
    """
    import machina.services as services_module

    _, tools = _install_estimate_fixture(monkeypatch, services_module, contents)
    estimate = await services_module._estimate_next_request(uuid.uuid4())
    gate = _gate_estimate(contents, tools)
    assert abs(estimate - gate) <= max(16, int(gate * 0.05)), (estimate, gate)


# ─── 3. reducer marker survives condensation (actual output) ───────────────


def _actual_reduced_view() -> str:
    from machina.core.reducers import reduce_tool_result

    lines = ["pytest output", "=" * 40]
    for index in range(200):
        lines.append(f"FAILED tests/test_sample.py::test_case_{index} - AssertionError")
    for index in range(40):
        lines.append(f"E   assert {index} == {index + 1}")
    lines.append("21 failed, 100 passed in 1.23s")
    result = reduce_tool_result(
        "\n".join(lines), budget_chars=900, artifact_id="artifact-1"
    )
    assert result.changed is True
    return result.output


def test_reducer_marker_is_single_bracket_and_matched():
    from machina.core.prompt import _REDUCED_VIEW_TAIL_RE

    view = _actual_reduced_view()
    assert view.count("[machina view:") == 1
    assert "[[" not in view
    assert _REDUCED_VIEW_TAIL_RE.search(view), view[-200:]


def test_accepted_reducer_view_passes_through_next_rounds():
    from machina.core.prompt import condense_history

    view = _actual_reduced_view()

    def _history(extra: list[dict]) -> list[dict]:
        return [
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "t"}}],
            },
            {"role": "tool", "content": view, "tool_call_id": "c0"},
            *extra,
        ]

    def _round(call_id: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "t"}}],
            },
            {"role": "tool", "content": "next round output", "tool_call_id": call_id},
        ]

    messages = _history([])
    condense_history(messages)
    assert messages[2]["content"] == view

    # A second, later round makes the view stale again; it must still survive.
    messages.extend(_round("c1"))
    condense_history(messages)
    assert messages[2]["content"] == view
    assert "condensed" not in messages[2]["content"]

    messages.extend(_round("c2"))
    condense_history(messages)
    assert messages[2]["content"] == view


# ─── 4. harness_status keeps notes and features under the cap ──────────────


async def test_harness_status_keeps_notes_and_features_under_cap(monkeypatch):
    from machina.core.models import Session
    from machina.tools import harness

    sid = uuid.uuid4()
    owner = "b3-status-owner"
    monkeypatch.setattr(harness, "active_agent_scope", AsyncMock(return_value=(owner, sid)))
    monkeypatch.setattr(
        harness.session_manager,
        "get",
        AsyncMock(return_value=Session(id=sid, agent_id=owner, execution_mode="ask")),
    )
    monkeypatch.setattr(
        harness.growth_store,
        "get",
        AsyncMock(
            return_value={
                "notes": {"style:tone": "Speak plainly"},
                "interactions": 3,
                "feedback_count": 1,
            }
        ),
    )
    monkeypatch.setattr(
        harness.growth_store,
        "rsi_status",
        AsyncMock(return_value={"enabled": True, "revision": 1, "trial": None}),
    )
    # Realistic persona payload: the regression appears when a normal saved
    # persona description is present, pushing the default snapshot over the
    # cap. The old ladder then dropped the whole FEATURES map (80% of the
    # payload) even though it is the least recoverable live section here.
    monkeypatch.setattr(
        harness,
        "_persona_summary",
        AsyncMock(
            return_value={
                "available": True,
                "effective_style": "mentor",
                "notes_applied": ["style:tone"],
                "rsi": {"revision": 1},
                "description": (
                    "A concise, practical tutor who explains trade-offs, "
                    "prefers small verified steps, and asks before changing "
                    "shared configuration. " * 8
                ),
            }
        ),
    )
    result = await harness.harness_status()
    payload = json.loads(result)
    assert len(result) <= harness._MAX_STATUS_BYTES
    assert payload["learning"]["notes"] == {"style:tone": "Speak plainly"}
    assert "notes_omitted" not in payload["learning"]
    assert payload["features"], "the capability map must not be dropped first"
    assert "features_omitted" not in payload


# ─── 5. provider reservation settlement ────────────────────────────────────


async def test_reservation_settlement_is_exactly_once():
    from machina.core.provider.transport import ReservationSettlement

    class _Store:
        def __init__(self):
            self.calls = []

        async def finish(self, reservation_id, usage=None, *, failed=False, model=None):
            self.calls.append({"id": reservation_id, "usage": usage, "failed": failed})
            return True

    store = _Store()
    settlement = ReservationSettlement(uuid.UUID(int=7), store)
    await settlement.settle({"prompt_tokens": 3, "completion_tokens": 2})
    await settlement.settle({"prompt_tokens": 99})
    await settlement.fail()
    assert len(store.calls) == 1
    assert store.calls[0]["failed"] is False

    failed_store = _Store()
    failed = ReservationSettlement(uuid.UUID(int=8), failed_store)
    await failed.fail()
    await failed.fail()
    assert len(failed_store.calls) == 1
    assert failed_store.calls[0]["failed"] is True


async def test_stream_generator_close_settles_reservation_failed_once(monkeypatch):
    """``aclose()`` mid-stream is GeneratorExit, not Exception: settle in finally."""
    from machina.core.provider import OpenRouterProvider

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.api_key = "test-key"
    provider.model = "test-model"
    provider._default_effort = ""
    provider._rate_limiter = AsyncMock()
    provider.client = MagicMock()

    response = MagicMock()
    response.raise_for_status = MagicMock()

    async def lines():
        yield 'data: {"model": "test-model", "choices": [{"delta": {"content": "one"}}]}'
        yield 'data: {"model": "test-model", "choices": [{"delta": {"content": "two"}}]}'
        yield "data: [DONE]"

    response.aiter_lines = lines

    class _Ctx:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *args):
            return False

    provider.client.stream = MagicMock(return_value=_Ctx())

    reserve = AsyncMock(return_value=uuid.uuid4())
    finish = AsyncMock()
    monkeypatch.setattr("machina.core.usage.usage_store.reserve", reserve)
    monkeypatch.setattr("machina.core.usage.usage_store.finish", finish)

    stream = provider.stream_complete(
        messages=[{"role": "user", "content": "hello"}],
        session_id=uuid.uuid4(),
        agent_id="agent",
    )
    first = await anext(stream)
    assert first.type == "text"
    await stream.aclose()

    reserve.assert_awaited_once()
    finish.assert_awaited_once()
    assert finish.call_args.kwargs.get("failed") is True


# ─── 6. pricing / usage-map fidelity ──────────────────────────────────────


def test_total_only_usage_has_unknown_cost_not_zero():
    from machina.core.pricing import estimate_cost

    cost, version = estimate_cost(
        "deepseek",
        "deepseek-chat",
        uncached_input_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        output_tokens=0,
    )
    assert cost is None and version is None


def test_deepseek_models_do_not_inherit_sibling_snapshot():
    from machina.core.pricing import lookup_price

    chat = lookup_price("deepseek", "deepseek-chat")
    assert chat is not None
    assert chat.cache_read_per_mtok is not None
    assert chat.cache_write_per_mtok is not None
    # An unlisted sibling stays unknown instead of being priced as chat.
    assert lookup_price("deepseek", "deepseek-reasoner") is None
    assert lookup_price("deepseek", "deepseek-v3") is None


def test_deepseek_hit_miss_split_wins_over_details():
    from machina.core.provider.usage_map import parse_openai_usage

    usage = parse_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    assert usage["cache_read_tokens"] == 40
    assert usage["uncached_input_tokens"] == 60
    assert usage["usage_basis"] == "provider"


@pytest.mark.asyncio
async def test_finish_total_only_row_records_null_cost(db_pool, monkeypatch):
    from machina.core.usage import UsageStore

    monkeypatch.setattr("machina.core.usage.db", db_pool)
    store = UsageStore()
    sid = uuid.uuid4()
    agent_id = f"b3-total-only-{sid}"
    try:
        reservation = await store.reserve(
            sid,
            agent_id,
            "deepseek",
            "deepseek-chat",
            [{"role": "user", "content": "x"}],
            [],
            8,
        )
        # Normalization turns a total-only provider block into 0/0/100; the
        # ledger must still treat the missing split as unknown cost.
        from machina.core.provider.usage_map import parse_openai_usage

        await store.finish(reservation, parse_openai_usage({"total_tokens": 100}))
        row = await db_pool.fetchrow("SELECT * FROM llm_usage WHERE id = $1", reservation)
        assert row["cost_usd"] is None  # missing split is unknown, never "free"
        assert row["accounted_tokens"] == 100
    finally:
        await db_pool.execute("DELETE FROM llm_usage WHERE session_id = $1", sid)
