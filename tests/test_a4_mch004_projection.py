"""A4 fix campaign — MCH-004 request-projection field retention.

``provider_messages_to_units`` rebuilt each unit from a fixed field subset and
``_render_unit`` re-serialized that subset, so a retained-but-unmodified
message lost every other key of the original provider mapping (``name``,
arbitrary extra metadata) — even when the whole request fit and the plan was
"verbatim" (C16). The contract: the unit carries the original mapping
(``raw``) and unmodified units re-render it; reconstructed units (view
replacement) merge the extra original fields back instead of erasing them.

Pins-ordering behavior (commit 8c9475c: governing pins → required
instruction → derived pins) must be unchanged.
"""

from __future__ import annotations

import copy
import json

from machina.core.request_projection import (
    ProjectionInput,
    ProjectionMessage,
    ProjectionPins,
    project_request,
    provider_messages_to_units,
)

PROVIDER_CONTENT_KEY = "provider_content"


class Counting:
    """Characters-based deterministic counter (keeps the request fitting)."""

    def __call__(self, messages) -> int:
        total = 0
        for message in messages:
            total += 4 + len(str(message.get("content") or ""))
        return total


def _fit_input(units: tuple[ProjectionMessage, ...], budget: int = 100000) -> ProjectionInput:
    return ProjectionInput(messages=units, budget_tokens=budget, count_tokens=Counting())


def test_roundtrip_preserves_extra_fields_when_request_fits():
    """MCH-004 acceptance criteria: representative extra fields survive."""
    raw = [
        {"role": "user", "content": "hello", "name": "alice", "extra": "preserve"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "custom_meta": {"trace": "t-1"},
        },
        {"role": "tool", "content": "ok", "tool_call_id": "c1", "obs_id": "obs-9"},
    ]
    units = provider_messages_to_units(raw)
    result = project_request(_fit_input(units))
    assert result.refusal is None
    planned = list(result.planned_messages)
    # Deep equality: nothing silently vanishes, ordered exactly as sent.
    assert planned == [copy.deepcopy(message) for message in raw]
    assert planned[0]["extra"] == "preserve"
    assert planned[0]["name"] == "alice"


def test_view_replacement_merges_extra_original_fields():
    """When reconstruction IS needed (bulky view), extra original fields must
    be merged into the reconstructed mapping instead of erased."""
    bulky = "x" * 1200
    raw = [
        {"role": "user", "content": "change me", "name": "alice"},
        {
            "role": "tool",
            "content": bulky,
            "tool_call_id": "c1",
            "name": "read",
            "custom_key": "keep-me",
        },
    ]
    units = provider_messages_to_units(
        raw,
        artifact_handles={1: "art-1"},
        protected_indexes=set(),
    )

    def artifact_lookup(handle: str):
        return {
            "available": True,
            "complete": True,
            "handle": handle,
            "revision": "rev-1",
            "summary_view": "short summary",
        }

    result = project_request(
        ProjectionInput(
            messages=units,
            budget_tokens=400,  # the bulky body must be replaced by the view
            count_tokens=Counting(),
            artifacts=artifact_lookup,
        )
    )
    assert result.refusal is None
    planned = [dict(message) for message in result.planned_messages]
    tool_message = planned[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["custom_key"] == "keep-me"
    assert tool_message["name"] == "read"
    assert tool_message["tool_call_id"] == "c1"
    # The replaced body is the view text, not the bulky original.
    assert "[machina view: reduced by request_projection" in str(tool_message["content"])
    assert bulky not in str(tool_message["content"])


def test_planned_messages_are_json_serializable():
    """Sanity: adapter round-trip output stays JSON-serializable."""
    raw = [{"role": "system", "content": "seed", "origin": "probe"}]
    units = provider_messages_to_units(raw)
    result = project_request(_fit_input(units))
    json.dumps(result.planned_messages, default=str)  # must not raise


def test_units_without_raw_still_render_the_reconstructed_shape():
    """Directly-built units (no ``raw``) keep today's reconstruction output."""
    unit = ProjectionMessage(unit_id="u1", role="user", content="hi")
    result = project_request(_fit_input((unit,)))
    assert result.refusal is None
    planned = list(result.planned_messages)
    assert planned == [{"role": "user", "content": "hi"}]


def test_pins_ordering_governing_then_instruction_then_derived():
    """The 8c9475c order must survive the raw-retention change."""
    message = {"role": "user", "content": "the instruction", "name": "alice"}
    units = provider_messages_to_units([message], kinds={0: "required"})
    pins = ProjectionPins(
        objective=("objective text",),
        constraints=("constraints text",),
        core_instructions=("core text",),
        read_only_scope=("scope text",),
        blockers=("blockers text",),
        task_state=("task state text",),
    )
    result = project_request(
        ProjectionInput(
            messages=units,
            pins=pins,
            budget_tokens=100000,
            count_tokens=Counting(),
        )
    )
    planned = [dict(m) for m in result.planned_messages]
    rendered = [str(m.get("content")) for m in planned]
    assert rendered == [
        "objective text",
        "constraints text",
        "core text",
        "scope text",
        "blockers text",
        "task state text",
        "the instruction",
    ]
    # The key shape survived verbatim through the raw round-trip.
    assert planned[6].get("name") == "alice"


def test_input_is_not_aliased_by_raw_retention():
    """Planned messages must not share storage with the caller's payload."""
    raw = [{"role": "user", "content": "hello", "name": "alice", "meta": {"k": "v"}}]
    units = provider_messages_to_units(raw)
    result = project_request(_fit_input(units))
    planned = list(result.planned_messages)
    planned[0]["meta"]["k"] = "mutated"
    planned[0]["content"] = "mutated"
    assert raw[0]["meta"]["k"] == "v"
    assert raw[0]["content"] == "hello"
