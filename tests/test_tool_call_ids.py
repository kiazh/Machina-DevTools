"""Combined #27: synthesized tool-call ids are persisted and turn-unique."""

from __future__ import annotations

from machina.core.agent import _ensure_tool_call_id


def test_missing_tool_call_ids_are_persisted_and_unique():
    first = {"function": {"name": "clock", "arguments": "{}"}}
    second = {"function": {"name": "clock", "arguments": "{}"}}
    first_id = _ensure_tool_call_id(first)
    second_id = _ensure_tool_call_id(second)

    assert first_id and second_id
    assert first_id != second_id
    assert first["id"] == first_id and second["id"] == second_id
    assert first_id != "call-0"
