"""Token-efficiency T4/T6: view-aware condensation, output policy, omissions.

- Reducer-produced views survive stale-round condensation unchanged.
- ``output_policy`` is configurable; the legacy default keeps prompts
  byte-identical to the pre-feature assembly.
- The assembler discloses budget-dropped evidence instead of silently
  presenting a trimmed section as the full set.
"""

from __future__ import annotations

from machina.core.assembler import PromptAssembler
from machina.core.config import config
from machina.core.models import ContextChunk
from machina.core.prompt import TOOL_RESULT_STALE_CAP, condense_history, output_policy_section


def test_output_policy_legacy_default_is_empty():
    config.set("output_policy", "legacy")
    assert output_policy_section() == ""


def test_output_policy_concise_preserves_exact_payloads():
    config.set("output_policy", "concise")
    policy = output_policy_section()
    assert "Answer directly" in policy
    assert "Preserve requirements" in policy
    assert "exact code" in policy
    assert "Match the user's requested detail" in policy


def test_output_policy_survives_broken_config_reads(monkeypatch):
    def broken_get(key, *args, **kwargs):
        if key == "output_policy":
            raise ValueError("boom")
        return "legacy"

    monkeypatch.setattr(config, "get", broken_get)
    assert output_policy_section() == ""


def test_reduced_views_survive_condensation():
    """The exemption is guarded with real producer output, not a hand-written
    string: ``reduce_tool_result`` must emit exactly the disclosure shape
    ``condense_history`` exempts, or the curated view gets mutilated here."""
    from machina.core.prompt import _REDUCED_VIEW_TAIL_RE
    from machina.core.reducers import reduce_tool_result

    payload = "HEAD-" + "h" * 5000 + "-TAIL"
    reduction = reduce_tool_result(payload, budget_chars=1000, artifact_id="abc")
    assert reduction.changed and reduction.lossy
    message_view = reduction.output
    # Long enough that condensation would fire without the exemption.
    assert len(message_view) > TOOL_RESULT_STALE_CAP
    assert _REDUCED_VIEW_TAIL_RE.search(message_view)
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": message_view, "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "latest", "tool_call_id": "t2"},
    ]
    condense_history(messages)
    assert messages[2]["content"] == message_view  # curated view untouched
    assert messages[2]["content"].endswith("artifact abc; artifact_read pages the exact original]")


def test_marker_quoted_inside_body_is_not_exempt():
    """A page quoting the marker mid-text must still condense normally."""
    body = "[machina view: reduced by tests] appears in this fetched page\n" + "P" * 900
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": body, "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "latest", "tool_call_id": "t2"},
    ]
    condense_history(messages)
    assert "condensed" in messages[2]["content"]


def test_regular_stale_rounds_still_condense():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "H" * 800 + "T" * 200, "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "latest", "tool_call_id": "t2"},
    ]
    condense_history(messages)
    assert "condensed" in messages[2]["content"]


def _chunk(text: str) -> ContextChunk:
    import uuid

    return ContextChunk(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        agent_id="harness",
        chunk_type="memory",
        payload={"content": text},
        token_count=10,
    )


def test_assembler_discloses_omitted_evidence():
    assembler = PromptAssembler(session_budget=8000)
    prompt = assembler.assemble(
        system_prompt="system",
        goal=None,
        recent_chunks=[],
        retrieved_chunks=[(_chunk("kept evidence"), 0.9)],
        query="q",
        omitted_evidence=4,
    )
    assert "kept evidence" in prompt
    assert "4 lower-ranked retrieved excerpts omitted by the token budget" in prompt


def test_assembler_without_omissions_is_unchanged():
    assembler = PromptAssembler(session_budget=8000)
    prompt = assembler.assemble(
        system_prompt="system",
        goal=None,
        recent_chunks=[],
        retrieved_chunks=[(_chunk("kept evidence"), 0.9)],
        query="q",
    )
    assert "omitted by the token budget" not in prompt


def test_omission_note_survives_a_tight_budget():
    """The disclosure leads the section, so truncation keeps it."""
    import uuid as _uuid

    from machina.core.models import ContextChunk

    chunk = ContextChunk(
        id=_uuid.uuid4(),
        session_id=_uuid.uuid4(),
        agent_id="harness",
        chunk_type="memory",
        payload={"content": "EVIDENCE " * 400},
        token_count=400,
    )
    assembler = PromptAssembler(session_budget=220)
    prompt = assembler.assemble(
        system_prompt="system",
        goal=None,
        recent_chunks=[],
        retrieved_chunks=[(chunk, 0.9)],
        query="q",
        omitted_evidence=5,
    )
    assert "5 lower-ranked retrieved excerpts omitted by the token budget" in prompt
