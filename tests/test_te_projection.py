"""Token-efficiency T5: evidence dedup + token-budgeted selection."""

from __future__ import annotations

from machina.core.models import ContextChunk
from machina.core.projection import dedupe_evidence, project_evidence, select_evidence


def _chunk(text: str, *, tokens: int = 0, chunk_type: str = "memory"):
    return ContextChunk(
        id=__import__("uuid").uuid4(),
        session_id=__import__("uuid").uuid4(),
        agent_id="harness",
        chunk_type=chunk_type,
        payload={"content": text},
        token_count=tokens,
    )


def test_exact_duplicates_removed_whitespace_normalized():
    entries = [
        (_chunk("same fact"), 0.9),
        (_chunk("same    fact\n"), 0.8),  # whitespace-only difference
        (_chunk("different fact"), 0.7),
    ]
    kept, removed = dedupe_evidence(entries)
    assert removed == 1
    assert [entry[0].payload["content"] for entry in kept] == ["same fact", "different fact"]
    assert kept[0][1] == 0.9  # first occurrence (highest-ranked) wins


def test_conflicting_variants_are_preserved():
    entries = [(_chunk("value is 5"), 0.9), (_chunk("value is 6"), 0.8)]
    kept, removed = dedupe_evidence(entries)
    assert removed == 0
    assert len(kept) == 2


def test_budget_selection_keeps_order_and_counts_omissions():
    entries = [(_chunk(f"evidence {i}", tokens=10), 0.9) for i in range(5)]
    kept, omitted = select_evidence(entries, 30)
    assert len(kept) == 3
    assert omitted == 2
    assert kept[0][0].payload["content"] == "evidence 0"


def test_oversized_entry_skipped_smaller_later_entry_fits():
    entries = [
        (_chunk("huge", tokens=100), 0.9),
        (_chunk("small", tokens=5), 0.8),
    ]
    kept, omitted = select_evidence(entries, 10)
    assert [entry[0].payload["content"] for entry in kept] == ["small"]
    assert omitted == 1


def test_budget_zero_disables_and_all_oversized_keeps_first():
    entries = [(_chunk("a", tokens=10), 0.9)]
    kept, omitted = select_evidence(entries, 0)
    assert kept == entries and omitted == 0

    kept2, omitted2 = select_evidence(entries, 1)
    assert len(kept2) == 1 and omitted2 == 0  # never drop all evidence


def test_project_evidence_returns_dupes_and_omissions():
    entries = [
        (_chunk("dup", tokens=10), 0.9),
        (_chunk("dup", tokens=10), 0.9),
        (_chunk("later", tokens=10), 0.5),
    ]
    kept, duplicates, omitted = project_evidence(entries, 10)
    assert duplicates == 1
    assert omitted == 1
    assert len(kept) == 1


def test_project_retrieved_respects_config_gate_and_fails_open(monkeypatch):
    from machina.core.config import config
    from machina.core.projection import project_retrieved

    entries = [(_chunk("a", tokens=10), 0.9), (_chunk("b", tokens=10), 0.8)]

    config.set("retrieval_selection_enabled", False)
    kept, omitted = project_retrieved(entries)
    assert kept == entries and omitted == 0

    config.set("retrieval_selection_enabled", True)
    config.set("retrieval_token_budget", 10)
    kept, omitted = project_retrieved(entries)
    assert len(kept) == 1 and omitted == 1

    class Boom:
        def __iter__(self):
            raise RuntimeError("boom")

    kept, omitted = project_retrieved(Boom())
    assert omitted == 0
