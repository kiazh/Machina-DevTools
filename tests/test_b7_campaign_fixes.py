"""B7 fix campaign — memory/RAG (P0-2 redaction, P0-3 persona leak,
P0-1 chunker tail loss, projection compaction-summary protection).

Red-first regression tests for the defects reported in
``review/memory-rag.md`` and the projection evidence contract.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from machina.core.models import ContextChunk
from machina.core.projection import project_evidence, select_evidence
from machina.memory import redaction as redaction_mod
from machina.memory.models import MemoryEntry, RetrievedMemory
from machina.memory.persona import PersonaMemory, PersonaMemoryStore
from machina.memory.redaction import StreamingSecretRedactor, redact_secrets
from machina.memory.retriever import MemoryRetriever
from machina.rag.chunker import RecursiveCharacterTextSplitter


def _is_subsequence(source: str, emitted: str) -> bool:
    it = iter(emitted)
    return all(char in it for char in source)


# ---------------------------------------------------------------------------
# P0-2 — short configured secret must never rewrite ordinary prose
# ---------------------------------------------------------------------------


def test_short_configured_secret_does_not_corrupt_prose(monkeypatch):
    import machina.security.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "known_secret_values", lambda: ("a",))
    text = "banana split cat"
    result = redact_secrets(text)
    assert result.text == text
    assert result.redactions == []


def test_one_to_three_character_secrets_are_not_replaceable(monkeypatch):
    import machina.security.secrets as secrets_mod

    for short in ("1", "ab", "xyz"):
        monkeypatch.setattr(secrets_mod, "known_secret_values", lambda s=short: (s,))
        text = f"ordinary prose mentioning {short} once"
        # The exact configured value must survive: it is a placeholder, not a
        # credential, and replacing it would corrupt unrelated text.
        assert redact_secrets(text).text == text


def test_long_configured_secret_is_still_redacted(monkeypatch):
    import machina.security.secrets as secrets_mod

    secret = "configured-value-0123456789"
    assert len(secret) >= redaction_mod.MIN_CONFIGURED_SECRET_LENGTH
    monkeypatch.setattr(secrets_mod, "known_secret_values", lambda: (secret,))
    text = f"prefix {secret} suffix"
    result = redact_secrets(text)
    assert secret not in result.text
    assert "[REDACTED_SECRET]" in result.text
    assert "configured_secret" in result.redactions


def test_streaming_short_secret_does_not_corrupt_prose(monkeypatch):
    import machina.security.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "known_secret_values", lambda: ("a",))
    text = "banana split cat"
    redactor = StreamingSecretRedactor()
    out = redactor.feed(text[:7]) + redactor.feed(text[7:]) + redactor.flush()
    assert out == text


def test_redaction_threshold_is_documented_constant():
    # The literal-replace path needs a documented floor; 1-3 char values are
    # explicitly out of scope (see review/memory-rag.md P0-2).
    assert redaction_mod.MIN_CONFIGURED_SECRET_LENGTH >= 4


# ---------------------------------------------------------------------------
# P0-3 — unscoped persona retrieval must not leak another persona's rows
# ---------------------------------------------------------------------------


def _candidate(agent_id: str = "alice") -> RetrievedMemory:
    return RetrievedMemory(
        memory=MemoryEntry(
            id=uuid.uuid4(),
            session_id=None,
            agent_id=agent_id,
            content="shared fact",
            category="fact",
        ),
        score=0.5,
    )


async def test_persona_store_refuses_unscoped_fact_search(monkeypatch):
    from machina.memory import persona

    fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(persona.db, "fetch", fetch)
    assert await PersonaMemoryStore().search_persona_for_facts([uuid.uuid4()], None, 10) == []
    fetch.assert_not_awaited()


async def test_unscoped_emotion_retrieval_skips_foreign_personas():
    candidates = [_candidate()]
    persona_store = AsyncMock()
    persona_store.search_persona_for_facts.return_value = [
        PersonaMemory(
            id=uuid.uuid4(),
            fact_id=candidates[0].memory.id,
            persona_id="bob",
            interpretation="bob private view",
            confidence=1.0,
        )
    ]
    retriever = MemoryRetriever(persona_store=persona_store)

    out = await retriever._retrieve_with_emotion(candidates, "joy")

    assert out == candidates
    assert out[0].persona_interpretation is None
    persona_store.search_persona_for_facts.assert_not_awaited()


async def test_scoped_emotion_retrieval_still_merges_own_persona():
    candidates = [_candidate("alice")]
    persona_store = AsyncMock()
    persona_store.search_persona_for_facts.return_value = [
        PersonaMemory(
            id=uuid.uuid4(),
            fact_id=candidates[0].memory.id,
            persona_id="alice",
            interpretation="alice view",
            confidence=1.0,
        )
    ]
    retriever = MemoryRetriever(persona_store=persona_store)

    out = await retriever._retrieve_with_emotion(candidates, "joy", persona_id="alice")

    persona_store.search_persona_for_facts.assert_awaited_once_with(
        [candidates[0].memory.id], "alice", 10
    )
    assert out[0].persona_interpretation == "alice view"


async def test_two_personas_unscoped_query_returns_no_rows(db_pool, monkeypatch):
    from machina.memory import persona

    monkeypatch.setattr(persona, "db", db_pool)
    fact_id = uuid.uuid4()
    try:
        await db_pool.execute(
            "INSERT INTO memories (id, agent_id, content, category) VALUES ($1, $2, $3, $4)",
            fact_id,
            "alice",
            "shared fact",
            "fact",
        )
        for owner, view in (("alice", "alice private view"), ("bob", "bob private view")):
            await db_pool.execute(
                "INSERT INTO persona_memories (fact_id, persona_id, interpretation) "
                "VALUES ($1, $2, $3)",
                fact_id,
                owner,
                view,
            )
        store = PersonaMemoryStore()
        assert await store.search_persona_for_facts([fact_id], None, 10) == []
        alice_rows = await store.search_persona_for_facts([fact_id], "alice", 10)
        assert [row.interpretation for row in alice_rows] == ["alice private view"]
    finally:
        await db_pool.execute("DELETE FROM memories WHERE id = $1", fact_id)


# ---------------------------------------------------------------------------
# P0-1 — chunker must not drop the trailing separator run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        ". ,aa,,bab...bab,a. ",
        "alpha. beta. gamma. ",
        "one\n\ntwo\n\nthree. ",
        "tail words here... ",
        "x. y. z. ",
    ],
)
def test_trailing_separator_run_is_emitted(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4, chunk_overlap=0)
    chunks = splitter.split_text(text)
    emitted = "".join(chunk.text for chunk in chunks)
    content = "".join(char for char in text if not char.isspace())
    assert _is_subsequence(content, emitted), f"lost source tail: {text!r} -> {emitted!r}"
    for chunk in chunks:
        assert chunk.text in text  # every emission is a literal source slice


def test_reproduced_recursive_tail_loss_is_fixed():
    """Exact review reproduction: source[18:20] == '. ' used to vanish."""
    text = ". ,aa,,bab...bab,a. "
    splitter = RecursiveCharacterTextSplitter(chunk_size=4, chunk_overlap=4)
    chunks = splitter.split_text(text)
    emitted = "".join(chunk.text for chunk in chunks)
    content = "".join(char for char in text if not char.isspace())
    assert _is_subsequence(content, emitted), f"lost source tail: {emitted!r}"
    assert emitted.count(text[18:20]) >= 1
    # No tail duplication beyond the requested overlap: the last emission is
    # still a literal, non-repeated source slice.
    for chunk in chunks:
        assert chunk.text in text


def test_trailing_gap_is_not_duplicated_across_chunks():
    text = "first part. second part. final part. "
    splitter = RecursiveCharacterTextSplitter(chunk_size=4, chunk_overlap=0)
    chunks = splitter.split_text(text)
    # Every chunk is a distinct source range; concatenation reconstructs the
    # source exactly when there is no overlap (no dropped and no duplicated
    # characters).
    assert "".join(chunk.text for chunk in chunks) == text


# ---------------------------------------------------------------------------
# Compaction summaries are protected evidence under projection
# ---------------------------------------------------------------------------


def _chunk(text: str, *, tokens: int = 0, chunk_type: str = "memory") -> ContextChunk:
    return ContextChunk(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        agent_id="harness",
        chunk_type=chunk_type,
        payload={"content": text},
        token_count=tokens,
    )


@pytest.mark.parametrize(
    "entry",
    [
        (_chunk("compaction summary", tokens=100, chunk_type="compression_summary"), 0.9),
        ({"type": "compression_summary", "payload": {"content": "compaction summary"}}, 0.9),
    ],
)
def test_over_budget_projection_retains_compaction_summary(entry):
    normal = (_chunk("ordinary evidence", tokens=100), 0.8)
    kept, omitted = select_evidence([normal, entry], 50)
    texts = [str(k) for k in kept]
    assert any("compaction summary" in text for text in texts)
    assert omitted == 1


def test_projection_reserves_budget_for_protected_summary():
    normal = (_chunk("ordinary evidence", tokens=40), 0.8)
    summary = (_chunk("compaction summary", tokens=30, chunk_type="compression_summary"), 0.9)
    kept, duplicates, omitted = project_evidence([normal, summary], 50)
    assert duplicates == 0
    assert len(kept) == 1
    assert kept[0][0].chunk_type == "compression_summary"
    assert omitted == 1


def test_protected_summary_survives_even_when_larger_than_budget():
    summary = (_chunk("compaction summary", tokens=500, chunk_type="compression_summary"), 0.9)
    kept, omitted = select_evidence([summary], 10)
    assert kept == [summary]
    # Never counted as omitted: the assembler may truncate, but the projector
    # must not silently drop protective evidence.
    assert omitted == 0
