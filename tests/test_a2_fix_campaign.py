"""A2 fix-campaign tests (MCH-001, MCH-010, MCH-011, C2 token basis, L-03).

F-01 (context-window override bypass) is pinned red-first in
``tests/test_context_windows.py``. Every test here fails against the *defect*
itself, not a missing seam: producer-side envelope budgeting for artifact
pages (C5/MCH-001), durable artifact-handle linkage on the ordinary successful
execution path (MCH-010), stable transcript event identity across pending-chunk
flush retries (C1/MCH-011), the tiktoken token basis on the assigned
compaction-input chunk sites (C2), and redacted logging of malformed tool
arguments (L-03). Consumer-side SQL changes (B3: ON CONFLICT/RETURNING in
``add_chunks_batch``) are deliberately NOT assumed — the flush seam is tested
against spoofed batch results.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent import BaseReActAgent, ReActAgent
from machina.core.artifacts import _decode_cursor, _encode_cursor
from machina.core.assembler import get_token_count
from machina.core.config import config
from machina.core.models import LLMResponse, Session, StreamEvent
from machina.permissions.broker import set_approval_handler


def _run(awaitable):
    return asyncio.run(awaitable)


# ─── shared agent-loop harness (mirrors test_te_views_integration.py) ───────


def _wire_session(monkeypatch, tmp_path):
    """Minimal DB-less agent-loop wiring; ``add_chunks_batch`` is captured.

    The consumer stand-in acknowledges every submitted record by id (a healthy
    store) and the capture wrapper deep-copies the batch: ``_flush_pending_chunks``
    clears the live list in place, so recording references would read back empty.
    """
    monkeypatch.setattr(config, "workspace_root", str(tmp_path / "work"))
    monkeypatch.setattr(config, "execution_mode", "ask")
    monkeypatch.setenv("MACHINA_SKILLS_DIR", str(tmp_path / "skills"))
    for field in ("memory_enabled", "rag_enabled", "personality_growth_enabled"):
        monkeypatch.setattr(config, field, False)
    session = Session(id=uuid.uuid4(), context_budget=8000)
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    monkeypatch.setattr("machina.core.agent.session_manager.update_activity", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text", AsyncMock(return_value=[])
    )

    def _ack(chunks):
        return [SimpleNamespace(id=chunk.get("id")) for chunk in chunks]

    add_batch = AsyncMock(side_effect=_ack)

    async def capturing_batch(chunks):
        return await add_batch([copy.deepcopy(chunk) for chunk in chunks])

    monkeypatch.setattr("machina.core.agent.context_manager.add_chunks_batch", capturing_batch)
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(BaseReActAgent, "_reserve_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(BaseReActAgent, "_finish_usage", AsyncMock())
    return session, add_batch


def _provider(calls_holder, tool_calls):
    """Scripted provider: one tool-call turn (if any), then terminal text."""

    class Provider:
        async def complete(self, **kwargs):
            calls_holder.append(copy.deepcopy(kwargs))
            if len(calls_holder) == 1:
                return LLMResponse(
                    content="",
                    model="fake",
                    usage={"total_tokens": 1},
                    tool_calls=tool_calls,
                )
            return LLMResponse(content="done", model="fake", usage={"total_tokens": 1})

        async def stream_complete(self, **kwargs):
            yield StreamEvent(type="done", response=await self.complete(**kwargs))

    return Provider()


class FakeArtifactStore:
    """Capture-only fake for the ``bounded_tool_view`` store seam."""

    def __init__(self, handle: str = "hand-a2-1"):
        self.handle = handle
        self.store_calls: list[dict] = []

    async def store(self, kind, content, *, session_id, agent_id, status=None, **kwargs):
        self.store_calls.append({"kind": kind, "length": len(content)})
        return {"artifact_id": self.handle, "kind": kind, "content_bytes": len(content)}


C2_TEXT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor." * 100


# ─── MCH-001: artifact page envelope budgeting (C5) ─────────────────────────


class FakeReadStore:
    """Imitates ``ArtifactStore.read``: emits ≤ limit chars, cursor-bound."""

    revision = "sha256:test0000000001"
    kind = "tool:grep"

    def __init__(self, total: int):
        self.total = total
        self.calls: list[dict] = []

    async def read(self, artifact_id, *, session_id, agent_id, offset=0, limit=4000, cursor=None):
        self.calls.append({"offset": offset, "limit": limit, "cursor": cursor})
        body_len = max(0, min(int(limit), self.total - offset))
        page = "p" * body_len
        truncated = offset + body_len < self.total
        next_cursor = None
        if truncated:
            next_cursor = _encode_cursor(
                {
                    "v": 1,
                    "t": "read",
                    "artifact": str(artifact_id),
                    "revision": self.revision,
                    "source_version": "machina:test",
                    "expires_at": None,
                    "offset": offset + body_len,
                    "total_chars": self.total,
                }
            )
        return {
            "artifact_id": str(artifact_id),
            "kind": self.kind,
            "status": "completed",
            "offset": offset,
            "limit": limit,
            "total_chars": self.total,
            "returned_chars": len(page),
            "truncated": truncated,
            "content": page,
            "source_version": "machina:test",
            "content_revision": self.revision,
            "expires_at": None,
            "next_cursor": next_cursor,
        }


def _read_tool(monkeypatch, store):
    from machina.tools import artifacts as tool_artifacts

    async def scope():
        return "a2-test-agent", uuid.uuid4()

    monkeypatch.setattr(tool_artifacts, "active_agent_scope", scope)
    monkeypatch.setattr(tool_artifacts, "artifact_store", store)
    return tool_artifacts


def test_default_read_page_fits_view_budget(monkeypatch):
    """A default 4000-char body plus header/cursor stays ≤ the visible cap."""
    from machina.core.tool_views import append_limit

    store = FakeReadStore(total=8000)
    _read_tool(monkeypatch, store)
    from machina.tools.artifacts import artifact_read

    text = _run(artifact_read("11111111-1111-1111-1111-111111111111"))
    budget = append_limit("artifact_read")
    assert budget is not None
    assert len(text) <= budget, f"envelope {len(text)} exceeds view budget {budget}"
    # The continuation cursor survives the envelope intact (its own last line).
    assert text.splitlines()[-1].startswith("next_cursor=")
    header = text.splitlines()[0]
    assert "chars 0-" in header and " of 8000" in header
    # The store's first page was too large once framed; the re-read sized the
    # body to the budget instead of the append seam blind-slicing the cursor.
    assert len(store.calls) > 1, "envelope budgeting must size the body from the budget"


def test_read_cursor_describes_emitted_bytes(monkeypatch):
    """The emitted next_cursor resumes exactly at the end of emitted content."""
    store = FakeReadStore(total=80_000)
    _read_tool(monkeypatch, store)
    from machina.tools.artifacts import artifact_read

    artifact_id = "22222222-2222-2222-2222-222222222222"
    text = _run(artifact_read(artifact_id, limit=20000))
    assert store.calls, "store.read was not called"
    assert store.calls[-1]["limit"] < 4000, (
        "artifact_read must count envelope bytes before choosing the body size "
        f"({store.calls[-1]['limit']} chars requested from the store)"
    )
    lines = text.splitlines()
    assert lines[-1].startswith("next_cursor=")
    content_block = "\n".join(lines[1:-1])
    decoded = _decode_cursor(lines[-1].split("=", 1)[1])
    assert decoded["offset"] == store.calls[-1]["offset"] + len(content_block), (
        f"next_cursor resumes at {decoded['offset']} but emitted content ends at "
        f"{store.calls[-1]['offset'] + len(content_block)}"
    )
    assert decoded["artifact"] == artifact_id


def test_read_refuses_smaller_page_instead_of_blind_slicing(monkeypatch):
    """If the envelope genuinely cannot fit, return a structured refusal."""
    store = FakeReadStore(total=8000)
    _read_tool(monkeypatch, store)
    # Pathological budget: framing + cursor alone cannot fit in 200 chars.
    monkeypatch.setattr("machina.core.tool_views.append_limit", lambda tool_name: 200)
    from machina.tools.artifacts import artifact_read

    text = _run(artifact_read("33333333-3333-3333-3333-333333333333"))
    assert len(text) <= 200
    # A refusal, not a sliced page: no raw body, no partial cursor line.
    assert "pppp" not in text
    assert "next_cursor=" not in text
    lowered = text.lower()
    assert "smaller" in lowered or "limit" in lowered


class FakeSearchStore:
    def __init__(self, matches, cursor):
        self.matches = matches
        self.cursor = cursor
        self.calls: list[dict] = []

    async def search_page(
        self,
        query,
        *,
        session_id,
        agent_id,
        artifact_id=None,
        limit=5,
        cursor=None,
        scan_limit=50,
    ):
        self.calls.append({"limit": limit, "cursor": cursor})
        return {
            "matches": self.matches[:limit],
            "query": query,
            "scope": {"kind": "session", "session_id": str(session_id)},
            "order": "newest",
            "scanned": len(self.matches),
            "scan_limit": scan_limit,
            "scan_truncated": True,
            "next_cursor": self.cursor,
        }


def _search_tool(monkeypatch, store):
    from machina.tools import artifacts as tool_artifacts

    async def scope():
        return "a2-test-agent", uuid.uuid4()

    monkeypatch.setattr(tool_artifacts, "active_agent_scope", scope)
    monkeypatch.setattr(tool_artifacts, "artifact_store", store)
    return tool_artifacts


def test_search_page_fits_view_budget_and_keeps_cursor(monkeypatch):
    """Search pages keep complete match lines inside the budget; cursor last."""
    from machina.core.tool_views import append_limit

    matches = [
        {
            "artifact_id": f"44444444-4444-4444-4444-{i:012d}",
            "kind": "tool:grep",
            "status": "completed",
            "total_chars": 90_000,
            "position": i * 1000,
            "excerpt": "E" * 480,  # store-side excerpt window bound
            "content_revision": "sha256:test0000000001",
            "source_version": "machina:test",
        }
        for i in range(20)
    ]
    cursor = _encode_cursor(
        {
            "v": 1,
            "t": "search",
            "after": {"created_at": "2026-01-01T00:00:00+00:00", "id": "1"},
        }
    )
    store = FakeSearchStore(matches, cursor)
    _search_tool(monkeypatch, store)
    from machina.tools.artifacts import artifact_search

    text = _run(artifact_search("needle", limit=20))
    budget = append_limit("artifact_search")
    assert budget is not None
    assert len(text) <= budget, f"search envelope {len(text)} exceeds budget {budget}"
    lines = text.splitlines()
    for line in lines:
        assert line.startswith(("artifact=", "scanned=", "next_cursor=")), (
            f"partial line emitted: {line[:60]!r}"
        )
    assert lines[-1] == f"next_cursor={cursor}", "cursor must end the search page"
    emitted = sum(1 for line in lines if line.startswith("artifact="))
    # The envelope is budgeted by re-querying a smaller match page: lines stay
    # whole and the cursor (anchored on the last scanned artifact) stays exact.
    assert 1 <= emitted < 20, "over-budget pages must shrink instead of cutting lines"
    assert emitted == store.calls[-1]["limit"]
    assert len(store.calls) > 1, "the tool must re-query with a smaller match limit"


# ─── MCH-010: durable chunk carries the artifact handle on the happy path ───


@pytest.mark.asyncio
async def test_oversized_tool_result_chunk_carries_artifact_handle(tmp_path, monkeypatch):
    """Ordinary successful execution: the durable tool_call chunk is linked."""
    session, add_batch = _wire_session(monkeypatch, tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    # Larger than the durable chunk copy cap (65536) so result_truncated is
    # genuine, and larger than read_file's append cap so an artifact is kept.
    (tmp_path / "work" / "huge.txt").write_text("Z" * 70_000)
    store = FakeArtifactStore("hand-mch10")
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    calls: list = []
    provider = _provider(
        calls,
        [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "huge.txt"})},
            }
        ],
    )

    async def unexpected_approval(card):
        pytest.fail("ordinary workflow requested approval")

    set_approval_handler(unexpected_approval)
    agent = ReActAgent(provider=provider, allowed_tools=["read_file"])
    try:
        response = await agent.run(session.id, "Dump the file", verbose=False)
        assert response.content == "done"
        assert store.store_calls, "oversized result was not retained as an artifact"
        tool_payloads = [
            chunk["payload"]
            for call in add_batch.call_args_list
            for chunk in call.args[0]
            if chunk.get("chunk_type") == "tool_call"
            and chunk["payload"].get("tool") == "read_file"
        ]
        assert tool_payloads, "no durable tool_call chunk was recorded"
        payload = tool_payloads[0]
        assert payload.get("result_truncated") is True
        assert payload.get("artifact_handle") == "hand-mch10", (
            "the durable tool record must carry the link to the retained original"
        )
    finally:
        set_approval_handler(None)


# ─── MCH-010: cancellation finalization shares the retention decision ──────


@pytest.mark.asyncio
async def test_cancellation_finalize_links_artifact_handle(monkeypatch):
    """Finalization (no Phase D view yet) retains + links before recording."""
    agent = _agent()
    store = FakeArtifactStore("hand-cancel")
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    item = {
        "name": "read_file",
        "args": {},
        "result": "Z" * 70_000,
        "elapsed_ms": 1,
        "tc": {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        "call_id": "1",
    }
    pending: list = []
    await agent._record_tool_action_joined(item, uuid.uuid4(), pending, [])
    assert store.store_calls, "finalization must retain the oversized original first"
    payload = pending[0]["payload"]
    assert payload["artifact_handle"] == "hand-cancel"
    assert payload["result_truncated"] is True


# ─── MCH-011: stable transcript event identity across flush retries ────────
class _Row:
    """ContextChunk stand-in exposing ``.id`` (batch RETURNING shape)."""

    def __init__(self, chunk_id):
        self.id = chunk_id


def _agent():
    return ReActAgent(provider=SimpleNamespace())


def _pending(n=3):
    ids = [uuid.uuid4() for _ in range(n)]
    pending = [
        {
            "id": str(chunk_id),
            "session_id": uuid.uuid4(),
            "agent_id": "a2-test-agent",
            "chunk_type": "assistant_message",
            "payload": {"content": f"c{i}"},
            "token_count": 3,
        }
        for i, chunk_id in enumerate(ids)
    ]
    return pending


class ScriptedConsumer:
    """add_chunks_batch stand-in with scripted outcomes; captures attempts."""

    def __init__(self, results):
        self.calls: list[list[dict]] = []
        self._results = list(results)

    async def add_chunks_batch(self, chunks):
        self.calls.append(copy.deepcopy(chunks))
        if not self._results:
            raise AssertionError("unexpected add_chunks_batch attempt")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_ambiguous_flush_verifies_missing_ids_before_clearing(monkeypatch):
    """A flush whose returned rows ⊂ records must verify by id, then clear."""
    agent = _agent()
    pending = _pending(3)
    ids = [p["id"] for p in pending]
    consumer = ScriptedConsumer(
        [
            RuntimeError("connection dropped"),
            [_Row(pending[0]["id"])],  # ambiguous success: only one row returned
        ]
    )
    verify_fetch = AsyncMock(
        return_value=[{"id": uuid.UUID(p["id"])} for p in pending[1:]]
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.add_chunks_batch", consumer.add_chunks_batch
    )
    monkeypatch.setattr(
        "machina.core.agent._context_db", SimpleNamespace(fetch=verify_fetch), raising=False
    )
    assert await agent._flush_pending_chunks(pending) is False
    assert len(pending) == 3, "failed flush must retain every chunk"
    assert await agent._flush_pending_chunks(pending) is True
    assert verify_fetch.await_count == 1, "ambiguous success must be verified by id"
    asked = [str(item) for item in verify_fetch.await_args.args[1]]
    assert set(asked) == set(ids) - {ids[0]}
    assert pending == [], "only AFTER the by-id verification may the queue clear"
    # The retry resubmitted the same identities (never re-allocated).
    assert [c["id"] for c in consumer.calls[1]] == ids


@pytest.mark.asyncio
async def test_unverified_flush_retains_only_unverified_chunks(monkeypatch):
    """Verified ids never re-submit; unverified ids are retained for retry."""
    agent = _agent()
    pending = _pending(3)
    ids = [p["id"] for p in pending]
    consumer = ScriptedConsumer(
        [
            [],  # ambiguous success: nothing reported inserted
            list(pending),
        ]
    )
    verify_fetch = AsyncMock(return_value=[{"id": uuid.UUID(item)} for item in ids[:2]])
    monkeypatch.setattr(
        "machina.core.agent.context_manager.add_chunks_batch", consumer.add_chunks_batch
    )
    monkeypatch.setattr(
        "machina.core.agent._context_db", SimpleNamespace(fetch=verify_fetch), raising=False
    )

    assert await agent._flush_pending_chunks(pending) is False
    assert [p["id"] for p in pending] == [ids[2]], (
        "only verified-durable chunks may leave the queue; unverified ones stay"
    )
    # The retained chunk is retried with the same id and then fully confirms.
    assert await agent._flush_pending_chunks(pending) is True
    assert pending == []


@pytest.mark.asyncio
async def test_flush_retries_resubmit_identical_records(monkeypatch):
    """A failed flush retains the chunk objects; a retry submits identical ids."""
    agent = _agent()
    pending = _pending(2)
    ids = [p["id"] for p in pending]
    consumer = ScriptedConsumer([RuntimeError("dropped"), list(pending)])
    monkeypatch.setattr(
        "machina.core.agent.context_manager.add_chunks_batch", consumer.add_chunks_batch
    )

    assert await agent._flush_pending_chunks(pending) is False
    assert len(pending) == 2
    assert await agent._flush_pending_chunks(pending) is True
    assert pending == []
    assert len(consumer.calls) == 2
    assert [c["id"] for c in consumer.calls[0]] == [c["id"] for c in consumer.calls[1]], (
        "retry attempts must submit identical event identities (never re-allocated)"
    )
    assert [c["id"] for c in consumer.calls[0]] == ids


@pytest.mark.asyncio
async def test_pending_chunks_carry_stable_event_ids(tmp_path, monkeypatch):
    """Enqueue assigns a durable, unique event ``id`` to every pending chunk."""
    session, add_batch = _wire_session(monkeypatch, tmp_path)
    provider = _provider([], [])
    agent = ReActAgent(provider=provider)
    await agent.run(session.id, "hello", verbose=False)
    records = [chunk for call in add_batch.call_args_list for chunk in call.args[0]]
    assert records, "no pending chunks were submitted"
    types = {chunk["chunk_type"]: chunk for chunk in records}
    for key in ("user_message", "assistant_message"):
        assert key in types, f"missing {key} record in the pending batch"
        raw = types[key].get("id")
        assert isinstance(raw, str) and raw, f"{key} pending chunk must carry a stable event id"
        try:
            uuid.UUID(raw)
        except ValueError:
            pytest.fail(f"{key} event id is not a UUID: {raw!r}")
    ids = [chunk["id"] for chunk in records]
    assert len(set(ids)) == len(ids), "event ids must be unique per logical event"


# ─── C2: tiktoken token basis on the assigned compaction-input sites ────────


@pytest.mark.asyncio
async def test_tool_call_chunk_token_count_uses_tiktoken_basis():
    """agent.py durable tool_call chunk counts tokens like compression.py."""
    agent = _agent()
    result_str = C2_TEXT
    item = {
        "name": "read_file",
        "args": {},
        "result": result_str,
        "tc": {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        "call_id": "1",
        "elapsed_ms": 1,
    }
    pending: list = []
    await agent._record_tool_action_joined(item, uuid.uuid4(), pending, [])
    assert pending, "tool_call chunk was not appended to the pending list"
    assert pending[0]["token_count"] == get_token_count(result_str), (
        f"expected tiktoken basis {get_token_count(result_str)}, got {pending[0]['token_count']} "
        f"(len//4 would be {len(result_str) // 4})"
    )


@pytest.mark.asyncio
async def test_terminal_assistant_chunk_token_count_uses_tiktoken_basis(tmp_path, monkeypatch):
    """agent.py terminal assistant_message chunk counts tokens like compression.py."""
    session, add_batch = _wire_session(monkeypatch, tmp_path)

    class Provider:
        async def complete(self, **kwargs):
            return LLMResponse(content=C2_TEXT, model="fake", usage={"total_tokens": 1})

        async def stream_complete(self, **kwargs):
            yield StreamEvent(type="done", response=await self.complete(**kwargs))

    agent = ReActAgent(provider=Provider())
    await agent.run(session.id, "hello", verbose=False)
    records = [chunk for call in add_batch.call_args_list for chunk in call.args[0]]
    assistant = [c for c in records if c["chunk_type"] == "assistant_message"]
    assert assistant, "no assistant_message chunk was recorded"
    assert assistant[0]["payload"]["content"] == C2_TEXT
    assert assistant[0]["token_count"] == get_token_count(C2_TEXT), (
        f"expected tiktoken basis {get_token_count(C2_TEXT)}, got "
        f"{assistant[0]['token_count']} (len//4 would be {len(C2_TEXT) // 4})"
    )


# ─── L-03: malformed tool arguments are logged redacted + bounded ───────────


@pytest.mark.asyncio
async def test_malformed_tool_args_logged_redacted_and_bounded(caplog):
    """Raw malformed tool arguments are logged as a bounded redacted preview.

    ``machina.safe_logging`` (installed at package init) maps secret-shaped
    fragments inside log ARGS, so a bare secret alone would not leak — but the
    raw argument dump is still UNBOUNDED and passes provider-controlled bytes
    to logging whole. L-03 wants the repo's tool-payload redaction approach
    (``machina.memory.redaction``) applied with a bounded preview at the site.
    """
    secret = "sk-proj-SecretChars1234567890SECRET"
    malformed = json.dumps({"cmd": secret, "blob": "b" * 200_000}, sort_keys=True)[:-3]
    response = LLMResponse(
        content="",
        model="fake",
        usage={"total_tokens": 1},
        tool_calls=[
            {"id": "1", "type": "function", "function": {"name": "echo_tool", "arguments": malformed}}
        ],
    )
    agent = _agent()
    with caplog.at_level(logging.DEBUG, logger="machina.core.agent"):
        agen = agent._execute_tool_calls_stream(response, [], [], uuid.uuid4())
        async for _event in agen:
            pass
        text = caplog.text
        records = [
            record for record in caplog.records if "Failed to parse tool arguments" in record.getMessage()
        ]
    assert records, "the parse failure should be logged"
    assert secret not in text, "raw malformed tool arguments leaked unredacted into logs"
    assert "echo_tool" in text
    message = records[0].getMessage()
    assert len(message) <= 2000, (
        f"logged raw tool-args preview is unbounded ({len(malformed)} raw bytes, "
        f"{len(message)} logged) — it must be a bounded redacted preview"
    )


# ─── MCH-011 flush behavior: unbounded-but-verified clearing is idempotent ──


@pytest.mark.asyncio
async def test_flush_of_empty_queue_is_success():
    agent = _agent()
    assert await agent._flush_pending_chunks([]) is True
