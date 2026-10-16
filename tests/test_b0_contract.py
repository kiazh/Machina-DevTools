"""B0 artifact contract — completeness, store-failure honesty, handle reader, cursors.

Spec §14.2/§14.6 and constraints C06/C07/C09. These tests pin the seams B1/B2
build on:

- the store records ``complete``/``upstream_truncated``/``stored_truncated``
  honestly, derives a content revision, and keeps unknown rows unknown;
- a refused/failed store yields an explicit ``no_artifact`` marker with no
  recoverable view (C06);
- audited producers' in-band truncation markers become
  ``upstream_truncated=true``/``complete=false`` (C07);
- a stored ``artifact_handle`` renders (bounded preview + pointer) for a
  prior-turn ``tool_call`` chunk;
- read/search cursors bind revision, scope, filters, order, and retention.
"""

from __future__ import annotations

import base64
import json
import uuid
from types import SimpleNamespace

import pytest

from machina.core.artifacts import ArtifactError, artifact_store
from machina.core.assembler import PromptAssembler
from machina.core.config import config
from machina.core.reducers import VIEW_MARKER
from machina.core.tool_views import bounded_tool_view


async def _mk_session(pool, *, agent_id: str = "harness", ancestors: list[uuid.UUID] | None = None):
    session_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO sessions (id, agent_id, ancestor_session_ids, ancestor_agent_ids) "
        "VALUES ($1, $2, $3::uuid[], $4::text[])",
        session_id,
        agent_id,
        ancestors or [],
        [agent_id] * len(ancestors or []),
    )
    return session_id


@pytest.fixture
async def artifact_env(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.artifacts.db", db_pool)
    session_id = await _mk_session(db_pool)
    yield SimpleNamespace(pool=db_pool, session_id=session_id)
    await db_pool.execute("DELETE FROM sessions WHERE id = $1", session_id)


class _RecordingStore:
    """Captures store inputs; returns a stable fake handle."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def store(self, kind, content, **kwargs):
        self.calls.append({"kind": kind, "content": content, **kwargs})
        return {"artifact_id": "hand-x", "kind": kind, "content_bytes": len(content)}


class _RefusingStore:
    async def store(self, *args, **kwargs):
        return None


def _decode_cursor(cursor: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))


def _encode_cursor(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


# ─── G1: audited producer truncation markers ────────────────────────────────


@pytest.mark.parametrize(
    ("tool_name", "result_tail"),
    [
        ("terminal", "boom\n[output truncated]"),
        ("read_file", "\n[read_file: showing lines 1-2 of 9; use offset=3 to continue]"),
        ("learn_directory", "\n[digest truncated]"),
    ],
)
async def test_audited_producer_truncation_marks_upstream_and_incomplete(
    tool_name, result_tail, monkeypatch
):
    store = _RecordingStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    result = "x" * 3000 + result_tail
    view, artifact_id = await bounded_tool_view(
        tool_name, result, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id == "hand-x"
    call = store.calls[-1]
    assert call["upstream_truncated"] is True
    assert call["complete"] is False
    # C07: the model-visible text never claims an exact original here.
    assert "exact original" not in view
    assert "artifact_read" in view
    assert VIEW_MARKER not in view  # upstream-truncated results take no reducer view


@pytest.mark.parametrize("tool_name", ["terminal", "read_file", "learn_directory"])
async def test_unmarked_audited_producer_is_complete(tool_name, monkeypatch):
    store = _RecordingStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    view, artifact_id = await bounded_tool_view(
        tool_name, "y" * 3000, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id == "hand-x"
    call = store.calls[-1]
    assert call["upstream_truncated"] is False
    assert call["complete"] is True
    assert "artifact_read" in view


async def test_unaudited_producer_stays_unknown_never_complete(monkeypatch):
    store = _RecordingStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    view, artifact_id = await bounded_tool_view(
        "grep", "z" * 3000, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id == "hand-x"
    call = store.calls[-1]
    assert call["upstream_truncated"] is None
    assert call["complete"] is None
    # Unknown completeness is disclosed; nothing claims an exact original.
    assert "exact original" not in view
    assert "artifact_read" in view
    assert VIEW_MARKER not in view


async def test_source_version_reaches_the_store(monkeypatch):
    store = _RecordingStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    _view, artifact_id = await bounded_tool_view(
        "terminal",
        "w" * 3000,
        session_id=uuid.uuid4(),
        agent_id="agent",
        source_version="machina:test",
    )
    assert artifact_id == "hand-x"
    assert store.calls[-1]["source_version"] == "machina:test"


# ─── G2: store failure → explicit unrecoverable marker, no lossy view ───────


async def test_store_failure_yields_no_artifact_marker_and_no_recoverable_view(monkeypatch):
    monkeypatch.setattr("machina.core.tool_views.artifact_store", _RefusingStore())
    view, artifact_id = await bounded_tool_view(
        "read_file", "B" * 5000, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id is None
    assert "no_artifact" in view
    assert "unrecoverable" in view
    assert "re-run" in view
    # C06: nothing here claims a recoverable original.
    assert "artifact_read" not in view
    assert VIEW_MARKER not in view
    assert len(view) <= 1200  # cap + the bounded marker (a constant)
    assert view.startswith("B" * 100)


async def test_store_disabled_yields_marker_without_calling_the_store(monkeypatch):
    class _UncalledStore:
        async def store(self, *args, **kwargs):
            pytest.fail("store must not be called when artifacts are disabled")

    monkeypatch.setattr("machina.core.tool_views.artifact_store", _UncalledStore())
    config.set("artifacts_enabled", False)
    view, artifact_id = await bounded_tool_view(
        "terminal", "T" * 4000, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id is None
    assert "no_artifact" in view
    assert len(view) <= 1200


async def test_exempt_paging_tools_keep_a_plain_cap_without_marker(monkeypatch):
    monkeypatch.setattr("machina.core.tool_views.artifact_store", _RecordingStore())
    view, artifact_id = await bounded_tool_view(
        "artifact_read", "A" * 20_000, session_id=uuid.uuid4(), agent_id="agent"
    )
    assert artifact_id is None
    assert len(view) <= 4000
    assert "no_artifact" not in view


# ─── G4: stored handle renders for prior-turn chunks ────────────────────────


def test_prior_turn_tool_call_renders_preview_and_handle():
    assembler = PromptAssembler(session_budget=8000)
    chunk = {
        "type": "tool_call",
        "payload": {
            "tool": "terminal",
            "args": {"command": "pytest -q"},
            "result_preview": "12 passed in 0.42s",
            "result": "12 passed in 0.42s",
            "artifact_handle": "hand-abc",
        },
    }
    text = assembler._compress_chunk(chunk)
    assert "[tool_call] terminal(" in text
    assert "12 passed in 0.42s" in text
    assert "hand-abc" in text
    assert "artifact_read" in text


def test_prior_turn_tool_call_without_handle_shape_is_unchanged():
    assembler = PromptAssembler(session_budget=8000)
    chunk = {"type": "tool_call", "payload": {"tool": "terminal", "args": {"command": "ls"}}}
    assert assembler._compress_chunk(chunk) == "[tool_call] terminal(command=ls)"


def test_prior_turn_truncated_stored_copy_discloses_the_cap():
    assembler = PromptAssembler(session_budget=8000)
    chunk = {
        "type": "tool_call",
        "payload": {
            "tool": "terminal",
            "args": {},
            "result_preview": "tail -5",
            "result": "x" * 65536,
            "result_truncated": True,
            "artifact_handle": "hand-trunc",
        },
    }
    text = assembler._compress_chunk(chunk)
    assert "hand-trunc" in text
    assert "truncated" in text.lower()


def test_prior_turn_handle_reader_stays_bounded_without_a_preview():
    assembler = PromptAssembler(session_budget=8000)
    chunk = {
        "type": "tool_call",
        "payload": {
            "tool": "read_file",
            "args": {"path": "big.py"},
            "result": "Q" * 65_536,
            "result_truncated": True,
            "artifact_handle": "hand-big",
        },
    }
    text = assembler._compress_chunk(chunk)
    assert len(text) < 1200  # never inlines the stored 65k copy
    assert "hand-big" in text


# ─── G5: recovery cursors (DB) ──────────────────────────────────────────────


async def test_lookup_refuses_across_sessions(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.artifacts.db", db_pool)
    owner = await _mk_session(db_pool)
    stranger = await _mk_session(db_pool)
    try:
        stored = await artifact_store.store(
            "tool:terminal",
            "S" * 1200,
            session_id=owner,
            agent_id="harness",
            complete=True,
            upstream_truncated=False,
        )
        assert (
            await artifact_store.lookup(
                stored["artifact_id"], session_id=stranger, agent_id="harness"
            )
            is None
        )
    finally:
        await db_pool.execute("DELETE FROM sessions WHERE id = ANY($1::uuid[])", [owner, stranger])


async def test_read_cursor_binds_and_resumes(artifact_env):
    content = "".join(str(i % 10) for i in range(5000))
    stored = await artifact_store.store(
        "tool:terminal",
        content,
        session_id=artifact_env.session_id,
        agent_id="harness",
        source_version="machina:test",
        complete=True,
        upstream_truncated=False,
    )
    first = await artifact_store.read(
        stored["artifact_id"], session_id=artifact_env.session_id, agent_id="harness", limit=2000
    )
    assert first["content"] == content[:2000]
    assert first["source_version"] == "machina:test"
    assert first["content_revision"] == stored["content_revision"]
    assert first["next_cursor"]

    resumed = await artifact_store.read(
        stored["artifact_id"],
        session_id=artifact_env.session_id,
        agent_id="harness",
        limit=2000,
        cursor=first["next_cursor"],
    )
    assert resumed["offset"] == 2000
    assert resumed["content"] == content[2000:4000]

    last = await artifact_store.read(
        stored["artifact_id"],
        session_id=artifact_env.session_id,
        agent_id="harness",
        limit=2000,
        cursor=resumed["next_cursor"],
    )
    assert last["content"] == content[4000:]
    assert last["next_cursor"] is None


async def test_read_cursor_rejects_foreign_artifact_and_malformed_tokens(artifact_env):
    first = await artifact_store.store(
        "tool:terminal", "A" * 100, session_id=artifact_env.session_id, agent_id="harness"
    )
    second = await artifact_store.store(
        "tool:terminal", "B" * 100, session_id=artifact_env.session_id, agent_id="harness"
    )
    cursor = (
        await artifact_store.read(
            first["artifact_id"], session_id=artifact_env.session_id, agent_id="harness", limit=10
        )
    )["next_cursor"]

    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.read(
            second["artifact_id"],
            session_id=artifact_env.session_id,
            agent_id="harness",
            cursor=cursor,
        )
    assert excinfo.value.code == "invalid_cursor"

    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.read(
            first["artifact_id"],
            session_id=artifact_env.session_id,
            agent_id="harness",
            cursor="not-a-cursor",
        )
    assert excinfo.value.code == "invalid_cursor"


async def test_read_cursor_rejects_revision_and_retention_drift(artifact_env):
    stored = await artifact_store.store(
        "tool:terminal", "C" * 300, session_id=artifact_env.session_id, agent_id="harness"
    )
    cursor = (
        await artifact_store.read(
            stored["artifact_id"], session_id=artifact_env.session_id, agent_id="harness", limit=10
        )
    )["next_cursor"]

    payload = _decode_cursor(cursor)
    payload["revision"] = "tampered"
    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.read(
            stored["artifact_id"],
            session_id=artifact_env.session_id,
            agent_id="harness",
            cursor=_encode_cursor(payload),
        )
    assert excinfo.value.code == "stale_cursor"

    payload = _decode_cursor(cursor)
    payload["expires_at"] = "2099-01-01T00:00:00+00:00"
    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.read(
            stored["artifact_id"],
            session_id=artifact_env.session_id,
            agent_id="harness",
            cursor=_encode_cursor(payload),
        )
    assert excinfo.value.code == "stale_cursor"


async def test_search_page_binds_filters_and_continues(artifact_env):
    for index in range(3):
        await artifact_store.store(
            "tool:terminal",
            f"MARK-{index} payload " * 10,
            session_id=artifact_env.session_id,
            agent_id="harness",
            source_version="machina:test",
            complete=True,
            upstream_truncated=False,
        )

    page = await artifact_store.search_page(
        "MARK-",
        session_id=artifact_env.session_id,
        agent_id="harness",
        scan_limit=2,
    )
    assert len(page["matches"]) == 2
    assert page["scanned"] == 2
    assert page["scan_limit"] == 2
    assert page["scan_truncated"] is True
    assert page["next_cursor"]
    assert page["scope"] == {
        "kind": "session",
        "session_id": str(artifact_env.session_id),
        "artifact_id": None,
    }
    assert page["order"] == "created_at_desc,id_desc"
    assert all(match["content_revision"] for match in page["matches"])
    assert all(match["source_version"] == "machina:test" for match in page["matches"])

    resumed = await artifact_store.search_page(
        "MARK-",
        session_id=artifact_env.session_id,
        agent_id="harness",
        scan_limit=2,
        cursor=page["next_cursor"],
    )
    assert len(resumed["matches"]) == 1
    assert resumed["scan_truncated"] is False
    assert resumed["next_cursor"] is None
    assert resumed["matches"][0]["artifact_id"] not in {
        match["artifact_id"] for match in page["matches"]
    }


async def test_search_cursor_rejects_tampered_filters_or_scope(artifact_env, db_pool, monkeypatch):
    for index in range(2):
        await artifact_store.store(
            "tool:terminal",
            f"MARK-{index} payload " * 10,
            session_id=artifact_env.session_id,
            agent_id="harness",
        )
    page = await artifact_store.search_page(
        "MARK-", session_id=artifact_env.session_id, agent_id="harness", scan_limit=1
    )
    cursor = page["next_cursor"]

    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.search_page(
            "OTHER",
            session_id=artifact_env.session_id,
            agent_id="harness",
            scan_limit=1,
            cursor=cursor,
        )
    assert excinfo.value.code == "invalid_cursor"

    stranger = await _mk_session(db_pool)
    try:
        with pytest.raises(ArtifactError) as excinfo:
            await artifact_store.search_page(
                "MARK-", session_id=stranger, agent_id="harness", scan_limit=1, cursor=cursor
            )
        assert excinfo.value.code == "invalid_cursor"
    finally:
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", stranger)


async def test_search_page_continues_after_an_early_match_stop(artifact_env):
    for index in range(3):
        await artifact_store.store(
            "tool:terminal",
            f"MARK-{index} payload " * 10,
            session_id=artifact_env.session_id,
            agent_id="harness",
        )
    # One match per page must still leave a cursor for the unscanned candidates.
    page = await artifact_store.search_page(
        "MARK-", session_id=artifact_env.session_id, agent_id="harness", limit=1, scan_limit=3
    )
    assert len(page["matches"]) == 1
    assert page["scanned"] == 1
    assert page["scan_truncated"] is True
    assert page["next_cursor"]

    resumed = await artifact_store.search_page(
        "MARK-",
        session_id=artifact_env.session_id,
        agent_id="harness",
        limit=2,
        scan_limit=3,
        cursor=page["next_cursor"],
    )
    assert len(resumed["matches"]) == 2
    assert resumed["next_cursor"] is None


async def test_search_cursor_rejects_a_changed_evidence_basis(artifact_env):
    for index in range(2):
        await artifact_store.store(
            "tool:terminal",
            f"MARK-{index} payload " * 10,
            session_id=artifact_env.session_id,
            agent_id="harness",
        )
    page = await artifact_store.search_page(
        "MARK-", session_id=artifact_env.session_id, agent_id="harness", scan_limit=1
    )
    cursor = page["next_cursor"]
    head_id = _decode_cursor(cursor)["head"]["id"]

    # A tampered revision on the head artifact must refuse the cursor.
    await artifact_env.pool.execute(
        "UPDATE tool_artifacts SET content_revision = 'drifted' WHERE id = $1", uuid.UUID(head_id)
    )
    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.search_page(
            "MARK-",
            session_id=artifact_env.session_id,
            agent_id="harness",
            scan_limit=1,
            cursor=cursor,
        )
    assert excinfo.value.code == "stale_cursor"

    # A vanished head artifact leaves the basis unverifiable: refuse, never guess.
    await artifact_env.pool.execute(
        "DELETE FROM tool_artifacts WHERE id = $1", uuid.UUID(head_id)
    )
    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.search_page(
            "MARK-",
            session_id=artifact_env.session_id,
            agent_id="harness",
            scan_limit=1,
            cursor=cursor,
        )
    assert excinfo.value.code == "stale_cursor"


async def test_search_scan_window_discloses_truncation(artifact_env):
    for index in range(3):
        await artifact_store.store(
            "tool:terminal",
            f"payload {index} " * 10,
            session_id=artifact_env.session_id,
            agent_id="harness",
        )
    page = await artifact_store.search_page(
        "never-there", session_id=artifact_env.session_id, agent_id="harness", scan_limit=1
    )
    assert page["matches"] == []
    assert page["scan_truncated"] is True
    assert page["next_cursor"]
