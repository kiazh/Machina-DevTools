"""Token-efficiency T2: durable artifacts — retention, ACLs, recovery.

Covers the store contract: verbatim originals, paging that reports the true
total, re-authorized reads across the delegation tree, honest expiry, quota
refusals that never evict live evidence, fail-open behavior when storage is
unavailable, and (B0) completeness metadata: ``complete``,
``upstream_truncated``, ``stored_truncated``, counts, and content revision —
with legacy rows reading as unknown, never silently complete.
"""

from __future__ import annotations

import copy
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.artifacts import ArtifactError, artifact_disclosure, artifact_store
from machina.core.config import config
from machina.core.exceptions import ToolError


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
    created: list[uuid.UUID] = [session_id]

    yield SimpleNamespace(pool=db_pool, session_id=session_id)

    await db_pool.execute("DELETE FROM sessions WHERE id = ANY($1::uuid[])", created)


@pytest.mark.asyncio
async def test_store_describe_read_paging(artifact_env):
    content = "".join(str(i % 10) for i in range(100))
    stored = await artifact_store.store(
        "tool:terminal",
        content,
        session_id=artifact_env.session_id,
        agent_id="harness",
        status="completed",
    )
    assert stored is not None
    artifact_id = stored["artifact_id"]
    assert stored["content_bytes"] == 100

    meta = await artifact_store.describe(
        artifact_id, session_id=artifact_env.session_id, agent_id="harness"
    )
    assert meta["kind"] == "tool:terminal"
    assert meta["status"] == "completed"
    assert meta["content_bytes"] == 100

    page = await artifact_store.read(
        artifact_id, session_id=artifact_env.session_id, agent_id="harness", offset=0, limit=10
    )
    assert page["content"] == content[:10]
    assert page["total_chars"] == 100
    assert page["truncated"] is True

    tail = await artifact_store.read(
        artifact_id,
        session_id=artifact_env.session_id,
        agent_id="harness",
        offset=95,
        limit=10,
    )
    assert tail["content"] == content[95:]
    assert tail["truncated"] is False

    with pytest.raises(ArtifactError):
        await artifact_store.read(
            artifact_id, session_id=artifact_env.session_id, agent_id="harness", offset=-1
        )
    with pytest.raises(ArtifactError):
        await artifact_store.read(
            artifact_id, session_id=artifact_env.session_id, agent_id="harness", limit=0
        )


@pytest.mark.asyncio
async def test_search_deterministic_positions_and_no_false_positives(artifact_env):
    content = "first line\nUNIQUE_CONFLICT here\nfiller\nunique_conflict again\n"
    stored = await artifact_store.store(
        "tool:terminal", content, session_id=artifact_env.session_id, agent_id="harness"
    )
    artifact_id = stored["artifact_id"]

    matches = await artifact_store.search(
        "unique_conflict", session_id=artifact_env.session_id, agent_id="harness"
    )
    assert len(matches) == 1  # one entry per artifact, earliest match
    assert matches[0]["artifact_id"] == artifact_id
    # 0-based character index of the earliest match (pins the strpos->index
    # conversion; an off-by-one would fail here).
    assert matches[0]["position"] == content.find("UNIQUE_CONFLICT")
    assert "unique_conflict" in matches[0]["excerpt"].lower()

    assert (
        await artifact_store.search(
            "no-such-token", session_id=artifact_env.session_id, agent_id="harness"
        )
        == []
    )
    with pytest.raises(ArtifactError):
        await artifact_store.search("", session_id=artifact_env.session_id, agent_id="harness")
    with pytest.raises(ArtifactError):
        await artifact_store.search(
            "x" * 500, session_id=artifact_env.session_id, agent_id="harness"
        )


@pytest.mark.asyncio
async def test_access_follows_delegation_tree_not_siblings(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.artifacts.db", db_pool)
    parent = await _mk_session(db_pool)
    child = await _mk_session(db_pool, ancestors=[parent])
    unrelated = await _mk_session(db_pool)
    try:
        stored = await artifact_store.store(
            "tool:terminal",
            "parent evidence " * 20,
            session_id=parent,
            agent_id="harness",
        )
        artifact_id = stored["artifact_id"]

        # Child reads parent artifacts (ancestor in its own snapshot).
        meta = await artifact_store.describe(
            artifact_id, session_id=child, agent_id="harness"
        )
        assert meta["artifact_id"] == artifact_id

        # Parent reads child artifacts (parent appears in the child's snapshot).
        child_stored = await artifact_store.store(
            "tool:terminal", "child evidence " * 20, session_id=child, agent_id="harness"
        )
        await artifact_store.describe(
            child_stored["artifact_id"], session_id=parent, agent_id="harness"
        )

        # Unrelated sessions are denied with a typed error, never a weaker path.
        with pytest.raises(ArtifactError) as excinfo:
            await artifact_store.describe(artifact_id, session_id=unrelated, agent_id="harness")
        assert excinfo.value.code == "denied"
        with pytest.raises(ArtifactError):
            await artifact_store.read(artifact_id, session_id=unrelated, agent_id="harness")
    finally:
        await db_pool.execute(
            "DELETE FROM sessions WHERE id = ANY($1::uuid[])", [parent, child, unrelated]
        )


@pytest.mark.asyncio
async def test_expiry_answers_unavailable_and_purge_removes(artifact_env):
    stored = await artifact_store.store(
        "tool:terminal", "expiring content", session_id=artifact_env.session_id, agent_id="harness"
    )
    artifact_id = stored["artifact_id"]
    await artifact_env.pool.execute(
        "UPDATE tool_artifacts SET expires_at = now() - interval '1 hour' WHERE id = $1",
        uuid.UUID(artifact_id),
    )
    with pytest.raises(ArtifactError) as excinfo:
        await artifact_store.describe(
            artifact_id, session_id=artifact_env.session_id, agent_id="harness"
        )
    assert excinfo.value.code == "expired"

    removed = await artifact_store.purge_expired()
    assert removed >= 1
    assert (
        await artifact_env.pool.fetchval(
            "SELECT COUNT(*) FROM tool_artifacts WHERE id = $1", uuid.UUID(artifact_id)
        )
        == 0
    )


@pytest.mark.asyncio
async def test_quota_refuses_new_stores_without_evicting(artifact_env):
    config.set("artifact_max_per_session", 1)
    first = await artifact_store.store(
        "tool:terminal", "first", session_id=artifact_env.session_id, agent_id="harness"
    )
    second = await artifact_store.store(
        "tool:terminal", "second", session_id=artifact_env.session_id, agent_id="harness"
    )
    assert first is not None
    assert second is None  # refuse, do not silently evict the first
    assert (
        await artifact_env.pool.fetchval(
            "SELECT COUNT(*) FROM tool_artifacts WHERE session_id = $1", artifact_env.session_id
        )
        == 1
    )


@pytest.mark.asyncio
async def test_expired_rows_do_not_consume_quota(artifact_env):
    config.set("artifact_max_per_session", 1)
    first = await artifact_store.store(
        "tool:terminal", "old", session_id=artifact_env.session_id, agent_id="harness"
    )
    assert first is not None
    # Expire the only row: it must stop counting against the quota, so new
    # evidence is never permanently refused by stale rows.
    await artifact_env.pool.execute(
        "UPDATE tool_artifacts SET expires_at = now() - interval '1 hour' WHERE id = $1",
        uuid.UUID(first["artifact_id"]),
    )
    replacement = await artifact_store.store(
        "tool:terminal", "new", session_id=artifact_env.session_id, agent_id="harness"
    )
    assert replacement is not None


@pytest.mark.asyncio
async def test_store_guards_disabled_oversize_and_unavailable(artifact_env, monkeypatch):
    config.set("artifacts_enabled", False)
    assert (
        await artifact_store.store(
            "tool:terminal", "x" * 10, session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )

    config.set("artifacts_enabled", True)
    config.set("artifact_max_chars", 1000)
    # Oversize refuses rather than storing a differently-sized "original".
    assert (
        await artifact_store.store(
            "tool:terminal", "x" * 2000, session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )

    monkeypatch.setattr(
        "machina.core.artifacts.db", SimpleNamespace(connected=False, fetchval=None)
    )
    assert (
        await artifact_store.store(
            "tool:terminal", "y" * 10, session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )
    assert await artifact_store.purge_expired() == 0


@pytest.mark.asyncio
async def test_artifact_tools_page_authorized_originals(db_pool, monkeypatch):
    """The model-facing tools return exact pages and typed denials."""
    monkeypatch.setattr("machina.core.artifacts.db", db_pool)
    import machina.tools.agents as agents_module

    monkeypatch.setattr(agents_module, "db", db_pool)
    session_id = await _mk_session(db_pool)
    try:
        content = "alpha\nbeta RUN_MARKER gamma\n" + "z" * 500
        stored = await artifact_store.store(
            ":tool:terminal", content, session_id=session_id, agent_id="harness"
        )
        artifact_id = stored["artifact_id"]

        token = agents_module.current_session_id.set(session_id)
        token_agent = agents_module.current_agent_id.set("harness")
        try:
            from machina.tools.artifacts import (
                artifact_describe,
                artifact_read,
                artifact_search,
            )

            page = await artifact_read(artifact_id, offset=0, limit=30)
            assert page.startswith(f"[artifact {artifact_id} kind=:tool:terminal chars 0-30 of ")
            assert content[:30] in page

            described = await artifact_describe(artifact_id)
            assert "kind=:tool:terminal" in described
            assert "content_bytes=" in described

            found = await artifact_search("RUN_MARKER", artifact_id=artifact_id)
            assert "pos=" in found and "RUN_MARKER" in found

            missing = await artifact_search("never-there", artifact_id=artifact_id)
            assert missing == "No artifact matches."

            with pytest.raises(ToolError):
                await artifact_read(str(uuid.uuid4()))
        finally:
            agents_module.current_session_id.reset(token)
            agents_module.current_agent_id.reset(token_agent)
    finally:
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", session_id)


def test_artifact_disclosure_shape_is_stable():
    text = artifact_disclosure("abc-123", 4242)
    assert "abc-123" in text and "4242" in text and "artifact_read" in text


# ─── B0: completeness metadata (G1) ─────────────────────────────────────────


async def test_schema_completeness_columns_exist_and_are_nullable(db_pool):
    rows = await db_pool.fetch(
        "SELECT column_name, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'tool_artifacts' AND column_name = ANY($1::text[])",
        [
            "complete",
            "upstream_truncated",
            "stored_truncated",
            "record_count",
            "content_revision",
        ],
    )
    found = {row["column_name"]: row["is_nullable"] for row in rows}
    assert set(found) == {
        "complete",
        "upstream_truncated",
        "stored_truncated",
        "record_count",
        "content_revision",
    }
    # Old rows read as unknown, never silently complete.
    assert all(value == "YES" for value in found.values())


async def test_store_records_and_describe_reads_completeness(artifact_env):
    stored = await artifact_store.store(
        "tool:read_file",
        "content",
        session_id=artifact_env.session_id,
        agent_id="harness",
        source_version="machina:test",
        content_revision="rev-7",
        complete=True,
        upstream_truncated=False,
        record_count=3,
    )
    assert stored is not None
    assert stored["complete"] is True
    assert stored["upstream_truncated"] is False
    assert stored["stored_truncated"] is False
    assert stored["content_revision"] == "rev-7"
    assert stored["record_count"] == 3

    meta = await artifact_store.describe(
        stored["artifact_id"], session_id=artifact_env.session_id, agent_id="harness"
    )
    assert meta["complete"] is True
    assert meta["upstream_truncated"] is False
    assert meta["stored_truncated"] is False
    assert meta["content_revision"] == "rev-7"
    assert meta["record_count"] == 3
    assert meta["source_version"] == "machina:test"


async def test_store_derives_a_deterministic_content_revision(artifact_env):
    first = await artifact_store.store(
        "tool:terminal", "same content", session_id=artifact_env.session_id, agent_id="harness"
    )
    second = await artifact_store.store(
        "tool:terminal", "same content", session_id=artifact_env.session_id, agent_id="harness"
    )
    assert first["content_revision"].startswith("sha256:")
    assert first["content_revision"] == second["content_revision"]


async def test_store_validates_revision_and_flag_input(artifact_env):
    stored = await artifact_store.store(
        "tool:terminal",
        "validated",
        session_id=artifact_env.session_id,
        agent_id="harness",
        source_version=123,  # non-string: unknown, never a fabricated version
        content_revision="   ",  # blank: derive from content instead of storing junk
        complete="yes",  # non-bool: unknown
        upstream_truncated=1,  # non-bool: unknown
        record_count=-5,  # negative: unknown
    )
    assert stored is not None
    meta = await artifact_store.describe(
        stored["artifact_id"], session_id=artifact_env.session_id, agent_id="harness"
    )
    assert meta["source_version"] is None
    assert meta["content_revision"].startswith("sha256:")
    assert meta["complete"] is None
    assert meta["upstream_truncated"] is None
    assert meta["record_count"] is None


async def test_store_derives_complete_from_upstream_flag(artifact_env):
    complete = await artifact_store.store(
        "tool:terminal",
        "derived-complete",
        session_id=artifact_env.session_id,
        agent_id="harness",
        upstream_truncated=False,
    )
    partial = await artifact_store.store(
        "tool:terminal",
        "derived-partial",
        session_id=artifact_env.session_id,
        agent_id="harness",
        upstream_truncated=True,
    )
    assert complete["complete"] is True
    assert partial["complete"] is False


async def test_legacy_rows_read_unknown_not_complete(artifact_env):
    artifact_id = uuid.uuid4()
    await artifact_env.pool.execute(
        "INSERT INTO tool_artifacts (id, session_id, agent_id, kind, content, content_bytes) "
        "VALUES ($1, $2, 'harness', 'tool:terminal', 'legacy row', 10)",
        artifact_id,
        artifact_env.session_id,
    )
    meta = await artifact_store.describe(
        str(artifact_id), session_id=artifact_env.session_id, agent_id="harness"
    )
    assert meta["complete"] is None
    assert meta["upstream_truncated"] is None
    assert meta["stored_truncated"] is None
    assert meta["content_revision"] is None
    assert (
        await artifact_store.lookup(
            str(artifact_id), session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )


async def test_lookup_contract_shape_and_honest_gates(artifact_env):
    complete = await artifact_store.store(
        "tool:read_file",
        "R" * 1200,
        session_id=artifact_env.session_id,
        agent_id="harness",
        complete=True,
        upstream_truncated=False,
    )
    found = await artifact_store.lookup(
        complete["artifact_id"], session_id=artifact_env.session_id, agent_id="harness"
    )
    assert set(found) == {"available", "complete", "handle", "revision", "summary_view"}
    assert found["available"] is True
    assert found["complete"] is True
    assert found["handle"] == complete["artifact_id"]
    assert isinstance(found["revision"], str)
    assert isinstance(found["summary_view"], str)
    assert found["summary_view"]

    partial = await artifact_store.store(
        "tool:terminal",
        "T" * 1200,
        session_id=artifact_env.session_id,
        agent_id="harness",
        upstream_truncated=True,
        complete=False,
    )
    assert (
        await artifact_store.lookup(
            partial["artifact_id"], session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )

    await artifact_env.pool.execute(
        "UPDATE tool_artifacts SET expires_at = now() - interval '1 hour' WHERE id = $1",
        uuid.UUID(complete["artifact_id"]),
    )
    assert (
        await artifact_store.lookup(
            complete["artifact_id"], session_id=artifact_env.session_id, agent_id="harness"
        )
        is None
    )


def test_artifact_disclosure_is_upstream_aware():
    upstream = artifact_disclosure(
        "abc-123", 4242, upstream_truncated=True, complete=False
    )
    assert "truncated" in upstream
    assert "exact original" not in upstream
    unknown = artifact_disclosure("abc-123", 4242, upstream_truncated=None, complete=None)
    assert "exact original" not in unknown
    assert "artifact_read" in unknown


# ─── B2: model-facing tool surface (describe completeness, search window) ───
#
# The tools must not flatten the store's honesty: describe surfaces the
# completeness/revision binding and search states its scan window so an empty
# page can never read as a definitive absence (B0 G5 / spec §14.6, C07/C09).


class _ToolStore:
    """Store double for the tool-surface contract (no DB)."""

    def __init__(self, *, meta: dict | None = None, page: dict | None = None) -> None:
        self.meta = meta or {}
        self.page = page or {}
        self.describe_calls: list[tuple] = []
        self.search_calls: list[dict] = []

    async def describe(self, artifact_id, *, session_id, agent_id):
        self.describe_calls.append((artifact_id, session_id, agent_id))
        return dict(self.meta)

    async def search_page(
        self, query, *, session_id, agent_id, artifact_id=None, limit=5, cursor=None
    ):
        self.search_calls.append(
            {
                "query": query,
                "artifact_id": artifact_id,
                "limit": limit,
                "cursor": cursor,
            }
        )
        return copy.deepcopy(self.page)


@pytest.fixture
def tool_scope(monkeypatch):
    """Active session/agent scope for the artifact tools without a database."""
    import machina.tools.agents as agents_module

    monkeypatch.setattr(
        agents_module, "db", SimpleNamespace(fetchval=AsyncMock(return_value="harness"))
    )
    token = agents_module.current_session_id.set(uuid.uuid4())
    token_agent = agents_module.current_agent_id.set("harness")
    yield
    agents_module.current_session_id.reset(token)
    agents_module.current_agent_id.reset(token_agent)


_DESCRIBE_META = {
    "artifact_id": "abc-123",
    "kind": "tool:read_file",
    "status": "completed",
    "source_version": "machina:test",
    "content_revision": "sha256:deadbeef",
    "complete": True,
    "upstream_truncated": False,
    "stored_truncated": False,
    "record_count": 3,
    "content_bytes": 100,
    "token_estimate": 25,
    "created_at": "2026-10-16T00:00:00+00:00",
    "expires_at": None,
    "access_count": 2,
}


def _kv(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


@pytest.mark.asyncio
async def test_artifact_describe_surfaces_completeness_revision_and_counts(
    tool_scope, monkeypatch
):
    from machina.tools import artifacts as artifact_tools

    store = _ToolStore(meta=_DESCRIBE_META)
    monkeypatch.setattr(artifact_tools, "artifact_store", store)
    fields = _kv(await artifact_tools.artifact_describe("abc-123"))
    assert store.describe_calls[0][0] == "abc-123"
    assert fields["source_version"] == "machina:test"
    assert fields["content_revision"] == "sha256:deadbeef"
    assert fields["complete"] == "True"
    assert fields["upstream_truncated"] == "False"
    assert fields["stored_truncated"] == "False"
    assert fields["record_count"] == "3"
    # Legacy fields stay present.
    assert fields["content_bytes"] == "100"
    assert fields["token_estimate"] == "25"
    assert fields["access_count"] == "2"


@pytest.mark.asyncio
async def test_artifact_describe_reads_tri_state_unknown_never_complete(tool_scope, monkeypatch):
    from machina.tools import artifacts as artifact_tools

    legacy = dict(
        _DESCRIBE_META,
        source_version=None,
        content_revision=None,
        complete=None,
        upstream_truncated=None,
        stored_truncated=None,
        record_count=None,
    )
    monkeypatch.setattr(artifact_tools, "artifact_store", _ToolStore(meta=legacy))
    fields = _kv(await artifact_tools.artifact_describe("legacy"))
    assert fields["complete"] == "unknown"
    assert fields["upstream_truncated"] == "unknown"
    assert fields["stored_truncated"] == "unknown"
    assert fields["record_count"] == "unknown"
    assert fields["source_version"] == "unknown"
    assert fields["content_revision"] == "unknown"


def _search_page(*, matches=None, scanned=50, scan_limit=50, truncated=False, cursor=None):
    return {
        "matches": matches or [],
        "query": "RUN_MARKER",
        "scope": {"kind": "session", "session_id": "s", "artifact_id": None},
        "order": "created_at_desc,id_desc",
        "scanned": scanned,
        "scan_limit": scan_limit,
        "scan_truncated": truncated,
        "next_cursor": cursor,
    }


@pytest.mark.asyncio
async def test_artifact_search_discloses_scan_window_and_cursor(tool_scope, monkeypatch):
    from machina.tools import artifacts as artifact_tools

    match = {
        "artifact_id": "a1",
        "kind": "tool:terminal",
        "total_chars": 100,
        "position": 3,
        "excerpt": "RUN_MARKER here",
    }
    store = _ToolStore(
        page=_search_page(matches=[match], scanned=50, scan_limit=50, truncated=True, cursor="CUR-1")
    )
    monkeypatch.setattr(artifact_tools, "artifact_store", store)
    output = await artifact_tools.artifact_search("RUN_MARKER")
    # Legacy match lines unchanged, with the window appended.
    assert "pos=3" in output and "RUN_MARKER" in output
    assert "scanned=50 scan_limit=50 scan_truncated=true" in output
    assert "next_cursor=CUR-1" in output
    # Legacy call shape: no cursor supplied, limit default preserved.
    assert store.search_calls[0]["cursor"] is None
    assert store.search_calls[0]["limit"] == 5


@pytest.mark.asyncio
async def test_artifact_search_empty_page_discloses_the_unscanned_window(tool_scope, monkeypatch):
    from machina.tools import artifacts as artifact_tools

    monkeypatch.setattr(
        artifact_tools,
        "artifact_store",
        _ToolStore(page=_search_page(scanned=50, scan_limit=50, truncated=True, cursor="CUR-2")),
    )
    output = await artifact_tools.artifact_search("never-there")
    assert output != "No artifact matches."
    assert "an unscanned artifact may still match" in output
    assert "next_cursor=CUR-2" in output
    assert "scan_truncated=true" in output


@pytest.mark.asyncio
async def test_artifact_search_empty_complete_scan_is_definitive_but_disclosed(
    tool_scope, monkeypatch
):
    from machina.tools import artifacts as artifact_tools

    monkeypatch.setattr(
        artifact_tools,
        "artifact_store",
        _ToolStore(page=_search_page(scanned=7, scan_limit=50, truncated=False, cursor=None)),
    )
    output = await artifact_tools.artifact_search("never-there")
    assert "every readable artifact in scope was scanned" in output
    assert "scanned=7 scan_limit=50 scan_truncated=false" in output
    assert "next_cursor" not in output


@pytest.mark.asyncio
async def test_artifact_search_scoped_empty_keeps_legacy_text_and_params(tool_scope, monkeypatch):
    from machina.tools import artifacts as artifact_tools

    artifact_id = str(uuid.uuid4())
    store = _ToolStore(page=_search_page(scanned=1, scan_limit=50, truncated=False, cursor=None))
    monkeypatch.setattr(artifact_tools, "artifact_store", store)
    assert await artifact_tools.artifact_search("never-there", artifact_id=artifact_id) == (
        "No artifact matches."
    )
    assert store.search_calls[0]["artifact_id"] == artifact_id


@pytest.mark.asyncio
async def test_artifact_search_cursor_resumes_the_same_evidence_basis(tool_scope, monkeypatch):
    from machina.tools import artifacts as artifact_tools

    store = _ToolStore(page=_search_page(scanned=10, scan_limit=50, truncated=False))
    monkeypatch.setattr(artifact_tools, "artifact_store", store)
    await artifact_tools.artifact_search("RUN_MARKER", cursor="CUR-9")
    assert store.search_calls[0]["cursor"] == "CUR-9"
