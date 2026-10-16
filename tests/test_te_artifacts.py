"""Token-efficiency T2: durable artifacts — retention, ACLs, recovery.

Covers the store contract: verbatim originals, paging that reports the true
total, re-authorized reads across the delegation tree, honest expiry, quota
refusals that never evict live evidence, and fail-open behavior when storage
is unavailable.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

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
            from machina.tools.artifacts import artifact_describe, artifact_read, artifact_search

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
