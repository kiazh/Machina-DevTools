"""H07 (holistic review 2026-10-15): approval-gated writes must return a
discriminated pending outcome, never a fabricated stored ``MemoryEntry``.

Pre-fix ``MemoryStore.add`` returned a ``MemoryEntry`` whose ``id`` belonged to
``pending_memories``: the ``remember`` tool printed "Memory stored", the
gateway/API echoed an ordinary memory object, and the returned id could never
resolve to the memory approval later created (approval allocates a different
durable id). These pins assert the fixed contract:

* store pending arm: the gate's ``PendingMemory`` — ``id`` = pending row,
  ``memory_id`` = the approved durable memory id once applicable;
* store stored/durable arm: unchanged ``MemoryEntry``;
* tool/gateway/API surfaces report ``pending`` and never claim stored;
* consolidator entries exclude queued (not yet durable) writes.
"""

from __future__ import annotations

import re
import uuid

from machina.core.config import config

_APPROVAL_TARGETS = (
    "machina.memory.store.db",
    "machina.memory.identity.db",
    "machina.memory.approval.db",
)


def _enable_approval(monkeypatch, db_pool) -> None:
    """Route store/identity/approval database work at the test DB, gate ON."""
    for target in _APPROVAL_TARGETS:
        monkeypatch.setattr(target, db_pool)
    monkeypatch.setattr(config, "memory_approval_enabled", True, raising=False)


async def _cleanup_agent(db_pool, agent_id: str) -> None:
    await db_pool.execute("DELETE FROM pending_memories WHERE agent_id = $1", agent_id)
    await db_pool.execute("DELETE FROM memories WHERE agent_id = $1", agent_id)


# ─── Store contract ──────────────────────────────────────────────────────────


async def test_store_add_returns_pending_outcome_under_approval(db_pool, monkeypatch):
    """The pending arm is a PendingMemory: pending id now, approved id later."""
    from machina.memory.approval import ApprovalStatus, PendingMemory, memory_approval_gate
    from machina.memory.models import MemoryEntry
    from machina.memory.store import MemoryStore

    _enable_approval(monkeypatch, db_pool)
    agent_id = f"h07-pending-{uuid.uuid4()}"
    store = MemoryStore()
    try:
        outcome = await store.add(None, agent_id, "Release notes stay concise", "preference")

        # Discriminated from the result type: not a stored-entry lookalike.
        assert isinstance(outcome, PendingMemory)
        assert not isinstance(outcome, MemoryEntry)
        assert outcome.status == ApprovalStatus.PENDING
        assert outcome.memory_id is None  # no durable id yet

        # The returned id names the pending row — and nothing durable.
        pending_row = await db_pool.fetchrow(
            "SELECT status, content FROM pending_memories WHERE id = $1", outcome.id
        )
        assert pending_row is not None
        assert pending_row["status"] == "pending"
        assert await store.get(outcome.id) is None

        # Approval allocates the durable id; the pending record exposes it.
        approved = await memory_approval_gate.approve(outcome.id)
        assert approved is not None
        assert approved.id != outcome.id
        assert approved.content == pending_row["content"]
        assert (
            await db_pool.fetchval(
                "SELECT memory_id FROM pending_memories WHERE id = $1", outcome.id
            )
            == approved.id
        )
        # The originally returned (pending) id still never resolves to a memory.
        assert await store.get(outcome.id) is None

        # The completed record is where the approved memory id is exposed.
        completed = [
            p
            for p in await memory_approval_gate.list_pending(ApprovalStatus.APPROVED, limit=500)
            if p.id == outcome.id
        ]
        assert completed and completed[0].memory_id == approved.id
    finally:
        await _cleanup_agent(db_pool, agent_id)


async def test_store_add_stored_arms_stay_memory_entries(db_pool, monkeypatch):
    """Durable writes (gate OFF, and bypass_approval=True) keep MemoryEntry."""
    from machina.memory.models import MemoryEntry
    from machina.memory.store import MemoryStore

    store = MemoryStore()
    for target in ("machina.memory.store.db", "machina.memory.identity.db"):
        monkeypatch.setattr(target, db_pool)

    # Gate OFF (default): unchanged direct write.
    agent_id = f"h07-direct-{uuid.uuid4()}"
    try:
        entry = await store.add(None, agent_id, "Direct durable fact", "fact")
        assert isinstance(entry, MemoryEntry)
        assert await store.get(entry.id) is not None
    finally:
        await _cleanup_agent(db_pool, agent_id)

    # Gate ON but bypass_approval=True: approval completion / shared-bus copy.
    _enable_approval(monkeypatch, db_pool)
    agent_id = f"h07-bypass-{uuid.uuid4()}"
    try:
        entry = await store.add(
            None, agent_id, "Bypassed durable fact", "fact", bypass_approval=True
        )
        assert isinstance(entry, MemoryEntry)
        assert await store.get(entry.id) is not None
        assert (
            await db_pool.fetchval(
                "SELECT COUNT(*) FROM pending_memories WHERE agent_id = $1", agent_id
            )
            == 0
        )
    finally:
        await _cleanup_agent(db_pool, agent_id)


# ─── Tool surface ────────────────────────────────────────────────────────────


async def test_remember_tool_reports_pending_approval(db_pool, monkeypatch):
    """``remember`` must not claim "Memory stored" while the write is pending."""
    from machina.tools.memory import remember

    _enable_approval(monkeypatch, db_pool)
    agent_id = f"h07-remember-{uuid.uuid4()}"
    try:
        message = await remember(
            content="The staging database resets every night",
            category="fact",
            agent_id=agent_id,
        )
        assert "Memory stored" not in message
        assert "pending approval" in message.lower()
        assert "memory-approve" in message  # actionable review pointer

        match = re.search(r"\[([0-9a-fA-F-]{36})\]", message)
        assert match, message
        pending_id = uuid.UUID(match.group(1))
        assert (
            await db_pool.fetchval(
                "SELECT status FROM pending_memories WHERE id = $1", pending_id
            )
            == "pending"
        )
        # Nothing durable exists; the id is a pending id, not a memory id.
        assert (
            await db_pool.fetchval("SELECT COUNT(*) FROM memories WHERE agent_id = $1", agent_id)
            == 0
        )
    finally:
        await _cleanup_agent(db_pool, agent_id)


# ─── Gateway surface ─────────────────────────────────────────────────────────


async def test_gateway_memory_add_reports_pending(db_pool, monkeypatch):
    """RPC ``memory.add`` exposes the pending write under a pending discriminant."""
    from machina.gateway.features.memory import memory_add

    _enable_approval(monkeypatch, db_pool)
    tag = f"h07-gateway-{uuid.uuid4().hex[:12]}"

    class _GatewayStub:
        def require_db(self) -> None:
            pass

    try:
        response = await memory_add(
            _GatewayStub(), {"content": f"{tag} fact", "category": "fact"}
        )
        assert response["status"] == "pending"
        assert "memory" not in response  # no fabricated stored object
        pending = response["pending"]
        assert pending["status"] == "pending"
        assert (
            await db_pool.fetchval(
                "SELECT COUNT(*) FROM pending_memories WHERE id = $1", uuid.UUID(pending["id"])
            )
            == 1
        )
    finally:
        await db_pool.execute("DELETE FROM pending_memories WHERE content LIKE $1", f"%{tag}%")
        await db_pool.execute("DELETE FROM memories WHERE content LIKE $1", f"%{tag}%")


# ─── Consolidator surface ────────────────────────────────────────────────────


async def test_consolidator_excludes_approval_queued_writes():
    """Queued-for-approval writes must not masquerade as durable entries."""
    from machina.memory.approval import ApprovalStatus, PendingMemory
    from machina.memory.consolidator import MemoryConsolidator
    from machina.memory.models import MemoryEntry

    candidate = MemoryEntry(
        id=uuid.uuid4(),
        session_id=None,
        agent_id="h07-consolidator",
        content="Queued consolidation fact",
        category="fact",
        importance=0.9,
    )
    queued_ids: list[uuid.UUID] = []

    class _ApprovalRoutedStore:
        async def search(self, **kwargs):
            return []

        async def search_by_embedding(self, **kwargs):
            return []

        async def update_access(self, *args, **kwargs):
            raise AssertionError("no dedup access bump expected")

        async def add(self, **kwargs):
            pending_id = uuid.uuid4()
            queued_ids.append(pending_id)
            return PendingMemory(
                id=pending_id,
                memory_id=None,
                content=kwargs["content"],
                category=kwargs["category"],
                importance=kwargs["importance"],
                agent_id=kwargs["agent_id"],
                status=ApprovalStatus.PENDING,
            )

    consolidator = MemoryConsolidator(store=_ApprovalRoutedStore())

    async def fake_extract(*args, **kwargs):
        return [candidate]

    consolidator._extract_memories = fake_extract
    written = await consolidator.consolidate_from_text(
        "The user queued a fact", agent_id="h07-consolidator"
    )

    assert queued_ids  # the write reached the store
    assert written == []  # ...and is not reported as a durable entry


# ─── HTTP API surface ────────────────────────────────────────────────────────


async def test_api_memory_create_reports_pending(db_pool, monkeypatch):
    """HTTP ``POST /api/v1/memory`` must not return a stored-shaped memory."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from httpx import ASGITransport, AsyncClient

    from machina.api.app import create_app

    _enable_approval(monkeypatch, db_pool)
    monkeypatch.setenv("MACHINA_API_KEY", "h07-test-key")
    agent_id = f"h07-api-{uuid.uuid4()}"

    with (
        patch("machina.api.app.db.connect", new_callable=AsyncMock),
        patch("machina.api.app.db.close", new_callable=AsyncMock),
    ):
        app = create_app()
        app.state._rpc_gateway = MagicMock()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory",
                headers={"Authorization": "Bearer h07-test-key"},
                json={"content": "API pending fact", "category": "fact", "agent": agent_id},
            )
    try:
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "pending"
        assert "memory" not in body
        pending_id = uuid.UUID(body["pendingId"])
        assert body["content"] == "API pending fact"
        assert (
            await db_pool.fetchval(
                "SELECT status FROM pending_memories WHERE id = $1", pending_id
            )
            == "pending"
        )
    finally:
        await _cleanup_agent(db_pool, agent_id)
