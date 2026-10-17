"""Combined #33: user forks are independent, not delegated subagents."""

from __future__ import annotations

import uuid

from machina.core.session import SessionManager
from machina.gateway.serializers import session_to_dict


async def test_fork_is_marked_and_scrubs_cursor_state(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.session.db", db_pool)
    manager = SessionManager()
    source = await manager.create(title="fork source", agent_id="harness")
    source.state = {
        "mem_consolidated_up_to": str(uuid.uuid4()),
        "stagnation_stop": {"count": 3},
        "last_auto_compact_tokens": 1234,
        "keep": "yes",
    }
    # Persist the state through the normal session state path.
    await db_pool.execute(
        "UPDATE sessions SET state_msgpack = $2 WHERE id = $1",
        source.id,
        __import__("msgpack").packb(source.state, use_bin_type=True),
    )
    try:
        fork = await manager.fork(source.id)
        assert fork.is_fork is True
        assert fork.state == {"keep": "yes"}
        payload = session_to_dict(fork)
        assert payload["parentSessionId"] is None
        assert payload["isFork"] is True
    finally:
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", source.id)
