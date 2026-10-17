"""Session control-plane mutations share the turn coordinator's ownership fence."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from machina.api.routes.models import UpdateSessionRequest
from machina.api.routes.sessions import update_session
from machina.core import turns
from machina.core.config import config
from machina.core.models import Session
from machina.core.session import session_manager
from machina.gateway.errors import TURN_IN_PROGRESS, RpcError
from machina.gateway.features import mode, sessions


@pytest.fixture
def mutation_target(monkeypatch):
    session = Session(
        id=uuid.uuid4(), agent_id="harness", goal="old", model="old", provider="openrouter"
    )
    gateway = SimpleNamespace(require_db=lambda: None, get_session=AsyncMock(return_value=session))
    monkeypatch.setattr("machina.db.connection.db", SimpleNamespace(connected=False))
    monkeypatch.setattr(session_manager, "get", AsyncMock(return_value=session))
    for method in ("set_title", "set_goal", "set_model", "update_state_fields"):
        monkeypatch.setattr(session_manager, method, AsyncMock())
    monkeypatch.setattr(
        "machina.permissions.broker.permission_broker.revoke", AsyncMock(return_value=0)
    )
    return session, gateway


async def mutate(name, session, gateway):
    params = {"sessionId": str(session.id)}
    if name == "http":
        return await update_session(session.id, UpdateSessionRequest(title="new"))
    handler, extra = {
        "mode_set": (mode.mode_set, {"mode": "ask"}),
        "mode_revoke": (mode.mode_revoke, {}),
        "rename": (sessions.session_rename, {"title": "new"}),
        "goal": (sessions.session_set_goal, {"goal": "new"}),
        "model": (sessions.session_set_model, {"model": "new"}),
    }[name]
    return await handler(gateway, {**params, **extra})


@pytest.mark.parametrize("name", ["http", "mode_set", "mode_revoke", "rename", "goal", "model"])
async def test_session_mutation_rejects_a_live_turn(mutation_target, name):
    session, gateway = mutation_target
    token = await turns.try_begin_turn(session.id)
    assert token
    try:
        with pytest.raises((HTTPException, RpcError)) as error:
            await mutate(name, session, gateway)
        if isinstance(error.value, HTTPException):
            assert error.value.status_code == 409
        else:
            assert error.value.code == TURN_IN_PROGRESS
        for method in ("set_title", "set_goal", "set_model", "update_state_fields"):
            getattr(session_manager, method).assert_not_awaited()
    finally:
        await turns.end_turn(session.id, token)


async def test_in_memory_revoke_stays_ask_when_global_default_is_full(mutation_target, monkeypatch):
    from machina.core.session_mode import get_effective_mode

    session, gateway = mutation_target
    monkeypatch.setattr(config, "execution_mode", "full")
    monkeypatch.setenv("MACHINA_EXECUTION_MODE", "full")
    result = await mode.mode_revoke(gateway, {"sessionId": str(session.id)})
    assert result["mode"] == get_effective_mode(str(session.id)) == "ask"


@pytest.mark.parametrize("name", ["http", "mode_set", "mode_revoke", "rename", "goal", "model"])
@pytest.mark.parametrize("outcome", ["ok", "error", "cancel"])
async def test_mutation_owns_slot_and_releases_after_every_exit(
    mutation_target, monkeypatch, name, outcome
):
    import asyncio

    from machina.permissions.broker import permission_broker

    session, gateway = mutation_target

    async def effect(*args, **kwargs):
        assert await turns.try_begin_turn(session.id) is None
        if outcome == "error":
            raise RuntimeError("mutation failed")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return 0

    target, method = (
        (permission_broker, "revoke")
        if name.startswith("mode_")
        else (
            session_manager,
            {"http": "set_title", "rename": "set_title", "goal": "set_goal", "model": "set_model"}[
                name
            ],
        )
    )
    monkeypatch.setattr(target, method, effect)
    if outcome == "ok":
        await mutate(name, session, gateway)
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancel" else RuntimeError):
            await mutate(name, session, gateway)
    token = await turns.try_begin_turn(session.id)
    assert token
    await turns.end_turn(session.id, token)


@pytest.mark.parametrize("name", ["http", "mode_set", "mode_revoke", "rename", "goal", "model"])
async def test_durable_session_mutations_reject_held_turn(db_pool, monkeypatch, name):
    monkeypatch.setattr("machina.db.connection.db", db_pool)
    monkeypatch.setattr("machina.core.session.db", db_pool)
    session = await session_manager.create(title="unchanged", execution_mode="ask")
    gateway = SimpleNamespace(require_db=lambda: None, get_session=AsyncMock(return_value=session))
    token = await turns.try_begin_turn(session.id)
    assert token
    try:
        with pytest.raises((HTTPException, RpcError)) as error:
            await mutate(name, session, gateway)
        if isinstance(error.value, HTTPException):
            assert error.value.status_code == 409
        else:
            assert error.value.code == TURN_IN_PROGRESS
        stored = await session_manager.get_fresh(session.id)
        assert stored.title == "unchanged"
        assert stored.execution_mode == "ask"
    finally:
        await turns.end_turn(session.id, token)
        await session_manager.delete(session.id)
