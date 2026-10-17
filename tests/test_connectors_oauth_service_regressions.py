"""Exercise the durable service wiring used by the runtime, not a resolver fake."""

import asyncio

from machina.connectors.oauth import OAuthCoordinator
from machina.connectors.oauth_store import OAuthCredentialStore
from machina.connectors.service import ConnectorService
from machina.connectors.store import ConnectorStore


async def test_durable_inventory_with_oauth_coordinator_does_not_reenter_refresh(tmp_path):
    coordinator = OAuthCoordinator(
        credential_store=OAuthCredentialStore(tmp_path / "oauth", key_provider=lambda: b"k" * 32)
    )
    store = ConnectorStore(tmp_path / "mcp.json")
    service = ConnectorService(store=store, oauth_coordinator=coordinator)
    try:
        configured = await asyncio.wait_for(
            service.configure({"id": "local", "command": "python3"}), timeout=1
        )
        assert configured["id"] == "local"
        assert (await asyncio.wait_for(service.list(), timeout=1))[0]["id"] == "local"
        restarted = ConnectorService(
            store=store, oauth_coordinator=OAuthCoordinator(credential_store=coordinator._creds)
        )
        assert (await asyncio.wait_for(restarted.list(), timeout=1))[0]["id"] == "local"
        assert await asyncio.wait_for(service.remove("local"), timeout=1)
        assert await service.list() == []
    finally:
        await coordinator.shutdown()


async def test_connect_revalidates_durable_oauth_after_approval(tmp_path, monkeypatch):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    import pytest
    from tests.test_connectors_oauth_refresh import _seed_refresh

    from machina.connectors.execution import connect_authorized, prepare_connect
    from machina.connectors.models import ConnectorToolError
    from machina.core.execution_context import ExecutionContext
    from machina.permissions import broker

    for change in ("forget", "replace", "in_flight"):
        coordinator, config, binding, identity = await _seed_refresh(tmp_path / change)
        service = ConnectorService(oauth_coordinator=coordinator)
        service.registry.add(config).revision = config.revision
        monkeypatch.setattr(
            "machina.core.execution_context.resolve_execution_context",
            AsyncMock(return_value=ExecutionContext("ask", "host", tmp_path)),
        )
        connect = AsyncMock()
        monkeypatch.setattr(service, "connect_and_discover", connect)
        monkeypatch.setattr(service.manager, "status", lambda _: {})

        async def change_authority(_request):
            if change == "forget":
                await coordinator._creds.forget(binding, expected_generation=identity["generation"])
            else:
                record = await coordinator._creds.load(binding)
                updated = (
                    replace(record, generation=record.generation + 1, grant_id="replacement")
                    if change == "replace"
                    else replace(record, refresh_state="in_flight")
                )
                await coordinator._creds.save(updated, expected_generation=record.generation)

        monkeypatch.setattr(broker.permission_broker, "guard", change_authority)
        preview = await prepare_connect(service, "web", "session", "agent")
        assert service.auth_identity("web") == identity
        with pytest.raises(ConnectorToolError, match="changed during approval"):
            await asyncio.wait_for(
                connect_authorized(service, "web", "session", "agent", preview["digest"]), 1
            )
        connect.assert_not_awaited()
        await coordinator.shutdown()
