"""Tests for RPC dispatch through the typed ``Gateway.call`` seam.

Machina 2.0 removed the private ``_RpcGateway`` frame bridge: ``/rpc`` now
reads the shared embedded ``Gateway`` from ``app.state`` and calls its typed
``Gateway.call`` method. These tests keep the HTTP-level contract (auth,
redaction, error mapping, shared instance, concurrency) while treating the
gateway itself as the boundary double.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("MACHINA_API_KEY", "test-key-123")

from machina.api.app import create_app
from machina.gateway.errors import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    NOT_FOUND,
    RpcError,
)

TEST_API_KEY = "test-key-123"
AUTH = {"Authorization": f"Bearer {TEST_API_KEY}"}


class _RouteGateway:
    """Typed fake: the route must call this seam, never ``handle_message``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result: Any = {"ok": True}
        self.error: RpcError | None = None
        self.unexpected: Exception | None = None
        self.delay = 0.0

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, params or {}))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.unexpected is not None:
            raise self.unexpected
        if method == "test.echo":
            return {"i": (params or {}).get("i")}
        return self.result


@pytest.fixture
async def rpc():
    """An app served through ASGI with one typed gateway on app.state."""
    app = create_app()
    fake = _RouteGateway()
    app.state.gateway = fake
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield SimpleNamespace(client=client, gateway=fake, app=app)


class TestRpcDispatch:
    async def test_route_dispatches_through_the_shared_gateway(self, rpc):
        first = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "session.fork", "params": {"a": 1}}
        )
        second = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "session.delete", "params": {"b": 2}}
        )

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first.json() == {"result": {"ok": True}}
        assert second.json() == {"result": {"ok": True}}
        assert rpc.gateway.calls == [
            ("session.fork", {"a": 1}),
            ("session.delete", {"b": 2}),
        ]
        # The route must use the app-state gateway, never build one per request.
        assert rpc.app.state.gateway is rpc.gateway

    async def test_route_applies_redaction_to_results(self, rpc):
        secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
        rpc.gateway.result = {"apiKey": secret, "nested": {"list": [secret]}}

        resp = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "config.get", "params": {}}
        )

        assert resp.status_code == 200, resp.text
        assert secret not in resp.text
        assert resp.json()["result"]["apiKey"] == "[REDACTED_OPENAI_KEY]"
        assert resp.json()["result"]["nested"]["list"] == ["[REDACTED_OPENAI_KEY]"]

    async def test_route_maps_gateway_errors_to_http_status(self, rpc):
        rpc.gateway.error = RpcError(METHOD_NOT_FOUND, "no such method")
        resp = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "nope.method", "params": {}}
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "no such method"

        rpc.gateway.error = RpcError(NOT_FOUND, "missing session")
        resp = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "session.get", "params": {}}
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "missing session"

    async def test_route_hides_internal_error_details(self, rpc):
        rpc.gateway.error = RpcError(INTERNAL_ERROR, "database password leaked")
        resp = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "session.list", "params": {}}
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "RPC dispatch failed"

        rpc.gateway.error = None
        rpc.gateway.unexpected = RuntimeError("database password leaked")
        resp = await rpc.client.post(
            "/rpc", headers=AUTH, json={"method": "session.list", "params": {}}
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "internal error"
        assert "password" not in resp.text

    async def test_route_rejects_prompt_methods(self, rpc):
        for method in ("prompt.submit", "prompt.cancel", "shutdown"):
            resp = await rpc.client.post(
                "/rpc", headers=AUTH, json={"method": method, "params": {}}
            )
            assert resp.status_code == 400, method
            assert "session prompt stream" in resp.json()["detail"]
        # Rejected before dispatch: the gateway never saw a call.
        assert rpc.gateway.calls == []

    async def test_route_handles_concurrent_requests_without_cross_talk(self, rpc):
        rpc.gateway.delay = 0.001

        responses = await asyncio.gather(
            *[
                rpc.client.post(
                    "/rpc",
                    headers=AUTH,
                    json={"method": "test.echo", "params": {"i": index}},
                )
                for index in range(20)
            ]
        )

        assert all(response.status_code == 200 for response in responses)
        assert [response.json()["result"]["i"] for response in responses] == list(
            range(20)
        )
        assert sorted(call[1]["i"] for call in rpc.gateway.calls) == list(range(20))


class TestRpcDispatchWithoutGateway:
    async def test_route_returns_503_when_gateway_missing(self):
        app = create_app()
        app.state.gateway = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/rpc", headers=AUTH, json={"method": "session.list", "params": {}}
            )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "RPC gateway not initialized"
