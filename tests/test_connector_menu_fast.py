"""Fast checks for the two local MCP menu accelerators.

`cached_tools_list` must never re-list over the network, and startup
auto-connect must never launch stdio servers or open an authorization flow.
"""

from __future__ import annotations

import asyncio

from machina.connectors.models import ConnectorConfig, DiscoveredTool
from machina.connectors.service import ConnectorService


def _http(connector_id: str) -> ConnectorConfig:
    return ConnectorConfig(
        id=connector_id,
        display_name=connector_id,
        transport="streamable_http",
        endpoint=f"https://{connector_id}.example.com/mcp",
    )


def test_cached_tools_read_the_registry_without_listing_the_server() -> None:
    async def run() -> None:
        service = ConnectorService(store=None)
        service.registry.add(_http("vercel"))
        service.registry.set_tools(
            "vercel",
            [
                DiscoveredTool(
                    connector_id="vercel",
                    name="",
                    remote_name="list_projects",
                    description="List projects",
                    schema={"type": "object"},
                )
            ],
        )
        page = await service.cached_tools_list("vercel")
        assert [tool["remote_name"] for tool in page["tools"]] == ["list_projects"]
        assert page["tools"][0]["name"] == "vercel__list_projects"

    asyncio.run(run())


def test_startup_auto_connect_never_opens_authorization_without_cached_credentials() -> None:
    async def run() -> None:
        class FakeOAuth:
            def auth_identity(self, connector_id: str):
                return None

            async def refresh_auth_cache(self, connector_id: str, *, config=None, wait_for_refresh=True):
                return None

        service = ConnectorService(store=None, oauth_coordinator=FakeOAuth())
        service.registry.add(
            ConnectorConfig(
                id="oauth-app",
                display_name="OAuth app",
                transport="streamable_http",
                endpoint="https://oauth.example.com/mcp",
                oauth={"client_mode": "auto"},
            )
        )
        called: list[str] = []

        async def fake_connect(connector_id: str, **_: object):
            called.append(connector_id)
            return []

        service.connect_and_discover = fake_connect  # type: ignore[method-assign]
        result = await service.auto_connect_authorized()
        assert result == {"oauth-app": "authorization_required"}
        assert called == []

    asyncio.run(run())


def test_startup_auto_connect_skips_stdio_and_never_needs_a_browser() -> None:
    async def run() -> None:
        service = ConnectorService(store=None)
        service.registry.add(_http("vercel"))
        service.registry.add(
            ConnectorConfig(id="local", transport="stdio", command="node")
        )
        connected: list[str] = []

        async def fake_connect(connector_id: str, **_: object):
            connected.append(connector_id)
            return []

        service.connect_and_discover = fake_connect  # type: ignore[method-assign]
        result = await service.auto_connect_authorized()
        assert result == {"vercel": "connected"}
        assert connected == ["vercel"]

    asyncio.run(run())


async def test_firecrawl_catalog_uses_hosted_oauth():
    from machina.connectors.store import ConnectorStore
    from machina.gateway.features import METHODS

    result = await METHODS["mcp.catalog"](object(), {})
    entry = next(item for item in result["entries"] if item["id"] == "firecrawl")
    config = ConnectorStore.validate_input(entry["config"])
    assert config.endpoint == "https://mcp.firecrawl.dev/v2/mcp-oauth"
    assert config.oauth["client_mode"] == "auto"
    assert entry["setup"]["type"] == "oauth"
    assert config.local_access is False


def test_capability_brief_lists_connected_mcp_connectors_for_routing(monkeypatch):
    import uuid

    from machina.core import runtime as runtime_module
    from machina.core.models import Session
    from machina.core.self_knowledge import capability_context

    class FakeConfig:
        id = "vercel"
        enabled = True

    class FakeTool:
        name = "vercel__list_projects"
        remote_name = "list_projects"

    class FakeRegistry:
        def list(self):
            return [FakeConfig()]

        def tools_for(self, connector_id):
            return [FakeTool()] if connector_id == "vercel" else []

    class FakeManager:
        def status(self, connector_id):
            return {"state": "connected"}

    class FakeService:
        registry = FakeRegistry()
        manager = FakeManager()

    monkeypatch.setattr(
        runtime_module.runtime_services,
        "_connector_service",
        FakeService(),
        raising=False,
    )
    context = capability_context(
        Session(id=uuid.uuid4()),
        tool_names={"mcp_list", "mcp_call"},
        growth_state="none",
    )
    assert "MCP connectors in this runtime" in context
    assert "vercel (connected; tools: list_projects)" in context
    assert "mcp_list with connector=<id>" in context
