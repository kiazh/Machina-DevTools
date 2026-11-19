"""OAuth discovery compatibility tests for remote MCP providers."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from machina.connectors import oauth_network
from machina.connectors.oauth_network import OAuthNetworkClient, normalize_resource


class _MetadataTransport:
    """Serve only the Google-shaped OAuth metadata used by these tests."""

    def __init__(self, endpoint: str, issuer: str, resource: str) -> None:
        self.endpoint = endpoint
        self.issuer = issuer
        self.resource = resource

    async def probe(self, endpoint: str, *, timeout: float):
        return 405, {}

    async def fetch(self, url: str, *, headers, timeout, max_bytes):
        path = url.split("://", 1)[-1]
        if url == f"{self.endpoint.split('/mcp/')[0]}/.well-known/oauth-protected-resource/mcp/v1":
            body = {
                "authorization_servers": [self.issuer],
                "resource": self.resource,
                "scopes_supported": ["https://www.googleapis.com/auth/drive.readonly"],
            }
            return 200, {}, json.dumps(body).encode()
        if path == "accounts.google.com/.well-known/oauth-authorization-server":
            return 404, {}, b""
        if path == "accounts.google.com/.well-known/openid-configuration":
            body = {
                "issuer": "https://accounts.google.com",
                "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_endpoint": "https://oauth2.googleapis.com/token",
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
            }
            return 200, {}, json.dumps(body).encode()
        return 404, {}, b""


@pytest.mark.asyncio
async def test_discovery_accepts_google_issuer_trailing_slash(monkeypatch):
    endpoint = "https://drivemcp.googleapis.com/mcp/v1"
    resource = normalize_resource(endpoint)
    transport = _MetadataTransport(
        endpoint=endpoint,
        issuer="https://accounts.google.com/",
        resource=resource,
    )
    config = SimpleNamespace(
        id="google-drive",
        endpoint=endpoint,
        local_access=False,
        oauth={
            "client_mode": "auto",
            "scope_allowlist": ["https://www.googleapis.com/auth/drive.readonly"],
        },
    )

    async def immediate_check(func, *, timeout):
        return func()

    monkeypatch.setattr(oauth_network, "run_dns_check", immediate_check)
    monkeypatch.setattr(
        oauth_network,
        "validate_outbound_url",
        lambda url, *, local_access, allow_http_loopback=True: url,
    )
    monkeypatch.setattr(oauth_network, "_host_is_public", lambda url: True)

    client = OAuthNetworkClient(local_access=False, transport=transport)
    try:
        discovered = await client.discover(
            config, issuer=None, deadline=time.monotonic() + 30
        )
    finally:
        await client.aclose()

    assert normalize_resource("https://accounts.google.com/") == normalize_resource(
        "https://accounts.google.com"
    )
    assert discovered.selected_issuer == "https://accounts.google.com"
    assert discovered.resource == resource
    assert discovered.registration_endpoint == ""
