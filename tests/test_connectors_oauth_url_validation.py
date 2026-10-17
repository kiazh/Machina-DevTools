"""Malformed OAuth URL authorities must use structured boundary errors."""

import asyncio

import pytest
from tests.test_connectors_oauth_network import StubTransport, _config, _discovery_routes

from machina.connectors.oauth_models import OAuthConfigError, OAuthSettings
from machina.connectors.oauth_network import (
    OAuthNetworkClient,
    OAuthNetworkError,
    issuer_metadata_candidates,
    normalize_resource,
    protected_resource_candidates,
    validate_outbound_url,
)

BAD_URLS = [
    "https://auth.example:99999/tenant",
    "https://auth.example:abc/tenant",
    "https://[::1/tenant",
]


@pytest.mark.parametrize("url", BAD_URLS)
@pytest.mark.parametrize("field", ["issuer", "client_metadata_url"])
def test_bad_configured_url_authority_is_typed(url, field):
    fields = {field: url}
    if field == "client_metadata_url":
        fields["client_mode"] = "metadata_document"
    with pytest.raises(OAuthConfigError):
        OAuthSettings.from_dict(fields)


@pytest.mark.parametrize("url", BAD_URLS)
@pytest.mark.parametrize(
    "helper", [normalize_resource, issuer_metadata_candidates, protected_resource_candidates]
)
def test_bad_metadata_url_authority_is_typed(url, helper):
    with pytest.raises(OAuthNetworkError):
        helper(url)


@pytest.mark.parametrize("url", BAD_URLS)
def test_outbound_bad_port_fails_before_dns(monkeypatch, url):
    from machina.connectors import oauth_network

    monkeypatch.setattr(
        oauth_network, "classify_url_host", lambda *a, **k: pytest.fail("DNS reached")
    )
    with pytest.raises(OAuthNetworkError):
        validate_outbound_url(url, local_access=False)


@pytest.mark.parametrize("source", ["issuer", "resource"])
async def test_discovery_bad_remote_url_is_typed(source):
    transport = StubTransport()
    if source == "resource":
        _discovery_routes(transport, resource=BAD_URLS[0])
    else:
        _discovery_routes(transport)
    client = OAuthNetworkClient(transport=transport, local_access=True)
    with pytest.raises(OAuthNetworkError):
        await client.discover(
            _config(),
            issuer=BAD_URLS[0] if source == "issuer" else None,
            deadline=asyncio.get_running_loop().time() + 5,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"issuer": "https://auth.example/tenant?unexpected=value"},
        {
            "client_mode": "preregistered",
            "client_id": "client",
            "client_metadata_url": "https://auth.example/doc",
        },
    ],
)
def test_conflicting_oauth_url_settings_are_rejected(fields):
    with pytest.raises(OAuthConfigError):
        OAuthSettings.from_dict(fields)
