"""Grok/Copilot auth control plane registration and status payloads."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from machina.gateway.features import METHODS
from machina.gateway.features import auth_subscriptions
from machina.security import copilot_auth, grok_auth


def test_subscription_auth_methods_are_registered():
    for method in (
        "auth.grok.status",
        "auth.grok.login",
        "auth.grok.logout",
        "auth.copilot.status",
        "auth.copilot.login",
        "auth.copilot.importGh",
        "auth.copilot.logout",
    ):
        assert method in METHODS


def test_grok_status_never_returns_tokens(monkeypatch):
    monkeypatch.setattr(
        grok_auth,
        "status",
        lambda: {
            "signed_in": True,
            "source": "machina",
            "expired": False,
            "expires_at": 123.0,
            "account_id": "acct",
            "last_refresh": "now",
            "access_token": "secret",
        },
    )
    result = asyncio.run(auth_subscriptions.auth_grok_status(SimpleNamespace(), {}))
    assert result["status"]["accountId"] == "acct"
    assert "access_token" not in result["status"]
    assert "secret" not in str(result)


def test_copilot_status_projects_public_fields(monkeypatch):
    monkeypatch.setattr(
        copilot_auth,
        "status",
        lambda: {
            "signed_in": True,
            "source": "gh",
            "token_family": "fine-grained",
            "account": "octocat",
            "last_refresh": "now",
            "token": "secret",
        },
    )
    result = asyncio.run(auth_subscriptions.auth_copilot_status(SimpleNamespace(), {}))
    assert result["status"]["accountId"] == "octocat"
    assert result["status"]["tokenFamily"] == "fine-grained"
    assert "token" not in result["status"]
