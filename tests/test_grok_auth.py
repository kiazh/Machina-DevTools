"""Grok subscription auth/provider regression checks."""

from __future__ import annotations

import time

from machina.core.provider.grok import GrokProvider
from machina.security import grok_auth


def test_grok_token_store_roundtrip(tmp_path):
    path = tmp_path / "grok.json"
    grok_auth._save(
        {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 3600,
            "account_id": "xai-account",
        },
        path,
    )
    loaded = grok_auth.load_tokens(path)
    assert loaded and loaded["access_token"] == "access-token"
    assert loaded["refresh_token"] == "refresh-token"
    status = grok_auth.status(path)
    assert status["signed_in"] is True
    assert status["account_id"] == "xai-account"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_grok_provider_uses_proxy_headers_and_responses_input(tmp_path):
    provider = GrokProvider(model="grok-4.7", auth_path=str(tmp_path / "grok.json"))
    headers = provider._headers({"access_token": "a.b.c", "account_id": "acct"})
    assert headers["x-grok-client-identifier"] == "grok-shell"
    assert headers["x-grok-user-id"] == "acct"
    assert headers["x-xai-token-auth"] == "xai-grok-cli"

    payload = provider._payload(
        [{"role": "user", "content": "hello"}],
        "grok-4.7",
        None,
        None,
    )
    assert payload["model"] == "grok-4.7"
    assert payload["input"][0]["role"] == "system"
    assert payload["input"][1]["role"] == "user"
    assert payload["reasoning"]["summary"] == "detailed"
