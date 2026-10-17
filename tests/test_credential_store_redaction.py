"""Stored opaque credentials must be redacted on all shared output surfaces."""

import pytest
from tests.test_connectors_oauth_store import _binding, _record

from machina.connectors.oauth_store import OAuthCredentialStore
from machina.memory.redaction import redact_secrets
from machina.security import copilot_auth, secrets


@pytest.mark.parametrize("operation", ["save", "load"])
def test_copilot_store_registers_exact_token_bytes(tmp_path, operation):
    token = "ghu_opaque-credential-for-redaction"
    path = tmp_path / "copilot.json"
    copilot_auth.save_tokens({"token": token}, path)
    if operation == "load":
        secrets._redaction_values.clear()
        assert copilot_auth.load_tokens(path)["token"] == token
    assert token in secrets.known_secret_values()
    assert token not in redact_secrets(f"provider echoed {token}").text


@pytest.mark.parametrize("operation", ["save", "load"])
async def test_mcp_store_registers_access_and_refresh_tokens(tmp_path, operation):
    store = OAuthCredentialStore(tmp_path / "oauth", key_provider=lambda: b"k" * 32)
    binding = _binding()
    record = _record(
        binding, access_token="opaque-access-credential", refresh_token="opaque-refresh-credential"
    )
    await store.save(record, expected_generation=0)
    if operation == "load":
        secrets._redaction_values.clear()
        assert (await store.load(binding)).access_token == record.access_token
    for token in (record.access_token, record.refresh_token):
        assert token in secrets.known_secret_values()
        assert token not in redact_secrets(f"provider echoed {token}").text
