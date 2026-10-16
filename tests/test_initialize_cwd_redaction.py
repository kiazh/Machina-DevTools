"""Initialize redaction contract for the workspace root (T11 upstream fix).

The workspace feature needs a real absolute root. The gateway's ``initialize``
historically returns only the redacted basename (so an unauthenticated caller
cannot learn usernames/machine layout). The fix discloses the absolute launch
directory as ``cwdFull`` ONLY to a handshake that presented the valid token;
the bare/token-less surfaces keep exactly the old redaction.
"""

from __future__ import annotations

import json
import os

import pytest

from machina.gateway.server import Gateway

# ``initialize`` connects the database, so these tests need a DSN. DevTools CI
# and local runs set MACHINA_TEST_DATABASE_URL; a DSN-less environment skips
# with an explicit reason instead of failing on DATABASE_UNAVAILABLE.
pytestmark = pytest.mark.skipif(
    not os.environ.get("MACHINA_TEST_DATABASE_URL"),
    reason="initialize connects the database (set MACHINA_TEST_DATABASE_URL)",
)


class _Harness:
    """Minimal in-process gateway driver (mirrors test_gateway.Harness)."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.gateway = Gateway(self.frames.append)
        self._next_id = 0

    async def call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        frame = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params or {},
        }
        await self.gateway.handle_line(json.dumps(frame))
        matches = [item for item in self.frames if item.get("id") == self._next_id]
        assert matches, f"no response frame for {method}"
        return matches[-1]


async def test_authenticated_initialize_discloses_the_absolute_cwd() -> None:
    h = _Harness()
    h.gateway._auth_token = "secret-token"
    response = await h.call("initialize", {"token": "secret-token"})
    result = response["result"]
    assert result["cwd"] == os.path.basename(os.getcwd())
    assert result["cwdFull"] == os.getcwd()


async def test_bare_initialize_keeps_only_the_redacted_basename() -> None:
    h = _Harness()
    h.gateway._auth_token = "secret-token"
    response = await h.call("initialize")  # bare handshake: no token
    result = response["result"]
    assert "cwdFull" not in result
    assert result["cwd"] == os.path.basename(os.getcwd())


async def test_tokenless_local_mode_keeps_only_the_redacted_basename() -> None:
    h = _Harness()
    response = await h.call("initialize")
    assert "cwdFull" not in response["result"]


async def test_wrong_token_does_not_disclose_the_absolute_cwd() -> None:
    h = _Harness()
    h.gateway._auth_token = "secret-token"
    response = await h.call("initialize", {"token": "wrong-token"})
    if "result" in response:
        assert "cwdFull" not in response["result"]
    else:
        assert response["error"]["code"] == 1005  # UNAUTHORIZED
