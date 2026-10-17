"""Agent B2 campaign regressions: approvals, OAuth logout/step-up, transport, env.

Every test names the defect it pins and is written to fail (or, for already
correct behavior, to pin the invariant) against the pre-fix code.
"""

from __future__ import annotations

import logging
import os
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# ─── C4 / P0-12: approval ownership is proven, never skipped ────────────────


def _approval_record(**overrides):
    record = {
        "request_id": "req-1",
        "session_id": "11111111-1111-1111-1111-111111111111",
        "agent_id": "harness",
        "turn_id": "turn-1",
        "target": "",
        "digest": "digest",
        "status": "pending",
    }
    record.update(overrides)
    return record


async def test_http_resolve_without_session_id_is_rejected(monkeypatch):
    from fastapi import HTTPException

    from machina.api.routes import approvals as route
    from machina.api.routes.models import ResolveApprovalRequest
    from machina.permissions import store as st

    async def get_approval(_request_id):
        return _approval_record()

    async def must_not_resolve(*_args, **_kwargs):
        raise AssertionError("resolve_decision ran without proven ownership")

    monkeypatch.setattr(st, "get_approval", get_approval)
    monkeypatch.setattr(st, "resolve_decision", must_not_resolve)

    with pytest.raises(HTTPException) as exc:
        await route.resolve_approval(
            "req-1", ResolveApprovalRequest(verdict="approved")
        )

    assert exc.value.status_code == 400
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail["type"] == "approval.ownership_required"


async def test_http_resolve_rejects_a_foreign_session_id(monkeypatch):
    from fastapi import HTTPException

    from machina.api.routes import approvals as route
    from machina.api.routes.models import ResolveApprovalRequest
    from machina.permissions import store as st

    async def get_approval(_request_id):
        return _approval_record()

    monkeypatch.setattr(st, "get_approval", get_approval)
    with pytest.raises(HTTPException) as exc:
        await route.resolve_approval(
            "req-1",
            ResolveApprovalRequest(
                verdict="approved", sessionId=uuid.UUID(int=2)
            ),
        )
    assert exc.value.status_code == 403


async def test_http_resolve_accepts_the_stored_binding(monkeypatch):
    from machina.api.routes import approvals as route
    from machina.api.routes.models import ResolveApprovalRequest
    from machina.permissions import store as st

    session_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    seen: list[tuple] = []

    async def get_approval(_request_id):
        return _approval_record()

    async def resolve_decision(request_id, verdict, principal):
        seen.append((request_id, verdict, principal))
        return True

    monkeypatch.setattr(st, "get_approval", get_approval)
    monkeypatch.setattr(st, "resolve_decision", resolve_decision)

    result = await route.resolve_approval(
        "req-1", ResolveApprovalRequest(verdict="approved", sessionId=session_id)
    )

    assert result["approval"] == {"requestId": "req-1", "status": "approved"}
    assert seen == [("req-1", "approved", "http")]


class _ApprovalGateway:
    def __init__(self):
        self._pending_approvals: dict = {}
        self.writes: list[dict] = []

    def require_db(self):
        return None

    def _write(self, payload):
        self.writes.append(payload)

    def event_payload(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


async def test_gateway_resolve_without_session_id_is_rejected(monkeypatch):
    from machina.gateway.errors import INVALID_PARAMS, RpcError
    from machina.gateway.features.mode import approvals_resolve
    from machina.permissions import store as st

    async def get_approval(_request_id):
        return _approval_record()

    async def must_not_resolve(*_args, **_kwargs):
        raise AssertionError("resolve_decision ran without proven ownership")

    monkeypatch.setattr(st, "get_approval", get_approval)
    monkeypatch.setattr(st, "resolve_decision", must_not_resolve)

    with pytest.raises(RpcError) as exc:
        await approvals_resolve(
            _ApprovalGateway(), {"requestId": "req-1", "verdict": "approved"}
        )

    assert exc.value.code == INVALID_PARAMS
    assert "sessionId is required" in exc.value.message


async def test_gateway_resolve_rejects_a_foreign_session_id(monkeypatch):
    from machina.gateway.errors import RpcError
    from machina.gateway.features.mode import approvals_resolve
    from machina.permissions import store as st

    async def get_approval(_request_id):
        return _approval_record()

    monkeypatch.setattr(st, "get_approval", get_approval)
    with pytest.raises(RpcError) as exc:
        await approvals_resolve(
            _ApprovalGateway(),
            {
                "requestId": "req-1",
                "verdict": "approved",
                "sessionId": str(uuid.UUID(int=2)),
            },
        )
    assert "does not belong" in exc.value.message


# ─── P0-14: forget() surfaces partial cleanup failures ─────────────────────


async def test_forget_reports_disconnect_and_drop_tool_failures(tmp_path, monkeypatch):
    from machina.connectors import oauth as oauth_module
    from tests.test_connectors_oauth_flow import _owner
    from tests.test_connectors_oauth_refresh import _seed_refresh

    coordinator, _config, _binding, identity = await _seed_refresh(tmp_path)
    recorded: list[str] = []
    monkeypatch.setattr(
        oauth_module, "record_failure", lambda op, _exc: recorded.append(op)
    )

    class Manager:
        async def disconnect(self, _connector_id):
            raise RuntimeError("disconnect failed")

    class Registry:
        def _drop_tools(self, _connector_id):
            raise RuntimeError("catalog drop failed")

    coordinator._service = SimpleNamespace(manager=Manager(), registry=Registry())

    result = await coordinator.forget(
        "web",
        _owner(),
        expected_generation=int(identity.get("generation") or 0),
        revoke=False,
    )

    assert result["partial"] is True
    assert {failure["step"] for failure in result["failures"]} == {
        "disconnect",
        "drop_tools",
    }
    assert "oauth.forget.disconnect" in recorded
    assert "oauth.forget.drop_tools" in recorded


async def test_forget_reports_a_clean_run_as_complete(tmp_path):
    from tests.test_connectors_oauth_flow import _owner
    from tests.test_connectors_oauth_refresh import _seed_refresh

    coordinator, _config, _binding, identity = await _seed_refresh(tmp_path)

    class Manager:
        async def disconnect(self, _connector_id):
            return True

    class Registry:
        def _drop_tools(self, _connector_id):
            return 0

    coordinator._service = SimpleNamespace(manager=Manager(), registry=Registry())
    result = await coordinator.forget(
        "web",
        _owner(),
        expected_generation=int(identity.get("generation") or 0),
        revoke=False,
    )
    assert result["partial"] is False
    assert result["failures"] == []


# ─── P0-15: /rpc DB errors are sanitized, correlated, and released ─────────


class _FakeDB:
    def __init__(self, *, fail_connect=False, fail_schema=False):
        self.connected = False
        self.close_calls = 0
        self.fail_connect = fail_connect
        self.fail_schema = fail_schema

    async def connect(self):
        self.connected = True
        if self.fail_connect:
            raise RuntimeError("host=db.internal user=admin password=do-not-leak")

    async def initialize_runtime_schema(self):
        if self.fail_schema:
            raise RuntimeError("relation secret_table does not exist")

    async def close(self):
        self.close_calls += 1
        self.connected = False


async def test_initialize_connect_failure_is_sanitized_and_closes_pool(
    monkeypatch, caplog
):
    from machina.gateway import server as server_module
    from machina.gateway.errors import DATABASE_UNAVAILABLE, RpcError

    fake = _FakeDB(fail_connect=True)
    monkeypatch.setattr(server_module, "db", fake)
    gateway = server_module.Gateway(write=lambda _frame: None, owns_db=False)

    with (
        caplog.at_level(logging.ERROR, logger="machina.gateway.server"),
        pytest.raises(RpcError) as exc,
    ):
        await gateway._initialize({})

    assert exc.value.code == DATABASE_UNAVAILABLE
    assert "do-not-leak" not in exc.value.message
    assert "db.internal" not in exc.value.message
    assert "reference:" in exc.value.message
    reference = exc.value.message.rsplit("reference:", 1)[1].strip(" )")
    assert reference and reference in caplog.text
    assert fake.close_calls == 1


async def test_initialize_schema_failure_is_sanitized_and_closes_pool(
    monkeypatch, caplog
):
    from machina.gateway import server as server_module
    from machina.gateway.errors import DATABASE_UNAVAILABLE, RpcError

    fake = _FakeDB(fail_schema=True)
    monkeypatch.setattr(server_module, "db", fake)
    gateway = server_module.Gateway(write=lambda _frame: None, owns_db=False)

    with (
        caplog.at_level(logging.ERROR, logger="machina.gateway.server"),
        pytest.raises(RpcError) as exc,
    ):
        await gateway._initialize({})

    assert exc.value.code == DATABASE_UNAVAILABLE
    assert "secret_table" not in exc.value.message
    assert "reference:" in exc.value.message
    reference = exc.value.message.rsplit("reference:", 1)[1].strip(" )")
    assert reference and reference in caplog.text
    assert gateway._db_ready is False
    assert fake.close_calls == 1


# ─── P0-1: remote MCP transports require https:// ──────────────────────────


@pytest.mark.parametrize(
    "endpoint,local_access",
    [
        ("http://127.0.0.1:8080/mcp", True),
        ("http://localhost:8080/mcp", True),
        ("http://[::1]:8080/mcp", True),
        ("https://mcp.example.com/mcp", False),
    ],
)
def test_connector_transport_accepts_encrypted_or_consented_loopback(
    endpoint, local_access
):
    from machina.connectors.models import ConnectorConfig

    ConnectorConfig(
        id="web",
        transport="streamable_http",
        endpoint=endpoint,
        local_access=local_access,
    ).validate()


@pytest.mark.parametrize(
    "endpoint,local_access",
    [
        ("http://mcp.example.com/mcp", False),
        ("http://10.0.0.5/mcp", True),
        ("http://127.0.0.1:8080/mcp", False),
        ("http://localhost:8080/mcp", False),
    ],
)
def test_connector_transport_rejects_plaintext_remote_or_unconsented_loopback(
    endpoint, local_access
):
    from machina.connectors.models import ConfigError, ConnectorConfig

    with pytest.raises(ConfigError):
        ConnectorConfig(
            id="web",
            transport="streamable_http",
            endpoint=endpoint,
            local_access=local_access,
        ).validate()


# ─── F-25: documented Machina OAuth identifiers ────────────────────────────


def test_oauth_identifiers_match_documented_machina_names():
    from machina.connectors import oauth_network
    from machina.connectors.oauth_store import OAuthCredentialStore
    from machina.security import oauth_keys

    assert oauth_keys.MASTER_KEY_ENV == "MACHINA_OAUTH_MASTER_KEY"
    assert oauth_keys.KEYRING_SERVICE == "Machina/MCP-OAuth"
    assert oauth_keys.KEYRING_ACCOUNT == "master-key-v1"
    assert oauth_network.CLIENT_NAME == "Machina"
    assert OAuthCredentialStore(key_provider=lambda: b"k" * 32).root == (
        Path.home() / ".machina" / "oauth"
    )


# ─── F-26: 403 insufficient_scope challenge is exposed, never replayed ─────


class _Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _HTTPError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"{status_code} Forbidden")
        self.response = _Response(status_code, headers)


async def test_403_insufficient_scope_is_captured_and_exposed(tmp_path):
    pytest.importorskip("mcp")
    from machina.connectors.mcp_client import MCPConnection
    from machina.connectors.models import ConnectionState
    from tests.test_connectors_oauth_flow import _owner
    from tests.test_connectors_oauth_refresh import _seed_refresh

    class Session:
        async def call_tool(self, _name, _arguments):
            raise _HTTPError(
                403,
                {
                    "WWW-Authenticate": (
                        'Bearer error="insufficient_scope", '
                        'scope="docs:read docs:write docs:read"'
                    )
                },
            )

    coordinator, config, _binding, _identity = await _seed_refresh(tmp_path)
    connection = MCPConnection(config)
    connection._session = Session()
    connection._schemas = {"echo": {"type": "object"}}
    outcome = await connection.call_tool("echo", {})

    assert outcome.outcome_unknown is True
    assert connection.state == ConnectionState.AUTH_REQUIRED
    assert connection.auth_status == 403
    assert connection.auth_challenge_scopes == ("docs:read", "docs:write")

    manager = SimpleNamespace(connection=lambda _connector_id: connection)
    coordinator._service = SimpleNamespace(manager=manager)

    view = await coordinator._public_auth(config)
    assert view["requiredScopes"] == ["docs:read", "docs:write"]
    assert view["authChallenge"]["error"] == "insufficient_scope"

    status = await coordinator.status("web", _owner())
    assert status["auth"]["requiredScopes"] == ["docs:read", "docs:write"]


async def test_401_does_not_claim_a_scope_challenge(tmp_path):
    pytest.importorskip("mcp")
    from machina.connectors.mcp_client import MCPConnection
    from machina.connectors.models import ConnectionState

    class Session:
        async def call_tool(self, _name, _arguments):
            raise _HTTPError(401, {"WWW-Authenticate": "Bearer"})

    from machina.connectors.models import ConnectorConfig

    connection = MCPConnection(
        ConnectorConfig(
            id="web",
            transport="streamable_http",
            endpoint="https://mcp.example.com/mcp",
        ).validate()
    )
    connection._session = Session()
    connection._schemas = {"echo": {"type": "object"}}
    outcome = await connection.call_tool("echo", {})

    assert outcome.outcome_unknown is True
    assert connection.state == ConnectionState.AUTH_REQUIRED
    assert connection.auth_challenge_scopes == ()
    view = connection.auth_challenge_view()
    assert view is not None and "requiredScopes" not in view


# ─── F-24: .env mode is 0600; redirects stay review-gated ─────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_env_file_read_tightens_a_legacy_world_readable_file(tmp_path):
    from machina.security import env_file

    target = tmp_path / ".env"
    target.write_text("LEGACY=1\n", encoding="utf-8")
    target.chmod(0o644)

    assert env_file.read_env_values(target) == {"LEGACY": "1"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_env_file_write_creates_and_repairs_0600(tmp_path):
    from machina.security import env_file

    target = tmp_path / ".env"
    env_file.set_env_values({"A_KEY": "1"}, target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    target.chmod(0o644)
    env_file.set_env_values({"B_KEY": "2"}, target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert env_file.read_env_values(target) == {"A_KEY": "1", "B_KEY": "2"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell grammar")
@pytest.mark.parametrize(
    "command",
    [
        "git log --oneline -5 2>/dev/null",
        "echo ok 2>&1",
        "echo ok > output.txt",
        "cat < input.txt",
        "echo $(pwd)",
        "echo `pwd`",
        "echo ok &",
    ],
)
def test_redirects_and_expansions_still_require_review(command):
    from machina.permissions.shell import parse_terminal_command, simple_shell_commands

    parsed = parse_terminal_command(command)
    assert parsed.shell == "sh"
    assert parsed.payload == command
    assert simple_shell_commands(command) is None


# ─── P0-17: web tools use one redacted identity for key and stored source ──


async def test_crawl_stores_the_same_redacted_source_used_for_dedup(monkeypatch):
    from machina.memory.redaction import redact_secrets
    from machina.tools import web_crawl as crawl_module

    raw = "https://example.com/guide?api_key=abcdefghijklmnopqrstuvw123456"
    stored = redact_secrets(raw).text
    sid = uuid.uuid4()
    pipeline = SimpleNamespace(
        index_document=AsyncMock(return_value=[object()])
    )
    monkeypatch.setattr(
        crawl_module, "active_agent_scope", AsyncMock(return_value=("harness", sid))
    )
    monkeypatch.setattr(
        crawl_module, "acquire_rag_pipeline", AsyncMock(return_value=pipeline)
    )
    monkeypatch.setattr(crawl_module, "release_rag_pipeline", AsyncMock())
    exists = AsyncMock(return_value=False)
    monkeypatch.setattr(crawl_module.db, "fetchval", exists)
    monkeypatch.setattr(
        crawl_module,
        "fetch_page_html",
        AsyncMock(return_value="<html><body>page text</body></html>"),
    )

    await crawl_module.web_crawl(raw, max_depth=0)

    assert exists.await_args.args[-1] == stored
    indexed = pipeline.index_document.await_args.kwargs
    assert indexed["source"].source == stored
    assert indexed["metadata"]["url"] == stored
    assert indexed["metadata"]["crawl_url"] == stored


async def test_research_stores_the_same_redacted_source_used_for_dedup(monkeypatch):
    from machina.memory.redaction import redact_secrets
    from machina.tools import web_research as research_module

    raw = "https://example.com/guide?api_key=abcdefghijklmnopqrstuvw123456"
    stored = redact_secrets(raw).text
    sid = uuid.uuid4()
    pipeline = SimpleNamespace(
        index_document=AsyncMock(return_value=[object()])
    )
    monkeypatch.setattr(
        research_module,
        "active_agent_scope",
        AsyncMock(return_value=("harness", sid)),
    )
    monkeypatch.setattr(
        research_module,
        "_search_web_results",
        AsyncMock(return_value=[("Reference", raw, "")]),
    )
    monkeypatch.setattr(
        research_module, "extract_url_text", AsyncMock(return_value="page text")
    )
    monkeypatch.setattr(
        research_module, "acquire_rag_pipeline", AsyncMock(return_value=pipeline)
    )
    monkeypatch.setattr(research_module, "release_rag_pipeline", AsyncMock())
    exists = AsyncMock(return_value=False)
    monkeypatch.setattr(research_module.db, "fetchval", exists)

    await research_module.web_research("guide")

    assert exists.await_args.args[-1] == stored
    indexed = pipeline.index_document.await_args.kwargs
    assert indexed["source"].source == stored
    assert indexed["metadata"]["url"] == stored
