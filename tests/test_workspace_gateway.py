"""T5 workspace gateway handler tests (plan §19/§24C relevant rows).

Handler-level tests with a minimal fake gateway exposing only what the
feature module touches: ``require_db`` (no-op), ``_write`` (frame capture) and
``add_method``. Params arrive as untrusted JSON; validation and closed
allowlists are asserted here, not in the service.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from machina.gateway import features
from machina.gateway.errors import INVALID_PARAMS, NOT_FOUND, RpcError
from machina.gateway.features import workspace as ws

WORKSPACE_METHODS = {
    "workspace.open",
    "workspace.list",
    "workspace.read",
    "workspace.save",
    "workspace.documentOpen",
    "workspace.documentChange",
    "workspace.documentClose",
    "workspace.intelligence",
    "workspace.lspStatus",
    "workspace.lspControl",
    "workspace.close",
}


class FakeGateway:
    """Minimal gateway surface for handler tests."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.methods: dict = {}
        self.workspace_service = None

    def require_db(self) -> None:
        return None

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)

    def add_method(self, name, handler) -> None:
        self.methods[name] = handler


async def _opened(tmp_path: Path, *, trust: bool = True) -> tuple[FakeGateway, dict, str]:
    gw = FakeGateway()
    opened = await ws.workspace_open(gw, {"root": str(tmp_path), "trust": trust})
    return gw, opened, opened["workspaceId"]


# ─── registry and open ───────────────────────────────────────────────────────
def test_workspace_methods_registered():
    assert WORKSPACE_METHODS <= set(features.METHODS)
    gw = FakeGateway()
    features.register(gw)
    assert WORKSPACE_METHODS <= set(gw.methods)


async def test_open_requires_explicit_trust(tmp_path):
    gw = FakeGateway()
    with pytest.raises(RpcError) as missing:
        await ws.workspace_open(gw, {"root": str(tmp_path)})
    assert missing.value.code == INVALID_PARAMS

    with pytest.raises(RpcError) as typed:
        await ws.workspace_open(gw, {"root": str(tmp_path), "trust": "yes please"})
    assert typed.value.code == INVALID_PARAMS

    untrusted = await ws.workspace_open(gw, {"root": str(tmp_path), "trust": False})
    assert untrusted["trusted"] is False
    assert "intelligence" not in untrusted["capabilities"]
    assert "lspControl" not in untrusted["capabilities"]
    assert "save" in untrusted["capabilities"]
    assert untrusted["resolvedRoot"] == os.path.realpath(tmp_path)
    assert untrusted["workspaceId"].startswith("ws_")
    assert untrusted["clientId"].startswith("cl_")
    assert isinstance(untrusted["epoch"], int)

    trusted = await ws.workspace_open(gw, {"root": str(tmp_path), "trust": True})
    assert trusted["trusted"] is True
    assert "intelligence" in trusted["capabilities"]
    assert trusted["epoch"] == untrusted["epoch"] + 1  # reopen mints a new epoch


async def test_open_missing_root_is_invalid_params(tmp_path):
    gw = FakeGateway()
    with pytest.raises(RpcError) as error:
        await ws.workspace_open(gw, {"root": str(tmp_path / "nope"), "trust": True})
    assert error.value.code == INVALID_PARAMS


async def test_wrong_workspace_id_not_found(tmp_path):
    gw = FakeGateway()
    with pytest.raises(RpcError) as not_opened:
        await ws.workspace_list(gw, {"workspaceId": "ws_absent"})
    assert not_opened.value.code == NOT_FOUND

    gw2, opened, wid = await _opened(tmp_path)
    with pytest.raises(RpcError) as wrong:
        await ws.workspace_list(gw2, {"workspaceId": "ws_other"})
    assert wrong.value.code == NOT_FOUND
    with pytest.raises(RpcError) as absent:
        await ws.workspace_read(gw2, {"workspaceId": "ws_other", "uri": "file:///x"})
    assert absent.value.code == NOT_FOUND

    # The correct id still works.
    page = await ws.workspace_list(gw2, {"workspaceId": wid})
    assert page["entries"] == []


async def test_actor_role_params_are_ignored(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "ok.txt").write_text("ok")

    gw = FakeGateway()
    opened = await ws.workspace_open(
        gw,
        {"root": str(root), "trust": True, "actor": "agent", "role": "admin"},
    )
    wid = opened["workspaceId"]

    allowed = await ws.workspace_read(
        gw, {"workspaceId": wid, "uri": (root / "ok.txt").as_uri(), "actor": "user"}
    )
    assert allowed["text"] == "ok"

    # A forged actor/role cannot authorize a path outside the root.
    refused = await ws.workspace_read(
        gw, {"workspaceId": wid, "uri": (outside / "secret.txt").as_uri(), "role": "admin"}
    )
    assert refused["refused"]["reason"] == "denied"


# ─── list/read/save validation and refusal shapes ────────────────────────────
async def test_list_validation_errors(tmp_path):
    gw, _, wid = await _opened(tmp_path)
    for params in (
        {"workspaceId": wid, "limit": 0},
        {"workspaceId": wid, "limit": 201},
        {"workspaceId": wid, "limit": "5"},
        {"workspaceId": wid, "cursor": "abc"},
        {"workspaceId": wid, "cursor": -1},
        {"workspaceId": wid, "cursor": "9" * 13},
        {"workspaceId": wid, "includeHidden": "maybe"},
        {"workspaceId": wid, "path": 7},
        {"workspaceId": ""},
        {},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_list(gw, params)
        assert error.value.code == INVALID_PARAMS


async def test_list_refusal_is_a_structured_result(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linkdir").symlink_to(outside)
    gw, _, wid = await _opened(root)
    refused = await ws.workspace_list(gw, {"workspaceId": wid, "path": "linkdir"})
    assert refused["refused"]["reason"] == "denied"
    assert "escapes" in refused["refused"]["message"]


async def test_read_save_roundtrip_conflict_and_refusals(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    gw, opened, wid = await _opened(tmp_path)
    uri = (tmp_path / "a.txt").as_uri()

    read = await ws.workspace_read(gw, {"workspaceId": wid, "uri": uri})
    assert read["text"] == "hello\n"
    assert read["lineEnding"] == "LF"
    assert read["encoding"] == "utf-8"
    assert read["readOnly"] is False

    saved = await ws.workspace_save(
        gw,
        {
            "workspaceId": wid,
            "uri": uri,
            "text": "bye\n",
            "expectedRevision": read["revision"],
            "uiVersion": 3,
        },
    )
    assert saved["ok"] is True
    assert saved["savedUiVersion"] == 3
    assert (tmp_path / "a.txt").read_text() == "bye\n"

    conflict = await ws.workspace_save(
        gw,
        {"workspaceId": wid, "uri": uri, "text": "again\n", "expectedRevision": read["revision"]},
    )
    assert conflict["ok"] is False
    assert conflict["conflict"]["diskRevision"] == saved["revision"]
    assert conflict["conflict"]["baseRevision"] == read["revision"]
    assert conflict["conflict"]["reason"] == "revision-mismatch"
    assert (tmp_path / "a.txt").read_text() == "bye\n"

    # Structured read refusals are results, not RPC errors.
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01")
    refused = await ws.workspace_read(
        gw, {"workspaceId": wid, "uri": (tmp_path / "bin.dat").as_uri()}
    )
    assert refused["refused"]["reason"] == "binary"
    missing = await ws.workspace_read(
        gw, {"workspaceId": wid, "uri": (tmp_path / "gone.txt").as_uri()}
    )
    assert missing["refused"]["reason"] == "missing"

    # Malformed save params fail before touching the service.
    for params in (
        {"workspaceId": wid, "uri": uri, "text": 5},
        {"workspaceId": wid, "uri": uri},
        {"workspaceId": wid, "uri": uri, "text": "x", "expectedRevision": 7},
        {"workspaceId": wid, "uri": uri, "text": "x", "uiVersion": -1},
        {"workspaceId": wid, "text": "x"},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_save(gw, params)
        assert error.value.code == INVALID_PARAMS


async def test_save_read_only_refusal_shape(tmp_path):
    target = tmp_path / "ro.txt"
    target.write_bytes(b"x")
    target.chmod(0o444)
    try:
        gw, _, wid = await _opened(tmp_path)
        uri = target.as_uri()
        read = await ws.workspace_read(gw, {"workspaceId": wid, "uri": uri})
        assert read["readOnly"] is True
        saved = await ws.workspace_save(
            gw,
            {
                "workspaceId": wid,
                "uri": uri,
                "text": "y\n",
                "expectedRevision": read["revision"],
            },
        )
        assert saved["ok"] is False
        assert saved["refused"]["reason"] == "denied"
        assert "read-only" in saved["refused"]["message"]
        assert target.read_bytes() == b"x"
    finally:
        target.chmod(0o644)


async def test_save_emits_workspace_file_changed_event(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    gw, opened, wid = await _opened(tmp_path)
    uri = (tmp_path / "a.txt").as_uri()

    read = await ws.workspace_read(gw, {"workspaceId": wid, "uri": uri})
    assert gw.frames == []  # reads never emit workspace events
    saved = await ws.workspace_save(
        gw,
        {
            "workspaceId": wid,
            "uri": uri,
            "text": "y\n",
            "expectedRevision": read["revision"],
        },
    )
    assert saved["ok"] is True
    assert len(gw.frames) == 1
    frame = gw.frames[0]
    assert frame["jsonrpc"] == "2.0"
    assert frame["method"] == "event"
    event = frame["params"]
    assert event["type"] == "workspace.fileChanged"
    assert event["workspaceId"] == wid
    assert event["epoch"] == opened["epoch"]
    assert event["uri"] == read["uri"]
    assert event["revision"] == saved["revision"]
    assert event["origin"] == "save"
    # Workspace events must not masquerade as chat events.
    assert "sessionId" not in event and "turnId" not in event


# ─── document mirror handlers ────────────────────────────────────────────────
async def test_document_flow_and_validation(tmp_path):
    gw, _, wid = await _opened(tmp_path)
    uri = (tmp_path / "a.py").as_uri()

    opened = await ws.workspace_document_open(
        gw, {"workspaceId": wid, "uri": uri, "languageId": "python", "version": 1, "text": "x = 1\n"}
    )
    assert opened["acceptedVersion"] == 1
    assert opened["state"] == "missing"  # no server manager is a truthful state
    assert isinstance(opened["serverGeneration"], int)

    changed = await ws.workspace_document_change(
        gw,
        {
            "workspaceId": wid,
            "uri": uri,
            "baseVersion": 1,
            "version": 2,
            "changes": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                    "text": "y",
                }
            ],
        },
    )
    assert changed == {"acceptedVersion": 2}

    resync = await ws.workspace_document_change(
        gw,
        {"workspaceId": wid, "uri": uri, "baseVersion": 1, "version": 3, "changes": []},
    )
    assert resync["resync"] is True
    assert resync["acceptedVersion"] == 2

    closed = await ws.workspace_document_close(gw, {"workspaceId": wid, "uri": uri})
    assert closed["closed"] is True

    for params in (
        {"workspaceId": wid, "uri": uri},  # version required
        {"workspaceId": wid, "uri": uri, "version": "1"},
        {"workspaceId": wid, "uri": uri, "version": 1, "text": 5},
        {"workspaceId": wid, "uri": uri, "version": -1},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_document_open(gw, params)
        assert error.value.code == INVALID_PARAMS

    for params in (
        {"workspaceId": wid, "uri": uri, "baseVersion": 1, "version": 2, "changes": "nope"},
        {"workspaceId": wid, "uri": uri, "baseVersion": 1, "version": 2, "changes": [{}]},
        {
            "workspaceId": wid,
            "uri": uri,
            "baseVersion": 1,
            "version": 2,
            "changes": [
                {
                    "range": {
                        "start": {"line": -1, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "text": "x",
                }
            ],
        },
        {
            "workspaceId": wid,
            "uri": uri,
            "baseVersion": 1,
            "version": 2,
            "changes": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "text": "x",
                }
            ]
            * 513,
        },
        {"workspaceId": wid, "uri": uri, "baseVersion": 1, "changes": []},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_document_change(gw, params)
        assert error.value.code == INVALID_PARAMS


async def test_document_stale_epoch_refused(tmp_path):
    gw, opened, wid = await _opened(tmp_path)
    refusal = await ws.workspace_document_open(
        gw,
        {
            "workspaceId": wid,
            "uri": (tmp_path / "a.py").as_uri(),
            "version": 1,
            "epoch": opened["epoch"] + 1,
        },
    )
    assert refusal["refused"]["reason"] == "denied"


# ─── intelligence and LSP surface ────────────────────────────────────────────
async def test_intelligence_without_manager_is_truthful(tmp_path):
    gw, _, wid = await _opened(tmp_path)
    result = await ws.workspace_intelligence(
        gw,
        {
            "workspaceId": wid,
            "operation": "hover",
            "uri": (tmp_path / "a.py").as_uri(),
            "position": {"line": 0, "character": 0},
        },
    )
    assert result["state"] == "missing"
    assert result["freshness"] == "unavailable"
    assert result["result"] is None
    assert result["serverGeneration"] is None

    with pytest.raises(RpcError) as bad_operation:
        await ws.workspace_intelligence(
            gw, {"workspaceId": wid, "operation": "rename", "uri": "file:///x.py"}
        )
    assert bad_operation.value.code == INVALID_PARAMS
    assert "completion" in bad_operation.value.message  # allowlist disclosed

    for params in (
        {"workspaceId": wid, "operation": "hover", "position": {"line": "0", "character": 0}},
        {"workspaceId": wid, "operation": "hover", "position": {"line": 0}},
        {"workspaceId": wid, "operation": "hover", "version": "1"},
        {},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_intelligence(gw, params)
        assert error.value.code == INVALID_PARAMS


async def test_lsp_status_and_control_without_manager(tmp_path):
    gw, _, wid = await _opened(tmp_path)
    assert await ws.workspace_lsp_status(gw, {"workspaceId": wid}) == {"servers": []}

    started = await ws.workspace_lsp_control(
        gw, {"workspaceId": wid, "action": "start", "profile": "python"}
    )
    assert started["state"] == "missing"
    assert started["action"] == "start"
    assert started["profile"] == "python"

    for params in (
        {"workspaceId": wid, "action": "install", "profile": "python"},
        {"workspaceId": wid, "action": "start"},
        {"workspaceId": wid, "profile": "python"},
        {"workspaceId": wid, "action": "start", "profile": 7},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_lsp_control(gw, params)
        assert error.value.code == INVALID_PARAMS

    untrusted_gw = FakeGateway()
    untrusted = await ws.workspace_open(
        untrusted_gw, {"root": str(tmp_path), "trust": False}
    )
    denied = await ws.workspace_lsp_control(
        untrusted_gw,
        {"workspaceId": untrusted["workspaceId"], "action": "start", "profile": "python"},
    )
    assert denied["refused"]["reason"] == "denied"


# ─── lifecycle and factory ───────────────────────────────────────────────────
async def test_close_retires_workspace_and_blocks_further_calls(tmp_path):
    gw, _, wid = await _opened(tmp_path)
    assert await ws.workspace_close(gw, {"workspaceId": wid}) == {"closed": True}
    with pytest.raises(RpcError) as error:
        await ws.workspace_list(gw, {"workspaceId": wid})
    assert error.value.code == NOT_FOUND


async def test_service_created_once_per_gateway(tmp_path):
    gw = FakeGateway()
    opened = await ws.workspace_open(gw, {"root": str(tmp_path), "trust": True})
    service = gw.workspace_service
    assert service is not None
    assert service.workspace_id == opened["workspaceId"]
    await ws.workspace_list(gw, {"workspaceId": opened["workspaceId"]})
    assert gw.workspace_service is service

    other_gw = FakeGateway()
    await ws.workspace_open(other_gw, {"root": str(tmp_path), "trust": True})
    assert other_gw.workspace_service is not service
