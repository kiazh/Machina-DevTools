"""T10 privacy proof (plan §20 I2): viewing a file never sends its text.

API-level: open a workspace, mirror a document whose text holds a planted
secret, then assert the always-on brief, ``configuration``, ``harness_status``
and the workspace event stream (the exact payloads the gateway wraps) carry no
buffer text. The mirror itself is read back to prove the secret really is in
process memory, so the absence assertions are meaningful.

Deliberately NOT a unit mock of the redaction layer: every surface below is
the real production function reading the real active ``WorkspaceService``.
"""

from __future__ import annotations

import contextlib
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from machina.core.models import Session
from machina.core.self_knowledge import capability_context, configuration
from machina.tools import harness
from machina.workspace import lsp as lsp_module
from machina.workspace.lsp import LspManager, LspProfile
from machina.workspace.service import (
    WorkspaceService,
    active_workspace_service,
    clear_active_workspace_service,
)

FAKE_SERVER = Path(__file__).resolve().parent / "support" / "fake_lsp_server.py"
SECRET = "planted-buffer-secret-9f3a7c21"
PLANTED_TEXT = f"API_KEY = '{SECRET}'\nprint('hello')\n"


def _fake_profile(language: str) -> LspProfile:
    return LspProfile(
        language_id=language,
        profile_id="pyright" if language == "python" else "typescript-language-server",
        executable=sys.executable,
        args=[str(FAKE_SERVER)],
    )


@pytest.fixture(autouse=True)
def deterministic_profiles(monkeypatch):
    def _resolve(language: str, explicit_path: str | None = None) -> LspProfile | None:
        if language in ("python", "typescript"):
            return _fake_profile(language)
        return None

    monkeypatch.setattr(lsp_module, "resolve_profile", _resolve)


@pytest.fixture(autouse=True)
def no_leaked_workspace():
    clear_active_workspace_service()
    yield
    clear_active_workspace_service()


@contextlib.asynccontextmanager
async def _open_workspace_with_mirrored_secret(root: Path):
    events: list[dict] = []
    manager = LspManager(root, profiles={"python": _fake_profile("python")}, auto_restart=False)
    service = WorkspaceService(send_event=events.append)
    service.set_lsp_manager(manager)
    identity = service.open(str(root), True)
    assert active_workspace_service() is service
    try:
        yield service, identity, manager, events
    finally:
        service.close()
        await service.wait_stopped()


def _owned_session(monkeypatch) -> Session:
    """A session harness_status accepts as the caller's own (read-only mock)."""
    session = Session(id=uuid.uuid4(), agent_id="privacy-owner")
    monkeypatch.setattr(harness, "active_agent_scope", AsyncMock(return_value=("privacy-owner", session.id)))
    monkeypatch.setattr(harness.session_manager, "get", AsyncMock(return_value=session))
    monkeypatch.setattr(
        harness.growth_store,
        "get",
        AsyncMock(return_value={"notes": {}, "interactions": 0, "feedback_count": 0}),
    )
    monkeypatch.setattr(harness.growth_store, "rsi_status", AsyncMock(return_value={}))
    return session


async def test_opening_a_file_never_leaks_its_text_to_any_agent_surface(tmp_path, monkeypatch):
    session = _owned_session(monkeypatch)
    async with _open_workspace_with_mirrored_secret(tmp_path) as (service, identity, manager, events):
        uri = (tmp_path / "secret.py").as_uri()
        service.document_open(uri, "python", 1, PLANTED_TEXT, epoch=service.epoch)
        # Flush the pending sync so the server has the mirror and refuses
        # nothing; the session data is now genuinely in process memory.
        await service.intelligence("symbols", uri=uri)
        connection = await manager.ensure_started("python")
        await connection.request(
            "publish-diagnostics",
            {
                "uri": uri,
                "version": 1,
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 7},
                        },
                        "severity": 2,
                        "message": "server-published-note",
                    }
                ],
            },
        )

        # Control: the mirror really holds the secret in this process.
        assert SECRET in service.document_snapshot(uri)["text"]

        brief = capability_context(session, tool_names={"harness_status"}, growth_state="active")
        assert SECRET not in brief, "the always-on brief must not carry buffer text"
        assert '"workspace":{"available":true' in brief.replace(" ", "")

        config_payload = json.dumps(configuration(session))
        assert SECRET not in config_payload
        assert "secret.py" not in config_payload, "no open-file names in the brief either"

        status = json.loads(await harness.harness_status())
        status_payload = json.dumps(status)
        assert SECRET not in status_payload, "harness_status must not carry buffer text"
        assert status["workspace"]["available"] is True
        assert status["workspace"]["root_label"] == tmp_path.name
        assert status["workspace"]["mirrors"] == 1
        assert status["workspace"]["profiles"]["python"]["state"] == "ready"
        assert status["workspace"]["profiles"]["python"]["generation"] == connection.generation
        assert status["workspace"]["operations"] == [
            "diagnostics",
            "definition",
            "references",
            "symbols",
            "hover",
        ]

        # The workspace event stream (what the gateway wraps in JSON-RPC) is a
        # positive control: diagnostics events exist and carry server text —
        # but never the buffer text.
        assert any(event.get("type") == "workspace.diagnostics" for event in events)
        events_payload = json.dumps(events)
        assert "server-published-note" in events_payload, "events really carried server data"
        assert SECRET not in events_payload, "no event may carry mirror/buffer text"
        for event in events:
            assert "sessionId" not in event and "turnId" not in event


async def test_unsaved_changes_stay_out_of_every_status_surface(tmp_path, monkeypatch):
    session = _owned_session(monkeypatch)
    async with _open_workspace_with_mirrored_secret(tmp_path) as (service, _, manager, events):
        uri = (tmp_path / "secret.py").as_uri()
        service.document_open(uri, "python", 1, "print('hello')\n", epoch=service.epoch)
        await service.intelligence("symbols", uri=uri)
        await manager.ensure_started("python")
        # A change carries the planted secret, accepted into the mirror only
        # (the editor typed it; nothing else asked for it).
        change = [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 0},
                },
                "text": f"# {SECRET}\n",
            }
        ]
        assert service.document_change(uri, 1, 2, change, epoch=service.epoch).accepted_version == 2
        assert SECRET in service.document_snapshot(uri)["text"]

        assert SECRET not in capability_context(
            session, tool_names={"harness_status"}, growth_state="active"
        )
        assert SECRET not in json.dumps(configuration(session))
        assert SECRET not in await harness.harness_status()
        assert SECRET not in json.dumps(events)
