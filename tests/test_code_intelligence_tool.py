"""T10 tests: the bounded read-only ``code_intelligence`` tool (plan §20).

Every behavior is driven through the real registry and the real
``WorkspaceService``; the LSP answers come from the scripted fake server
(``tests/support/fake_lsp_server.py``) over real stdio pipes, mirroring
``test_workspace_lsp_integration.py``. Honesty is asserted, not vocabulary:
no workspace / refused paths / missing positions are structured JSON errors,
and results carry the frozen §20 contract with truthful states.
"""

from __future__ import annotations

import contextlib
import json
import sys
import uuid
from pathlib import Path

import pytest

from machina.core.exceptions import ValidationError
from machina.core.models import Session
from machina.core.self_knowledge import FEATURES, capability_context, configuration
from machina.tools import code_intelligence as tool_module
from machina.tools.base import registry
from machina.workspace import lsp as lsp_module
from machina.workspace.lsp import LspManager, LspProfile
from machina.workspace.service import (
    CODE_INTELLIGENCE_OPERATIONS,
    WorkspaceService,
    active_workspace_service,
    clear_active_workspace_service,
)

FAKE_SERVER = Path(__file__).resolve().parent / "support" / "fake_lsp_server.py"


def _fake_profile(language: str, *extra: str) -> LspProfile:
    profile_id = "pyright" if language == "python" else "typescript-language-server"
    return LspProfile(
        language_id=language,
        profile_id=profile_id,
        executable=sys.executable,
        args=[str(FAKE_SERVER), *extra],
    )


@pytest.fixture(autouse=True)
def deterministic_profiles(monkeypatch):
    """Resolve profiles deterministically: dev binaries may or may not exist."""

    def _resolve(language: str, explicit_path: str | None = None) -> LspProfile | None:
        if language in ("python", "typescript"):
            return _fake_profile(language)
        return None

    monkeypatch.setattr(lsp_module, "resolve_profile", _resolve)


@pytest.fixture(autouse=True)
def no_leaked_workspace():
    """The active-service registry is process state; every test starts clean."""
    clear_active_workspace_service()
    yield
    clear_active_workspace_service()


def _manager(root: Path, *, python_args: tuple[str, ...] = ()) -> LspManager:
    return LspManager(
        root,
        profiles={
            "python": _fake_profile("python", *python_args),
            "typescript": _fake_profile("typescript"),
        },
        auto_restart=False,
    )


@contextlib.asynccontextmanager
async def _workspace(root: Path, manager: LspManager | None = None, *, trust: bool = True):
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    if manager is not None:
        service.set_lsp_manager(manager)
    identity = service.open(str(root), trust)
    assert active_workspace_service() is service, "open registers the active service"
    try:
        yield service, identity, events
    finally:
        service.close()
        await service.wait_stopped()
        assert active_workspace_service() is None, "close clears the active service"


async def _execute(**kwargs) -> dict:
    return json.loads(await registry.execute("code_intelligence", **kwargs))


# ─── registration, schema, effects ───────────────────────────────────────────
def test_tool_is_registered_read_only_with_a_closed_operation_enum():
    tool = registry.get_tool("code_intelligence")
    assert tool is not None
    # Read-only declaration: no mutating flag can ever authorize anything.
    assert tool.effects == ("fs.read",)
    assert not set(tool.effects) & {"fs.write", "exec", "net", "delegate", "mutating-tool"}
    schema = tool.parameters
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["operation", "path"]
    assert schema["properties"]["operation"]["enum"] == list(CODE_INTELLIGENCE_OPERATIONS)
    assert schema["properties"]["position"]["additionalProperties"] is False
    assert schema["properties"]["position"]["required"] == ["line", "character"]
    # The definition sent to the model keeps the closed schema too.
    definition = next(
        entry for entry in registry.get_tool_definitions() if entry.name == "code_intelligence"
    )
    assert definition.parameters["additionalProperties"] is False
    assert definition.parameters["properties"]["operation"]["enum"] == list(
        CODE_INTELLIGENCE_OPERATIONS
    )


async def test_unknown_arguments_are_rejected_by_registry_validation():
    with pytest.raises(ValidationError):
        await registry.execute("code_intelligence", operation="diagnostics", path="a.py", bogus=1)
    with pytest.raises(ValidationError):
        await registry.execute("code_intelligence", operation="diagnostics")


# ─── honest errors, never fabricated answers ─────────────────────────────────
async def test_no_workspace_is_a_structured_error_never_a_claim():
    payload = await _execute(operation="diagnostics", path="a.py")
    assert payload == {
        "error": {
            "code": "no-workspace",
            "message": (
                "no workspace is open in this process; the user opens one with /workspace"
            ),
        }
    }


async def test_unknown_operation_and_query_are_refused_honestly(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        assert service.workspace_id is not None
        payload = await _execute(operation="completion", path="a.py")
        assert payload["error"]["code"] == "unsupported-operation"
        payload = await _execute(operation="symbols", path="a.py", query="Alpha")
        assert payload["error"]["code"] == "unsupported-query"
        assert "omit query" in payload["error"]["message"]


async def test_position_is_required_for_cursor_operations(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        assert service.workspace_id is not None
        for operation in ("definition", "references", "hover"):
            payload = await _execute(operation=operation, path="a.py")
            assert payload["error"]["code"] == "position-required", operation
        payload = await _execute(
            operation="hover", path="a.py", position={"line": -1, "character": 0}
        )
        assert payload["error"]["code"] == "invalid-position"
        payload = await _execute(operation="hover", path="a.py", position={"line": 0})
        assert payload["error"]["code"] == "invalid-position"


async def test_untrusted_workspace_is_refused_outright(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager, trust=False) as (service, _, _events):
        assert service.workspace_id is not None
        payload = await _execute(operation="diagnostics", path="a.py")
        assert payload["error"]["code"] == "untrusted"
        assert manager.status() == [], "nothing may spawn for an untrusted workspace"


async def test_paths_outside_the_root_are_refused_structured(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("x = 1\n", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)
    manager = _manager(root)
    async with _workspace(root, manager) as (service, _, _events):
        assert service.workspace_id is not None
        for bad in ("../outside/secret.py", "/etc/passwd", "..", "a\\b.py"):
            payload = await _execute(operation="diagnostics", path=bad)
            assert payload["error"]["code"] == "denied", bad
        # A blank path is malformed input, refused before any resolution.
        payload = await _execute(operation="diagnostics", path="")
        assert payload["error"]["code"] == "invalid-path"
        # A symlink inside the root that resolves outside is refused too.
        payload = await _execute(operation="diagnostics", path="link/secret.py")
        assert payload["error"]["code"] == "denied"
        # Nothing was spawned by the refused calls.
        assert manager.status() == []


# ─── the frozen §20 contract over a real service + scripted LSP ──────────────
_CONTRACT_FIELDS = frozenset(
    {
        "workspaceId",
        "source",
        "uri",
        "documentVersion",
        "serverGeneration",
        "operation",
        "state",
        "freshness",
        "truncated",
        "result",
    }
)


async def test_every_operation_returns_the_frozen_contract(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, identity, _events):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
        await service.intelligence("symbols", uri=uri)  # flush the sync chain
        connection = await manager.ensure_started("python")

        base = await _execute(operation="symbols", path="a.py")
        assert _CONTRACT_FIELDS <= set(base)
        # The service envelope carries exactly the frozen fields plus its
        # epoch scope (plan §19: the epoch accompanies every result), nothing
        # else may sneak in.
        assert set(base) == _CONTRACT_FIELDS | {"workspaceEpoch"}
        assert base["workspaceId"] == identity.workspace_id
        assert base["source"] == "editor-buffer"
        assert base["uri"] == uri
        assert base["documentVersion"] == 1
        assert base["serverGeneration"] == connection.generation
        assert base["operation"] == "symbols"
        assert base["state"] == "ready"
        assert base["freshness"] == "versioned"
        assert base["truncated"] is False
        assert base["result"]["flat"][0]["name"] == "Alpha"

        position = {"line": 0, "character": 1}
        hover = await _execute(operation="hover", path="a.py", position=position)
        assert hover["result"]["contents"].startswith("# Hover")
        definition = await _execute(operation="definition", path="a.py", position=position)
        assert definition["result"]["locations"] == [
            {
                "uri": (tmp_path / "target.py").as_uri(),
                "range": {
                    "start": {"line": 3, "character": 2},
                    "end": {"line": 3, "character": 7},
                },
            }
        ]
        references = await _execute(operation="references", path="a.py", position=position)
        assert len(references["result"]["locations"]) == 3

        # Diagnostics come from the notification store: publish, then read.
        await connection.request(
            "publish-diagnostics",
            {
                "uri": uri,
                "version": 1,
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 5},
                        },
                        "severity": 1,
                        "message": "boom",
                    }
                ],
            },
        )
        diagnostics = await _execute(operation="diagnostics", path="a.py")
        assert diagnostics["operation"] == "diagnostics"
        assert diagnostics["state"] == "ready"
        assert diagnostics["freshness"] == "versioned"
        assert [entry["message"] for entry in diagnostics["result"]["diagnostics"]] == ["boom"]
        assert diagnostics["result"]["producers"][0]["source"] == "pyright"


async def test_disk_sourced_analysis_is_unversioned_not_fabricated(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        assert service.workspace_id is not None
        # The server is running for the language (started by an editor open).
        first = tmp_path / "a.py"
        first.write_text("x = 1\n", encoding="utf-8")
        service.document_open(first.as_uri(), "python", 1, "x = 1\n", epoch=service.epoch)
        await service.intelligence("symbols", uri=first.as_uri())
        # A file nobody opened: disk-sourced, unversioned — never fabricated.
        (tmp_path / "other.py").write_text("y = 2\n", encoding="utf-8")
        payload = await _execute(operation="symbols", path="other.py")
        assert payload["source"] == "disk"
        assert payload["documentVersion"] is None
        assert payload["state"] == "ready"
        assert payload["freshness"] == "unversioned"


async def test_missing_language_server_is_an_honest_unavailable_state(tmp_path):
    # A manager with no configured profiles: nothing can answer, and nothing
    # is spawned; the tool reports missing/unavailable instead of inventing.
    manager = LspManager(tmp_path, profiles={})
    async with _workspace(tmp_path, manager) as (service, _, _events):
        assert service.workspace_id is not None
        payload = await _execute(
            operation="hover", path="other.py", position={"line": 0, "character": 0}
        )
        assert payload["state"] == "missing"
        assert payload["freshness"] == "unavailable"
        assert payload["result"] is None
        assert manager.status() == []


async def test_byte_cap_truncates_whole_entries_with_a_flag(tmp_path):
    manager = _manager(tmp_path, python_args=("--references-count", "600"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
        payload = await _execute(
            operation="references", path="a.py", position={"line": 0, "character": 1}
        )
        serialized = json.dumps(payload, separators=(",", ":"))
        assert len(serialized.encode("utf-8")) <= tool_module.MAX_RESULT_BYTES
        assert payload["truncated"] is True
        locations = payload["result"]["locations"]
        assert 0 < len(locations) < 500, "the service cap alone (500) was not enough to fit"
        # The drop is disclosed through the operation's own counter, and every
        # surviving entry remains a whole normalized location.
        assert payload["result"]["omitted"] >= 500 - len(locations)
        for entry in locations:
            assert set(entry) == {"uri", "range"}


async def test_hover_text_respects_both_result_caps(tmp_path):
    manager = _manager(tmp_path, python_args=("--hover-repeats", "2000"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
        payload = await _execute(operation="hover", path="a.py", position={"line": 0, "character": 1})
        serialized = json.dumps(payload, separators=(",", ":"))
        assert len(serialized.encode("utf-8")) <= tool_module.MAX_RESULT_BYTES
        # The service caps hover at 4096 chars and discloses the cut; the tool
        # bound must never exceed it (both caps hold together).
        assert len(payload["result"]["contents"]) == 4096
        assert payload["truncated"] is True


def test_tool_byte_cap_shrinks_escaped_hover_text_with_a_flag():
    """Unit-level: escaping can inflate a capped string past the tool bound."""

    def envelope(result):
        return {
            "workspaceId": "ws_x",
            "source": "editor-buffer",
            "uri": "file:///root/a.py",
            "documentVersion": 1,
            "serverGeneration": 1,
            "workspaceEpoch": 1,
            "operation": "hover",
            "state": "ready",
            "freshness": "versioned",
            "truncated": False,
            "result": result,
        }

    payload = envelope({"contents": "\x01" * 5000})
    serialized = tool_module._bounded(payload)
    assert len(serialized.encode("utf-8")) <= tool_module.MAX_RESULT_BYTES
    parsed = json.loads(serialized)
    assert parsed["truncated"] is True
    assert isinstance(parsed["result"]["contents"], str)
    assert 0 < len(parsed["result"]["contents"]) < 5000
    assert parsed["state"] == "ready" and parsed["operation"] == "hover"


def test_tool_byte_cap_trims_symbol_views_with_exact_omitted_count():
    """Unit-level: a symbols payload trims both views and discloses the drop."""
    def node(index: int) -> dict:
        return {
            "name": f"symbol-{index:04d}",
            "kind": 13,
            "depth": 0,
            "range": {
                "start": {"line": index, "character": 0},
                "end": {"line": index, "character": 8},
            },
        }
    payload = {
        "workspaceId": "ws_x",
        "source": "editor-buffer",
        "uri": "file:///root/a.py",
        "documentVersion": 1,
        "serverGeneration": 1,
        "workspaceEpoch": 1,
        "operation": "symbols",
        "state": "ready",
        "freshness": "versioned",
        "truncated": False,
        "result": {
            "hierarchical": [node(index) for index in range(500)],
            "flat": [node(index) for index in range(500)],
            "omitted": 0,
        },
    }
    serialized = tool_module._bounded(payload)
    assert len(serialized.encode("utf-8")) <= tool_module.MAX_RESULT_BYTES
    parsed = json.loads(serialized)
    assert parsed["truncated"] is True
    kept = len(parsed["result"]["flat"])
    assert 0 < kept < 500
    assert len(parsed["result"]["hierarchical"]) == kept, "both views stay aligned"
    assert parsed["result"]["omitted"] == 500 - kept, "the drop is disclosed exactly"


# ─── permissions mapping (least-risk, read-only) ─────────────────────────────
def test_permission_mapping_is_a_routine_project_read():
    from machina.permissions.tools import action_for_tool

    request = action_for_tool("code_intelligence", {"operation": "diagnostics", "path": "src/a.py"})
    assert request == {"operation": "file.read", "targets": ["src/a.py"]}
    assert "capabilities" not in request  # never a mutating capability


def test_permission_decide_allows_inside_and_pends_outside(tmp_path, monkeypatch):
    from machina.permissions import policy
    from machina.permissions.policy import ActionRequest, decide, normalize_request

    monkeypatch.setattr(policy, "workspace_root", lambda: tmp_path)
    inside = normalize_request(
        ActionRequest(
            mode="workspace",
            tool="code_intelligence",
            operation="file.read",
            targets=["src/a.py"],
            workspace_root=str(tmp_path),
        )
    )
    assert decide(inside).verdict == "allowed"
    outside = normalize_request(
        ActionRequest(
            mode="workspace",
            tool="code_intelligence",
            operation="file.read",
            targets=["../elsewhere/a.py"],
            workspace_root=str(tmp_path),
        )
    )
    assert decide(outside).verdict == "pending"


# ─── self-knowledge pins (completeness + live facts + no file text) ──────────
def test_existing_self_knowledge_completeness_pins_still_hold():
    from tests.test_self_knowledge_completeness import (
        test_every_backticked_machina_cli_reference_exists,
        test_every_snake_case_token_is_a_config_key_or_registered_tool,
        test_every_slash_command_named_in_the_brief_is_registered,
        test_every_tui_feature_command_is_covered_or_allowlisted,
        test_facts_are_single_line_entries,
    )

    test_every_snake_case_token_is_a_config_key_or_registered_tool()
    test_every_backticked_machina_cli_reference_exists()
    test_every_tui_feature_command_is_covered_or_allowlisted()
    test_every_slash_command_named_in_the_brief_is_registered()
    test_facts_are_single_line_entries()


def test_workspace_entry_states_the_sharing_and_authority_stance():
    entry = FEATURES["workspace"]
    assert "code_intelligence" in entry, "the tool must be discoverable from the brief"
    assert "/attach" in entry, "explicit attachment is the only unsaved-share path"
    assert "not runtime or test success" in entry, "analysis is not a test verdict"
    assert "no edit, command or file permission" in entry, "a visible pane grants nothing"


async def test_configuration_reports_live_facts_only_while_open(tmp_path):
    session = Session(id=uuid.uuid4())
    assert configuration(session)["workspace"] == "none"

    root = tmp_path / "brief-root"
    root.mkdir()
    manager = _manager(root)
    async with _workspace(root, manager) as (service, _, _events):
        target = root / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
        await service.intelligence("symbols", uri=uri)
        await manager.ensure_started("python")

        facts = configuration(session)["workspace"]
        assert facts["available"] is True
        assert facts["root_label"] == "brief-root", "basename only"
        assert facts["trusted"] is True
        assert facts["profiles"]["python"] == "ready"
        assert facts["operations"] == list(CODE_INTELLIGENCE_OPERATIONS)
        assert str(tmp_path) not in json.dumps(facts), "no full private path in the brief"
    assert configuration(session)["workspace"] == "none"


async def test_brief_never_carries_buffer_or_disk_text(tmp_path):
    planted = "planted-secret-6f2c0b17"
    session = Session(id=uuid.uuid4())
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        target = tmp_path / "a.py"
        target.write_text(f"api_key = '{planted}'\n", encoding="utf-8")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, f"api_key = '{planted}'\n", epoch=service.epoch)
        await service.intelligence("symbols", uri=uri)
        assert planted in service.document_snapshot(uri)["text"], "the mirror really holds it"

        brief = capability_context(session, tool_names={"harness_status"}, growth_state="active")
        assert planted not in brief
        assert planted not in json.dumps(configuration(session))
        # The live facts ARE present (the test proves they exist without text).
        assert '"workspace":{"available":true' in brief.replace(" ", "")
