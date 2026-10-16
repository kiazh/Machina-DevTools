"""§24-E real supported-server integration tests (plan §24.E, boxes 1–9).

Drives the provisioned pinned servers through the real service layer:

* python  -> ``ui/node_modules/.bin/pyright-langserver`` (1.1.414)
* typescript -> ``ui/node_modules/.bin/typescript-language-server`` (6.0.2)

Only the filesystem fixtures are synthetic. The service, its document
mirrors, the T7 transport and both server processes are real; every asserted
value (diagnostic messages/ranges, completion labels, definition/reference
URIs and ranges, rename edits) is the server's own response. The one
transparent shim is ``tests/support/lsp_real_workspace.py``, which implements
the document-sync/intelligence seam T8 owns — and steps aside when the
attached service performs that sync itself (see its docstring).

Run:

    cd /Users/shadow/Machina-DevTools && \
    MACHINA_REPO=/Users/shadow/machina PYTHON=/Users/shadow/machina/.venv/bin/python \
    ./run-tests.sh tests/test_lsp_integration_real.py -q -s

Every wait is a bounded poll (deadline <= 20 s); pyright's first start is the
slow operation. When a server binary is absent the affected tests skip with
the resolved-path evidence instead of passing silently (plan §24.E: "a
disabled/skipped integration test is not a pass" — so the report states
exactly which ran).
"""

from __future__ import annotations

import contextlib
import json
import shutil

import pytest
import pytest_asyncio

from machina.workspace import lsp as lsp_module
from machina.workspace import revision_from_bytes
from machina.workspace import service as workspace_service_module
from machina.workspace.lsp import LspProfile, resolve_profile
from machina.workspace.models import WorkspaceError
from tests.support.lsp_real_workspace import (
    RealLspWorkspace,
    diagnostics_capable_client,
    uri_key,
    wait_for,
)

pytestmark = pytest.mark.integration

# ─── fixtures (synthetic filesystem only; servers are real) ─────────────────

MAIN_SOURCE = 'import util\n\nmessage = util.greet("world")\nprint(message)\n'
MAIN_BAD = "import util\n\nmessage = util.greet(123)\nprint(message)\n"
UTIL_SOURCE = 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'

APP_SOURCE = 'import { name } from "./config";\n\nconsole.log(name);\n'
APP_BAD = 'import { name } from "./config";\n\nconst n: number = name;\nconsole.log(n);\n'
APP_DOT = 'import { name } from "./config";\n\nname.\n'
CONFIG_SOURCE = 'export const name: string = "machina";\n'

# LSP (line, character) positions into the fixtures above (zero-based).
PY_ATTR_POSITION = {"line": 2, "character": 15}  # directly after ``util.``
PY_NAME_POSITION = {"line": 2, "character": 18}  # inside ``greet`` at the call
PY_DEF_POSITION = {"line": 0, "character": 5}  # inside ``greet`` in util.py
TS_NAME_POSITION = {"line": 2, "character": 20}  # inside ``name`` in APP_BAD
TS_DOT_POSITION = {"line": 2, "character": 5}  # directly after ``name.``


def _replace_all(old_text: str, new_text: str) -> list[dict]:
    """One full-document LSP change ranging over ``old_text``."""
    lines = old_text.split("\n")
    end = {"line": len(lines) - 1, "character": len(lines[-1])}
    return [{"range": {"start": {"line": 0, "character": 0}, "end": end}, "text": new_text}]


def _require_profile(language: str) -> LspProfile:
    """Resolve a real server or skip with the exact paths tried (never fake it)."""
    profile = resolve_profile(language)
    if profile is not None:
        print(f"[lsp-real] {language} profile: {profile.profile_id} -> {profile.executable}")
        return profile
    _, executable_name = lsp_module._KNOWN_PROFILES[language]
    fallback = lsp_module._DEV_FALLBACKS.get(executable_name, "(none)")
    pytest.skip(
        f"{language}: no language server resolved; PATH lookup {executable_name!r} -> "
        f"{shutil.which(executable_name)!r}; dev fallback {fallback!r} is not an executable "
        "file. Provisioning is an explicit environment step."
    )


@pytest_asyncio.fixture
async def make_workspace(tmp_path):
    """Factory for real workspaces on ``tmp_path``; closes every one on exit."""
    created: list[RealLspWorkspace] = []

    async def factory(profiles=None) -> RealLspWorkspace:
        print(f"[lsp-real] backend service module: {workspace_service_module.__file__}")
        workspace = RealLspWorkspace(
            tmp_path,
            profiles=profiles,
            connection_kwargs={"initialize_timeout": 20.0, "request_timeout": 15.0},
        )
        created.append(workspace)
        return workspace

    yield factory
    for workspace in created:
        with contextlib.suppress(Exception):
            await workspace.aclose()


def _diagnostics_with(result, code: object) -> dict | None:
    for diagnostic in result.result or []:
        if diagnostic.get("code") == code:
            return diagnostic
    return None


# ─── 1. Python: unsaved-text diagnostics ────────────────────────────────────
async def test_python_unsaved_diagnostics_target_the_call(make_workspace):
    """An unsaved type error reaches pyright and comes back as a diagnostic.

    Proves: documentOpen/Change sync of editor-buffer text (never saved), the
    polled intelligence(diagnostics) operation, and content-based targeting of
    the call site (line + range slice + message).
    """
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    util_uri = workspace.uri("util.py")
    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    await workspace.document_open(util_uri, "python", 1, UTIL_SOURCE)
    changed = await workspace.document_change(
        main_uri, 1, 2, _replace_all(MAIN_SOURCE, MAIN_BAD)
    )
    assert changed.resync is False and changed.accepted_version == 2

    result = await wait_for(
        lambda: workspace.service.intelligence("diagnostics", uri=main_uri),
        lambda value: _diagnostics_with(value, "reportArgumentType") is not None,
        label="pyright publishes reportArgumentType for greet(123)",
    )
    payload = result.as_dict()
    assert payload["source"] == "editor-buffer"
    assert payload["state"] == "ready"
    assert payload["documentVersion"] == 2
    assert payload["freshness"] == "versioned"
    assert payload["workspaceEpoch"] == workspace.identity.epoch

    diagnostic = _diagnostics_with(result, "reportArgumentType")
    message = diagnostic["message"]
    assert "cannot be assigned" in message
    assert "greet" in message and "str" in message and "123" in message
    assert diagnostic["severity"] == 1
    start, end = diagnostic["range"]["start"], diagnostic["range"]["end"]
    assert start["line"] == end["line"] == 2
    line = MAIN_BAD.splitlines()[2]
    assert line[start["character"] : end["character"]] == "123"
    # The edit was never saved: disk still holds the clean text.
    assert workspace.path("main.py").read_text(encoding="utf-8") == MAIN_SOURCE


# ─── 2. Python: completion against the mirror ───────────────────────────────
async def test_python_completion_after_util_attribute(make_workspace):
    """Attribute completion after ``util.`` offers the imported function.

    Asserted position: line 2, character 15 (immediately after the dot in
    ``message = util.greet(...)``) — chosen because pyright serves attribute
    completion reliably there (verified against the pinned 1.1.414; not the
    flakier ``greet(`` parameter position). Item content: label ``greet``,
    kind 3 (Function).
    """
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)

    result = await wait_for(
        lambda: workspace.service.intelligence(
            "completion", uri=main_uri, position=PY_ATTR_POSITION
        ),
        lambda value: any(
            item.get("label") == "greet" for item in (value.result or {}).get("items", [])
        ),
        label="pyright completion lists util.greet",
    )
    assert result.state == "ready"
    assert result.source == "editor-buffer"
    item = next(
        entry for entry in result.result["items"] if entry.get("label") == "greet"
    )
    assert item.get("kind") == 3  # CompletionItemKind.Function


# ─── 3. Python: definition + references ─────────────────────────────────────
async def test_python_definition_and_references(make_workspace):
    """Definition lands on util.py's ``greet``; references include the call."""
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    util_uri = workspace.uri("util.py")
    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    await workspace.document_open(util_uri, "python", 1, UTIL_SOURCE)

    definition = await wait_for(
        lambda: workspace.service.intelligence(
            "definition", uri=main_uri, position=PY_NAME_POSITION
        ),
        lambda value: any(
            uri_key(location["uri"]) == uri_key(util_uri) for location in value.result or []
        ),
        label="pyright definition resolves into util.py",
    )
    target = next(
        location
        for location in definition.result
        if uri_key(location["uri"]) == uri_key(util_uri)
    )
    span = target["range"]
    assert span["start"]["line"] == 0 and span["end"]["line"] == 0
    util_line = UTIL_SOURCE.splitlines()[0]
    assert util_line[span["start"]["character"] : span["end"]["character"]] == "greet"

    references = await wait_for(
        lambda: workspace.service.intelligence(
            "references", uri=util_uri, position=PY_DEF_POSITION
        ),
        lambda value: any(
            uri_key(location["uri"]) == uri_key(main_uri) for location in value.result or []
        ),
        label="pyright references include the main.py call",
    )
    main_location = next(
        location
        for location in references.result
        if uri_key(location["uri"]) == uri_key(main_uri)
    )
    span = main_location["range"]
    assert span["start"]["line"] == 2
    main_line = MAIN_SOURCE.splitlines()[2]
    assert main_line[span["start"]["character"] : span["end"]["character"]] == "greet"
    # includeDeclaration=True also returns util.py's own definition.
    assert any(
        uri_key(location["uri"]) == uri_key(util_uri) for location in references.result
    )


# ─── 4. Python: rename (service allowlist gap, documented) ──────────────────
async def test_python_rename_not_in_service_allowlist_raw_probe(make_workspace):
    """Service-level rename is pending; the real capability is proven raw.

    ``INTELLIGENCE_OPERATIONS`` does not include ``rename``, so the service
    refuses it with a structured denial (asserted). The real server capability
    is then exercised over the real connection via the harness accessor —
    prepareRename range and the two-file WorkspaceEdit come verbatim from
    pyright. No service-level rename result is fabricated.
    """
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    util_uri = workspace.uri("util.py")
    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    await workspace.document_open(util_uri, "python", 1, UTIL_SOURCE)
    await workspace.document_change(main_uri, 1, 2, _replace_all(MAIN_SOURCE, MAIN_BAD))

    with pytest.raises(WorkspaceError) as denied:
        await workspace.service.intelligence(
            "rename", uri=main_uri, position=PY_NAME_POSITION
        )
    assert denied.value.reason == "denied"
    assert "rename" in denied.value.message

    connection = workspace.connection_for("python")
    assert connection is not None and connection.state == "ready"

    async def prepare_rename():
        return await connection.request(
            "textDocument/prepareRename",
            {"textDocument": {"uri": main_uri}, "position": PY_NAME_POSITION},
            timeout=15,
        )

    # Bounded poll: the server may answer null until its view of the document
    # has caught up with the mirror (sync order under the service's chain).
    prepared = await wait_for(
        prepare_rename,
        lambda value: value is not None,
        label="pyright prepareRename resolves after document sync",
    )
    prepared_range = prepared["range"] if isinstance(prepared, dict) and "range" in prepared else prepared
    main_bad_line = MAIN_BAD.splitlines()[2]
    assert main_bad_line[
        prepared_range["start"]["character"] : prepared_range["end"]["character"]
    ] == "greet"

    edit = await connection.request(
        "textDocument/rename",
        {
            "textDocument": {"uri": main_uri},
            "position": PY_NAME_POSITION,
            "newName": "salute",
        },
        timeout=15,
    )
    edits_by_file: dict[str, list[dict]] = {}
    if isinstance(edit, dict) and "documentChanges" in edit:
        for document_change in edit["documentChanges"]:
            edits_by_file[uri_key(document_change["textDocument"]["uri"])] = document_change[
                "edits"
            ]
    else:
        edits_by_file = {
            uri_key(uri): edits for uri, edits in (edit or {}).get("changes", {}).items()
        }
    assert set(edits_by_file) == {uri_key(main_uri), uri_key(util_uri)}
    for edits in edits_by_file.values():
        assert edits and all(entry["newText"] == "salute" for entry in edits)
    main_edit = edits_by_file[uri_key(main_uri)][0]
    assert main_bad_line[
        main_edit["range"]["start"]["character"] : main_edit["range"]["end"]["character"]
    ] == "greet"
    util_edit = edits_by_file[uri_key(util_uri)][0]
    assert UTIL_SOURCE.splitlines()[0][
        util_edit["range"]["start"]["character"] : util_edit["range"]["end"]["character"]
    ] == "greet"


# ─── 5. Python: save revision + mirror consistency ──────────────────────────
async def test_python_save_revision_and_mirror_consistency(make_workspace):
    """workspace.save writes disk, mints a new revision, keeps the mirror.

    didSave forwarding differs by backend and neither real server exposes an
    ack: on the T5 baseline the harness seam forwards it (asserted); on the
    newer service the service forwards it itself (recorded, not server-proven).
    The follow-up hover only succeeds while the connection is healthy.
    """
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    read = workspace.service.read(main_uri)
    assert read.text == MAIN_SOURCE

    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    await workspace.document_change(main_uri, 1, 2, _replace_all(MAIN_SOURCE, MAIN_BAD))

    saved = await workspace.save(main_uri, MAIN_BAD, read.revision, ui_version=2)
    assert saved.ok is True
    assert saved.revision == revision_from_bytes(MAIN_BAD.encode("utf-8"))
    assert saved.revision != read.revision
    assert saved.saved_ui_version == 2

    assert workspace.path("main.py").read_text(encoding="utf-8") == MAIN_BAD
    read_back = workspace.service.read(main_uri)
    assert read_back.text == MAIN_BAD and read_back.revision == saved.revision
    snapshot = workspace.service.document_snapshot(main_uri)
    assert snapshot is not None and snapshot["text"] == MAIN_BAD

    events = [event for event in workspace.events if event.get("type") == "workspace.fileChanged"]
    assert len(events) == 1
    assert events[0]["revision"] == saved.revision
    assert uri_key(events[0]["uri"]) == uri_key(main_uri)

    # didSave: neither pyright nor tsls exposes an ack, so this records which
    # seam forwarded it — the harness on the T5 baseline (asserted below), the
    # service itself on the newer backend — and reports it instead of claiming
    # server-observable behavior.
    if workspace.service_owns_document_sync:
        print("[lsp-real] didSave forwarded by the workspace service itself")
    else:
        assert workspace.did_save_calls == [main_uri]

    # The connection stays healthy after the save notification.
    hovered = await wait_for(
        lambda: workspace.service.intelligence("hover", uri=main_uri, position=PY_NAME_POSITION),
        lambda value: value.result is not None and "greet" in json.dumps(value.result),
        label="pyright hover still serves after didSave",
    )
    assert hovered.state == "ready"


# ─── 6. Python: stop/restart reopens the latest (unsaved) text ──────────────
async def test_python_stop_restart_reopens_latest_unsaved_text(make_workspace):
    """Stop -> stopped (observed), restart -> a new generation reopens mirrors.

    The decisive proof that the reopen carries the *unsaved* text: disk holds
    the clean source, the mirror holds ``greet(123)``; the diagnostic can only
    reappear after the restart if the full latest mirror text was replayed.
    """
    workspace = await make_workspace({"python": _require_profile("python")})
    workspace.write("util.py", UTIL_SOURCE)
    workspace.write("main.py", MAIN_SOURCE)
    await workspace.ensure_started("python")

    main_uri = workspace.uri("main.py")
    util_uri = workspace.uri("util.py")
    await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    await workspace.document_open(util_uri, "python", 1, UTIL_SOURCE)
    await workspace.document_change(main_uri, 1, 2, _replace_all(MAIN_SOURCE, MAIN_BAD))
    await wait_for(
        lambda: workspace.service.intelligence("diagnostics", uri=main_uri),
        lambda value: _diagnostics_with(value, "reportArgumentType") is not None,
        label="pre-restart diagnostic",
    )

    servers = (await workspace.service.lsp_status())["servers"]
    before = next(entry for entry in servers if entry["languageId"] == "python")
    assert before["state"] == "ready"
    generation = before["generation"]

    stopped = await workspace.service.lsp_control("stop", "python")
    assert stopped["state"] == "stopped"
    assert stopped["generation"] == generation
    servers = (await workspace.service.lsp_status())["servers"]
    entry = next(item for item in servers if item["languageId"] == "python")
    assert entry["state"] == "stopped"
    unavailable = await workspace.service.intelligence("diagnostics", uri=main_uri)
    assert unavailable.state == "stopped" and unavailable.result is None
    assert workspace.path("main.py").read_text(encoding="utf-8") == MAIN_SOURCE

    restarted = await workspace.service.lsp_control("restart", "python")
    assert restarted["state"] == "ready"
    assert restarted["generation"] == generation + 1
    servers = (await workspace.service.lsp_status())["servers"]
    entry = next(item for item in servers if item["languageId"] == "python")
    assert entry["state"] == "ready" and entry["generation"] == generation + 1

    result = await wait_for(
        lambda: workspace.service.intelligence("diagnostics", uri=main_uri),
        lambda value: _diagnostics_with(value, "reportArgumentType") is not None,
        label="post-restart diagnostic from reopened unsaved text",
    )
    assert result.document_version == 2
    assert result.source == "editor-buffer"
    diagnostic = _diagnostics_with(result, "reportArgumentType")
    main_line = MAIN_BAD.splitlines()[2]
    span = diagnostic["range"]
    assert main_line[span["start"]["character"] : span["end"]["character"]] == "123"

    completed = await wait_for(
        lambda: workspace.service.intelligence(
            "completion", uri=main_uri, position=PY_ATTR_POSITION
        ),
        lambda value: any(
            item.get("label") == "greet" for item in (value.result or {}).get("items", [])
        ),
        label="completion works on the restarted generation",
    )
    assert completed.state == "ready"


# ─── 7. TypeScript: diagnostics + completion + definition ───────────────────
async def test_typescript_unsaved_diagnostics_completion_definition(
    make_workspace, monkeypatch
):
    """tsls publishes the unsaved TS2322 and serves completion/definition.

    The pinned typescript-language-server suppresses all diagnostics unless
    the client declares ``textDocument.publishDiagnostics``; T7's capability
    list omits the key, so this test injects the standard capability (see
    ``diagnostics_capable_client``) and reports the gap instead of faking a
    result.
    """
    with diagnostics_capable_client(monkeypatch):
        workspace = await make_workspace(
            {"typescript": _require_profile("typescript")}
        )
        workspace.write("config.ts", CONFIG_SOURCE)
        workspace.write("app.ts", APP_SOURCE)
        await workspace.ensure_started("typescript")

        app_uri = workspace.uri("app.ts")
        config_uri = workspace.uri("config.ts")
        await workspace.document_open(app_uri, "typescript", 1, APP_SOURCE)
        await workspace.document_open(config_uri, "typescript", 1, CONFIG_SOURCE)
        await workspace.document_change(app_uri, 1, 2, _replace_all(APP_SOURCE, APP_BAD))

        result = await wait_for(
            lambda: workspace.service.intelligence("diagnostics", uri=app_uri),
            lambda value: _diagnostics_with(value, 2322) is not None,
            label="tsls publishes TS2322 for the unsaved const n: number = name",
        )
        assert result.state == "ready"
        assert result.source == "editor-buffer"
        diagnostic = _diagnostics_with(result, 2322)
        assert "not assignable" in diagnostic["message"]
        assert "string" in diagnostic["message"] and "number" in diagnostic["message"]
        assert diagnostic["severity"] == 1
        start, end = diagnostic["range"]["start"], diagnostic["range"]["end"]
        assert start["line"] == 2
        assert APP_BAD.splitlines()[2][start["character"] : end["character"]] == "n"
        assert workspace.path("app.ts").read_text(encoding="utf-8") == APP_SOURCE

        definition = await wait_for(
            lambda: workspace.service.intelligence(
                "definition", uri=app_uri, position=TS_NAME_POSITION
            ),
            lambda value: any(
                uri_key(location["uri"]) == uri_key(config_uri)
                for location in value.result or []
            ),
            label="tsls definition resolves name into config.ts",
        )
        target = next(
            location
            for location in definition.result
            if uri_key(location["uri"]) == uri_key(config_uri)
        )
        span = target["range"]
        assert span["start"]["line"] == 0
        config_line = CONFIG_SOURCE.splitlines()[0]
        assert config_line[span["start"]["character"] : span["end"]["character"]] == "name"

        await workspace.document_change(app_uri, 2, 3, _replace_all(APP_BAD, APP_DOT))
        completed = await wait_for(
            lambda: workspace.service.intelligence(
                "completion", uri=app_uri, position=TS_DOT_POSITION
            ),
            lambda value: any(
                item.get("label") == "charAt" for item in (value.result or {}).get("items", [])
            ),
            label="tsls member completion after name. lists string methods",
        )
        labels = [item.get("label") for item in completed.result["items"]]
        assert "charAt" in labels and "substring" in labels
        assert completed.state == "ready"


# ─── 8. Server absence remains honest ───────────────────────────────────────
async def test_missing_server_is_honest_but_editing_works(make_workspace, monkeypatch):
    """A bogus explicit path resolves to no profile: missing, never fabricated.

    Resolution is pointed at a bogus explicit path with PATH lookup and the
    dev fallback disabled (the absent-server environment); the mirror still
    accepts opens/changes and intelligence reports missing/unavailable with no
    result and no crash.
    """
    monkeypatch.setattr(lsp_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(lsp_module, "_DEV_FALLBACKS", {})
    workspace = await make_workspace()
    bogus = str(workspace.path("bogus-server"))
    print(f"[lsp-real] python: resolve_profile(explicit_path={bogus!r}) -> no profile")

    connection = await workspace.ensure_started("python", explicit_path=bogus)
    assert connection.state == "missing"

    servers = (await workspace.service.lsp_status())["servers"]
    entry = next(item for item in servers if item["languageId"] == "python")
    assert entry["profile"] == "missing-python"
    assert entry["state"] == "missing"
    assert "no language server" in entry["detail"]

    main_uri = workspace.uri("main.py")
    opened = await workspace.document_open(main_uri, "python", 1, MAIN_SOURCE)
    assert opened.accepted_version == 1
    assert opened.state == "missing"

    result = await workspace.service.intelligence("diagnostics", uri=main_uri)
    payload = result.as_dict()
    assert payload["state"] == "missing"
    assert payload["freshness"] == "unavailable"
    assert payload["result"] is None
    assert payload["source"] == "editor-buffer"

    changed = await workspace.document_change(main_uri, 1, 2, _replace_all(MAIN_SOURCE, MAIN_BAD))
    assert changed.resync is False and changed.accepted_version == 2
    snapshot = workspace.service.document_snapshot(main_uri)
    assert snapshot is not None and snapshot["text"] == MAIN_BAD
    completion = await workspace.service.intelligence(
        "completion", uri=main_uri, position=PY_ATTR_POSITION
    )
    assert completion.state == "missing" and completion.result is None
