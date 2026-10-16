"""T8 end-to-end tests: document sync + core intelligence through WorkspaceService.

Every test drives a scripted fake LSP server (``tests/support/fake_lsp_server.py``)
over real stdio pipes through the real ``LspManager``/``LspConnection``, with
``WorkspaceService`` and its document mirrors in front, and a list capturing
the exact ``send_event`` payloads the gateway would wrap. Synchronization is
event-driven (waits on observations), never based on fixed sleeps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import pytest

from machina.workspace import lsp as lsp_module
from machina.workspace.intelligence import (
    lsp_request_for,
    lsp_sync_kind,
    normalize_completion_list,
    normalize_for,
    normalize_hover,
    normalize_locations,
    sanitize_log_text,
)
from machina.workspace.lsp import (
    STATE_FAILED,
    STATE_READY,
    LspManager,
    LspProfile,
    LspUnavailableError,
)
from machina.workspace.service import WorkspaceService

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
    """Resolve profiles deterministically: dev binaries may or may not exist.

    The manager's ``profiles`` override is what actually launches the fake
    server; this only makes availability probes (open/control) stable.
    """

    def _resolve(language: str, explicit_path: str | None = None) -> LspProfile | None:
        if language in ("python", "typescript"):
            return _fake_profile(language)
        return None

    monkeypatch.setattr(lsp_module, "resolve_profile", _resolve)


def _manager(
    root: Path,
    *,
    python_args: tuple[str, ...] = (),
    typescript_args: tuple[str, ...] = (),
    auto_restart: bool = False,
) -> LspManager:
    return LspManager(
        root,
        profiles={
            "python": _fake_profile("python", *python_args),
            "typescript": _fake_profile("typescript", *typescript_args),
        },
        auto_restart=auto_restart,
    )


@contextlib.asynccontextmanager
async def _workspace(
    root: Path,
    manager: LspManager | None = None,
    *,
    trust: bool = True,
    service_kwargs: dict | None = None,
):
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append, **(service_kwargs or {}))
    if manager is not None:
        service.set_lsp_manager(manager)
    identity = service.open(str(root), trust)
    try:
        yield service, identity, events
    finally:
        service.close()
        await service.wait_stopped()


async def _until(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


async def _records(connection) -> list[dict]:
    log = await connection.request("record-log")
    return log["records"]


def _by_method(records: list[dict], method: str) -> list[dict]:
    return [entry["params"] for entry in records if entry["method"] == method]


async def _flush(service: WorkspaceService, uri: str) -> None:
    """Await the document's pending LSP synchronization through a real request."""
    await service.intelligence("symbols", uri=uri)


def _lsp_state_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "workspace.lspState"]


def _diagnostics_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "workspace.diagnostics"]


# ─── lazy start and didOpen ordering ─────────────────────────────────────────
async def test_open_does_not_spawn_and_first_document_open_starts_lazily(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, identity, _events):
        assert manager.status() == []  # open() learned profiles, spawned nothing

        uri = (tmp_path / "a.py").as_uri()
        opened = service.document_open(uri, "python", 1, "x = 1\n")
        assert opened.accepted_version == 1
        assert opened.state == "starting"  # a lazy start was scheduled
        assert opened.server_generation == identity.epoch  # no server generation yet

        # Flushing through a real request proves the start completed.
        await _flush(service, uri)
        connection = await manager.ensure_started("python")
        records = await _records(connection)
        methods = [entry["method"] for entry in records]
        assert methods == [
            "initialize",
            "initialized",
            "textDocument/didOpen",
            "textDocument/documentSymbol",
            "record-log",
        ]
        did_open = _by_method(records, "textDocument/didOpen")[0]["textDocument"]
        assert did_open == {
            "uri": uri,
            "languageId": "python",
            "version": 1,
            "text": "x = 1\n",
        }

        retried = service.document_open(uri, "python", 1, "x = 1\n")
        assert retried.state == "ready"
        assert retried.server_generation == connection.generation
        # Re-read the log AFTER the retry: asserting on the pre-retry snapshot
        # would pass even if the retry sent a duplicate didOpen (pass-2 fix).
        after_retry = [entry["method"] for entry in await _records(connection)]
        assert after_retry.count("textDocument/didOpen") == 1

        status = (await service.lsp_status())["servers"]
        assert len(status) == 1
        row = status[0]
        assert row["profile"] == "pyright"
        assert row["languageId"] == "python"
        assert row["state"] == "ready"
        assert row["generation"] == connection.generation
        assert row["capabilities"] == {
            "completion": True,
            "hover": True,
            "definition": True,
            "references": True,
            "documentSymbol": True,
            "rename": False,
            "signatureHelp": False,
            "formatting": False,
            "codeAction": False,
            "syncKind": 1,
            "saveIncludeText": True,
        }


async def test_language_detection_by_extension_not_client_language_id(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        ts_uri = (tmp_path / "a.ts").as_uri()
        opened = service.document_open(ts_uri, "typescriptreact", 1, "const x = 1\n")
        assert opened.state == "starting"
        await _flush(service, ts_uri)
        languages = {row["languageId"] for row in (await service.lsp_status())["servers"]}
        assert languages == {"typescript"}
        connection = await manager.ensure_started("typescript")
        did_open = _by_method(await _records(connection), "textDocument/didOpen")[0]
        assert did_open["textDocument"]["languageId"] == "typescript"

        # A client-declared languageId cannot route an unmapped extension to a
        # server: notes.txt is mirrored but never reaches one.
        txt_uri = (tmp_path / "notes.txt").as_uri()
        mirror_only = service.document_open(txt_uri, "python", 1, "hello\n")
        assert mirror_only.state == "missing"
        assert service.document_snapshot(txt_uri)["text"] == "hello\n"
        languages = {row["languageId"] for row in (await service.lsp_status())["servers"]}
        assert languages == {"typescript"}


# ─── document change: incremental vs full sync ──────────────────────────────
async def test_incremental_sync_sends_exact_utf16_ranges(tmp_path):
    manager = _manager(tmp_path, python_args=("--sync-kind", "2"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "a😀b\nsecond\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        # Replacing the surrogate pair (UTF-16 units 1..3) with "X".
        changes = [
            {
                "range": {
                    "start": {"line": 0, "character": 1},
                    "end": {"line": 0, "character": 3},
                },
                "text": "X",
            }
        ]
        accepted = service.document_change(uri, 1, 2, changes)
        assert accepted.accepted_version == 2
        assert accepted.resync is False

        # A second, sequential change must reuse its own computed range.
        second = [
            {
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 6},
                },
                "text": "SECOND",
            }
        ]
        assert service.document_change(uri, 2, 3, second).accepted_version == 3
        assert service.document_snapshot(uri)["text"] == "aXb\nSECOND\n"

        records = await _records(connection)
        did_changes = _by_method(records, "textDocument/didChange")
        assert [entry["textDocument"]["version"] for entry in did_changes] == [2, 3]
        assert did_changes[0]["contentChanges"] == changes
        assert did_changes[1]["contentChanges"] == second


async def test_full_sync_server_receives_full_snapshots(tmp_path):
    manager = _manager(tmp_path)  # fake declares textDocumentSync.change = 1
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        change = [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 1},
                },
                "text": "y",
            }
        ]
        assert service.document_change(uri, 1, 2, change).accepted_version == 2
        records = await _records(connection)
        did_change = _by_method(records, "textDocument/didChange")[0]
        assert did_change["textDocument"]["version"] == 2
        assert did_change["contentChanges"] == [{"text": "y = 1\n"}]


async def test_request_after_change_ordering_and_reopen_on_ready(tmp_path):
    manager = _manager(tmp_path, python_args=("--sync-kind", "2"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        # Change while the server is still handshaking: accepted into the
        # mirror, nothing is sent (never a didChange against a dead generation).
        change = [
            {
                "range": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 5},
                },
                "text": "2",
            }
        ]
        assert service.document_change(uri, 1, 2, change).accepted_version == 2

        await service.intelligence("completion", uri=uri, position={"line": 0, "character": 5})
        connection = await manager.ensure_started("python")
        records = await _records(connection)
        methods = [entry["method"] for entry in records]
        assert methods.index("textDocument/didOpen") < methods.index("textDocument/completion")
        assert "textDocument/didChange" not in methods  # reopen carried the latest text
        did_open = _by_method(records, "textDocument/didOpen")[0]["textDocument"]
        assert did_open["version"] == 2
        assert did_open["text"] == "x = 2\n"

        # Once ready, a change is queued inline and still precedes the request
        # that depends on it.
        assert service.document_change(uri, 2, 3, [dict(change[0], text="3")]).accepted_version == 3
        await service.intelligence("completion", uri=uri, position={"line": 0, "character": 5})
        records = await _records(connection)
        methods = [entry["method"] for entry in records]
        last_completion = max(i for i, m in enumerate(methods) if m == "textDocument/completion")
        did_change = methods.index("textDocument/didChange")
        assert did_change < last_completion
        changes = _by_method(records, "textDocument/didChange")
        assert changes == [
            {
                "textDocument": {"uri": uri, "version": 3},
                "contentChanges": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 4},
                            "end": {"line": 0, "character": 5},
                        },
                        "text": "3",
                    }
                ],
            }
        ]

        # A request for a version the mirror does not hold is stale, not a
        # window into older text.
        stale = await service.intelligence(
            "completion", uri=uri, version=9, position={"line": 0, "character": 5}
        )
        assert stale.state == STATE_READY
        assert stale.freshness == "stale"
        assert stale.result is None
        assert stale.document_version == 3

        # A file the server never opened still gets a bounded result, from
        # disk-sourced (unversioned) analysis.
        disk_result = await service.intelligence("symbols", uri=(tmp_path / "other.py").as_uri())
        assert disk_result.source == "disk"
        assert disk_result.document_version is None
        assert disk_result.freshness == "unversioned"
        assert disk_result.result["flat"][0]["name"] == "Alpha"


# ─── completion / hover / definition / references / symbols ─────────────────
async def test_completion_normalization_and_no_snippet_capability(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence(
            "completion", uri=uri, position={"line": 0, "character": 1}
        )
        assert result.state == STATE_READY
        assert result.freshness == "versioned"
        assert result.truncated is False
        payload = result.result
        assert payload["isIncomplete"] is True
        assert payload["omitted"] == 0
        items = payload["items"]
        assert [item["label"] for item in items] == ["alpha", "beta", "gamma"]

        alpha = items[0]
        assert alpha["kind"] == 3
        assert alpha["detail"] == "detail-alpha"
        assert alpha["documentation"] == "Docs for **alpha**"
        assert alpha["filterText"] == "alpha"
        assert alpha["sortText"] == "1alpha"
        assert alpha["insertTextFormat"] == 1
        assert alpha["textEdit"] == {
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 5},
            },
            "newText": "alpha",
        }
        assert alpha["additionalTextEdits"] == [
            {
                "range": {
                    "start": {"line": 1, "character": 0},
                    "end": {"line": 1, "character": 4},
                },
                "newText": "edit",
            }
        ]

        # InsertReplaceEdit: normalized to the insert range (never fabricated).
        beta = items[1]
        assert beta["textEdit"] == {
            "range": {
                "start": {"line": 2, "character": 0},
                "end": {"line": 2, "character": 0},
            },
            "newText": "beta",
        }

        # A malformed edit range is dropped; the item survives without an edit.
        gamma = items[2]
        assert "textEdit" not in gamma

        # The initialize handshake advertised snippetSupport=false, so servers
        # must never send snippets; assert the parameter we sent.
        connection = await manager.ensure_started("python")
        log = await connection.request("record-log")
        text_document = log["initializeParams"]["capabilities"]["textDocument"]
        assert text_document["completion"]["completionItem"]["snippetSupport"] is False


async def test_completion_cap_100_with_omitted(tmp_path):
    manager = _manager(tmp_path, python_args=("--completion-count", "150"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence(
            "completion", uri=uri, position={"line": 0, "character": 1}
        )
        assert result.truncated is True
        assert len(result.result["items"]) == 100
        assert result.result["omitted"] == 50
        assert result.result["items"][99]["label"] == "item0099"


async def test_hover_flattening_and_cap(tmp_path):
    manager = _manager(tmp_path, python_args=("--hover-repeats", "2000"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence("hover", uri=uri, position={"line": 0, "character": 1})
        assert result.state == STATE_READY
        assert result.truncated is True
        assert len(result.result["contents"]) == 4096
        assert result.result["contents"].startswith("# Hover\n\n")
        assert result.result["range"] == {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        }


async def test_intelligence_timeout_is_bounded_and_configurable(tmp_path):
    manager = _manager(tmp_path, python_args=("--hang-on", "textDocument/hover"))
    async with _workspace(
        tmp_path, manager, service_kwargs={"intelligence_timeouts": {"hover": 0.3}}
    ) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await service.intelligence("hover", uri=uri, position={"line": 0, "character": 1})
        elapsed = loop.time() - started
        assert result.state == STATE_FAILED
        assert result.freshness == "unavailable"
        assert result.result is None
        assert elapsed < 5.0  # bounded by the configured deadline, not the 10s default
        # The connection survives a timed-out request.
        follow_up = await service.intelligence("symbols", uri=uri)
        assert follow_up.state == STATE_READY


async def test_definition_location_link_normalization(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence(
            "definition", uri=uri, position={"line": 0, "character": 1}
        )
        assert result.state == STATE_READY
        locations = result.result["locations"]
        assert locations == [
            {
                "uri": (tmp_path / "target.py").as_uri(),
                "range": {
                    "start": {"line": 3, "character": 2},
                    "end": {"line": 3, "character": 7},
                },
            }
        ]
        assert result.result["omitted"] == 0


async def test_references_cap_500(tmp_path):
    manager = _manager(tmp_path, python_args=("--references-count", "600"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence(
            "references", uri=uri, position={"line": 0, "character": 1}
        )
        assert result.truncated is True
        locations = result.result["locations"]
        assert len(locations) == 500
        assert result.result["omitted"] == 100
        assert locations[0]["range"]["start"] == {"line": 0, "character": 0}
        assert locations[-1]["range"]["start"] == {"line": 499, "character": 0}


async def test_symbols_hierarchical_form_computes_flat_view(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence("symbols", uri=uri)
        assert result.state == STATE_READY
        hierarchical = result.result["hierarchical"]
        assert [node["name"] for node in hierarchical] == ["Alpha"]
        alpha = hierarchical[0]
        assert [child["name"] for child in alpha["children"]] == ["method_a", "method_b"]
        assert [leaf["name"] for leaf in alpha["children"][1]["children"]] == ["nested"]
        flat = result.result["flat"]
        assert [(node["name"], node["depth"]) for node in flat] == [
            ("Alpha", 0),
            ("method_a", 1),
            ("method_b", 1),
            ("nested", 2),
        ]
        assert result.result["omitted"] == 0


async def test_symbols_flat_form_computes_hierarchical_view(tmp_path):
    manager = _manager(tmp_path, python_args=("--symbols-form", "flat"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        result = await service.intelligence("symbols", uri=uri)
        flat = result.result["flat"]
        assert [node["name"] for node in flat] == ["Alpha", "method_a", "method_b", "standalone"]
        assert flat[1]["containerName"] == "Alpha"
        assert flat[1]["uri"] == uri
        hierarchical = result.result["hierarchical"]
        assert [node["name"] for node in hierarchical] == ["Alpha", "standalone"]
        assert [child["name"] for child in hierarchical[0]["children"]] == [
            "method_a",
            "method_b",
        ]


# ─── diagnostics store ───────────────────────────────────────────────────────
def _diagnostic(line: int, message: str, severity: int = 1) -> dict:
    return {
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 4},
        },
        "severity": severity,
        "message": message,
        "source": "fake",
        "code": f"E{line}",
    }


async def test_diagnostics_replace_reject_unversioned_and_clear(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, identity, events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        await connection.request(
            "publish-diagnostics",
            {
                "uri": uri,
                "version": 1,
                "diagnostics": [_diagnostic(0, "first"), _diagnostic(1, "second")],
            },
        )
        frames = _diagnostics_events(events)
        assert len(frames) == 1
        frame = frames[0]
        assert frame == {
            "type": "workspace.diagnostics",
            "workspaceId": identity.workspace_id,
            "epoch": identity.epoch,
            "serverGeneration": connection.generation,
            "uri": uri,
            "version": 1,
            "source": "pyright",
            "diagnostics": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                    "severity": 1,
                    "message": "first",
                    "source": "fake",
                    "code": "E0",
                },
                {
                    "range": {
                        "start": {"line": 1, "character": 0},
                        "end": {"line": 1, "character": 4},
                    },
                    "severity": 1,
                    "message": "second",
                    "source": "fake",
                    "code": "E1",
                },
            ],
        }
        assert "sessionId" not in frame and "turnId" not in frame

        read = await service.intelligence("diagnostics", uri=uri)
        assert read.freshness == "versioned"
        assert read.document_version == 1
        assert [item["message"] for item in read.result["diagnostics"]] == ["first", "second"]
        assert read.result["producers"] == [
            {"source": "pyright", "version": 1, "freshness": "versioned", "count": 2}
        ]

        # A publish replaces the producer's whole set.
        await connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 1, "diagnostics": [_diagnostic(2, "third")]},
        )
        read = await service.intelligence("diagnostics", uri=uri)
        assert [item["message"] for item in read.result["diagnostics"]] == ["third"]

        # An older version is rejected entirely: no event, no mutation.
        before = len(_diagnostics_events(events))
        await connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 0, "diagnostics": [_diagnostic(3, "ancient")]},
        )
        assert len(_diagnostics_events(events)) == before
        read = await service.intelligence("diagnostics", uri=uri)
        assert [item["message"] for item in read.result["diagnostics"]] == ["third"]

        # Without a version the set is stored but clearly unversioned.
        await connection.request(
            "publish-diagnostics",
            {"uri": uri, "diagnostics": [_diagnostic(4, "freshness-unknown")]},
        )
        read = await service.intelligence("diagnostics", uri=uri)
        assert read.freshness == "unversioned"
        assert read.result["producers"][0]["version"] is None

        # An empty list clears only this producer and emits a clearing event.
        await connection.request("publish-diagnostics", {"uri": uri, "diagnostics": []})
        frame = _diagnostics_events(events)[-1]
        assert frame["diagnostics"] == []
        # No version on any later publish: the tombstone still rejects nothing,
        # but the store no longer serves a set for this producer.
        read = await service.intelligence("diagnostics", uri=uri)
        assert read.result is None
        assert read.freshness == "unversioned"  # ready server, nothing published


async def test_diagnostics_per_file_cap_1000_with_omitted(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        many = [_diagnostic(index, f"message-{index}") for index in range(1200)]
        await connection.request(
            "publish-diagnostics", {"uri": uri, "version": 1, "diagnostics": many}
        )
        frame = _diagnostics_events(events)[0]
        assert len(frame["diagnostics"]) == 1000
        read = await service.intelligence("diagnostics", uri=uri)
        assert read.truncated is True
        assert len(read.result["diagnostics"]) == 1000
        assert read.result["omitted"] == 200
        assert read.result["producers"][0]["count"] == 1000


async def test_diagnostics_are_per_producer(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        python_connection = await manager.ensure_started("python")

        started = await service.lsp_control("start", "typescript")
        assert started["state"] == "ready"
        ts_connection = await manager.ensure_started("typescript")

        await python_connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 1, "diagnostics": [_diagnostic(0, "from-python")]},
        )
        await ts_connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 1, "diagnostics": [_diagnostic(1, "from-ts")]},
        )
        read = await service.intelligence("diagnostics", uri=uri)
        assert [item["message"] for item in read.result["diagnostics"]] == [
            "from-python",
            "from-ts",
        ]
        assert [entry["source"] for entry in read.result["producers"]] == [
            "pyright",
            "typescript-language-server",
        ]

        # Clearing one producer keeps the other producer's set.
        await ts_connection.request("publish-diagnostics", {"uri": uri, "diagnostics": []})
        read = await service.intelligence("diagnostics", uri=uri)
        assert [item["message"] for item in read.result["diagnostics"]] == ["from-python"]
        assert [entry["source"] for entry in read.result["producers"]] == ["pyright"]
        assert [event["source"] for event in _diagnostics_events(events)] == [
            "pyright",
            "typescript-language-server",
            "typescript-language-server",
        ]


async def test_crash_marks_diagnostics_stale_and_restart_discards_old_generation(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, identity, events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")
        await connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 1, "diagnostics": [_diagnostic(0, "before-crash")]},
        )
        assert len(_diagnostics_events(events)) == 1

        with pytest.raises(LspUnavailableError):
            await connection.request("boom", {})
        await _until(lambda: connection.state == STATE_FAILED)

        # Pending intelligence is unavailable, never fabricated.
        completion = await service.intelligence(
            "completion", uri=uri, position={"line": 0, "character": 1}
        )
        assert completion.state == STATE_FAILED
        assert completion.freshness == "unavailable"
        assert completion.result is None

        # Diagnostics survive the crash, marked stale (not silently cleared).
        read = await service.intelligence("diagnostics", uri=uri)
        assert read.state == STATE_FAILED
        assert read.freshness == "stale"
        assert [item["message"] for item in read.result["diagnostics"]] == ["before-crash"]

        # Restart: generation increments, the old set is never served again.
        restarted = await service.lsp_control("restart", "pyright")
        assert restarted["state"] == "ready"
        assert restarted["generation"] == connection.generation
        assert connection.generation >= 2
        read = await service.intelligence("diagnostics", uri=uri)
        assert read.state == STATE_READY
        assert read.server_generation == connection.generation
        assert read.result is None or read.result["diagnostics"] == []
        assert read.freshness != "versioned"

        ready_events = [event for event in _lsp_state_events(events) if event["state"] == "ready"]
        assert ready_events[-1]["generation"] == connection.generation
        assert ready_events[-1]["profile"] == "pyright"
        assert ready_events[-1]["epoch"] == identity.epoch
        assert any(event["state"] == "failed" for event in _lsp_state_events(events))


async def test_restart_reopens_documents_with_latest_snapshots(tmp_path):
    manager = _manager(tmp_path, python_args=("--sync-kind", "2"))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        a_uri = (tmp_path / "a.py").as_uri()
        b_uri = (tmp_path / "b.py").as_uri()
        service.document_open(a_uri, "python", 1, "a = 1\n")
        service.document_open(b_uri, "python", 1, "b = 1\n")
        await _flush(service, b_uri)
        connection = await manager.ensure_started("python")

        change = [
            {
                "range": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 5},
                },
                "text": "2",
            }
        ]
        assert service.document_change(a_uri, 1, 2, change).accepted_version == 2

        restarted = await service.lsp_control("restart", "pyright")
        assert restarted["generation"] == connection.generation == 2

        # The restarted server is a fresh process: its log holds only the
        # initialize handshake and the reopened documents, in stable URI order.
        records = await _records(connection)
        methods = [entry["method"] for entry in records]
        assert methods[:2] == ["initialize", "initialized"]
        did_opens = _by_method(records, "textDocument/didOpen")
        assert [entry["textDocument"]["uri"] for entry in did_opens] == [a_uri, b_uri]
        assert did_opens[0]["textDocument"] == {
            "uri": a_uri,
            "languageId": "python",
            "version": 2,
            "text": "a = 2\n",
        }
        assert did_opens[1]["textDocument"]["version"] == 1
        assert did_opens[1]["textDocument"]["text"] == "b = 1\n"


# ─── save ────────────────────────────────────────────────────────────────────
async def test_save_sends_did_save_with_text_when_requested(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, events):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")
        read = service.read(uri)

        result = await service.save(uri, "x = 2\n", read.revision)
        assert result.ok is True
        saves = _by_method(await _records(connection), "textDocument/didSave")
        assert saves == [{"textDocument": {"uri": uri}, "text": "x = 2\n"}]

        changed = [event for event in events if event["type"] == "workspace.fileChanged"]
        assert len(changed) == 1
        assert changed[0]["origin"] == "save"
        assert changed[0]["revision"] == result.revision


async def test_save_omits_text_when_server_does_not_request_it(tmp_path):
    manager = _manager(tmp_path, python_args=("--no-save-include-text",))
    async with _workspace(tmp_path, manager) as (service, _, _events):
        target = tmp_path / "a.py"
        target.write_text("x = 1\n")
        uri = target.as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")
        read = service.read(uri)

        result = await service.save(uri, "x = 2\n", read.revision)
        assert result.ok is True
        saves = _by_method(await _records(connection), "textDocument/didSave")
        assert saves == [{"textDocument": {"uri": uri}}]


# ─── close, missing servers, honesty ────────────────────────────────────────
async def test_document_close_sends_did_close_only_when_open_on_generation(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        assert service.document_close(uri) == {"closed": True, "uri": uri}
        closes = _by_method(await _records(connection), "textDocument/didClose")
        assert closes == [{"textDocument": {"uri": uri}}]
        service.document_close(uri)  # idempotent: nothing to close any more
        closes = _by_method(await _records(connection), "textDocument/didClose")
        assert len(closes) == 1

        # A document whose generation is gone is closed silently (no didClose
        # against a dead generation); closed mirrors are not reopened by a
        # restart, while documents still open are.
        other = (tmp_path / "b.py").as_uri()
        kept = (tmp_path / "c.py").as_uri()
        service.document_open(other, "python", 1, "b = 1\n")
        service.document_open(kept, "python", 1, "c = 1\n")
        await _flush(service, kept)
        with pytest.raises(LspUnavailableError):
            await connection.request("boom", {})
        await _until(lambda: connection.state == STATE_FAILED)
        assert service.document_close(other)["closed"] is True
        restarted = await service.lsp_control("restart", "pyright")
        assert restarted["state"] == STATE_READY
        did_opens = _by_method(await _records(connection), "textDocument/didOpen")
        assert [entry["textDocument"]["uri"] for entry in did_opens] == [kept]


async def test_lsp_control_stop_and_lazy_restart(tmp_path):
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        # Stopping a profile that never started is honest, not an error.
        idle = await service.lsp_control("stop", "python")
        assert idle["state"] == "stopped"
        assert "no connection" in idle["detail"]
        assert manager.status() == []

        started = await service.lsp_control("start", "pyright")
        assert started["state"] == STATE_READY
        assert started["generation"] == 1
        assert (await service.lsp_status())["servers"][0]["state"] == STATE_READY

        stopped = await service.lsp_control("stop", "python")
        assert stopped["state"] == "stopped"
        assert (await service.lsp_status())["servers"][0]["state"] == "stopped"

        # The next documentOpen starts the stopped server again, lazily — the
        # service itself, without the test touching the manager.
        uri = (tmp_path / "a.py").as_uri()
        reopened = service.document_open(uri, "python", 1, "x = 1\n")
        assert reopened.state == "starting"
        await _until(lambda: manager.status() and manager.status()[0]["state"] == STATE_READY)
        connection = await manager.ensure_started("python")
        assert connection.generation == 2
        did_open = _by_method(await _records(connection), "textDocument/didOpen")[-1]
        assert did_open["textDocument"]["version"] == 1

        unsupported = await service.lsp_control("start", "cobol")
        assert unsupported["state"] == "missing"


async def test_missing_server_is_honest_and_mirrors_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(lsp_module, "resolve_profile", lambda language, explicit_path=None: None)
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, events):
        uri = (tmp_path / "a.py").as_uri()
        opened = service.document_open(uri, "python", 1, "x = 1\n")
        assert opened.state == "missing"
        assert opened.accepted_version == 1  # the mirror is accepted regardless
        assert manager.status() == []  # nothing was spawned or even created

        result = await service.intelligence(
            "completion", uri=uri, position={"line": 0, "character": 1}
        )
        assert result.state == "missing"
        assert result.freshness == "unavailable"
        assert result.result is None
        assert result.server_generation is None
        assert result.source == "editor-buffer"
        assert result.document_version == 1

        # Changes and closes still work as transport state, with no fake frames.
        assert (
            service.document_change(
                uri,
                1,
                2,
                [
                    {
                        "range": {
                            "start": {"line": 0, "character": 4},
                            "end": {"line": 0, "character": 5},
                        },
                        "text": "2",
                    }
                ],
            ).accepted_version
            == 2
        )
        assert service.document_snapshot(uri)["text"] == "x = 2\n"
        assert service.document_close(uri)["closed"] is True
        assert events == []

        # A position is required for cursor operations; a typed denial, never
        # a guessed answer.
        from machina.workspace.models import WorkspaceError

        with pytest.raises(WorkspaceError):
            await service.intelligence("hover", uri=uri)


async def test_reopen_epoch_drops_old_workspace_server_results(tmp_path):
    """Plan §16/§19: an old workspace's server can never write into the new one."""
    manager = _manager(tmp_path)
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    service.set_lsp_manager(manager)
    service.open(str(tmp_path), True)
    try:
        uri = (tmp_path / "a.py").as_uri()
        service.document_open(uri, "python", 1, "x = 1\n")
        await _flush(service, uri)
        connection = await manager.ensure_started("python")

        reopened = service.open(str(tmp_path), True)
        assert reopened.epoch == 2
        assert service.document_snapshot(uri) is None  # mirrors dropped

        before = len(events)
        await connection.request(
            "publish-diagnostics",
            {"uri": uri, "version": 1, "diagnostics": [_diagnostic(0, "stale-epoch")]},
        )
        assert events[before:] == []  # never stored, never emitted
    finally:
        service.close()
        await service.wait_stopped()


async def test_reopen_after_close_is_honest_about_the_retired_manager(tmp_path):
    """close() retires the manager permanently; a reopen says so truthfully."""
    manager = _manager(tmp_path)
    service = WorkspaceService(send_event=lambda _event: None)
    service.set_lsp_manager(manager)
    service.open(str(tmp_path), True)
    uri = (tmp_path / "a.py").as_uri()
    service.document_open(uri, "python", 1, "x = 1\n")
    await _flush(service, uri)
    assert service.close() == {"closed": True}
    await service.wait_stopped()

    identity = service.open(str(tmp_path), True)
    assert identity.epoch == 2
    reopened = service.document_open(uri, "python", 1, "x = 1\n")
    assert reopened.state == "missing"
    control = await service.lsp_control("start", "python")
    assert control["state"] == "missing"
    assert "no language-server manager" in control["detail"]
    assert (await service.lsp_status()) == {"servers": []}
    service.close()
    await service.wait_stopped()


async def test_close_stops_everything_and_is_idempotent(tmp_path):
    manager = _manager(tmp_path)
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    service.set_lsp_manager(manager)
    identity = service.open(str(tmp_path), True)
    uri = (tmp_path / "a.py").as_uri()
    service.document_open(uri, "python", 1, "x = 1\n")
    await _flush(service, uri)
    assert (await manager.ensure_started("python")).state == STATE_READY

    assert service.close() == {"closed": True}
    assert service.close() == {"closed": True}  # idempotent
    await service.wait_stopped()
    assert [row["state"] for row in manager.status()] == ["stopped"]
    assert service.workspace_id is None
    assert identity.workspace_id != service.workspace_id
    from machina.workspace.models import WorkspaceError

    with pytest.raises(WorkspaceError):
        service.document_open(uri, "python", 1, "x = 1\n")


# ─── pure normalization units (untrusted server data) ───────────────────────
def test_sync_kind_and_request_mapping_units():
    assert lsp_sync_kind({"textDocumentSync": 2}) == 2
    assert lsp_sync_kind({"textDocumentSync": {"change": 1}}) == 1
    assert lsp_sync_kind({"textDocumentSync": {"change": 9}}) is None
    assert lsp_sync_kind({"textDocumentSync": True}) is None
    assert lsp_sync_kind({}) is None

    method, params = lsp_request_for("references", "file:///a.py", {"line": 1, "character": 2})
    assert method == "textDocument/references"
    assert params["context"] == {"includeDeclaration": True}
    method, params = lsp_request_for("symbols", "file:///a.py", None)
    assert method == "textDocument/documentSymbol"
    assert params == {"textDocument": {"uri": "file:///a.py"}}
    with pytest.raises(ValueError):
        lsp_request_for("rename", "file:///a.py", None)


def test_normalizers_reject_malformed_ranges_never_fabricate():
    malformed = {
        "range": {
            "start": {"line": -1, "character": 0},
            "end": {"line": 0, "character": 0},
        },
        "message": "bad",
    }
    # Unit-level check through the hover/completion/locations normalizers.
    assert normalize_hover({"contents": "ok", "range": malformed["range"]}).result == {
        "contents": "ok"
    }
    items = normalize_completion_list(
        [
            {"label": "no-edit", "textEdit": {"range": malformed["range"], "newText": "x"}},
            {
                "label": "insert-replace-fallback",
                "textEdit": {
                    "newText": "y",
                    "insert": malformed["range"],
                    "replace": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                },
            },
        ]
    ).result["items"]
    assert "textEdit" not in items[0]
    assert items[1]["textEdit"]["range"] == {
        "start": {"line": 0, "character": 0},
        "end": {"line": 0, "character": 1},
    }
    locations = normalize_locations(
        [
            {"uri": "file:///a.py", "range": malformed["range"]},
            {"targetUri": "file:///b.py", "targetSelectionRange": malformed["range"]},
            {"uri": "file:///c.py", "range": None},
        ]
    ).result
    assert locations == {"locations": [], "omitted": 0}
    assert normalize_completion_list({"items": "nope"}) is None
    assert json.dumps(locations)  # json-able


def test_symbols_cap_bounds_both_views_exactly():
    raw = [
        {
            "name": f"item{index:04d}",
            "kind": 13,
            "range": {
                "start": {"line": index, "character": 0},
                "end": {"line": index, "character": 1},
            },
        }
        for index in range(600)
    ]
    normalized = normalize_for("symbols", raw)
    assert normalized.truncated is True
    assert normalized.result["omitted"] == 100
    assert len(normalized.result["hierarchical"]) == 500
    assert len(normalized.result["flat"]) == 500


def test_sanitize_log_text_strips_control_characters():
    assert sanitize_log_text("a\x1b[31mb\x07c\x00d") == "abcd"
    assert len(sanitize_log_text("x" * 1000, limit=10)) == 10


async def test_did_open_sends_the_documents_protocol_language(tmp_path):
    """didOpen languageId follows the DOCUMENT, not the profile language.

    CodeRabbit pass-1 finding: sending the canonical profile language (e.g.
    "typescript" for every TS-family file) makes servers treat .tsx/.jsx/.js
    with the wrong grammar family. The extension map carries the precise
    protocol ids; the profile language still selects the server (unchanged).
    """
    manager = _manager(tmp_path)
    async with _workspace(tmp_path, manager) as (service, _, _events):
        cases = (
            ("app.tsx", "typescriptreact", "typescriptreact"),
            ("widget.jsx", "javascriptreact", "javascriptreact"),
            ("esm.mjs", "javascript", "javascript"),
            ("legacy.cts", "typescript", "typescript"),
            ("main.py", "python", "python"),
        )
        for name, declared, expected in cases:
            uri = (tmp_path / name).as_uri()
            service.document_open(uri, declared, 1, "x = 1\n")
            await _flush(service, uri)
        connection = await manager.ensure_started("typescript")
        records = _by_method(await _records(connection), "textDocument/didOpen")
        sent = {r["textDocument"]["uri"]: r["textDocument"]["languageId"] for r in records}
        for name, _declared, expected in cases:
            if name.endswith(".py"):
                continue  # python goes to its own profile connection
            assert sent[(tmp_path / name).as_uri()] == expected, name
        python_connection = await manager.ensure_started("python")
        py_records = _by_method(await _records(python_connection), "textDocument/didOpen")
        py_sent = {r["textDocument"]["uri"]: r["textDocument"]["languageId"] for r in py_records}
        assert py_sent[(tmp_path / "main.py").as_uri()] == "python"
