"""Real language-server workspace harness for §24-E integration tests.

Composes the real ``WorkspaceService`` (T5/mirrors/intelligence dispatch) with
the real ``LspManager``/``LspConnection`` transport (T7) and drives the
provisioned pinned servers (pyright-langserver, typescript-language-server)
over real stdio pipes.

Why this module exists: at baseline 9427928 the backend has both halves but no
document-sync/intelligence adapter between them (T8 owns that seam). This
harness implements exactly that seam as a transparent pass-through — the
service's ``set_lsp_manager`` protocol is satisfied by ``RealLspWorkspace``
itself, its document wrappers forward the mirror's truth as didOpen/didChange/
didClose/didSave frames, ``publishDiagnostics`` notifications are collected
verbatim, and every completion/hover/definition/references/symbols payload is
the server's own response. No LSP behavior is mocked here; remove this file
when the T8 adapter lands and point the tests at it.

Dual-backend note: when the attached ``WorkspaceService`` already owns document
sync and the on_ready reopen (detected by capability, not version), the
harness stops forwarding frames and only supplies the manager surface the
service calls (``ensure_started``, ``restart``, ``status``, ``intelligence``)
plus diagnostics collection. That keeps the same tests truthful against both
the T5 service and the newer service without double-syncing documents.

Freshness values follow ``ui/src/workspace/protocol.ts``:
``"versioned" | "unversioned" | "pending" | "stale" | "unavailable"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import urllib.parse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from machina.workspace import lsp as lsp_module
from machina.workspace.lsp import (
    STATE_MISSING,
    STATE_READY,
    STATE_STOPPED,
    LspConnection,
    LspError,
    LspManager,
    LspProfile,
    resolve_profile,
)
from machina.workspace.models import language_id_for
from machina.workspace.service import WorkspaceService

__all__ = [
    "RealLspWorkspace",
    "diagnostics_capable_client",
    "uri_key",
    "wait_for",
]


def uri_key(uri: str) -> str:
    """Filesystem-realpath key for a file URI.

    macOS temp dirs are symlinked (``/tmp`` -> ``/private/tmp``); pyright
    answers some requests with realpath URIs and others with the URI it was
    sent, so comparisons must normalize both sides.
    """
    try:
        path = urllib.parse.unquote(urllib.parse.urlsplit(uri).path)
    except ValueError:
        return uri
    return os.path.realpath(path)


async def wait_for(
    step: Callable[[], Any],
    predicate: Callable[[Any], bool],
    *,
    timeout: float = 20.0,
    interval: float = 0.25,
    label: str = "condition",
) -> Any:
    """Poll ``await step()`` until ``predicate(value)`` holds (bounded wait).

    On deadline the last value is included in the error so a real-server
    mismatch is debuggable from the failure text alone.
    """

    async def _run() -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last: Any = None
        while True:
            last = await step()
            if predicate(last):
                return last
            if loop.time() >= deadline:
                raise AssertionError(
                    f"{label}: not satisfied within {timeout:g}s; last value: {last!r}"
                )
            await asyncio.sleep(interval)

    return await _run()


@contextlib.contextmanager
def diagnostics_capable_client(monkeypatch):
    """Declare ``textDocument.publishDiagnostics`` for the client when absent.

    Observed 2026-10-15 against typescript-language-server 6.0.2 (pinned):
    ``lib/cli.mjs`` sets ``features.diagnosticsSupport =
    Boolean(clientCapabilities.textDocument.publishDiagnostics)`` and both
    ``FileDiagnostics.publishDiagnostics()`` and ``clearDiagnosticsForFile``
    return early without it, so the server publishes zero diagnostics to this
    client. ``machina/workspace/lsp.py::_client_capabilities()`` (T7) does not
    declare the key. This context manager injects the standard capability for
    the duration of one test so the real diagnostics pipeline can be
    exercised; when the backend declares it, this is a no-op. Reported as a T7
    gap — not patched in the backend by this test task.
    """
    shipped = lsp_module._client_capabilities()
    if "publishDiagnostics" in shipped.get("textDocument", {}):
        print("[lsp-real] client already declares textDocument.publishDiagnostics")
        yield False
        return

    print(
        "[lsp-real] GAP: machina.workspace.lsp._client_capabilities() omits "
        "textDocument.publishDiagnostics; typescript-language-server 6.0.2 suppresses "
        "all diagnostics without it. Injecting the standard capability for this test "
        "only (backend fix pending, owned by the LSP task)."
    )
    original = lsp_module._client_capabilities

    def augmented() -> dict[str, Any]:
        capabilities = original()
        capabilities.setdefault("textDocument", {})["publishDiagnostics"] = {
            "relatedInformation": True
        }
        return capabilities

    monkeypatch.setattr(lsp_module, "_client_capabilities", augmented)
    yield True


class RealLspWorkspace:
    """One temporary workspace wired to real language servers.

    Service-facing manager protocol: ``ensure_started``, ``restart``,
    ``state_for_language``, ``status``, ``intelligence`` and (T5 baseline)
    ``control``. Document sync is forwarded only when the attached service
    does not do it itself: the T5 service only accepts mirrors and delegates
    intelligence, while the later backend service applies didOpen/didChange/
    didClose/didSave and reopens mirrors on its own. The mirror text is always
    the source of truth sent to the server, and a ``ready`` callback reopens
    every tracked document with its latest full text after restarts (on the
    baseline, where the service has no reopen of its own).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        profiles: Mapping[str, LspProfile] | None = None,
        connection_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.events: list[dict[str, Any]] = []
        self.service = WorkspaceService(send_event=self.events.append)
        self.identity = self.service.open(str(self.root), True)
        self.manager = LspManager(self.root, profiles=profiles, **dict(connection_kwargs or {}))
        self.service.set_lsp_manager(self)
        self._service_owns_sync = callable(
            getattr(self.service, "_apply_document_sync", None)
        )
        if not self._service_owns_sync:
            self.manager.on_ready(self._reopen_documents)

        self._connections: dict[str, LspConnection] = {}
        self._tracked_ids: set[int] = set()
        self._open_docs: dict[str, dict[str, Any]] = {}
        self._diagnostics: dict[str, dict[str, Any]] = {}
        self._diag_seq = 0
        self.did_save_calls: list[str] = []

    @property
    def service_owns_document_sync(self) -> bool:
        """True when the attached service performs document sync itself."""
        return self._service_owns_sync

    # ─── filesystem and canonical URIs ──────────────────────────────────────
    def path(self, rel: str) -> Path:
        return self.root / rel

    def write(self, rel: str, text: str) -> None:
        self.path(rel).write_text(text, encoding="utf-8")

    def uri(self, rel: str) -> str:
        """Canonical (realpath-rooted) URI, matching the service's canonical form."""
        return (self.root / rel).as_uri()

    # ─── lifecycle ──────────────────────────────────────────────────────────
    async def ensure_started(
        self, language: str, *, explicit_path: str | None = None
    ) -> LspConnection:
        connection = await self.manager.ensure_started(language, explicit_path=explicit_path)
        self._track(language, connection)
        return connection

    def connection_for(self, language: str) -> LspConnection | None:
        return self._connections.get(language)

    async def stop_language(self, language: str) -> LspConnection | None:
        """Stop one connection and wait (bounded) until the state is stopped.

        The pinned typescript-language-server can report ``cleanup_pending``
        first: its launcher exits with live tsserver children in the group, so
        the transport's immediate post-SIGKILL group check races their death.
        A short observe_termination poll observes the honest final state.
        """
        connection = self._connections.get(language)
        if connection is None:
            return None
        await connection.stop()
        await self._await_stopped(connection)
        return connection

    async def restart_language(self, language: str) -> LspConnection:
        connection = await self.manager.restart(language)
        self._track(language, connection)
        return connection

    async def restart(self, language: str) -> LspConnection:
        """Manager-protocol restart (the newer backend service calls this)."""
        return await self.restart_language(language)

    async def _await_stopped(self, connection: LspConnection, timeout: float = 10.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while connection.state not in (STATE_STOPPED, STATE_MISSING):
            if loop.time() >= deadline:
                return
            await asyncio.sleep(0.25)
            with contextlib.suppress(Exception):
                await connection.observe_termination(1.0)

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self.manager.stop_all()
        for connection in list(self._connections.values()):
            await self._await_stopped(connection, timeout=5.0)
            if connection.state not in (STATE_STOPPED, STATE_MISSING):
                print(
                    f"[lsp-real] WARNING: connection ({connection.profile_id}) still "
                    f"{connection.state} after aclose"
                )
        with contextlib.suppress(Exception):
            self.service.close()

    # ─── document sync (forwarded only when the service lacks its own) ──────
    def _language_for(self, uri: str | None) -> str:
        if uri is None:
            return ""
        record = self._open_docs.get(uri_key(uri))
        if record is not None:
            return str(record["language"])
        return language_id_for(Path(uri_key(uri)).name)

    async def document_open(self, uri: str, language_id: str, version: int, text: str):
        result = self.service.document_open(uri, language_id, version, text)
        snapshot = self.service.document_snapshot(uri)
        assert snapshot is not None
        record = {
            "language": language_id,
            "uri": self.uri_of(snapshot["uri"]),
            "version": snapshot["version"],
        }
        self._open_docs[uri_key(snapshot["uri"])] = record
        if not self._service_owns_sync:
            connection = self._connections.get(language_id)
            if connection is not None and connection.state == STATE_READY:
                connection.did_open(
                    record["uri"], language_id, snapshot["version"], snapshot["text"]
                )
        return result

    def uri_of(self, uri: str) -> str:
        """Canonicalize any file URI onto this workspace's resolved root."""
        path = Path(uri_key(uri))
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return uri
        return (self.root / rel).as_uri()

    async def document_change(
        self, uri: str, base_version: int, version: int, changes: list[dict[str, Any]]
    ):
        result = self.service.document_change(uri, base_version, version, changes)
        record = self._open_docs.get(uri_key(uri))
        if record is not None:
            record["version"] = result.accepted_version
        if self._service_owns_sync:
            return result
        connection = self._connections.get(self._language_for(uri))
        snapshot = self.service.document_snapshot(uri)
        if (
            record is not None
            and connection is not None
            and connection.state == STATE_READY
            and snapshot is not None
        ):
            # The mirror is the source of truth: send the full latest text at
            # the accepted version (this also resyncs after a rejected change).
            connection.did_change(
                self.uri_of(snapshot["uri"]), result.accepted_version, [{"text": snapshot["text"]}]
            )
        return result

    async def document_close(self, uri: str):
        record = self._open_docs.pop(uri_key(uri), None)
        connection = self._connections.get(record["language"]) if record else None
        result = self.service.document_close(uri)
        if (
            not self._service_owns_sync
            and record is not None
            and connection is not None
            and connection.state == STATE_READY
        ):
            connection.did_close(self.uri_of(uri))
        return result

    async def save(self, uri: str, text: str, expected_revision: str | None = None, **kwargs):
        result = await self.service.save(uri, text, expected_revision, **kwargs)
        if result.ok and not self._service_owns_sync:
            record = self._open_docs.get(uri_key(uri))
            connection = self._connections.get(record["language"]) if record else None
            if connection is not None and connection.state == STATE_READY:
                connection.did_save(self.uri_of(uri), text)
                self.did_save_calls.append(uri)
        return result

    async def _reopen_documents(self, connection: LspConnection) -> None:
        """After every successful initialize, resend the latest mirror text."""
        for record in list(self._open_docs.values()):
            if record["language"] != connection.language_id:
                continue
            snapshot = self.service.document_snapshot(record["uri"])
            if snapshot is None:
                continue
            with contextlib.suppress(LspError):
                connection.did_open(
                    record["uri"], record["language"], snapshot["version"], snapshot["text"]
                )
                record["version"] = snapshot["version"]

    # ─── diagnostics collection ─────────────────────────────────────────────
    def _track(self, language: str, connection: LspConnection) -> None:
        self._connections[language] = connection
        if id(connection) in self._tracked_ids:
            return
        self._tracked_ids.add(id(connection))

        def on_notification(method: str, params: Any) -> None:
            if method != "textDocument/publishDiagnostics" or not isinstance(params, dict):
                return
            uri = params.get("uri")
            if not isinstance(uri, str):
                return
            version = params.get("version")
            self._diag_seq += 1
            self._diagnostics[uri_key(uri)] = {
                "language": language,
                "version": version if isinstance(version, int) else None,
                "diagnostics": [
                    item for item in params.get("diagnostics", []) if isinstance(item, dict)
                ],
                "sequence": self._diag_seq,
            }

        def on_state(_connection: LspConnection, state: str) -> None:
            if state != STATE_READY:
                # A stopped/failed generation's pushed diagnostics are stale
                # for the next generation; drop them so a post-restart poll
                # cannot pass on pre-restart data.
                self._clear_diagnostics(language)

        connection.on_notification(on_notification)
        connection.on_state_change(on_state)

    def _clear_diagnostics(self, language: str) -> None:
        for key, entry in list(self._diagnostics.items()):
            if entry["language"] == language:
                del self._diagnostics[key]

    # ─── WorkspaceService manager protocol ──────────────────────────────────
    def state_for_language(self, language_id: str) -> str:
        connection = self._connections.get(language_id)
        if connection is not None:
            return connection.state
        return STATE_MISSING if resolve_profile(language_id) is None else STATE_STOPPED

    async def status(self) -> list[dict[str, Any]]:
        return self.manager.status()

    async def intelligence(self, request: dict[str, Any]) -> dict[str, Any] | None:
        operation = str(request.get("operation") or "")
        uri = request.get("uri")
        language = self._language_for(uri if isinstance(uri, str) else None)
        connection = self._connections.get(language) if language else None
        if connection is None:
            state = STATE_MISSING if resolve_profile(language) is None else STATE_STOPPED
            return {"state": state, "freshness": "unavailable", "result": None}
        if connection.state != STATE_READY:
            return {"state": connection.state, "freshness": "unavailable", "result": None}
        if operation == "diagnostics":
            entry = self._diagnostics.get(uri_key(uri)) if isinstance(uri, str) else None
            requested = request.get("version")
            fresh = entry is not None and (
                requested is None or entry["version"] is None or entry["version"] == requested
            )
            return {
                "state": "ready",
                "freshness": "versioned" if fresh else "unversioned",
                "truncated": False,
                "result": entry["diagnostics"] if entry is not None else [],
                "serverGeneration": connection.generation,
            }
        try:
            payload = await self._dispatch_request(connection, operation, request)
        except Exception as error:  # noqa: BLE001 — degrade honestly, with the cause
            # Transport failures and harness defects both degrade honestly; the
            # type+message travels in the detail so a poll timeout shows why.
            return {
                "state": "failed",
                "freshness": "unavailable",
                "truncated": False,
                "result": None,
                "detail": f"{type(error).__name__}: {error}"[:300],
            }
        return {
            "state": "ready",
            "freshness": "unversioned",
            "truncated": False,
            "result": payload,
            "serverGeneration": connection.generation,
        }

    async def _dispatch_request(
        self, connection: LspConnection, operation: str, request: dict[str, Any]
    ) -> Any:
        uri = str(request.get("uri") or "")
        position = request.get("position")
        if operation == "symbols":
            return await connection.request(
                "textDocument/documentSymbol", {"textDocument": {"uri": uri}}, timeout=15
            )
        if not isinstance(position, dict):
            raise LspError("position is required for this operation")
        params: dict[str, Any] = {"textDocument": {"uri": uri}, "position": position}
        if operation == "completion":
            return self._normalize_completion(
                await connection.request("textDocument/completion", params, timeout=15)
            )
        if operation == "hover":
            return await connection.request("textDocument/hover", params, timeout=15)
        if operation == "definition":
            return self._normalize_locations(
                await connection.request("textDocument/definition", params, timeout=15)
            )
        if operation == "references":
            params["context"] = {"includeDeclaration": True}
            return self._normalize_locations(
                await connection.request("textDocument/references", params, timeout=15)
            )
        raise LspError(f"operation {operation!r} is not supported by this harness")

    @staticmethod
    def _normalize_completion(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            items = result.get("items")
            return {
                "isIncomplete": bool(result.get("isIncomplete")),
                "items": [item for item in (items or []) if isinstance(item, dict)],
            }
        if isinstance(result, list):
            return {
                "isIncomplete": False,
                "items": [item for item in result if isinstance(item, dict)],
            }
        return {"isIncomplete": False, "items": []}

    @staticmethod
    def _normalize_locations(result: Any) -> list[dict[str, Any]]:
        entries = result if isinstance(result, list) else [result]
        locations: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            uri = entry.get("uri") or entry.get("targetUri")
            span = entry.get("range") or entry.get("targetSelectionRange") or entry.get(
                "targetRange"
            )
            if isinstance(uri, str) and isinstance(span, dict):
                locations.append({"uri": uri, "range": span})
        return locations

    async def control(self, action: str, profile: str) -> dict[str, Any]:
        if action == "start":
            connection = await self.ensure_started(profile)
            return {"state": connection.state, "generation": connection.generation}
        connection = self._connections.get(profile)
        if connection is None:
            return {"state": "missing", "detail": f"no connection for profile {profile!r}"}
        if action == "stop":
            await self.stop_language(profile)
            return {"state": connection.state, "generation": connection.generation}
        if action == "restart":
            connection = await self.restart_language(profile)
            return {"state": connection.state, "generation": connection.generation}
        return {"state": "missing", "detail": f"unsupported action {action!r}"}
