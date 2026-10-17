"""Fixes from the 2026-10-16 adversarial workspace review.

Each test pins one reviewed defect:

1. Closing a document clears its DiagnosticsStore sets, so a reopen (a fresh
   mirror at version 0) is not stuck behind the old version gate.
2. ``save`` returns right after the disk commit even when the document's LSP
   sync chain is parked; ``didSave`` still follows the flush in the background.
3. Connection notifications resolve the epoch at delivery time, so a
   connection re-tracked after a workspace reopen publishes under the new
   epoch (a genuinely stale connection is still dropped).
4. ``intelligence``/``lspStatus`` refuse an untrusted workspace with the same
   structured refusal as ``lspControl``, even with a manager attached.
5. A retired manager's slow ``stop_all`` is never cancelled by a reopen; it
   completes in the background and is drained at gateway shutdown.
6. LSP framing escapes lone surrogates and a server request always receives a
   response frame.
7. The diagnostics store's ``(uri, producer)`` table is capped with
   least-recently-published eviction, disclosed via ``evicted``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from machina.gateway.features import workspace as ws
from machina.workspace import lsp as lsp_mod
from machina.workspace.intelligence import DiagnosticsStore
from machina.workspace.lsp import STATE_READY, LspConnection, LspManager
from machina.workspace.models import WorkspaceError
from machina.workspace.service import WorkspaceService, clear_active_workspace_service


def _diagnostic(message: str, *, line: int = 0) -> dict:
    return {
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 1},
        },
        "severity": 1,
        "message": message,
    }


def _diagnostics_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "workspace.diagnostics"]


class FakeConnection:
    """Duck-typed LSP connection for service bookkeeping (no process)."""

    def __init__(
        self,
        *,
        profile_id: str = "pyright",
        generation: int = 1,
        state: str = STATE_READY,
    ) -> None:
        self.profile_id = profile_id
        self.generation = generation
        self.state = state
        self.error = ""
        self.cleanup_pending = False
        self.capabilities: dict = {}
        self.state_callbacks: list = []
        self.notification_callbacks: list = []
        self.did_open_calls: list = []
        self.did_change_calls: list = []
        self.did_save_calls: list = []
        self.did_close_calls: list = []

    def on_state_change(self, callback) -> None:
        self.state_callbacks.append(callback)

    def on_notification(self, callback) -> None:
        self.notification_callbacks.append(callback)

    def notify(self, method: str, params) -> None:
        for callback in list(self.notification_callbacks):
            callback(method, params)

    def did_open(self, uri: str, language_id: str, version: int, text: str) -> None:
        self.did_open_calls.append((uri, language_id, version, text))

    def did_change(self, uri: str, version: int, changes) -> None:
        self.did_change_calls.append((uri, version, changes))

    def did_save(self, uri: str, text: str | None = None) -> None:
        self.did_save_calls.append((uri, text))

    def did_close(self, uri: str) -> None:
        self.did_close_calls.append(uri)


class FakeManager:
    """Bare attached manager: no dispatch/start methods, so the service keeps
    its orchestrated path while a manager is present (as in production)."""


@pytest.fixture(autouse=True)
def _isolate_active_workspace():
    yield
    clear_active_workspace_service()


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


# ─── 1. close → reopen resets the diagnostics version gate ───────────────────
async def test_document_close_clears_diagnostics_so_reopen_publishes_again(tmp_path):
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(FakeManager())
    uri = (tmp_path / "a.py").as_uri()
    service.document_open(uri, "python", 0, "x = 1\n", epoch=service.epoch)
    connection = FakeConnection()
    service._track_connection("python", connection)

    def publish(version: int, message: str) -> bool:
        before = len(_diagnostics_events(events))
        service._handle_connection_notification(
            connection,
            "textDocument/publishDiagnostics",
            {"uri": uri, "version": version, "diagnostics": [_diagnostic(message)]},
        )
        return len(_diagnostics_events(events)) > before

    assert publish(42, "newer") is True
    assert publish(10, "older") is False  # while open: the version gate holds
    assert _diagnostics_events(events)[0]["epoch"] == service.epoch

    assert service.document_close(uri, epoch=service.epoch)["closed"] is True
    service.document_open(uri, "python", 0, "x = 1\n", epoch=service.epoch)
    assert publish(0, "fresh") is True  # a reopen starts at version 0 again
    read = await service.intelligence("diagnostics", uri=uri)
    assert read.result["producers"] == [
        {"source": "pyright", "version": 0, "freshness": "versioned", "count": 1}
    ]
    assert [item["message"] for item in read.result["diagnostics"]] == ["fresh"]


# ─── 2. save never blocks on a parked LSP sync chain ─────────────────────────
async def test_save_returns_promptly_with_a_parked_sync_chain(tmp_path):
    service = WorkspaceService(send_event=lambda _event: None)
    service.open(str(tmp_path), trust=True)
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    uri = target.as_uri()
    read = service.read(uri)

    service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
    connection = FakeConnection()
    service._track_connection("python", connection)  # didOpen + open record

    gate = asyncio.Event()

    async def parked() -> None:
        await gate.wait()

    parked_task = asyncio.get_running_loop().create_task(parked())
    service._sync_tasks[uri] = parked_task

    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await service.save(uri, "x = 2\n", read.revision, epoch=service.epoch)
    elapsed = loop.time() - started
    assert result.ok is True
    assert elapsed < 2.0  # the parked chain never holds the save RPC
    assert connection.did_save_calls == []  # didSave still follows the flush

    gate.set()
    await _until(lambda: bool(connection.did_save_calls))
    assert connection.did_save_calls == [(uri, "x = 2\n")]
    assert not parked_task.cancelled()


# ─── 3. notification epoch is resolved at delivery time ──────────────────────
async def test_reused_connection_notifications_deliver_under_the_new_epoch(tmp_path):
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    service.open(str(tmp_path), trust=True)
    connection = FakeConnection()
    service._track_connection("python", connection)
    assert len(connection.notification_callbacks) == 1

    service.open(str(tmp_path), trust=True)  # epoch bump; tracking cleared
    assert service.epoch == 2
    params = {
        "uri": (tmp_path / "a.py").as_uri(),
        "version": 1,
        "diagnostics": [_diagnostic("late")],
    }

    connection.notify("textDocument/publishDiagnostics", params)  # not re-tracked
    assert _diagnostics_events(events) == []  # genuinely stale: dropped

    service._track_connection("python", connection)  # reuse the same object
    assert len(connection.notification_callbacks) == 1  # no duplicate handler
    connection.notify("textDocument/publishDiagnostics", params)
    published = _diagnostics_events(events)
    assert len(published) == 1
    assert published[0]["epoch"] == 2


# ─── 4. trusted-only capability enforcement ──────────────────────────────────
async def test_untrusted_workspace_refuses_intelligence_and_lsp_status(tmp_path):
    service = WorkspaceService(send_event=lambda _event: None)
    service.open(str(tmp_path), trust=False)
    service.set_lsp_manager(FakeManager())  # attached despite untrusted

    uri = (tmp_path / "a.py").as_uri()
    with pytest.raises(WorkspaceError) as intelligence_refusal:
        await service.intelligence("symbols", uri=uri)
    assert intelligence_refusal.value.reason == "denied"
    assert "refused" in intelligence_refusal.value.as_refusal()

    with pytest.raises(WorkspaceError) as status_refusal:
        await service.lsp_status()
    assert status_refusal.value.reason == "denied"
    assert "refused" in status_refusal.value.as_refusal()


# ─── 5. a slow retired manager's stop is never cancelled ─────────────────────
class _SlowManager:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.started = asyncio.Event()
        self.completed = False
        self.cancelled = False

    async def stop_all(self) -> None:
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True


class _FakeGateway:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.workspace_service = None

    def require_db(self) -> None:
        return None

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)


async def test_reopen_retires_slow_manager_without_cancelling_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "MANAGER_STOP_TIMEOUT", 0.05)
    gw = _FakeGateway()
    slow = _SlowManager(delay=0.6)
    gw._workspace_lsp_manager = slow  # a manager left behind by a previous open

    loop = asyncio.get_running_loop()
    started = loop.time()
    await ws.workspace_open(gw, {"root": str(tmp_path), "trust": True})
    elapsed = loop.time() - started
    assert elapsed < 0.45  # the bounded observation ended, the RPC returned
    assert gw._workspace_lsp_manager is not slow
    assert isinstance(gw._workspace_lsp_manager, LspManager)

    await slow.started.wait()
    assert slow.completed is False  # still stopping after the RPC returned
    assert slow.cancelled is False

    await ws.drain_retired_lsp_managers(gw, timeout=5.0)
    assert slow.completed is True
    assert slow.cancelled is False


# ─── 6. Unicode-safe LSP framing ─────────────────────────────────────────────
def test_framing_encodes_a_lone_surrogate_in_server_data() -> None:
    frame = lsp_mod.LspFraming.encode(
        {"jsonrpc": "2.0", "id": 4, "method": "\ud800", "params": {}}
    )
    header, _, body = frame.partition(b"\r\n\r\n")
    assert header.startswith(b"Content-Length: ")
    assert int(header.split(b":")[1]) == len(body)
    assert json.loads(body.decode("utf-8"))["method"] == "\ud800"


async def test_server_request_with_surrogate_method_still_gets_a_frame(tmp_path):
    connection = LspConnection(profile=None, root=tmp_path, enabled=False)
    frames: list[bytes] = []

    async def capturing_send(message):
        frames.append(lsp_mod.LspFraming.encode(message))

    connection._send = capturing_send
    await connection._answer_server_request(9, "\ud800", None)

    assert len(frames) == 1
    message = json.loads(frames[0].partition(b"\r\n\r\n")[2].decode("utf-8"))
    assert message["id"] == 9
    assert message["error"]["code"] == lsp_mod.JSONRPC_METHOD_NOT_FOUND
    assert "\ud800" in message["error"]["message"]


async def test_server_request_encoding_failure_gets_ascii_fallback(tmp_path):
    connection = LspConnection(profile=None, root=tmp_path, enabled=False)
    calls: list[dict] = []

    async def flaky_send(message):
        calls.append(message)
        if len(calls) == 1:
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    connection._send = flaky_send
    await connection._answer_server_request(11, "workspace/unknown", None)

    assert len(calls) == 2
    fallback = calls[1]
    assert fallback["id"] == 11
    assert fallback["error"] == {
        "code": lsp_mod.JSONRPC_INTERNAL_ERROR,
        "message": "server request response could not be encoded",
    }


# ─── 7. bounded diagnostics-store key count ──────────────────────────────────
def test_diagnostics_store_caps_set_count_with_lru_eviction():
    store = DiagnosticsStore(max_sets=2)

    def publish(uri: str, message: str) -> None:
        outcome = store.publish(
            uri=uri,
            producer="pyright",
            diagnostics=[_diagnostic(message)],
            version=1,
            generation=1,
            epoch=1,
        )
        assert outcome is not None

    publish("file:///a", "a")
    publish("file:///b", "b")
    publish("file:///a", "a2")  # touch: b is now least-recently published
    assert store.evicted == 0

    publish("file:///c", "c")  # new key at the cap evicts b
    assert store.evicted == 1

    def entries(uri: str):
        return store.entries(uri, epoch=1, generation_for=lambda _producer: 1)

    assert len(entries("file:///a")) == 1
    assert len(entries("file:///c")) == 1
    assert entries("file:///b") == []
    assert entries("file:///a")[0].diagnostics[0]["message"] == "a2"


def test_diagnostics_store_stays_bounded_across_many_uris():
    store = DiagnosticsStore(max_sets=4)
    for index in range(10):
        store.publish(
            uri=f"file:///f{index}",
            producer="pyright",
            diagnostics=[_diagnostic(f"m{index}")],
            version=1,
            generation=1,
            epoch=1,
        )
    assert store.evicted == 6
    kept = sum(
        1
        for index in range(10)
        if store.entries(f"file:///f{index}", epoch=1, generation_for=lambda _p: 1)
    )
    assert kept == 4


# ─── 8. incremental-sync fallback must describe the server's old text ────────
async def test_incremental_sync_fallback_replaces_server_text_without_stale_range(tmp_path):
    """combined #1: a fallback full replacement must not use the NEW text's
    extent; when the document shrinks, that range leaves old server text."""
    from machina.workspace.documents import utf16_full_range

    service = WorkspaceService(send_event=lambda _event: None)
    service.open(str(tmp_path), trust=True)
    uri = (tmp_path / "a.py").as_uri()
    service.document_open(uri, "python", 1, "abcdef\nlast", epoch=service.epoch)
    connection = FakeConnection()
    connection.capabilities = {"textDocumentSync": 2}
    service._track_connection("python", connection)
    assert connection.did_open_calls[-1] == (uri, "python", 1, "abcdef\nlast")
    connection.did_change_calls.clear()

    service._documents.change_document(
        uri,
        1,
        2,
        [{"range": utf16_full_range("abcdef\nlast"), "text": "a"}],
        epoch=service.epoch,
    )
    service._apply_document_sync(connection, uri, "python", None)

    assert connection.did_change_calls == [(uri, 2, [{"text": "a"}])]
