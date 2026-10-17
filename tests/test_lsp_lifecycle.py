"""Real-subprocess lifecycle tests for ``machina.workspace.lsp``.

Every test drives a scripted fake LSP server (`tests/support/fake_lsp_server.py`)
over real stdio pipes. Synchronization is event-driven (state-change callbacks
and ``$/fake/progress`` notifications); there are no fixed sleeps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

from machina.workspace import lsp as lsp_module
from machina.workspace.lsp import (
    JSONRPC_METHOD_NOT_FOUND,
    LSP_CONTENT_MODIFIED,
    LSP_REQUEST_CANCELLED,
    LSP_REQUEST_FAILED,
    STATE_DISABLED,
    STATE_FAILED,
    STATE_INITIALIZING,
    STATE_MISSING,
    STATE_READY,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    STOP_OUTCOME_CLEANUP_PENDING,
    STOP_OUTCOME_STOPPED,
    LspConnection,
    LspManager,
    LspPendingLimitError,
    LspProfile,
    LspProtocolError,
    LspRequestError,
    LspTimeoutError,
    LspUnavailableError,
)

FAKE_SERVER = Path(__file__).resolve().parent / "support" / "fake_lsp_server.py"
REPO_ROOT = Path(lsp_module.__file__).resolve().parents[2]


def fake_profile(*extra_args: str) -> LspProfile:
    return LspProfile(
        language_id="python",
        profile_id="fake",
        executable=sys.executable,
        args=[str(FAKE_SERVER), *extra_args],
    )


class Condition:
    """Await a predicate; ``poke()`` re-checks it from sync callbacks."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def poke(self, *_: object) -> None:
        self._event.set()

    async def wait(self, predicate, timeout: float = 10.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if predicate():
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError("condition not met within timeout")
            self._event.clear()
            if predicate():
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._event.wait(), remaining)


class ProgressBus:
    """Collect ``$/fake/progress`` notifications and await named events."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self._waiters: dict[str, list[asyncio.Future[None]]] = {}

    def on_notification(self, method: str, params: object) -> None:
        if method != "$/fake/progress" or not isinstance(params, dict):
            return
        name = str(params.get("event", ""))
        self.events.append(name)
        for future in self._waiters.pop(name, []):
            if not future.done():
                future.set_result(None)

    async def wait(self, name: str, timeout: float = 10.0) -> None:
        if name in self.events:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(name, []).append(future)
        try:
            await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise AssertionError(
                f"progress event {name!r} not received; saw {self.events}"
            ) from None


async def _until(predicate, timeout: float = 5.0) -> None:
    """Bounded yield loop for the rare cases without a callback to hook."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0)


# --- handshake --------------------------------------------------------------


async def test_handshake_order_capabilities_and_no_false_ready(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path)
    await conn.start()
    try:
        assert conn.state == STATE_READY
        assert conn.generation == 1
        log = await conn.request("record-log")
        methods = [entry["method"] for entry in log["records"]]
        assert methods == ["initialize", "initialized", "record-log"]
        assert log["initializeCount"] == 1

        params = log["initializeParams"]
        assert params["processId"] == os.getpid()
        assert params["rootUri"] == tmp_path.resolve().as_uri()
        assert params["workspaceFolders"] == [
            {"uri": tmp_path.resolve().as_uri(), "name": tmp_path.name}
        ]
        capabilities = params["capabilities"]
        assert capabilities["general"]["positionEncodings"] == ["utf-16"]
        # I-6: production never registers on_server_request, so the client must
        # not advertise workspace-edit support it always refuses.
        assert capabilities["workspace"]["applyEdit"] is False
        assert "workspaceEdit" not in capabilities["workspace"]
        assert capabilities["textDocument"]["synchronization"] == {"didSave": True}
        assert (
            capabilities["textDocument"]["completion"]["completionItem"]["snippetSupport"] is False
        )
        assert capabilities["textDocument"]["rename"]["prepareSupport"] is True
        for absent in ("dynamicRegistration", "executeCommand", '"configuration"'):
            assert absent not in json.dumps(capabilities)

        assert conn.capabilities["textDocumentSync"]["save"]["includeText"] is True
        assert conn.server_info["name"] == "fake-lsp"
    finally:
        await conn.stop()


async def test_spawn_success_is_not_ready_and_pre_ready_requests_rejected(tmp_path):
    profile = fake_profile("--mode", "slow-initialize", "--delay", "0.6")
    condition = Condition()
    states: list[str] = []
    conn = LspConnection(profile, tmp_path)

    def on_state(connection: LspConnection, state: str) -> None:
        states.append(state)
        condition.poke()

    conn.on_state_change(on_state)
    task = asyncio.create_task(conn.start())
    try:
        await condition.wait(lambda: conn.state == STATE_INITIALIZING)
        assert conn.pid is not None  # spawn succeeded...
        assert conn.state != STATE_READY  # ...but readiness is not claimed yet
        with pytest.raises(LspUnavailableError):
            await conn.request("ping")
        await asyncio.wait_for(task, 10)
        assert conn.state == STATE_READY
        assert states[:2] == [STATE_STARTING, STATE_INITIALIZING]
        assert states[-1] == STATE_READY
    finally:
        if not task.done():
            task.cancel()
        await conn.stop()


# --- requests ---------------------------------------------------------------


async def test_out_of_order_responses_match_by_id(tmp_path):
    profile = fake_profile("--mode", "out-of-order", "--delay", "0.4")
    progress = ProgressBus()
    conn = LspConnection(profile, tmp_path)
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        slow = asyncio.create_task(conn.request("slow", {"n": 1}, timeout=5))
        await progress.wait("slow-received")
        ping = asyncio.create_task(conn.request("ping", {"n": 2}, timeout=5))
        done, _ = await asyncio.wait({slow, ping}, return_when=asyncio.FIRST_COMPLETED)
        assert done == {ping}
        assert await ping == {"method": "ping", "params": {"n": 2}}
        assert await slow == {"method": "slow", "params": {"n": 1}}
        assert conn.pending_count == 0
    finally:
        await conn.stop()


async def test_cancelled_request_late_response_is_dropped_without_leak(tmp_path):
    progress = ProgressBus()
    conn = LspConnection(fake_profile("--delay", "0.4"), tmp_path)
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        request_id = conn.allocate_request_id()
        task = asyncio.create_task(conn.request("slow", {"n": 9}, 5, request_id=request_id))
        await progress.wait("slow-received")
        assert conn.pending_count == 1
        assert conn.cancel(request_id) is True
        with pytest.raises(LspRequestError) as info:
            await task
        assert info.value.code == LSP_REQUEST_CANCELLED
        assert conn.pending_count == 0
        assert conn.cancel(request_id) is False  # the waiter is gone

        # The server still sends the late response; it must be dropped without
        # touching state or leaking a pending entry. The "slow-replied"
        # notification is written before the response on the same stream, and
        # the following ping response lands after it, so by the time ping
        # resolves the late frame has been consumed.
        await progress.wait("slow-replied")
        assert await conn.request("ping") == {"method": "ping", "params": None}
        assert conn.pending_count == 0
        assert conn.state == STATE_READY
        log = await conn.request("record-log")
        cancels = [r for r in log["records"] if r["method"] == "$/cancelRequest"]
        assert cancels and cancels[0]["params"] == {"id": request_id}
    finally:
        await conn.stop()


async def test_caller_cancellation_cleans_pending_and_sends_cancel(tmp_path):
    progress = ProgressBus()
    conn = LspConnection(fake_profile("--delay", "0.4"), tmp_path)
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        task = asyncio.create_task(conn.request("slow", {"n": 3}, timeout=30))
        await progress.wait("slow-received")
        assert conn.pending_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert conn.pending_count == 0
        log = await conn.request("record-log")
        assert any(r["method"] == "$/cancelRequest" for r in log["records"])
    finally:
        await conn.stop()


async def test_pending_request_limit_rejects_structurally(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path, pending_limit=2)
    await conn.start()
    try:
        tasks = [
            asyncio.create_task(conn.request("hang", {"n": index}, timeout=30))
            for index in range(2)
        ]
        await _until(lambda: conn.pending_count == 2)
        with pytest.raises(LspPendingLimitError) as info:
            await conn.request("ping")
        assert info.value.limit == 2
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert conn.pending_count == 0
    finally:
        await conn.stop()


async def test_requests_time_out_and_errors_preserve_codes(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path)
    await conn.start()
    try:
        with pytest.raises(LspTimeoutError) as info:
            await conn.request("hang", {}, timeout=0.25)
        assert info.value.code is None
        assert conn.pending_count == 0
        log = await conn.request("record-log")
        cancels = [r for r in log["records"] if r["method"] == "$/cancelRequest"]
        assert len(cancels) == 1

        with pytest.raises(LspRequestError) as info:
            await conn.request(
                "error", {"code": LSP_CONTENT_MODIFIED, "message": "ContentModified: stale"}
            )
        assert info.value.code == LSP_CONTENT_MODIFIED
        assert info.value.content_modified is True
        assert str(info.value) == "ContentModified: stale"
        assert info.value.data == {"injected": True}

        with pytest.raises(LspRequestError) as info:
            await conn.request(
                "error", {"code": LSP_REQUEST_FAILED, "message": "RequestFailed: broken"}
            )
        assert info.value.code == LSP_REQUEST_FAILED
        assert "RequestFailed" in str(info.value)
    finally:
        await conn.stop()


# --- ordered document sync --------------------------------------------------


async def test_write_order_didchange_cannot_be_overtaken_by_request(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path)
    await conn.start()
    try:
        uri = (tmp_path / "a.py").as_uri()
        conn.did_open(uri, "python", 1, "x = 1\n")
        change = {
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
            "text": "y",
        }
        for version in (2, 3):
            conn.did_change(uri, version, [change])
        conn.did_save(uri)
        log = await conn.request("record-log")
        methods = [entry["method"] for entry in log["records"]]
        assert methods == [
            "initialize",
            "initialized",
            "textDocument/didOpen",
            "textDocument/didChange",
            "textDocument/didChange",
            "textDocument/didSave",
            "record-log",
        ]
        versions = [
            entry["params"]["textDocument"]["version"]
            for entry in log["records"]
            if entry["method"] == "textDocument/didChange"
        ]
        assert versions == [2, 3]
    finally:
        await conn.stop()


async def test_did_save_include_text_honors_server_capabilities(tmp_path):
    uri = (tmp_path / "a.py").as_uri()
    for include, extra in ((True, []), (False, ["--no-save-include-text"])):
        conn = LspConnection(fake_profile(*extra), tmp_path)
        await conn.start()
        try:
            conn.did_save(uri, "print('x')\n")
            log = await conn.request("record-log")
            saves = [r for r in log["records"] if r["method"] == "textDocument/didSave"]
            assert len(saves) == 1
            params = saves[0]["params"]
            if include:
                assert params["text"] == "print('x')\n"
            else:
                assert "text" not in params
        finally:
            await conn.stop()


# --- environment and stderr ------------------------------------------------


async def test_child_environment_excludes_app_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MACHINA_GATEWAY_TOKEN", "super-secret")
    monkeypatch.setenv("MACHINA_OTHER_SECRET", "also-secret")
    profile = LspProfile(
        language_id="python",
        profile_id="fake",
        executable=sys.executable,
        args=[str(FAKE_SERVER), "--dump-env"],
        env_extra={"MACHINA_LSP_TEST_FLAG": "1"},
    )
    conn = LspConnection(profile, tmp_path)
    await conn.start()
    try:
        assert await conn.wait_for_stderr("ENVKEYS-DONE", 5.0)
        text = conn.stderr_text()
        env_keys = {
            line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("ENVKEY: ")
        }
        assert "MACHINA_GATEWAY_TOKEN" not in env_keys
        assert "MACHINA_OTHER_SECRET" not in env_keys
        assert "MACHINA_LSP_TEST_FLAG" in env_keys
        assert "PATH" in env_keys
        assert not any(key.startswith("MACHINA_") for key in env_keys - {"MACHINA_LSP_TEST_FLAG"})
    finally:
        await conn.stop()


async def test_stderr_flood_is_bounded_and_sanitized(tmp_path):
    progress = ProgressBus()
    conn = LspConnection(
        fake_profile("--mode", "stderr-flood", "--flood-bytes", "1500000"), tmp_path
    )
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        await progress.wait("flood-done", 20)
        assert await conn.wait_for_stderr("FLOOD-END", 20)
        text = conn.stderr_text()
        assert "FLOOD-START" not in text  # old bytes were evicted
        assert "FLOOD-END" in text
        assert "\x1b" not in text
        assert conn.stderr_size <= 256 * 1024
        assert await conn.request("ping") == {"method": "ping", "params": None}
    finally:
        await conn.stop()


# --- crash, restart, stop ---------------------------------------------------


async def test_crash_rejects_pending_and_marks_failed(tmp_path):
    condition = Condition()
    conn = LspConnection(fake_profile(), tmp_path)
    conn.on_state_change(lambda connection, state: condition.poke())
    await conn.start()
    try:
        task = asyncio.create_task(conn.request("boom", {"x": 1}, 10))
        with pytest.raises(LspUnavailableError):
            await asyncio.wait_for(task, 10)
        await condition.wait(lambda: conn.state == STATE_FAILED)
        assert conn.pending_count == 0
        assert conn.error.startswith("language server exited")
        assert isinstance(conn.last_failure, LspUnavailableError)
    finally:
        await conn.stop()


async def test_crash_after_initialize_is_observed_as_failed(tmp_path):
    condition = Condition()
    conn = LspConnection(fake_profile("--mode", "crash-after-initialize"), tmp_path)
    conn.on_state_change(lambda connection, state: condition.poke())
    with contextlib.suppress(LspUnavailableError):
        await conn.start()
    await condition.wait(lambda: conn.state == STATE_FAILED)
    assert "language server exited" in conn.error
    assert await conn.stop() == STOP_OUTCOME_STOPPED
    assert conn.state == STATE_STOPPED


async def test_restart_increments_generation_and_reinitializes(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path)
    await conn.start()
    try:
        assert conn.generation == 1
        first_pid = conn.pid
        assert await conn.stop() == STOP_OUTCOME_STOPPED
        assert conn.state == STATE_STOPPED
        assert conn.cleanup_pending is False

        await conn.start()
        assert conn.state == STATE_READY
        assert conn.generation == 2
        assert conn.pid != first_pid
        log = await conn.request("record-log")
        assert log["initializeCount"] == 1  # the new process initialized once
        methods = [entry["method"] for entry in log["records"]]
        assert methods[:2] == ["initialize", "initialized"]
    finally:
        await conn.stop()


async def test_manager_bounded_restart_and_manual_restart(tmp_path):
    manager = LspManager(
        tmp_path, profiles={"python": fake_profile()}, max_restarts=1, restart_delay=0.0
    )
    ready_generations: list[int] = []
    manager.on_ready(lambda connection: ready_generations.append(connection.generation))
    condition = Condition()
    conn = await manager.ensure_started("python")
    assert conn is await manager.ensure_started("python")  # never spawn per request
    assert conn.generation == 1 and conn.state == STATE_READY
    conn.on_state_change(lambda connection, state: condition.poke())

    with pytest.raises(LspUnavailableError):
        await conn.request("boom", {}, 10)
    await condition.wait(lambda: conn.state == STATE_READY and conn.generation == 2)
    assert ready_generations == [1, 2]

    with pytest.raises(LspUnavailableError):
        await conn.request("boom", {}, 10)
    await condition.wait(lambda: conn.state == STATE_FAILED and conn.generation == 2)
    with pytest.raises(LspUnavailableError):
        await manager.ensure_started("python")  # automatic budget exhausted

    conn = await manager.restart("python")
    assert conn.state == STATE_READY and conn.generation == 3
    assert ready_generations == [1, 2, 3]
    entry = manager.status()[0]
    assert entry["profile"] == "fake"
    assert entry["languageId"] == "python"
    assert entry["state"] == STATE_READY
    assert entry["generation"] == 3
    await manager.stop_all()
    assert manager.status()[0]["state"] == STATE_STOPPED


async def test_stop_observes_process_termination(tmp_path):
    conn = LspConnection(fake_profile(), tmp_path)
    await conn.start()
    pid = conn.pid
    assert pid is not None
    assert await conn.stop() == STOP_OUTCOME_STOPPED
    assert conn.state == STATE_STOPPED
    assert conn.last_returncode == 0
    assert conn.cleanup_pending is False
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert await conn.stop() == STOP_OUTCOME_STOPPED  # idempotent


async def test_stop_reports_cleanup_pending_when_server_refuses_exit(tmp_path):
    conn = LspConnection(
        fake_profile("--mode", "refuse-exit"),
        tmp_path,
        shutdown_timeout=0.2,
        terminate_timeout=0.2,
        term_timeout=0.2,
        kill_timeout=0.0,
    )
    await conn.start()
    try:
        outcome = await conn.stop()
        # SIGKILL was sent, but exit was not observed: report pending cleanup
        # instead of claiming "stopped" (never trust a signal).
        assert outcome == STOP_OUTCOME_CLEANUP_PENDING
        assert conn.state == STATE_STOPPING
        assert conn.cleanup_pending is True
        assert await conn.observe_termination(3.0) == STOP_OUTCOME_STOPPED
        assert conn.state == STATE_STOPPED
        assert conn.cleanup_pending is False
    finally:
        with contextlib.suppress(Exception):
            await conn.stop()


async def test_manager_status_reports_cleanup_pending_honestly(tmp_path):
    manager = LspManager(
        tmp_path,
        profiles={"python": fake_profile("--mode", "refuse-exit")},
        shutdown_timeout=0.2,
        terminate_timeout=0.2,
        term_timeout=0.2,
        kill_timeout=0.0,
    )
    conn = await manager.ensure_started("python")
    assert await conn.stop() == STOP_OUTCOME_CLEANUP_PENDING
    entry = manager.status()[0]
    assert entry["state"] == STATE_STOPPING
    assert "cleanup pending" in entry["detail"]
    assert await conn.observe_termination(3.0) == STOP_OUTCOME_STOPPED
    assert manager.status()[0]["state"] == STATE_STOPPED
    await manager.stop_all()


# --- server-initiated requests ---------------------------------------------


async def test_server_initiated_requests_answered_structurally(tmp_path):
    progress = ProgressBus()
    conn = LspConnection(fake_profile("--mode", "server-requests"), tmp_path)
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        await progress.wait("responses-collected", 10)
        responses = await conn.request("fetch-responses")
        assert responses["srv-1"]["result"] == {
            "applied": False,
            "failureReason": "workspace edits are not applied by this client",
        }
        assert responses["srv-2"]["result"] == []
        assert responses["srv-3"]["error"]["code"] == JSONRPC_METHOD_NOT_FOUND
        assert responses["srv-4"]["result"] is None
        assert responses["srv-5"]["error"]["code"] == JSONRPC_METHOD_NOT_FOUND

        assert await conn.wait_for_stderr("FAKE-LOG-RED", 5.0)
        assert "\x1b" not in conn.stderr_text()
        assert "\x07" not in conn.stderr_text()
    finally:
        await conn.stop()


async def test_apply_edit_refusal_survives_capability_withdrawal(tmp_path):
    """I-6: even a server that ignores ``applyEdit: false`` gets an honest refusal.

    The refusal branch is defense-in-depth: it must keep answering
    ``applied: false`` with a failureReason, and the advertised capability
    must match that behavior in the very same handshake.
    """
    progress = ProgressBus()
    conn = LspConnection(fake_profile("--mode", "server-requests"), tmp_path)
    conn.on_notification(progress.on_notification)
    await conn.start()
    try:
        log = await conn.request("record-log")
        workspace_capability = log["initializeParams"]["capabilities"]["workspace"]
        assert workspace_capability["applyEdit"] is False
        assert "workspaceEdit" not in workspace_capability

        await progress.wait("responses-collected", 10)
        responses = await conn.request("fetch-responses")
        refusal = responses["srv-1"]["result"]
        assert refusal["applied"] is False
        assert isinstance(refusal["failureReason"], str)
        assert refusal["failureReason"]
    finally:
        await conn.stop()


async def test_server_request_handler_overrides_defaults(tmp_path):
    progress = ProgressBus()
    conn = LspConnection(fake_profile("--mode", "server-requests"), tmp_path)
    conn.on_notification(progress.on_notification)
    seen: list[str] = []

    def handler(method: str, params: object):
        seen.append(method)
        if method == "workspace/configuration":
            return ([{"section": {"flag": True}}], None)
        if method == "machina/unknown":
            return ("custom", None)
        return None  # fall back to the structured defaults

    conn.on_server_request(handler)
    await conn.start()
    try:
        await progress.wait("responses-collected", 10)
        responses = await conn.request("fetch-responses")
        assert responses["srv-2"]["result"] == [{"section": {"flag": True}}]
        assert responses["srv-3"]["result"] == "custom"
        assert responses["srv-1"]["result"]["applied"] is False
        assert "workspace/configuration" in seen
    finally:
        await conn.stop()


# --- containment and honest missing ----------------------------------------


async def test_garbage_or_oversized_output_fails_only_that_connection(tmp_path):
    bad_watcher = Condition()
    bad = LspConnection(fake_profile("--mode", "oversized"), tmp_path)
    bad.on_state_change(lambda connection, state: bad_watcher.poke())
    garbage_watcher = Condition()
    garbage = LspConnection(fake_profile("--mode", "garbage"), tmp_path)
    garbage.on_state_change(lambda connection, state: garbage_watcher.poke())
    good = LspConnection(fake_profile(), tmp_path)
    await good.start()
    with contextlib.suppress(LspUnavailableError):
        await bad.start()
    with contextlib.suppress(LspUnavailableError):
        await garbage.start()
    try:
        await bad_watcher.wait(lambda: bad.state == STATE_FAILED)
        assert "Content-Length" in bad.error
        assert isinstance(bad.last_failure, LspProtocolError)
        assert bad.pending_count == 0

        await garbage_watcher.wait(lambda: garbage.state == STATE_FAILED)
        assert "garbage" in garbage.error
        assert isinstance(garbage.last_failure, LspProtocolError)

        assert good.state == STATE_READY
        assert await good.request("ping") == {"method": "ping", "params": None}
    finally:
        await bad.stop()
        await garbage.stop()
        await good.stop()


def test_resolve_profile_order_and_missing(tmp_path, monkeypatch):
    assert lsp_module.resolve_profile("cobol") is None

    explicit = tmp_path / "my-server"
    explicit.write_text("#!/bin/sh\n")
    explicit.chmod(0o755)
    monkeypatch.setattr(lsp_module.shutil, "which", lambda name: "/usr/bin/unused")
    profile = lsp_module.resolve_profile("python", explicit_path=str(explicit))
    assert profile is not None
    assert profile.executable == str(explicit.resolve())
    assert profile.profile_id == "pyright"

    profile = lsp_module.resolve_profile("python")
    assert profile is not None and profile.executable == "/usr/bin/unused"

    monkeypatch.setattr(lsp_module.shutil, "which", lambda name: None)
    profile = lsp_module.resolve_profile("typescript")
    assert profile is not None
    assert profile.executable == str(
        REPO_ROOT / "ui" / "node_modules" / ".bin" / "typescript-language-server"
    )
    assert profile.language_id == "typescript"
    assert list(profile.args) == ["--stdio"]


async def test_missing_and_disabled_states_are_honest(tmp_path):
    conn = LspConnection(None, tmp_path, language="cobol")
    with pytest.raises(LspUnavailableError):
        await conn.start()
    assert conn.state == STATE_MISSING
    assert "no language server" in conn.error

    disabled = LspConnection(None, tmp_path, language="cobol", enabled=False)
    assert disabled.state == STATE_DISABLED
    with pytest.raises(LspUnavailableError):
        await disabled.start()
    assert disabled.state == STATE_DISABLED


async def test_manager_missing_language_stays_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(lsp_module.shutil, "which", lambda name: None)
    manager = LspManager(tmp_path)
    conn = await manager.ensure_started("cobol")
    assert conn.state == STATE_MISSING
    entry = manager.status()[0]
    assert entry["profile"] == "missing-cobol"
    assert entry["state"] == STATE_MISSING
    await manager.stop_all()
    with pytest.raises(LspUnavailableError):
        await manager.ensure_started("cobol")


async def test_request_timeout_starts_before_the_send_phase(tmp_path, monkeypatch):
    """combined #2: a blocked stdin drain must not outlive the request timeout."""
    conn = LspConnection(fake_profile(), tmp_path)
    entered = asyncio.Event()

    async def blocked_send(_message):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(conn, "_transport_active", lambda: True)
    monkeypatch.setattr(conn, "_send", blocked_send)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LspTimeoutError):
        await asyncio.wait_for(conn._request_core("textDocument/hover", {}, 0.05), 1.0)
    elapsed = loop.time() - started
    assert entered.is_set()
    assert elapsed < 0.5, f"send phase ignored the 50ms timeout ({elapsed:.3f}s)"
    assert conn.pending_count == 0


async def test_cancelled_writer_settles_the_in_progress_frame(tmp_path, monkeypatch):
    """combined #3: a frame popped before a blocked drain must still settle."""
    conn = LspConnection(fake_profile(), tmp_path)
    gate = asyncio.Event()

    class BlockingStdin:
        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            await gate.wait()

    conn._stdin = BlockingStdin()
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    conn._outbox.append((b"frame", future))
    task = asyncio.create_task(conn._writer_loop())
    try:
        await _until(lambda: not conn._outbox)  # popped, now blocked in drain
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert future.done(), "cancelled writer abandoned the in-progress write future"
        assert isinstance(future.exception(), LspUnavailableError)
    finally:
        gate.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
