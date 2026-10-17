"""B6 fix campaign: fork ancestry, fail-closed state, lifespan, LSP stop.

Each test pins one campaign defect:

1. ``SessionManager.fork`` records the source session as the fork's parent and
   in its ``ancestor_session_ids`` (child semantics identical to ``create``),
   so artifact ACLs let the fork read the source's retained artifacts while a
   sibling branch stays denied.
2. ``update_state_fields`` fails closed on a corrupt/undecodable or non-mapping
   ``state_msgpack`` blob: a typed error, no UPDATE, blob untouched.
3. The FastAPI lifespan runs every initialized cleanup stage when the body
   raises (normal exception and cancellation), without masking the primary
   exception; partial startup still shuts the runtime down.
4. A workspace stop that exceeds its bounded wait escalates to a best-effort
   force-kill instead of leaving language-server processes orphaned; a
   cancelled stop task escalates too and never raises from shutdown.
5. Workspace close -> reopen clears diagnostics (epoch/identity scoped), so a
   previous workspace's sets can never attach to the new one.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import FastAPI

from machina.api import app as app_mod
from machina.core.artifacts import artifact_store
from machina.core.session import CorruptSessionStateError, SessionManager
from machina.workspace import lsp as lsp_mod
from machina.workspace.lsp import STATE_READY, LspConnection
from machina.workspace.service import WorkspaceService, clear_active_workspace_service


def _diagnostic(message: str) -> dict:
    return {
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
        "severity": 1,
        "message": message,
    }


class _FakeConnection:
    """Duck-typed connection (no process) for service bookkeeping."""

    def __init__(self, *, profile_id: str = "pyright", generation: int = 1) -> None:
        self.profile_id = profile_id
        self.generation = generation
        self.state = STATE_READY
        self.error = ""
        self.cleanup_pending = False
        self.capabilities: dict = {}
        self._notification_callbacks: list = []

    def on_state_change(self, callback) -> None:
        return None

    def on_notification(self, callback) -> None:
        self._notification_callbacks.append(callback)

    def notify(self, method: str, params) -> None:
        for callback in list(self._notification_callbacks):
            callback(method, params)

    def did_open(self, uri: str, language_id: str, version: int, text: str) -> None:
        return None

    def did_change(self, uri: str, version: int, changes) -> None:
        return None

    def did_save(self, uri: str, text: str | None = None) -> None:
        return None

    def did_close(self, uri: str) -> None:
        return None


class _BareManager:
    """An attached manager without dispatch/start overrides."""


@pytest.fixture(autouse=True)
def _isolate_active_workspace():
    yield
    clear_active_workspace_service()


# ─── 1. fork ancestry (DB) ───────────────────────────────────────────────────


async def _mk_session(pool, *, parent: uuid.UUID | None = None) -> uuid.UUID:
    session_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO sessions (id, agent_id, parent_session_id, ancestor_session_ids, "
        "ancestor_agent_ids) "
        "SELECT $1, 'harness', $2, "
        "  COALESCE((SELECT ARRAY[id] || ancestor_session_ids FROM sessions WHERE id = $2), '{}'), "
        "  COALESCE((SELECT ARRAY[agent_id] || ancestor_agent_ids FROM sessions WHERE id = $2), '{}')",
        session_id,
        parent,
    )
    return session_id


async def test_fork_inherits_source_ancestry_for_artifact_access(db_pool, monkeypatch):
    monkeypatch.setattr("machina.core.session.db", db_pool)
    monkeypatch.setattr("machina.core.artifacts.db", db_pool)
    manager = SessionManager()
    parent = await _mk_session(db_pool)
    source = await _mk_session(db_pool, parent=parent)
    sibling = await _mk_session(db_pool, parent=parent)
    fork = None
    try:
        source_artifact = await artifact_store.store(
            "tool:terminal",
            "S" * 1200,
            session_id=source,
            agent_id="harness",
            complete=True,
            upstream_truncated=False,
        )
        sibling_artifact = await artifact_store.store(
            "tool:terminal",
            "T" * 1200,
            session_id=sibling,
            agent_id="harness",
            complete=True,
            upstream_truncated=False,
        )

        fork = await manager.fork(source)

        # Child semantics identical to create(): the source is the parent and
        # an ancestor of the fork.
        assert fork.parent_session_id == source
        ancestors = await db_pool.fetchval(
            "SELECT ancestor_session_ids FROM sessions WHERE id = $1", fork.id
        )
        assert list(ancestors) == [source, parent]

        # The fork resolves the artifacts of the session it was forked from.
        resolved = await artifact_store.lookup(
            source_artifact["artifact_id"], session_id=fork.id, agent_id="harness"
        )
        assert resolved is not None and resolved["complete"] is True

        # A sibling branch is not ancestry: its artifacts stay denied.
        assert (
            await artifact_store.lookup(
                sibling_artifact["artifact_id"], session_id=fork.id, agent_id="harness"
            )
            is None
        )
    finally:
        if fork is not None:
            await manager.delete(fork.id)
        await manager.delete(source)
        await db_pool.execute("DELETE FROM sessions WHERE id = ANY($1::uuid[])", [parent, sibling])


# ─── 2. fail-closed update_state_fields (mock DB) ───────────────────────────


def _mock_state_db(monkeypatch, raw: object):
    from unittest.mock import AsyncMock, MagicMock

    from machina.core import session as session_mod

    connection = MagicMock()
    connection.fetchrow = AsyncMock(return_value={"state_msgpack": raw})
    connection.execute = AsyncMock(return_value="UPDATE 1")
    connection.transaction = MagicMock()
    database = MagicMock()
    database.acquire.return_value.__aenter__.return_value = connection
    monkeypatch.setattr(session_mod, "db", database)
    return connection


async def _call_update(monkeypatch, raw: object) -> None:
    connection = _mock_state_db(monkeypatch, raw)
    with pytest.raises(CorruptSessionStateError) as excinfo:
        await SessionManager().update_state_fields(uuid.uuid4(), {"emotion": "trust"})
    assert "state" in str(excinfo.value).lower()
    connection.execute.assert_not_awaited()


async def test_update_state_fields_refuses_undecodable_blob(monkeypatch):
    # b"\xc1" is never valid msgpack.
    await _call_update(monkeypatch, b"\xc1")


async def test_update_state_fields_refuses_non_mapping_blob(monkeypatch):
    # A decodable list is still not session state: merging would erase it.
    await _call_update(monkeypatch, __import__("msgpack").packb([1, 2, 3], use_bin_type=True))


# ─── 3. lifespan cleanup on exceptional exit ─────────────────────────────────


class _FakeRuntime:
    def __init__(self, order: list[str], *, fail_startup: bool = False) -> None:
        self.order = order
        self.fail_startup = fail_startup

    async def startup(self) -> dict:
        self.order.append("runtime.startup")
        if self.fail_startup:
            raise RuntimeError("startup failed")
        return {"db": "ready"}

    async def shutdown(self) -> None:
        self.order.append("runtime.shutdown")


class _FakeRunner:
    def __init__(self, order: list[str], *, fail_stop: bool = False) -> None:
        self.order = order
        self.fail_stop = fail_stop

    def start(self) -> None:
        self.order.append("runner.start")

    async def stop(self) -> None:
        self.order.append("runner.stop")
        if self.fail_stop:
            raise RuntimeError("runner stop failed")


class _FakeGateway:
    def __init__(self, order: list[str], *, fail_close: bool = False) -> None:
        self.order = order
        self.fail_close = fail_close

    async def close(self) -> None:
        self.order.append("gateway.close")
        if self.fail_close:
            raise RuntimeError("gateway close failed")


def _install_lifespan_fakes(
    monkeypatch,
    order: list[str],
    *,
    runtime: _FakeRuntime | None = None,
    runner: _FakeRunner | None = None,
    gateway: _FakeGateway | None = None,
) -> None:
    import machina.core.runtime as runtime_mod
    import machina.core.scheduler as scheduler_mod
    import machina.plugins.loader as loader_mod

    runtime = runtime or _FakeRuntime(order)
    runner = runner or _FakeRunner(order)
    gateway = gateway or _FakeGateway(order)
    monkeypatch.setattr(runtime_mod, "runtime_services", runtime)
    monkeypatch.setattr(scheduler_mod, "JobRunner", lambda: runner)
    monkeypatch.setattr(app_mod, "_RpcGateway", lambda: gateway)
    monkeypatch.setattr(loader_mod, "load_plugins", lambda: None)


async def test_lifespan_cleans_up_when_the_body_raises(monkeypatch):
    order: list[str] = []
    _install_lifespan_fakes(monkeypatch, order)

    with pytest.raises(ValueError, match="body boom"):
        async with app_mod.lifespan(FastAPI()):
            raise ValueError("body boom")

    assert order == [
        "runtime.startup",
        "runner.start",
        "runner.stop",
        "gateway.close",
        "runtime.shutdown",
    ]


async def test_lifespan_cleans_up_on_cancellation(monkeypatch):
    order: list[str] = []
    _install_lifespan_fakes(monkeypatch, order)

    with pytest.raises(asyncio.CancelledError):
        async with app_mod.lifespan(FastAPI()):
            raise asyncio.CancelledError

    assert "runner.stop" in order and "gateway.close" in order and "runtime.shutdown" in order


async def test_lifespan_cleanup_failure_does_not_mask_the_primary_error(monkeypatch):
    order: list[str] = []
    _install_lifespan_fakes(
        monkeypatch,
        order,
        runner=_FakeRunner(order, fail_stop=True),
        gateway=_FakeGateway(order, fail_close=True),
    )

    with pytest.raises(ValueError, match="body boom"):
        async with app_mod.lifespan(FastAPI()):
            raise ValueError("body boom")

    # Every stage still ran, in order, despite two cleanup failures.
    assert order == [
        "runtime.startup",
        "runner.start",
        "runner.stop",
        "gateway.close",
        "runtime.shutdown",
    ]


async def test_lifespan_partial_startup_still_shuts_down(monkeypatch):
    order: list[str] = []
    _install_lifespan_fakes(
        monkeypatch, order, runtime=_FakeRuntime(order, fail_startup=True)
    )

    with pytest.raises(ValueError, match="body boom"):
        async with app_mod.lifespan(FastAPI()):
            raise ValueError("body boom")

    # No runner/gateway was initialized; the runtime still gets its shutdown.
    assert order == ["runtime.startup", "runtime.shutdown"]


# ─── 4. LSP stop timeout escalation ─────────────────────────────────────────


class _HangingManager:
    def __init__(self, *, fail_force_kill: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False
        self.force_killed = False
        self.fail_force_kill = fail_force_kill

    async def stop_all(self) -> None:
        self.started.set()
        await self.release.wait()
        self.finished = True

    def force_kill_all(self) -> None:
        self.force_killed = True
        if self.fail_force_kill:
            raise RuntimeError("force kill failed")


async def test_wait_stopped_escalates_a_timed_out_stop(tmp_path):
    manager = _HangingManager()
    service = WorkspaceService(send_event=lambda _event: None, manager_stop_timeout=0.05)
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(manager)

    service.close()
    await service.wait_stopped()

    assert manager.started.is_set()
    assert manager.force_killed is True  # escalation beats leaving an orphan
    assert manager.finished is False  # the graceful stop itself was not faked

    manager.release.set()
    await asyncio.wait_for(service._stop_task, timeout=2.0)


async def test_wait_stopped_never_raises_when_escalation_fails(tmp_path):
    manager = _HangingManager(fail_force_kill=True)
    service = WorkspaceService(send_event=lambda _event: None, manager_stop_timeout=0.05)
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(manager)

    service.close()
    await service.wait_stopped()  # shutdown path must never raise

    assert manager.force_killed is True
    manager.release.set()
    await asyncio.wait_for(service._stop_task, timeout=2.0)


async def test_cancelled_stop_task_still_escalates(tmp_path):
    manager = _HangingManager()
    service = WorkspaceService(send_event=lambda _event: None, manager_stop_timeout=5.0)
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(manager)

    service.close()
    await manager.started.wait()
    service._stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await service._stop_task

    assert manager.force_killed is True


def test_connection_force_kill_is_best_effort_and_never_raises(tmp_path, monkeypatch):
    calls: list[object] = []

    class _Process:
        returncode = None

        def kill(self) -> None:
            calls.append("process.kill")

    connection = LspConnection(profile=None, root=tmp_path, enabled=True)
    connection._process = _Process()
    connection._pgid = 4242

    def _kill_tree(handle, pgid=None) -> None:
        calls.append(("kill_tree", pgid))
        raise OSError("signal failed")

    monkeypatch.setattr(lsp_mod, "kill_tree", _kill_tree)

    connection.force_kill()  # must not raise

    assert ("kill_tree", 4242) in calls
    assert connection.cleanup_pending is True


async def test_cancelled_connection_stop_force_kills_before_unwinding(tmp_path, monkeypatch):
    calls: list[object] = []

    class _Process:
        returncode = None

        def kill(self) -> None:
            calls.append("process.kill")

    async def _parked_shutdown() -> None:
        await asyncio.Event().wait()

    connection = LspConnection(profile=None, root=tmp_path, enabled=True)
    connection.state = lsp_mod.STATE_READY
    connection._process = _Process()
    connection._pgid = 4242
    connection._graceful_shutdown = _parked_shutdown
    monkeypatch.setattr(
        lsp_mod, "kill_tree", lambda handle, pgid=None: calls.append(("kill_tree", pgid))
    )

    task = asyncio.get_running_loop().create_task(connection.stop())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ("kill_tree", 4242) in calls  # the process group got a SIGKILL sweep


# ─── 5. workspace close -> reopen cannot serve old diagnostics ───────────────


async def test_close_reopen_clears_diagnostics_across_workspace_epochs(tmp_path):
    service = WorkspaceService(send_event=lambda _event: None)
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(_BareManager())
    uri = (tmp_path / "a.py").as_uri()
    service.document_open(uri, "python", 0, "x = 1\n")
    old = _FakeConnection(profile_id="old")
    service._track_connection("python", old)
    service._handle_connection_notification(
        old,
        "textDocument/publishDiagnostics",
        {"uri": uri, "version": 1, "diagnostics": [_diagnostic("old")]},
    )
    canonical = service._rel_to_uri(service._uri_to_rel(uri))
    assert service._diagnostics.entries(
        canonical, epoch=service.epoch, generation_for=lambda _p: 1
    )

    service.close()
    service.open(str(tmp_path), trust=True)
    service.set_lsp_manager(_BareManager())
    service.document_open(uri, "python", 0, "x = 1\n")
    new = _FakeConnection(profile_id="new")
    service._track_connection("python", new)

    # A publish delivered by the previous epoch's connection is not live.
    old.notify(
        "textDocument/publishDiagnostics",
        {"uri": uri, "version": 9, "diagnostics": [_diagnostic("stale")]},
    )
    assert (
        service._diagnostics.entries(
            canonical, epoch=service.epoch, generation_for=lambda _p: 1
        )
        == []
    )

    read = await service.intelligence("diagnostics", uri=uri)
    # An empty store is reported honestly (unversioned / no result), never as
    # the previous workspace's "clean" diagnostics.
    assert read.result is None
    assert read.freshness == "unversioned"
