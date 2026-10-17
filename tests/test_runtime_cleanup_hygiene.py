"""Evidence tests for runtime.py hygiene (core-loop P1-3 / P1-4, assigned as
item 6 of this campaign).

(a) ``_bounded_cleanup`` was a dead pass-through wrapper around
    ``machina.core.cleanup.bounded_cleanup`` adding only indirection. The fix
    inlines the shared helper at the call sites.

(b) ``RuntimeServices.track()`` cancelled every task registered after shutdown
    began. A deferred close (provider close + claim release) registered by
    ``TurnEngine._defer_close`` during shutdown was therefore killed before its
    first step: the provider leaked and the claim fell back to TTL expiry.
    Essential cleanup registered via ``track(..., essential=True)`` must not be
    cancelled; shutdown waits for it bounded before closing services.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from machina.core.runtime import RuntimeServices

# ─── (a) bounded-cleanup indirection ────────────────────────────────────────


def test_runtime_has_no_dead_bounded_cleanup_wrapper():
    import machina.core.runtime as runtime_module

    assert not hasattr(
        runtime_module, "_bounded_cleanup"
    ), "call sites must use machina.core.cleanup.bounded_cleanup directly"


async def test_shutdown_connectors_uses_module_bounded_cleanup(monkeypatch):
    """The wrapper's deferred import is gone: the module-level helper is used."""
    import machina.core.runtime as runtime_module

    calls: list[str] = []

    async def fake_bounded_cleanup(cleanup, timeout, label):
        cleanup.close()
        calls.append(label)
        return True

    monkeypatch.setattr(
        runtime_module, "bounded_cleanup", fake_bounded_cleanup, raising=False
    )

    async def noop():
        return None

    service = runtime_module.RuntimeServices()
    service._connector_service = SimpleNamespace(
        oauth=SimpleNamespace(shutdown=lambda deadline=None: noop()),
        manager=SimpleNamespace(shutdown_all=lambda: noop()),
    )

    await service.shutdown_connectors(timeout=2.0)

    assert calls == ["oauth shutdown", "connector shutdown"]


# ─── (b) tracked deferred close survives shutdown ───────────────────────────


async def test_track_during_shutdown_does_not_cancel_essential_cleanup():
    runtime = RuntimeServices(_shutting_down=True)
    ran = asyncio.Event()

    async def deferred_close():
        ran.set()

    task = asyncio.create_task(deferred_close())
    runtime.track(task, essential=True)
    await asyncio.wait_for(task, timeout=1)

    assert ran.is_set(), "essential cleanup must get to run, not be cancelled at registration"
    assert not task.cancelled()


async def test_nonessential_track_during_shutdown_keeps_the_reject_policy():
    runtime = RuntimeServices(_shutting_down=True)
    task = asyncio.create_task(asyncio.sleep(60))
    runtime.track(task)
    await asyncio.sleep(0)
    assert task.cancelled()


async def test_shutdown_runs_essential_close_registered_during_background_join(monkeypatch):
    """A deferred close registered mid-shutdown executes bounded before closure."""
    import machina.core.runtime as runtime_module

    order: list[str] = []
    runtime = runtime_module.RuntimeServices(_started=True)

    async def background():
        await asyncio.sleep(60)

    runtime.track(asyncio.create_task(background()))

    async def fake_join(tasks, timeout, label):
        # First shutdown pass: cancel/join background work, and emulate a turn
        # teardown deferring its owner cleanup while ``_shutting_down`` is set.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        async def deferred_close():
            await asyncio.sleep(0)
            order.append("deferred-close")

        runtime.track(asyncio.create_task(deferred_close()), essential=True)
        return []

    monkeypatch.setattr(runtime_module, "_bounded_join", fake_join)
    monkeypatch.setattr(runtime_module, "drain_background_tasks", AsyncMock())

    async def audit_stop():
        order.append("audit-stop")

    async def db_close():
        order.append("db-close")

    monkeypatch.setattr("machina.observability.audit.audit_persistence.stop", audit_stop)
    monkeypatch.setattr("machina.rag.embedder.close_shared_embedder", AsyncMock())
    monkeypatch.setattr("machina.tools.rag.close_active_rag_pipeline", AsyncMock())
    monkeypatch.setattr("machina.db.connection.db.close", db_close)

    await runtime.shutdown(timeout=2.0)

    assert "deferred-close" in order, "shutdown cancelled the deferred owner cleanup"
    assert order.index("deferred-close") < order.index("audit-stop")
    assert order.index("deferred-close") < order.index("db-close")
