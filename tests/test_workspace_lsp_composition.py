"""Production composition seam: workspace.open attaches a real LspManager.

Fallen-between-tasks gap found by the real-PTY completion scenario: the T8
service wiring existed but nothing in production ever constructed a manager,
so `/lsp status` reported "no language servers reported" and completion was
unavailable forever. These tests pin the composition:

- trusted open attaches a manager bound to the resolved root;
- untrusted open detaches (trust governs language servers, plan §14);
- reopening a different root replaces the manager;
- attaching does not start processes (lazy start on first documentOpen).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from machina.gateway.features import workspace as ws
from machina.workspace.lsp import LspManager
from machina.workspace.models import WorkspaceError


class FakeGateway:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.workspace_service = None

    def require_db(self) -> None:
        return None

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)


async def _open(tmp_path: Path, root: Path | None = None, *, trust: bool = True):
    gw = FakeGateway()
    opened = await ws.workspace_open(gw, {"root": str(root or tmp_path), "trust": trust})
    return gw, opened


async def test_trusted_open_attaches_a_manager_bound_to_the_root(tmp_path):
    gw, opened = await _open(tmp_path)
    manager = getattr(gw, "_workspace_lsp_manager", None)
    assert isinstance(manager, LspManager)
    assert Path(manager.root) == Path(opened["resolvedRoot"]).resolve()
    service = gw.workspace_service
    assert service._lsp_manager is manager


async def test_attaching_does_not_start_processes(tmp_path):
    gw, _ = await _open(tmp_path)
    manager: LspManager = gw._workspace_lsp_manager
    # `status()` is a list of *started* connections; attaching must not spawn.
    assert manager.status() == []


async def test_lsp_status_is_empty_until_a_server_starts(tmp_path):
    gw, _ = await _open(tmp_path)
    service = gw.workspace_service
    # Lazy start: no document opened yet, so no connection rows exist even
    # though the manager is attached (the distinguishing assertion is the
    # manager identity, covered in the first test).
    status = await service.lsp_status()
    assert status.get("servers") == []
    assert service._lsp_manager is not None


async def test_untrusted_open_leaves_the_service_managerless(tmp_path):
    gw, opened = await _open(tmp_path, trust=False)
    assert opened["trusted"] is False
    assert getattr(gw, "_workspace_lsp_manager", None) is None
    service = gw.workspace_service
    assert service._lsp_manager is None
    with pytest.raises(WorkspaceError, match="not trusted"):
        await service.lsp_status()


async def test_reopen_with_a_different_root_replaces_the_manager(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    gw, first = await _open(tmp_path)
    first_manager = gw._workspace_lsp_manager
    second = await ws.workspace_open(gw, {"root": str(other), "trust": True})
    second_manager = gw._workspace_lsp_manager
    assert second_manager is not first_manager
    assert Path(second_manager.root) == Path(second["resolvedRoot"]).resolve()
    assert gw.workspace_service._lsp_manager is second_manager
