"""T6b Quick Open tests (plan §12): bounded recursive index + ranked search.

Covers the bounded walk (caps, deadline, per-directory entry cap, symlink
safety, cycle guard, exclusions), ranking tiers and tie stability, the
single-flight/TTL/cancel indexer, the service partial/stale/epoch semantics,
the ``workspace.quickOpen`` handlers and the disclosed ``indexProgress``
event. Every bound asserted here is a disclosure, never a silent cut; every
stale answer is discarded, never mixed across epochs.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from machina.gateway import features
from machina.gateway.errors import INVALID_PARAMS, NOT_FOUND, RpcError
from machina.gateway.features import workspace as ws
from machina.workspace.models import (
    DEFAULT_EXCLUSIONS,
    DEFAULT_QUICK_OPEN_LIMIT,
    MAX_QUICK_OPEN_LIMIT,
    MAX_QUICK_OPEN_QUERY_CHARS,
    WorkspaceError,
)
from machina.workspace.quickopen import (
    REASON_DEADLINE,
    REASON_DIRECTORY_ENTRY_CAP,
    REASON_FILE_CAP,
    REASON_RESULT_LIMIT,
    REASON_UNREADABLE,
    QuickOpenIndex,
    QuickOpenIndexer,
    rank_entries,
)
from machina.workspace.service import WorkspaceService

QUICK_OPEN_METHODS = {"workspace.quickOpen", "workspace.quickOpenCancel"}


def _service(
    root: Path | str, trust: bool = True
) -> tuple[WorkspaceService, object, list]:
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    identity = service.open(str(root), trust)
    return service, identity, events


def _root(tmp_path: Path, name: str = "ws") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _write(root: Path, rel: str, text: str = "x") -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def _nested_tree(root: Path, dirs: str = "abcd", per_dir: int = 10) -> int:
    """A tree whose walk needs several slices: root file + one dir per letter."""
    total = 1
    _write(root, "root.txt")
    for letter in dirs:
        for index in range(per_dir):
            _write(root, f"{letter}/f{index:02d}.txt")
            total += 1
    return total


class FakeGateway:
    """Minimal gateway surface for handler tests (mirrors test_workspace_gateway)."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.methods: dict = {}
        self.workspace_service = None

    def require_db(self) -> None:
        return None

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)

    def add_method(self, name, handler) -> None:
        self.methods[name] = handler


async def _opened(tmp_path: Path) -> tuple[FakeGateway, dict, str]:
    gw = FakeGateway()
    opened = await ws.workspace_open(gw, {"root": str(tmp_path), "trust": True})
    return gw, opened, opened["workspaceId"]


# ─── bounded walk: paths, exclusions, caps, deadline ─────────────────────────
async def test_index_walks_nested_dirs_and_collects_root_relative_paths(tmp_path):
    root = _root(tmp_path)
    _write(root, "src/main.ts")
    _write(root, "src/deep/helper.ts")
    _write(root, "README.md")
    (root / "src" / "empty").mkdir()

    index = QuickOpenIndex(root)
    await index.scan()
    assert sorted(index.entries) == ["README.md", "src/deep/helper.ts", "src/main.ts"]
    assert index.done is True
    assert index.truncated is False
    assert index.reasons == []
    assert index.unreadable == 0
    assert index.scanned == 3


async def test_index_default_exclusions_are_not_traversed_or_listed(tmp_path):
    root = _root(tmp_path)
    _write(root, "node_modules/pkg/index.js")
    _write(root, ".git/config")
    _write(root, ".venv/lib/site.py")
    _write(root, "dist/bundle.js")
    _write(root, "src/ok.ts")

    index = QuickOpenIndex(root)
    await index.scan()
    assert index.entries == ["src/ok.ts"]
    assert index.exclusions == tuple(DEFAULT_EXCLUSIONS)
    assert all(
        part not in ("node_modules", ".git", ".venv", "dist") for part in index.entries
    )


async def test_index_symlink_loop_terminates_and_dedupes_targets(tmp_path):
    root = _root(tmp_path)
    _write(root, "a.txt")
    _write(root, "real/inside.txt")
    (root / "sub").mkdir()
    (root / "sub" / "loop").symlink_to(root)  # a cycle back to the root
    (root / "alias").symlink_to(root / "real")  # an inside-root directory link

    index = QuickOpenIndex(root)
    await asyncio.wait_for(index.scan(), timeout=5.0)
    assert index.done is True
    # The cycle back to the root is visited once; the aliased directory is not
    # walked twice under two names (scandir order decides which name wins).
    assert len(index.entries) == 2
    assert "a.txt" in index.entries
    assert len([path for path in index.entries if path.endswith("inside.txt")]) == 1
    assert not any(path.startswith("sub/loop") for path in index.entries)


async def test_index_follows_inside_root_symlinked_files_only(tmp_path):
    root = _root(tmp_path)
    _write(root, "data/target.txt")
    (root / "data" / "link.txt").symlink_to(root / "data" / "target.txt")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("secret")
    (root / "escape.py").symlink_to(outside / "secret.py")
    (root / "escapedir").symlink_to(outside)

    index = QuickOpenIndex(root)
    await index.scan()
    assert sorted(index.entries) == ["data/link.txt", "data/target.txt"]
    assert all("escape" not in path for path in index.entries)
    assert index.skipped_symlinks == 2


async def test_directory_entry_cap_is_disclosed_and_the_walk_continues(tmp_path):
    root = _root(tmp_path)
    for index in range(120):
        _write(root, f"many/f{index:03d}.txt")
    _write(root, "other/after.txt")

    index = QuickOpenIndex(root, dir_entry_cap=100)
    await index.scan()
    assert len(index.entries) == 101  # 100 from the capped dir + other/after.txt
    assert index.truncated is True
    assert REASON_DIRECTORY_ENTRY_CAP in index.reasons
    assert index.directory_caps == ["many"]
    assert "other/after.txt" in index.entries
    assert index.done is True


async def test_file_cap_stops_the_scan_and_marks_truncated(tmp_path):
    root = _root(tmp_path)
    for index in range(15):
        _write(root, f"a/f{index:02d}.txt")
    for index in range(15):
        _write(root, f"b/f{index:02d}.txt")

    index = QuickOpenIndex(root, max_files=10)
    await index.scan()
    assert len(index.entries) == 10
    assert index.truncated is True
    assert index.reasons == [REASON_FILE_CAP]
    assert index.done is True


async def test_scan_deadline_is_disclosed(tmp_path):
    root = _root(tmp_path)
    _write(root, "a.txt")

    index = QuickOpenIndex(root, deadline=0.0)
    await index.scan()
    assert index.entries == []
    assert index.truncated is True
    assert index.reasons == [REASON_DEADLINE]
    assert index.done is True


def _is_root_user() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


async def test_unreadable_directory_is_disclosed_not_fatal(tmp_path):
    if _is_root_user():
        pytest.skip("permission-denied semantics do not apply to root")
    root = _root(tmp_path)
    _write(root, "open/ok.txt")
    locked = root / "locked"
    locked.mkdir()
    (locked / "hidden.txt").write_text("x")
    locked.chmod(0)
    try:
        index = QuickOpenIndex(root)
        await index.scan()
        assert index.done is True
        assert index.entries == ["open/ok.txt"]
        assert index.unreadable == 1
        assert REASON_UNREADABLE in index.reasons
        assert index.truncated is True, (
            "an incomplete walk is never presented as complete"
        )
    finally:
        locked.chmod(0o700)


async def test_progress_callback_reports_done_once_and_survives_failures(tmp_path):
    root = _root(tmp_path)
    _write(root, "a.txt")
    _write(root, "b/c.txt")
    calls: list[tuple[int, bool]] = []

    def on_progress(scanned: int, done: bool) -> None:
        calls.append((scanned, done))
        raise RuntimeError("channel died")

    index = QuickOpenIndex(root, on_progress=on_progress)
    await index.scan()
    assert index.done is True, "a failing progress callback never fails the index"
    assert calls[-1] == (2, True)


async def test_indexer_reports_intermediate_progress_for_large_walks(tmp_path):
    root = _root(tmp_path, "big")
    for index in range(8):
        _write(root, f"d{index}/f.txt")
    calls: list[tuple[int, bool]] = []
    indexer = QuickOpenIndexer(
        root,
        key=("big", 1),
        yield_every=1,
        progress_every=1,
        on_progress=lambda s, d: calls.append((s, d)),
    )
    index, indexing = await indexer.ensure(wait=5.0)
    assert indexing is False
    assert index.done is True
    assert calls[-1] == (len(index.entries), True)
    assert any(not done for _scanned, done in calls), "progress is disclosed mid-walk"


# ─── ranking ─────────────────────────────────────────────────────────────────
def test_rank_tiers_exact_then_prefix_then_substring_then_subsequence():
    entries = (
        "src/main.ts",
        "src/mainUtils.ts",
        "src/someMain.ts",
        "src/m_a_i_n.ts",
        "src/other.ts",
    )
    matches, total = rank_entries(entries, "main", 10)
    assert [entry.path for entry in matches] == [
        "src/main.ts",
        "src/mainUtils.ts",
        "src/someMain.ts",
        "src/m_a_i_n.ts",
    ]
    assert total == 4
    assert matches[0].name == "main.ts"
    assert matches[0].directory == "src"
    assert matches[1].as_dict() == {
        "path": "src/mainUtils.ts",
        "name": "mainUtils.ts",
        "directory": "src",
    }

    exact, exact_total = rank_entries(("main", "main.txt"), "main", 10)
    assert [entry.path for entry in exact] == ["main", "main.txt"]
    assert exact_total == 2


def test_rank_is_case_insensitive_and_excludes_non_matches():
    entries = ("SRC/Main.TS", "src/other.ts")
    matches, total = rank_entries(entries, "MAIN", 10)
    assert [entry.path for entry in matches] == ["SRC/Main.TS"]
    assert total == 1
    assert rank_entries(entries, "zzz", 10) == ([], 0)


def test_rank_ties_are_stable_by_path_length_then_lexicographic():
    entries = ("b/xx.ts", "a/longer/xx.ts", "a/xx.ts")
    matches, total = rank_entries(entries, "xx", 10)
    assert [entry.path for entry in matches] == ["a/xx.ts", "b/xx.ts", "a/longer/xx.ts"]
    assert total == 3


def test_rank_empty_query_lists_all_and_limit_returns_total_before_cut():
    entries = tuple(f"src/file{index:03d}.ts" for index in range(30))
    matches, total = rank_entries(entries, "", 5)
    assert len(matches) == 5
    assert total == 30
    assert [entry.path for entry in matches] == list(entries[:5])


# ─── indexer: single-flight, TTL, explicit rebuild, cancel ───────────────────
async def test_concurrent_ensures_share_one_single_flight_scan(tmp_path, monkeypatch):
    root = _root(tmp_path)
    for index in range(6):
        _write(root, f"d{index}/f.txt")
    scans = 0
    original = QuickOpenIndex.scan

    async def counting_scan(self):  # type: ignore[no-untyped-def]
        nonlocal scans
        scans += 1
        await original(self)

    monkeypatch.setattr(QuickOpenIndex, "scan", counting_scan)
    indexer = QuickOpenIndexer(root, key=("shared", 1))
    results = await asyncio.gather(
        indexer.ensure(wait=5.0),
        indexer.ensure(wait=5.0),
        indexer.ensure(wait=5.0),
    )
    assert scans == 1, "concurrent requests never restart the walk"
    assert all(index is results[0][0] for index, _indexing in results)
    assert all(indexing is False for _index, indexing in results)
    assert len(results[0][0].entries) == 6


async def test_ttl_expiry_rebuilds_the_cached_index(tmp_path):
    root = _root(tmp_path)
    _write(root, "first.txt")
    indexer = QuickOpenIndexer(root, key=("ttl", 1), ttl=0.05)
    index, indexing = await indexer.ensure(wait=5.0)
    assert indexing is False
    assert index.entries == ["first.txt"]

    _write(root, "second.txt")
    cached, _ = await indexer.ensure(wait=0.0)
    assert cached.entries == ["first.txt"], "a fresh index is reused inside its TTL"

    await asyncio.sleep(0.06)
    rebuilt, indexing = await indexer.ensure(wait=5.0)
    assert indexing is False
    assert sorted(rebuilt.entries) == ["first.txt", "second.txt"]


async def test_explicit_rebuild_refreshes_without_waiting_for_ttl(tmp_path):
    root = _root(tmp_path)
    _write(root, "first.txt")
    indexer = QuickOpenIndexer(root, key=("rebuild", 1), ttl=600.0)
    index, _ = await indexer.ensure(wait=5.0)
    assert index.entries == ["first.txt"]

    _write(root, "second.txt")
    rebuilt, indexing = await indexer.ensure(wait=5.0, rebuild=True)
    assert indexing is False
    assert sorted(rebuilt.entries) == ["first.txt", "second.txt"]


async def test_cancel_retires_the_partial_index_and_returns_truthfully(tmp_path):
    root = _root(tmp_path)
    _nested_tree(root)
    indexer = QuickOpenIndexer(root, key=("cancel", 1))
    index, indexing = await indexer.ensure(wait=0.0)
    assert indexing is True
    assert index.done is False
    assert index.scanned < 41

    assert indexer.cancel() is True
    assert indexer.cancel() is False, "a settled scan is not cancelled twice"
    assert indexer.current() is None, "a cancelled index is never served as current"

    restarted, indexing = await indexer.ensure(wait=5.0)
    assert indexing is False
    assert len(restarted.entries) == 41, "the next request starts a fresh scan"


# ─── service: partial, cancel, stale epoch, disclosure ──────────────────────
async def test_service_quick_open_ranks_limits_and_discloses(tmp_path):
    root = _root(tmp_path)
    _write(root, "src/main.ts")
    _write(root, "src/mainUtils.ts")
    _write(root, "src/other.ts")
    service, identity, _ = _service(root)

    result = await service.quick_open("main", limit=1)
    assert result.indexing is False
    assert [entry.path for entry in result.entries] == ["src/main.ts"]
    assert result.total == 2
    assert result.truncated is True
    assert REASON_RESULT_LIMIT in result.reasons
    assert result.stale is False
    assert result.workspace_epoch == identity.epoch
    assert result.scanned == 3
    assert result.exclusions == tuple(DEFAULT_EXCLUSIONS)
    payload = result.as_dict()
    assert payload["entries"][0] == {
        "path": "src/main.ts",
        "name": "main.ts",
        "directory": "src",
    }
    assert payload["workspaceEpoch"] == identity.epoch
    assert payload["indexing"] is False


async def test_service_quick_open_default_limit_is_bounded(tmp_path):
    root = _root(tmp_path)
    for index in range(DEFAULT_QUICK_OPEN_LIMIT + 5):
        _write(root, f"f{index:03d}.txt")
    service, _, _ = _service(root)
    result = await service.quick_open("")
    assert len(result.entries) == DEFAULT_QUICK_OPEN_LIMIT
    assert result.total == DEFAULT_QUICK_OPEN_LIMIT + 5
    assert result.truncated is True


async def test_service_quick_open_returns_partial_with_indexing_flag(tmp_path):
    root = _root(tmp_path)
    total = _nested_tree(root)
    service, _, _ = _service(root)

    partial = await service.quick_open("", wait=0.0)
    assert partial.indexing is True
    assert 0 < partial.scanned < total, "the walk is disclosed as incomplete"
    assert len(partial.entries) == partial.scanned
    assert partial.total == partial.scanned, "the partial total is index-relative"
    assert partial.stale is False

    complete = await service.quick_open("", wait=5.0)
    assert complete.indexing is False
    assert complete.scanned == total
    assert complete.total == total
    assert len(complete.entries) == total


async def test_service_quick_open_cancel_mid_scan_and_noop_when_idle(tmp_path):
    root = _root(tmp_path)
    total = _nested_tree(root)
    service, _, _ = _service(root)

    assert service.quick_open_cancel() == {"cancelled": False}, (
        "no scan was started yet"
    )
    partial = await service.quick_open("", wait=0.0)
    assert partial.indexing is True

    assert service.quick_open_cancel() == {"cancelled": True}
    assert service.quick_open_cancel() == {"cancelled": False}

    after = await service.quick_open("", wait=5.0)
    assert after.indexing is False
    assert after.scanned == total, "a cancelled walk is restarted on the next query"


async def test_service_quick_open_discards_results_when_the_epoch_changes(tmp_path):
    root = _root(tmp_path)
    total = _nested_tree(root)
    service, identity, _ = _service(root)

    pending = asyncio.create_task(service.quick_open("", wait=5.0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # the scan is running and has not finished
    service.open(str(root), True)  # reopen mid-scan mints a new epoch
    result = await pending

    assert result.stale is True
    assert result.entries == ()
    assert result.total == 0
    assert result.indexing is False
    assert result.workspace_epoch == identity.epoch, "tagged with the queried epoch"
    assert result.reasons == ("stale-epoch",)
    assert service.quick_open_cancel() == {"cancelled": False}, (
        "the stale scan was retired"
    )

    fresh = await service.quick_open("", wait=5.0)
    assert fresh.stale is False
    assert fresh.scanned == total


async def test_service_quick_open_epoch_and_validation_refusals(tmp_path):
    root = _root(tmp_path)
    service, _, _ = _service(root)
    with pytest.raises(WorkspaceError) as stale_epoch:
        await service.quick_open("a", epoch=99)
    assert stale_epoch.value.reason == "denied"

    for query, limit in (
        ("x" * (MAX_QUICK_OPEN_QUERY_CHARS + 1), 10),
        ("ok", 0),
        ("ok", MAX_QUICK_OPEN_LIMIT + 1),
        ("ok", "5"),
    ):
        with pytest.raises(WorkspaceError) as error:
            await service.quick_open(query, limit)  # type: ignore[arg-type]
        assert error.value.reason == "denied"

    closed = WorkspaceService()
    with pytest.raises(WorkspaceError) as error:
        await closed.quick_open("")
    assert error.value.reason == "denied"
    assert closed.quick_open_cancel() == {"cancelled": False}


async def test_service_quick_open_emits_index_progress_disclosure(tmp_path):
    root = _root(tmp_path)
    _write(root, "a.txt")
    _write(root, "b/c.txt")
    service, identity, events = _service(root)
    result = await service.quick_open("", wait=5.0)
    assert result.indexing is False

    assert events, "the completed walk is disclosed"
    assert all(event["type"] == "workspace.indexProgress" for event in events)
    assert all(event["epoch"] == identity.epoch for event in events)
    assert all(event["done"] is False for event in events[:-1])
    final = events[-1]
    assert final == {
        "type": "workspace.indexProgress",
        "workspaceId": identity.workspace_id,
        "epoch": identity.epoch,
        "scanned": 2,
        "total": 2,
        "done": True,
    }
    assert "sessionId" not in final and "turnId" not in final


async def test_service_quick_open_close_then_denied_and_scan_retired(tmp_path):
    root = _root(tmp_path)
    _nested_tree(root)
    service, _, _ = _service(root)
    partial = await service.quick_open("", wait=0.0)
    assert partial.indexing is True
    service.close()
    with pytest.raises(WorkspaceError) as error:
        await service.quick_open("")
    assert error.value.reason == "denied"
    # The scan is not a service method once closed; cancellation stays
    # reachable for explicit cleanup and is honest about its outcome.
    assert service.quick_open_cancel() == {"cancelled": True}
    assert service.quick_open_cancel() == {"cancelled": False}


# ─── gateway handlers ────────────────────────────────────────────────────────
def test_quick_open_methods_registered():
    assert QUICK_OPEN_METHODS <= set(features.METHODS)
    gw = FakeGateway()
    features.register(gw)
    assert QUICK_OPEN_METHODS <= set(gw.methods)


async def test_quick_open_handler_ranks_and_returns_wire_shape(tmp_path):
    _write(tmp_path, "src/main.ts")
    _write(tmp_path, "src/other.ts")
    gw, opened, wid = await _opened(tmp_path)

    result = await ws.workspace_quick_open(
        gw, {"workspaceId": wid, "query": "main", "limit": 5}
    )
    assert result["entries"][0]["path"] == "src/main.ts"
    assert result["entries"][0]["name"] == "main.ts"
    assert result["entries"][0]["directory"] == "src"
    assert result["total"] == 1
    assert result["stale"] is False
    assert result["workspaceEpoch"] == opened["epoch"]
    assert result["indexing"] is False
    assert set(result["exclusions"]) == set(DEFAULT_EXCLUSIONS)


async def test_quick_open_handler_validation_and_workspace_identity(tmp_path):
    _write(tmp_path, "a.py")
    gw, _, wid = await _opened(tmp_path)

    for params in (
        {"workspaceId": wid, "query": "x" * (MAX_QUICK_OPEN_QUERY_CHARS + 1)},
        {"workspaceId": wid, "query": 7},
        {"workspaceId": wid, "limit": 0},
        {"workspaceId": wid, "limit": MAX_QUICK_OPEN_LIMIT + 1},
        {"workspaceId": wid, "limit": "5"},
        {"workspaceId": wid, "rebuild": "maybe"},
        {"workspaceId": wid, "epoch": -1},
    ):
        with pytest.raises(RpcError) as error:
            await ws.workspace_quick_open(gw, params)
        assert error.value.code == INVALID_PARAMS

    with pytest.raises(RpcError) as missing_id:
        await ws.workspace_quick_open(gw, {"query": "x"})
    assert missing_id.value.code == INVALID_PARAMS
    with pytest.raises(RpcError) as wrong:
        await ws.workspace_quick_open(gw, {"workspaceId": "ws_other", "query": "x"})
    assert wrong.value.code == NOT_FOUND

    stale = await ws.workspace_quick_open(
        gw, {"workspaceId": wid, "query": "x", "epoch": 12345}
    )
    assert stale["refused"]["reason"] == "denied"


async def test_quick_open_cancel_handler(tmp_path):
    _write(tmp_path, "a.py")
    gw, _, wid = await _opened(tmp_path)
    assert await ws.workspace_quick_open_cancel(gw, {"workspaceId": wid}) == {
        "cancelled": False
    }
    with pytest.raises(RpcError) as absent:
        await ws.workspace_quick_open_cancel(gw, {"workspaceId": "ws_absent"})
    assert absent.value.code == NOT_FOUND


async def test_quick_open_handlers_emit_index_progress_frames(tmp_path):
    _write(tmp_path, "a.py")
    gw, opened, wid = await _opened(tmp_path)
    frames_before = len(gw.frames)
    await ws.workspace_quick_open(gw, {"workspaceId": wid})
    frames = gw.frames[frames_before:]
    assert frames, "the walk discloses progress as a workspace event"
    frame = frames[-1]
    assert frame["jsonrpc"] == "2.0" and frame["method"] == "event"
    event = frame["params"]
    assert event["type"] == "workspace.indexProgress"
    assert event["workspaceId"] == wid
    assert event["epoch"] == opened["epoch"]
    assert event["done"] is True
    assert event["scanned"] == event["total"] == 1
    assert "sessionId" not in event and "turnId" not in event
