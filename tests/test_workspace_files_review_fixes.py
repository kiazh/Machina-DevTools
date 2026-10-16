"""Review fixes — machina/workspace/files.py + documents.py.

Covers three findings:

1. TOCTOU (HIGH): the earlier post-open realpath re-check was itself
   race-unsafe — ``os.path.realpath`` is non-strict and can return the
   unresolved path while its walk races an in-root symlink toggle, so
   ``resolve()`` and ``_verify_containment()`` both passed while ``open()``
   followed the swapped symlink. ``read_text``/``_disk_revision`` now open
   the root once and walk every component with ``dir_fd``-anchored
   ``O_NOFOLLOW|O_DIRECTORY`` opens; the two race tests below fire the swap
   at the component-open seam (no timing dependence, no dev/ino luck).
2. Pagination (MED): ``list_dir`` derived ``next_cursor`` from
   ``scan_truncated or offset+limit < len(entries)``, so a directory larger
   than ``MAX_LIST_SCAN`` eventually emitted a cursor the next call refused as
   out of range. The cursor now follows the page slice only; the scan cap
   remains disclosed through ``truncated``.
3. Encoding (LOW): ``DocumentMirrorStore.open_document``/``change_document``
   called ``text.encode("utf-8")`` unguarded, so lone-surrogate text escaped
   as an internal error; it is now the same structured ``unsupported-encoding``
   refusal ``WorkspaceFiles.save_text`` already produces.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import machina.workspace.files as fmod
from machina.workspace import WorkspaceError, WorkspaceFiles
from machina.workspace.documents import DocumentMirrorStore


# ─── helpers ────────────────────────────────────────────────────────────────
def _require_symlinks(tmp_path) -> None:
    probe = tmp_path / ".symlink_probe"
    target = tmp_path / ".symlink_target"
    target.mkdir(exist_ok=True)
    try:
        probe.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    finally:
        try:
            if os.path.islink(probe):
                os.unlink(probe)
        except OSError:
            pass


def _real(path: Path | str) -> Path:
    return Path(os.path.realpath(path))


def _swap_ancestor_hook(root: Path, ancestor: str, outside: Path):
    """Deterministic race hook: swap the *ancestor* directory for an outside
    symlink exactly when the descriptor walk opens that component (the
    guarded seam), before the real ``openat`` runs. The ``ELOOP``/``ENOTDIR``
    refusal is then forced — no dev/ino luck is involved."""
    real_open = fmod._os_open
    state = {"swapped": False}

    def hook(path, flags, *args, **kwargs):
        if not state["swapped"] and os.fspath(path) == ancestor and "dir_fd" in kwargs:
            os.rename(root / ancestor, root / f".{ancestor}.parked")
            os.symlink(outside, root / ancestor, target_is_directory=True)
            state["swapped"] = os.path.islink(root / ancestor)
        return real_open(path, flags, *args, **kwargs)

    return hook, state


# ─── fix 1: post-open containment re-check ──────────────────────────────────
def test_verify_containment_helper_inside_allowed_outside_refused(tmp_path):
    root = _real(tmp_path) / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "in.txt").write_text("in")
    outside = _real(tmp_path) / "outside"
    outside.mkdir()
    (outside / "out.txt").write_text("out")
    files = WorkspaceFiles(root)

    # Inside (and the root itself) passes; the outside path is a denial.
    files._verify_containment(root, display=".")
    files._verify_containment(_real(root / "sub" / "in.txt"), display="sub/in.txt")
    with pytest.raises(WorkspaceError) as error:
        files._verify_containment(_real(outside / "out.txt"), display="out.txt")
    assert error.value.reason == "denied"
    assert "TOCTOU" in error.value.message


def test_read_text_ancestor_swap_between_resolve_and_open_refused(tmp_path, monkeypatch):
    """The exact window: the ancestor becomes an outside symlink after
    resolve() but before the descriptor walk opens that component. The
    ``O_NOFOLLOW|O_DIRECTORY`` openat fails with ELOOP/ENOTDIR, so the read
    is refused before any bytes can flow (dev/ino cannot be consulted)."""
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "file.txt").write_text("INSIDE")
    outside = _real(tmp_path) / "outside"
    (outside / "b").mkdir(parents=True)
    (outside / "b" / "file.txt").write_text("SECRET")
    files = WorkspaceFiles(root)

    hook, state = _swap_ancestor_hook(root, "a", outside)
    monkeypatch.setattr(fmod, "_os_open", hook)

    with pytest.raises(WorkspaceError) as error:
        files.read_text("a/b/file.txt")
    assert state["swapped"], "swap seam (component openat) was never reached"
    assert error.value.reason == "denied"
    assert "symlink" in error.value.message


def test_save_text_ancestor_swap_during_disk_revision_refused(tmp_path, monkeypatch):
    """The same window on the write path's disk-revision probe must not
    auto-accept or conflict-report against the outside file."""
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "file.txt").write_text("INSIDE")
    outside = _real(tmp_path) / "outside"
    (outside / "b").mkdir(parents=True)
    outside_file = outside / "b" / "file.txt"
    outside_file.write_text("SECRET")
    files = WorkspaceFiles(root)

    hook, state = _swap_ancestor_hook(root, "a", outside)
    monkeypatch.setattr(fmod, "_os_open", hook)

    with pytest.raises(WorkspaceError) as error:
        asyncio.run(files.save_text("a/b/file.txt", "ATTACKER"))
    assert state["swapped"], "swap seam (component openat) was never reached"
    assert error.value.reason == "denied"
    assert "symlink" in error.value.message
    assert outside_file.read_text() == "SECRET"


# ─── fix 2: list pagination terminates at the scan cap ──────────────────────
def test_scan_cap_exhaustion_emits_no_refused_cursor(tmp_path, monkeypatch):
    """A directory larger than the scan cap must page to a terminal page with
    ``next_cursor=None`` (scan cap still disclosed via ``truncated``). The
    pre-fix code emitted a cursor at the cap that the next call refused as
    out of range, so this walk raised instead of terminating."""
    root = tmp_path / "ws"
    root.mkdir()
    for index in range(8):
        (root / f"f{index}.txt").write_text("x")
    files = WorkspaceFiles(root)
    monkeypatch.setattr(fmod, "MAX_LIST_SCAN", 5)

    cursor = None
    pages = 0
    scanned: list[str] = []
    while True:
        page = files.list_dir("", cursor=cursor, limit=2)
        pages += 1
        assert pages <= 10, "paging must terminate at the scan cap"
        scanned.extend(entry.name for entry in page.entries)
        if page.next_cursor is None:
            assert page.truncated is True  # scan cap fired, disclosed
            break
        assert page.entries, "a cursor must never lead to a non-terminal empty page"
        cursor = page.next_cursor

    assert pages == 3
    assert len(scanned) == 5  # first MAX_LIST_SCAN scanned entries only
    assert len(set(scanned)) == 5

    # The largest legal cursor (== the cap) answers an empty terminal page,
    # never a further cursor. Pre-fix this returned next_cursor="7" and the
    # following call was refused ("cursor is out of range").
    exhausted = files.list_dir("", cursor="5", limit=2)
    assert exhausted.entries == ()
    assert exhausted.next_cursor is None
    assert exhausted.truncated is True


def test_pagination_without_scan_truncation_unchanged(tmp_path):
    """Sanity: ordinary paging still reports truncated/cursor from the page."""
    root = tmp_path / "ws"
    root.mkdir()
    for index in range(5):
        (root / f"f{index}.txt").write_text("x")
    files = WorkspaceFiles(root)

    first = files.list_dir("", limit=2)
    assert first.next_cursor == "2"
    assert first.truncated is True
    second = files.list_dir("", cursor=first.next_cursor, limit=2)
    assert second.next_cursor == "4"
    third = files.list_dir("", cursor=second.next_cursor, limit=2)
    assert third.next_cursor is None
    assert third.truncated is False
    assert [entry.name for entry in third.entries] == ["f4.txt"]


# ─── fix 3: document mirrors refuse lone surrogates structurally ────────────
def test_open_document_lone_surrogate_refused_as_unsupported_encoding():
    store = DocumentMirrorStore(epoch=1)
    with pytest.raises(WorkspaceError) as error:
        store.open_document("file:///ws/a.txt", "plaintext", 1, "ok\ud800bad", epoch=1)
    assert error.value.reason == "unsupported-encoding"
    assert "UTF-8" in error.value.message
    assert store.snapshot("file:///ws/a.txt") is None  # nothing was mirrored


def test_change_document_lone_surrogate_refused_without_mutation():
    store = DocumentMirrorStore(epoch=1)
    store.open_document("file:///ws/a.txt", "plaintext", 1, "abc", epoch=1)
    with pytest.raises(WorkspaceError) as error:
        store.change_document(
            "file:///ws/a.txt",
            1,
            2,
            [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "text": "\ud800",
                }
            ],
            epoch=1,
        )
    assert error.value.reason == "unsupported-encoding"
    snapshot = store.snapshot("file:///ws/a.txt")
    assert snapshot["text"] == "abc"
    assert snapshot["version"] == 1
