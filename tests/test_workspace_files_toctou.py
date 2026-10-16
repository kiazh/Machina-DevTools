"""Descriptor-anchored TOCTOU tests for machina/workspace/files.py.

The post-open realpath re-check this fix replaces was proven race-unsafe: an
atomic symlink toggle raced against ``WorkspaceFiles.read_text`` leaked
outside-root bytes. Mechanism: ``os.path.realpath`` is non-strict and can
return the *unresolved* path while its own walk races the swap, and
``Path.is_relative_to`` is a pure string check — so both ``resolve()`` and
``_verify_containment()`` passed; meanwhile ``open()`` followed the swapped
symlink and fstat/lstat both saw the same outside inode, so ``_verify_fd``
passed too.

The read path now opens the workspace root once and walks every directory
component with ``openat``-equivalent ``os.open(..., dir_fd=...)`` +
``O_NOFOLLOW|O_DIRECTORY``, then opens the final component ``O_NOFOLLOW``
from its parent descriptor. These tests pin that control:

1. Deterministic refusal of a symlink directory component (pre-placed, and
   the strictly stronger inside-root case where the realpath pre-filter and
   the dev/ino comparison would both pass — proving the walker itself owns
   the refusal, not timing luck).
2. A bounded concurrent toggle stress: every successful read returned the
   in-root bytes and outside content never appeared.
3. Behavior that must not regress: inside-root *final-component* symlinks
   stay readable, links/loops/outside targets are refused, and refusal
   paths do not leak descriptors.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

import machina.workspace.files as fmod
from machina.workspace import WorkspaceError, WorkspaceFiles


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


# ─── 1a: the walker refuses a symlink component even when everything else ───
# ─── would pass (inside-root target: no realpath escape, dev/ino match) ─────
def test_walker_refuses_inside_root_symlink_directory_component(tmp_path):
    """Swap root/sub for a symlink that still points inside the root.

    ``resolve()`` passes (the target is inside), the opened fd would be a
    real in-root regular file and dev+ino would match its lstat — so only
    the descriptor walk's ``O_NOFOLLOW|O_DIRECTORY`` component check can
    refuse this read. No timing is involved: the swap is complete before the
    call, and the refusal proves the walker owns containment.
    """
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "real_sub").mkdir(parents=True)
    (root / "real_sub" / "f.txt").write_text("INSIDE")
    os.symlink(root / "real_sub", root / "sub", target_is_directory=True)
    files = WorkspaceFiles(root)

    with pytest.raises(WorkspaceError) as error:
        files.read_text("sub/f.txt")
    assert error.value.reason == "denied"
    assert "symlink" in error.value.message


# ─── 1b: the task's deterministic swap: component replaced by an outside ────
# ─── symlink before the call (no race, no dev/ino luck) ─────────────────────
def test_read_refuses_preplaced_outside_symlink_component(tmp_path):
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("INSIDE")
    outside = _real(tmp_path) / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("SECRET-OUTSIDE")

    # Atomic swap complete before the call; a plain pathname open follows it.
    os.rename(root / "sub", root / ".sub.parked")
    os.symlink(outside, root / "sub", target_is_directory=True)
    assert (root / "sub" / "f.txt").read_text() == "SECRET-OUTSIDE"

    files = WorkspaceFiles(root)
    with pytest.raises(WorkspaceError) as error:
        files.read_text("sub/f.txt")
    assert error.value.reason == "denied"


def test_disk_revision_refuses_component_swapped_mid_walk(tmp_path, monkeypatch):
    """The revision probe walks resolved components descriptor-anchored: a
    swap fired at the component-open seam (deterministic hook) is an ELOOP
    denial, never a hash of outside bytes."""
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("INSIDE")
    outside = _real(tmp_path) / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("SECRET-OUTSIDE")
    files = WorkspaceFiles(root)

    real_open = fmod._os_open
    state = {"swapped": False}

    def hook(path, flags, *args, **kwargs):
        if not state["swapped"] and os.fspath(path) == "sub" and "dir_fd" in kwargs:
            os.rename(root / "sub", root / ".sub.parked")
            os.symlink(outside, root / "sub", target_is_directory=True)
            state["swapped"] = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fmod, "_os_open", hook)
    with pytest.raises(WorkspaceError) as error:
        files._disk_revision(root / "sub" / "f.txt", display="sub/f.txt")
    assert state["swapped"], "swap seam (component openat) was never reached"
    assert error.value.reason == "denied"
    assert "symlink" in error.value.message


# ─── 2: bounded concurrent toggle stress (the original leak reproduction) ───
def test_concurrent_symlink_toggle_never_leaks_outside_content(tmp_path):
    """Toggle root/sub between a real dir and a symlink to an outside dir
    while reading ``sub/f.txt``. Every successful read must return the
    in-root bytes; outside bytes must never appear. Bounded (deadline based),
    no unfixed sleeps on the reader side."""
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("INSIDE")
    outside = _real(tmp_path) / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("SECRET-OUTSIDE")
    files = WorkspaceFiles(root)

    assert files.read_text("sub/f.txt").text == "INSIDE"  # warm-up mechanics

    parked = root / ".sub.parked"
    stop = threading.Event()
    toggles = [0]

    def toggler() -> None:
        linked = False
        while not stop.is_set():
            try:
                if linked:
                    if os.path.islink(root / "sub"):
                        os.unlink(root / "sub")
                    if os.path.isdir(parked) and not os.path.islink(parked):
                        os.rename(parked, root / "sub")
                    linked = False
                elif (
                    os.path.isdir(root / "sub")
                    and not os.path.islink(root / "sub")
                    and not parked.exists()
                ):
                    os.rename(root / "sub", parked)
                    os.symlink(outside, root / "sub", target_is_directory=True)
                    linked = True
                    toggles[0] += 1
            except OSError:
                pass
            time.sleep(0.0005)  # bound each state's hold; keeps reads live

    thread = threading.Thread(target=toggler, daemon=True)
    thread.start()
    attempts = 0
    reads_ok = 0
    refusals = 0
    deadline = time.monotonic() + 1.5
    try:
        while time.monotonic() < deadline:
            attempts += 1
            try:
                result = files.read_text("sub/f.txt")
            except WorkspaceError:
                refusals += 1  # fail-closed denials are acceptable
                continue
            reads_ok += 1
            # A successful read can only ever carry the in-root bytes; any
            # outside byte here is the leak this test exists to catch.
            assert result.text == "INSIDE", "outside content leaked"
    finally:
        stop.set()
        thread.join(timeout=10)
        # Restore the workspace layout before tmp_path teardown.
        if os.path.islink(root / "sub"):
            os.unlink(root / "sub")
        if (
            os.path.isdir(parked)
            and not os.path.islink(parked)
            and not (root / "sub").exists()
        ):
            os.rename(parked, root / "sub")

    assert toggles[0] > 0, "toggler never completed a swap cycle"
    assert attempts > 0
    assert reads_ok > 0, "stress never observed a successful read (vacuous?)"
    # Zero escapes: the outside file and its directory are untouched.
    assert (outside / "f.txt").read_text() == "SECRET-OUTSIDE"
    assert sorted(os.listdir(outside)) == ["f.txt"]
    assert (root / "sub" / "f.txt").read_text() == "INSIDE"
    print(
        f"toggle stress: cycles={toggles[0]} attempts={attempts} "
        f"reads_ok={reads_ok} refusals={refusals}"
    )


# ─── 3: no behavior regressions around the new walker ───────────────────────
def test_inside_root_final_symlink_still_readable(tmp_path):
    """Documented rule: final-component symlinks that stay inside the root
    are followed (absolute and relative targets)."""
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    root.mkdir()
    (root / "real.txt").write_text("hello")
    os.symlink(root / "real.txt", root / "abs.txt")
    os.symlink("real.txt", root / "rel.txt")
    files = WorkspaceFiles(root)

    assert files.read_text("abs.txt").text == "hello"
    assert files.read_text("rel.txt").text == "hello"
    assert files.read_text("rel.txt").revision == files.read_text("real.txt").revision


def test_leaf_symlink_loop_and_outside_targets_refused(tmp_path):
    _require_symlinks(tmp_path)
    root = _real(tmp_path) / "ws"
    root.mkdir()
    outside = _real(tmp_path) / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET-OUTSIDE")
    os.symlink(outside / "secret.txt", root / "evil.txt")
    os.symlink("loop_b", root / "loop_a")
    os.symlink("loop_a", root / "loop_b")
    # A relative target that climbs out of the root lexically.
    os.symlink("../outside/secret.txt", root / "climb.txt")
    files = WorkspaceFiles(root)

    for name in ("evil.txt", "loop_a", "climb.txt"):
        with pytest.raises(WorkspaceError) as error:
            files.read_text(name)
        assert error.value.reason == "denied", name


def test_walk_handles_nesting_root_level_final_and_missing(tmp_path):
    root = _real(tmp_path) / "ws"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "b" / "c" / "deep.txt").write_text("deep")
    (root / "top.txt").write_text("top")
    files = WorkspaceFiles(root)

    assert files.read_text("a/b/c/deep.txt").text == "deep"
    assert files.read_text("top.txt").text == "top"
    with pytest.raises(WorkspaceError) as missing:
        files.read_text("a/nope/c/deep.txt")
    assert missing.value.reason == "missing"
    with pytest.raises(WorkspaceError) as missing_leaf:
        files.read_text("a/b/c/missing.txt")
    assert missing_leaf.value.reason == "missing"
    with pytest.raises(WorkspaceError) as not_regular:
        files.read_text("a/b")
    assert not_regular.value.reason == "not-regular"


def test_walk_refusals_do_not_leak_descriptors(tmp_path):
    """Every walk path closes its descriptors (try/finally): 200 refusals
    must not grow the process fd table."""
    _require_symlinks(tmp_path)
    if not os.path.isdir("/dev/fd"):
        pytest.skip("/dev/fd unavailable for fd counting")
    root = _real(tmp_path) / "ws"
    (root / "real_sub").mkdir(parents=True)
    (root / "real_sub" / "f.txt").write_text("INSIDE")
    os.symlink(root / "real_sub", root / "sub", target_is_directory=True)
    os.symlink("loop_b", root / "loop_a")
    os.symlink("loop_a", root / "loop_b")
    files = WorkspaceFiles(root)

    def open_fd_count() -> int:
        return len(os.listdir("/dev/fd"))

    before = open_fd_count()
    for _ in range(200):
        with pytest.raises(WorkspaceError):
            files.read_text("sub/f.txt")  # component refusal path
        with pytest.raises(WorkspaceError):
            files.read_text("loop_a")  # bounded leaf-symlink hop path
        with pytest.raises(WorkspaceError):
            files.read_text("missing/f.txt")  # missing component path
    after = open_fd_count()
    assert after - before <= 3, f"descriptor leak: {before} -> {after}"
