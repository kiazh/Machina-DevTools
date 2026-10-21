"""Descriptor-anchored read containment for machina/tools/file.py (read port).

The read path resolved inside-base paths with ``realpath`` and then opened
the resolved candidate by pathname, verifying the descriptor with a
post-open dev/ino + realpath re-check (``_open_secure_read`` +
``_verify_fd_matches_path``). That is the same race-unsafe pattern proven to
leak outside-root bytes in ``machina/workspace/files.py``: ``os.path.realpath``
is non-strict and can return the unresolved path while its walk races an
ancestor symlink toggle, and ``is_relative_to`` is a pure string check.

``_open_secure_read`` now opens inside-base paths by walking from the base
root with ``openat`` + ``O_NOFOLLOW|O_DIRECTORY`` and opening the leaf
``O_NOFOLLOW`` (the read-path counterpart of the F01 write walk); the same
walk is used for the descriptor actually read, so an ancestor swapped for a
symlink at any point fails closed. Inside-base symlinks are still followed
because ``resolve_path`` canonicalizes them before the walk.

These tests pin the refusal at the component-open seam (deterministic, no
timing) and the no-escape property under a bounded concurrent toggle, going
through ``read_file`` itself.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

import machina.tools.file as fmod
from machina.core.exceptions import ToolError


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


def test_read_ancestor_swap_at_component_open_fails_closed(tmp_path, monkeypatch):
    """The exact window: an ancestor becomes an outside symlink right before
    the read walk opens that component. The ``O_NOFOLLOW|O_DIRECTORY`` openat
    fails closed and the outside file stays untouched."""
    _require_symlinks(tmp_path)
    base = tmp_path / "ws"
    (base / "a" / "b").mkdir(parents=True)
    (base / "a" / "b" / "file.txt").write_text("INSIDE")
    outside = tmp_path / "outside"
    (outside / "b").mkdir(parents=True)
    (outside / "b" / "file.txt").write_text("SECRET")
    monkeypatch.setattr(fmod, "_BASE_DIR", base.resolve())

    real_open = fmod._os_open
    state = {"fired": False}

    def hook(path, flags, *args, **kwargs):
        name = path if isinstance(path, str) else getattr(path, "name", "")
        if not state["fired"] and kwargs.get("dir_fd") is not None and name == "a":
            state["fired"] = True
            os.rename(base / "a", base / ".parking")
            os.symlink(outside, base / "a", target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fmod, "_os_open", hook)
    with pytest.raises(ToolError) as error:
        asyncio.run(fmod.read_file("a/b/file.txt"))

    assert state["fired"], "swap seam (component open) was never reached"
    assert "symlink" in str(error.value)
    assert (outside / "b" / "file.txt").read_text() == "SECRET"
    assert sorted(os.listdir(outside)) == ["b"]


def test_read_still_follows_inside_base_links(tmp_path, monkeypatch):
    """Ordinary inside-base symlinks stay readable: resolve_path canonicalizes
    them, and the walk then opens the canonical target (no false negatives)."""
    _require_symlinks(tmp_path)
    base = tmp_path / "ws"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "real.txt").write_text("hello\n")
    os.symlink(base / "sub" / "real.txt", base / "alias.txt")
    os.symlink("sub", base / "dirlink", target_is_directory=True)
    monkeypatch.setattr(fmod, "_BASE_DIR", base.resolve())

    assert asyncio.run(fmod.read_file("sub/real.txt")) == "hello\n"
    assert asyncio.run(fmod.read_file("alias.txt")) == "hello\n"
    assert asyncio.run(fmod.read_file("dirlink/real.txt")) == "hello\n"


def test_concurrent_toggle_read_file_never_leaks_outside_content(tmp_path, monkeypatch):
    """Toggle base/sub between a real dir and a symlink to an outside dir
    while ``read_file("sub/f.txt")`` runs. Every successful read must be the
    in-root bytes; outside bytes must never appear. Bounded (≤2s)."""
    _require_symlinks(tmp_path)
    base = tmp_path / "ws"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "f.txt").write_text("INSIDE")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("SECRET-OUTSIDE")
    monkeypatch.setattr(fmod, "_BASE_DIR", base.resolve())

    assert asyncio.run(fmod.read_file("sub/f.txt")) == "INSIDE"  # warm-up

    parked = base / ".sub.parked"
    stop = threading.Event()
    cycles = [0]

    def toggler() -> None:
        linked = False
        while not stop.is_set():
            try:
                if linked:
                    if os.path.islink(base / "sub"):
                        os.unlink(base / "sub")
                    if os.path.isdir(parked) and not os.path.islink(parked):
                        os.rename(parked, base / "sub")
                    linked = False
                elif (
                    os.path.isdir(base / "sub")
                    and not os.path.islink(base / "sub")
                    and not parked.exists()
                ):
                    os.rename(base / "sub", parked)
                    os.symlink(outside, base / "sub", target_is_directory=True)
                    linked = True
                    cycles[0] += 1
            except OSError:
                pass
            time.sleep(0.0005)

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
                text = asyncio.run(fmod.read_file("sub/f.txt"))
            except (ToolError, FileNotFoundError):
                refusals += 1  # fail-closed refusals are the expected outcome
                continue
            reads_ok += 1
            assert text == "INSIDE", "outside content leaked"
    finally:
        stop.set()
        thread.join(timeout=10)
        if os.path.islink(base / "sub"):
            os.unlink(base / "sub")
        if (
            os.path.isdir(parked)
            and not os.path.islink(parked)
            and not (base / "sub").exists()
        ):
            os.rename(parked, base / "sub")
        # The toggler can stop between its two rename steps. Restore the
        # in-root fixture deterministically before the final post-condition.
        if not (base / "sub").is_dir():
            (base / "sub").mkdir(parents=True, exist_ok=True)
            (base / "sub" / "f.txt").write_text("INSIDE")

    assert cycles[0] > 0, "toggler never completed a swap cycle"
    assert attempts > 0
    assert reads_ok > 0, "stress never observed a successful read (vacuous?)"
    assert (outside / "f.txt").read_text() == "SECRET-OUTSIDE"
    assert sorted(os.listdir(outside)) == ["f.txt"]
    assert asyncio.run(fmod.read_file("sub/f.txt")) == "INSIDE"
    print(
        f"read_file toggle stress: cycles={cycles[0]} attempts={attempts} "
        f"reads_ok={reads_ok} refusals={refusals}"
    )
