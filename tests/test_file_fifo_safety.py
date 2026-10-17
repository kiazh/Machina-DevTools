"""Combined #18: the write probe must not block on a FIFO.

``resolve_path`` rejects a FIFO it observes, but the write probe re-opens the
target later. A regular file swapped for a FIFO in that window blocks an
``O_WRONLY`` open before descriptor validation, and the joined worker cannot
be cancelled out of the syscall. Both the dir-fd-anchored probe and the
fallback probe must open nonblocking and verify ``S_ISREG`` on the descriptor.
"""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import machina


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(machina.__file__).resolve().parents[1])
    env["MACHINA_HOME"] = str(tmp_path / "home")
    return env


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs unavailable")
@pytest.mark.parametrize("probe", ["anchored", "fallback"])
def test_write_probe_rejects_fifo_without_blocking(tmp_path, probe):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    code = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path

        import machina.tools.file as file_tool

        fifo = Path(sys.argv[1])
        root = Path(sys.argv[2])
        file_tool._base_dir = lambda: root

        if sys.argv[3] == "anchored":
            try:
                file_tool._write_anchored_inside(fifo, "payload", "pipe")
                print("WROTE")
            except Exception as error:
                print(type(error).__name__, error)
        else:
            file_tool.resolve_path = lambda path, allow_outside=False: fifo
            file_tool._dir_fd_supported = lambda: False
            try:
                print(asyncio.run(file_tool.write_file(str(fifo), "payload")))
            except Exception as error:
                print(type(error).__name__, error)
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(fifo), str(tmp_path), probe],
            cwd=tmp_path,
            env=_env(tmp_path),
            capture_output=True,
            text=True,
            timeout=4,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{probe} write probe blocked opening a FIFO")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "WROTE" not in output
    assert "Successfully wrote" not in output
    assert "regular file" in output or "not accessible" in output
    assert stat.S_ISFIFO(fifo.stat().st_mode), "the FIFO target was replaced"


def test_write_file_still_writes_a_regular_file(tmp_path, monkeypatch):
    import machina.tools.file as file_tool

    monkeypatch.setattr(file_tool, "_base_dir", lambda: tmp_path)
    target = tmp_path / "note.txt"
    message = asyncio.run(file_tool.write_file(str(target), "hello"))
    assert "Successfully wrote" in message
    assert target.read_text() == "hello"
