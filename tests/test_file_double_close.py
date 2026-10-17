"""Combined #20: read-error cleanup must not close an fd twice.

``with os.fdopen(fd)`` owns and closes the descriptor. The exception handler
also called ``os.close(fd)``; if the number was reused between the two closes,
it could close an unrelated descriptor.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import machina.tools.file as file_tool
from machina.core.exceptions import ToolError


class _FakeHandle:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_read_error_does_not_double_close_the_descriptor(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("content\n")
    monkeypatch.setattr(file_tool, "_base_dir", lambda: tmp_path)
    closed: list[int] = []
    real_close = os.close

    def tracking_close(fd: int) -> None:
        if fd == 123:
            closed.append(fd)
            return
        real_close(fd)

    monkeypatch.setattr(file_tool, "_open_secure_read", lambda *args, **kwargs: 123)
    monkeypatch.setattr(file_tool.os, "fdopen", lambda fd, **kwargs: _FakeHandle())
    monkeypatch.setattr(file_tool.os, "close", tracking_close)

    def boom(*args, **kwargs):
        raise RuntimeError("read failed")

    monkeypatch.setattr(file_tool, "_read_page", boom)

    with pytest.raises(ToolError, match="Error reading file"):
        asyncio.run(file_tool.read_file(str(target)))

    assert closed == [], "read-error cleanup closed a descriptor already owned by fdopen"
