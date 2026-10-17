"""Bounded process output reports truncation and owns every async reader."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.child_process import run_bounded


@pytest.mark.parametrize("size", [8, 9, 20000])
def test_stderr_truncation_is_reported_without_failing_successful_child(tmp_path, size):
    result = run_bounded(
        [sys.executable, "-c", f"import sys; sys.stderr.write('x' * {size})"],
        cwd=tmp_path,
        timeout=3,
        max_stderr=8,
    )
    assert result.stderr == "x" * min(size, 8)
    assert result.truncated is (size > 8)
    assert result.returncode == 0


async def test_script_overflow_joins_the_other_reader(monkeypatch, tmp_path):
    from machina.core import job_scripts

    stdout, stderr = object(), object()
    process = SimpleNamespace(
        stdout=stdout, stderr=stderr, wait=AsyncMock(return_value=0), returncode=0
    )
    monkeypatch.setattr(
        job_scripts.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr(job_scripts, "_terminate_tree", AsyncMock())
    started, retired = asyncio.Event(), asyncio.Event()
    readers = []

    async def read(stream):
        readers.append(asyncio.current_task())
        if stream is stdout:
            await started.wait()
            raise ValueError("script output exceeded 64 KiB")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            retired.set()

    monkeypatch.setattr(job_scripts, "_read_capped", read)
    try:
        with pytest.raises(ValueError, match="output exceeded"):
            await job_scripts._execute_script_command(["script"], tmp_path)
        assert retired.is_set()
        assert all(reader.done() for reader in readers)
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
