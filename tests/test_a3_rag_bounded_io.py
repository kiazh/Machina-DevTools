"""A3 campaign: descriptor-anchored, bounded RAG ingestion (MCH-007/MCH-008).

Red-first evidence for:
* MCH-007 — validation must bind the bytes actually read (descriptor walk),
  never a later pathname open, for text and PDF paths.
* MCH-008 — ``rag.max_document_bytes`` is enforced before and during the
  descriptor read; PDF page/text budgets refuse oversized parses; the loader
  runs off the event loop.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from machina.core.exceptions import ValidationError
from machina.rag.loaders import Document, FileLoader

OUTSIDE = "OUTSIDE-SECRET-MARKER"


@pytest.fixture
def loader(tmp_path):
    return FileLoader(tmp_path)


def _swap_final_component(loader: FileLoader, outside: bytes):
    """Return a ``_resolve_path`` wrapper that swaps the leaf *after* checking."""
    original = loader._resolve_path

    def resolve_then_swap(path, *args, **kwargs):
        resolved = original(path, *args, **kwargs)
        resolved.unlink()
        resolved.symlink_to(outside)
        return resolved

    loader._resolve_path = resolve_then_swap  # type: ignore[method-assign]


def test_text_final_component_swap_after_validation_is_refused(loader, tmp_path):
    inside = tmp_path / "report.txt"
    inside.write_text("INSIDE", encoding="utf-8")
    outside = tmp_path.parent / "outside-a3.txt"
    outside.write_text(OUTSIDE, encoding="utf-8")

    _swap_final_component(loader, outside)

    with pytest.raises(ValidationError):
        loader.load("report.txt")
    assert inside.is_symlink()  # proof the swap really happened


def test_text_ancestor_swap_after_validation_is_refused(loader, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.txt").write_text("INSIDE", encoding="utf-8")
    outside_dir = tmp_path.parent / "outside-a3-dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "f.txt").write_text(OUTSIDE, encoding="utf-8")

    original = loader._resolve_path

    def resolve_then_swap(path, *args, **kwargs):
        resolved = original(path, *args, **kwargs)
        os.rename(tmp_path / "sub", tmp_path / "sub-moved")
        os.symlink(outside_dir, tmp_path / "sub")
        return resolved

    loader._resolve_path = resolve_then_swap  # type: ignore[method-assign]

    with pytest.raises(ValidationError):
        loader.load("sub/f.txt")


def test_non_regular_substitution_is_refused(loader, tmp_path):
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)
    with pytest.raises(ValidationError, match="Not a file|regular"):
        loader.load("pipe.txt")


def test_oversized_text_is_refused_before_any_read(tmp_path, monkeypatch):
    big = tmp_path / "big.log"
    big.write_text("x" * 4096, encoding="utf-8")
    loader = FileLoader(tmp_path, max_document_bytes=1024)

    reads = 0
    real_read = os.read

    def counting_read(fd, n):
        nonlocal reads
        reads += 1
        return real_read(fd, n)

    monkeypatch.setattr("machina.rag.loaders._os_read", counting_read)

    with pytest.raises(ValidationError, match="exceed|limit|large"):
        loader.load("big.log")
    assert reads == 0, "an oversized file must be refused before reading its body"


def test_read_that_grows_past_the_budget_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "grow.txt"
    target.write_text("x" * 32, encoding="utf-8")
    loader = FileLoader(tmp_path, max_document_bytes=64)
    calls = 0

    def growing_read(fd, n):
        nonlocal calls
        calls += 1
        # Pretend the file grew after the pre-read size check.
        return b"y" * n if calls == 1 else b""

    monkeypatch.setattr("machina.rag.loaders._os_read", growing_read)

    with pytest.raises(ValidationError, match="exceed|limit|large"):
        loader.load("grow.txt")
    assert calls >= 1


def _fake_pdf_module(pages: list[str]):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage(text) for text in pages]

    module = types.ModuleType("PyPDF2")
    module.PdfReader = FakeReader
    return module


def test_pdf_page_budget_is_enforced(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setitem(sys.modules, "PyPDF2", _fake_pdf_module(["a", "b", "c"]))
    loader = FileLoader(tmp_path, max_pdf_pages=2)

    with pytest.raises(ValidationError, match="page"):
        loader.load("doc.pdf")


def test_pdf_extracted_text_budget_is_enforced(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setitem(sys.modules, "PyPDF2", _fake_pdf_module(["abcdefgh", "ijklmnop"]))
    loader = FileLoader(tmp_path, max_pdf_text_bytes=12)

    with pytest.raises(ValidationError, match="extracted text|text budget"):
        loader.load("doc.pdf")


async def test_index_document_refuses_oversized_file_without_writing(tmp_path):
    from machina.rag.pipeline import RAGConfig, RAGPipeline

    big = tmp_path / "big.log"
    big.write_text("z" * 4096, encoding="utf-8")
    pipeline = RAGPipeline(
        config=RAGConfig(
            enable_embeddings=False, enable_reranking=False, max_document_bytes=1024
        )
    )
    pipeline._loader._base_dir = tmp_path
    mock_db = AsyncMock()
    from tests.support.database import attach_connection

    attach_connection(mock_db)

    with (
        patch("machina.rag.pipeline.db", mock_db),
        pytest.raises(ValidationError, match="exceed|limit|large"),
    ):
        await pipeline.index_document(str(big), uuid.uuid4())
    mock_db.fetch.assert_not_awaited()
    mock_db.execute.assert_not_awaited()


async def test_index_document_keeps_the_loop_responsive_with_a_slow_loader():
    from machina.rag.pipeline import RAGConfig, RAGPipeline

    class SlowLoader:
        def __init__(self):
            self.finished = False

        def load(self, source):
            time.sleep(0.2)
            self.finished = True
            return Document(content="alpha beta gamma", source=str(source), doc_type="text")

    slow = SlowLoader()
    pipeline = RAGPipeline(
        config=RAGConfig(enable_embeddings=False, enable_reranking=False), loader=slow
    )
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value="DELETE 0")
    mock_db.fetch = AsyncMock(return_value=[])
    from tests.support.database import attach_connection

    attach_connection(mock_db)

    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker enter its loop
    ticks = 0
    try:
        with patch("machina.rag.pipeline.db", mock_db):
            task = asyncio.create_task(
                pipeline.index_document("slow.txt", uuid.uuid4())
            )
            await asyncio.sleep(0.05)
            assert not slow.finished, "loader blocked the event loop"
            assert ticks > 0, "event loop did not progress during loader I/O"
            await task
    finally:
        stop = True
        tick.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tick
    assert slow.finished
