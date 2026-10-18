"""Learned-compression protector, quality guard, and rollback contracts."""

from __future__ import annotations

import pytest

from machina.core.config import config
from machina.core.optimization.compression.learned import (
    LlMLinguaCompressor,
    compress_messages,
    maybe_compress_request,
)
from machina.core.optimization.compression.quality_guard import (
    anchor_retention,
    evaluate_compression,
)
from machina.core.optimization.compression.span_protector import (
    compressible_spans,
    protected_reasons,
)


class FakeCompressor:
    revision = "fake:v1"
    available = True

    def __init__(self, value=None):
        self.value = value

    def compress(self, text: str, *, ratio: float) -> str | None:
        if self.value is not None:
            return self.value
        return ("This is a note. " * 40).strip()


def test_span_protector_never_selects_system_current_turn_or_code():
    messages = [
        {"role": "system", "content": "system " * 200},
        {"role": "assistant", "content": "```python\nprint('x')\n```"},
        {"role": "assistant", "content": "plain old assistant " * 50},
        {"role": "assistant", "content": "another old assistant " * 50},
        {"role": "user", "content": "current request " * 50},
        {"role": "user", "content": "latest request " * 50},
    ]
    indices = [span.message_index for span in compressible_spans(messages)]
    assert 0 not in indices  # system
    assert 1 not in indices  # code fence
    assert 2 in indices
    assert 3 in indices
    assert 4 not in indices  # recent
    assert 5 not in indices  # recent
    assert "system" in protected_reasons(messages[0], index=0, total=len(messages))
    assert "code" in protected_reasons(messages[1], index=1, total=len(messages))

    block_form = [
        {"role": "assistant", "content": [{"type": "text", "text": "old " * 100}]},
        {"role": "user", "content": "current"},
        {"role": "user", "content": "latest"},
    ]
    assert compressible_spans(block_form) == []


@pytest.mark.asyncio
async def test_compress_messages_replaces_only_eligible_span():
    original = [
        {"role": "system", "content": "system " * 200},
        {"role": "assistant", "content": "This is an old assistant note. " * 40},
        {"role": "assistant", "content": "This is a middle assistant note. " * 40},
        {"role": "assistant", "content": "recent assistant"},
        {"role": "user", "content": "current request"},
    ]
    result = await compress_messages(original, compressor=FakeCompressor())
    assert result.applied is True
    assert result.messages[1]["content"] != original[1]["content"]
    assert result.messages[0] == original[0]
    assert original[1]["content"].startswith("This is an old")
    assert result.tokens_after < result.tokens_before


@pytest.mark.asyncio
async def test_anchor_loss_reverts_span():
    original = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "Please inspect /tmp/example.py:42 " * 30},
        {"role": "user", "content": "current"},
    ]
    result = await compress_messages(
        original, compressor=FakeCompressor(value="Please inspect the file.")
    )
    assert result.applied is False
    assert result.messages == original


def test_anchor_retention_and_quality_guard():
    assert anchor_retention("see /tmp/a.py:12", "see /tmp/a.py:12") == 1.0
    assert anchor_retention("see /tmp/a.py:12", "see the file") < 1.0


@pytest.mark.asyncio
async def test_quality_guard_rejects_semantic_drift():
    async def embed(text):
        return [1.0, 0.0] if "original" in text else [0.0, 1.0]

    result = await evaluate_compression(
        "original", "compressed", embed=embed, min_similarity=0.9
    )
    assert result.allowed is False
    assert result.reason == "semantic_drift"


@pytest.mark.asyncio
async def test_shadow_mode_returns_original_messages():
    original = [
        {"role": "assistant", "content": "This is an old assistant note. " * 40},
        {"role": "assistant", "content": "This is a middle assistant note. " * 40},
        {"role": "assistant", "content": "recent assistant"},
        {"role": "user", "content": "current"},
    ]
    result = await compress_messages(original, compressor=FakeCompressor(), shadow=True)
    assert result.applied is False
    assert result.messages == original
    assert result.tokens_after < result.tokens_before


@pytest.mark.asyncio
async def test_compression_off_is_a_noop(monkeypatch):
    monkeypatch.setattr(config, "optimization_compression_mode", "off")
    original = [{"role": "user", "content": "hello"}]
    result = await maybe_compress_request(
        original,
        [],
        provider=type("P", (), {"model": "m"})(),
        model="m",
    )
    assert result.messages == original
    assert result.applied is False


def test_optional_backend_is_not_imported_eagerly():
    compressor = LlMLinguaCompressor()
    # Whether installed or not, constructing it must not load model weights or
    # make a network request.
    assert compressor._compressor is None
    assert compressor.revision.startswith("llmlingua2:")
