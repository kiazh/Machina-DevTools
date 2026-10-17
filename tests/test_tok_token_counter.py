"""TOK lane: the canonical token counter must be correct and honest.

RED before the fix:

* ``get_token_count`` is exact under tiktoken, but the "tiktoken unavailable"
  fallback is ``len(text) // 4`` — roughly 4x low for CJK and worse for emoji
  (0 tokens for a short emoji-only string);
* ``truncate_to_tokens`` on that fallback slices ``max_tokens * 4`` characters,
  so a CJK result is ~4x over the requested budget;
* a literal special-token spelling (``<|endoftext|>``) routes to the same
  broken fallback even when tiktoken is installed;
* the fallback emits no observable signal at all.

The documented fallback heuristic (``assembler._fallback_token_count``) is
expected to stay within ±25% of tiktoken for dense scripts (CJK/kana/hangul/
emoji and mixed text) and never be 4x low for any sample, while ASCII text
keeps the documented "one token per four characters" basis (rounded up, never
zero for non-empty text).
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from machina.core.assembler import (
    TokenCounter,
    get_token_count,
    truncate_to_tokens,
)
from tests.support.repo_paths import repo_path, repo_root

# ─── samples ──────────────────────────────────────────────────────────────

ASCII = "The quick brown fox jumps over the lazy dog. " * 4
CODE = "def f(x):\n    return {'a': x + 1}\n" * 5
JSON_SAMPLE = '{"a": 1, "b": [true, false, null], "c": "hello"}' * 5
PUNCTUATION = "!!!???...,,,;;;:::'\"'\"()[]{}<>---___" * 4
EMOJI = "🎉🚀😀👍🏽❤️🔥" * 4
CJK = "你好世界，这是一个测试。今天天气很好。" * 3
KANA = "こんにちは、これはテストです。カタカナも。" * 3
HANGUL = "안녕하세요 이것은 테스트입니다." * 3
MIXED = "Hello 世界! 🎉 你好，这是 mixed 文本。" * 4

SAMPLES = {
    "ascii": ASCII,
    "code": CODE,
    "json": JSON_SAMPLE,
    "punctuation": PUNCTUATION,
    "emoji": EMOJI,
    "cjk": CJK,
    "kana": KANA,
    "hangul": HANGUL,
    "mixed": MIXED,
}

# Dense scripts where the fallback must track tiktoken closely.
DENSE = ("emoji", "cjk", "kana", "hangul", "mixed")

# Documented tolerance of the deterministic fallback against tiktoken.
TOLERANCE = 0.25


def _tiktoken_length(text: str) -> int:
    tiktoken = pytest.importorskip("tiktoken")
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


@pytest.fixture
def fallback_counter(monkeypatch):
    """A counter that cannot see tiktoken (the offline/edge fallback path)."""
    from machina.core import assembler

    monkeypatch.setattr(assembler, "_TIKTOKEN_AVAILABLE", False)
    return assembler.TokenCounter()


# ─── 1. canonical counter is byte-for-byte tiktoken ───────────────────────


@pytest.mark.parametrize("name", list(SAMPLES))
def test_get_token_count_equals_direct_tiktoken_encoding(name):
    text = SAMPLES[name]
    assert get_token_count(text) == _tiktoken_length(text)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_token_counter_equals_direct_tiktoken_encoding(name):
    text = SAMPLES[name]
    assert TokenCounter().count(text) == _tiktoken_length(text)


def test_get_token_count_empty_is_zero():
    assert get_token_count("") == 0


# ─── 2. deterministic fallback heuristic ──────────────────────────────────


@pytest.mark.parametrize("name", DENSE)
def test_fallback_within_documented_tolerance_of_tiktoken(fallback_counter, name):
    text = SAMPLES[name]
    expected = _tiktoken_length(text)
    estimate = fallback_counter.count(text)
    assert abs(estimate - expected) / expected <= TOLERANCE, (
        f"{name}: fallback={estimate} tiktoken={expected} "
        f"delta={100 * (estimate - expected) / expected:+.1f}%"
    )


@pytest.mark.parametrize("name", list(SAMPLES))
def test_fallback_is_never_four_times_low(fallback_counter, name):
    expected = _tiktoken_length(SAMPLES[name])
    assert fallback_counter.count(SAMPLES[name]) * 4 >= expected


def test_fallback_minimum_one_for_non_empty_text(fallback_counter):
    for text in ("a", "🎉", "你", "ab", "!!"):
        assert fallback_counter.count(text) >= 1


def test_fallback_ascii_keeps_four_characters_per_token(fallback_counter):
    assert fallback_counter.count("") == 0
    assert fallback_counter.count("abcd") == 1
    assert fallback_counter.count("abcdefgh") == 2


def test_fallback_dense_scripts_count_per_character(fallback_counter):
    # Four CJK ideographs are ~four tiktoken tokens, not one.
    assert fallback_counter.count("你好世界") >= 4
    assert fallback_counter.count("🎉🎉") >= 4


def test_special_token_literal_uses_the_same_fallback_heuristic(monkeypatch):
    """Even with tiktoken installed, rejected text must not revert to len//4."""
    from machina.core import assembler

    text = "你好世界" * 5 + "<|endoftext|>"
    assert assembler._fallback_token_count(text) >= 20
    assert get_token_count(text) == assembler._fallback_token_count(text)


def test_prompt_assembler_budget_holds_for_special_token_and_cjk():
    """Assembly's internal fallback must use the constrained prefix, not chars*4."""
    from machina.core.assembler import PromptAssembler

    assembler = PromptAssembler(session_budget=60)
    prompt = assembler.assemble(
        "system <|endoftext|> " + "规则" * 40,
        None,
        [],
        [],
        "你好世界" * 40,
    )
    assert get_token_count(prompt) <= 60


def test_prompt_assembler_never_emits_replacement_characters():
    """A token cut inside an emoji must back off to a clean character boundary."""
    from machina.core.assembler import PromptAssembler

    assembler = PromptAssembler(session_budget=30)
    prompt = assembler.assemble("🎉" * 200, None, [], [], "hi")
    assert "\ufffd" not in prompt
    assert get_token_count(prompt) <= 30


def test_counter_lazy_init_is_thread_safe():
    import threading

    from machina.core.assembler import TokenCounter

    counter = TokenCounter()
    expected = _tiktoken_length(CJK)
    results: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(counter.count(CJK))
        except BaseException as error:  # noqa: BLE001 - surfaced by the assertion
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, errors
    assert results == [expected] * len(threads)


# ─── 3. truncation never exceeds the requested budget ─────────────────────


@pytest.mark.parametrize("name", DENSE)
@pytest.mark.parametrize("budget", [1, 5, 20])
def test_fallback_truncate_never_exceeds_budget(fallback_counter, name, budget):
    text = SAMPLES[name]
    out = fallback_counter.truncate(text, budget)
    assert text.startswith(out)
    assert fallback_counter.count(out) <= budget


def test_fallback_truncate_zero_or_negative_budget(fallback_counter):
    assert fallback_counter.truncate("你好世界", 0) == ""
    assert fallback_counter.truncate("你好世界", -3) == ""


@pytest.mark.parametrize("name", list(SAMPLES))
def test_tiktoken_truncate_never_exceeds_budget(name):
    text = SAMPLES[name] * 3
    out = truncate_to_tokens(text, 7)
    assert text.startswith(out)
    assert get_token_count(out) <= 7


def test_truncate_returns_full_text_when_within_budget():
    text = "你好世界"
    assert truncate_to_tokens(text, 100) == text


# ─── 4. first-fallback observability ──────────────────────────────────────


def test_first_fallback_engagement_logs_once(monkeypatch):
    from machina.core import assembler

    calls: list[tuple[str, str, dict]] = []
    fake = types.ModuleType("machina.observability.instrumentation")

    def log_event(event, *, level="info", **fields):
        calls.append((event, level, fields))

    fake.log_event = log_event
    monkeypatch.setitem(sys.modules, "machina.observability.instrumentation", fake)
    monkeypatch.setattr(assembler, "_fallback_notified", False)
    monkeypatch.setattr(assembler, "_TIKTOKEN_AVAILABLE", False)

    counter = assembler.TokenCounter()
    counter.count("你好世界")
    counter.count("世界你好")
    counter.truncate("你好世界", 2)

    assert [call[0] for call in calls] == ["tokens.count"]
    assert calls[0][1] == "debug"
    assert calls[0][2]["basis"] == "fallback"
    assert "reason" in calls[0][2]


def test_missing_instrumentation_module_never_breaks_counting(monkeypatch):
    from machina.core import assembler

    monkeypatch.setitem(sys.modules, "machina.observability.instrumentation", None)
    monkeypatch.setattr(assembler, "_fallback_notified", False)
    monkeypatch.setattr(assembler, "_TIKTOKEN_AVAILABLE", False)

    assert assembler.TokenCounter().count("你好世界") == 4


# ─── 5. verification tool ─────────────────────────────────────────────────


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root())
    env.pop("MACHINA_TEST_DATABASE_URL", None)
    return subprocess.run(
        [sys.executable, str(repo_path("tools/check_token_counter.py")), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root()),
        timeout=120,
        check=False,
    )


def test_check_token_counter_tool_passes_and_is_db_free():
    result = _run_tool()
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "tiktoken" in result.stdout
    for name in ("cjk", "emoji", "mixed"):
        assert name in result.stdout


def test_check_token_counter_tool_exits_nonzero_under_tolerance():
    result = _run_tool("--tolerance", "0.0001")
    assert result.returncode != 0
    assert "FAIL" in result.stdout or "fail" in result.stdout
