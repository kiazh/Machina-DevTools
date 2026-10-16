"""Token-efficiency T3: deterministic format reducers.

Contract under test: deterministic output, unchanged when unsafe (no net
saving), omission/recovery disclosure always present for lossy views, and
tokens counted after wrappers.
"""

from __future__ import annotations

import json

from machina.core.reducers import VIEW_MARKER, reduce_tool_result


def _pytest_output(passed: int = 412, failed: int = 2) -> str:
    lines = [f"tests/test_mod.py::test_case_{i} PASSED" for i in range(passed)]
    lines += [
        f"tests/test_other.py::test_fail_{i} FAILED" for i in range(failed)
    ]
    lines += [
        f"===== {passed} passed, {failed} failed, 7 skipped in 12.34s =====",
        "FAILED tests/test_save.py::test_conflict - AssertionError: expected conflict, received success",
        "E   assert 1 == 2",
    ]
    return "\n".join(lines)


def test_reduction_contract_shape():
    result = reduce_tool_result(_pytest_output(), budget_chars=1000, artifact_id="art-1")
    assert result.changed
    assert result.method == "tests"
    assert result.lossy is True
    assert result.recoverable is True
    assert result.artifact_id == "art-1"
    assert result.tokens_before is not None and result.tokens_after is not None
    assert result.tokens_after < result.tokens_before
    assert len(result.output) <= 1000
    assert VIEW_MARKER in result.output


def test_tests_reducer_keeps_counts_and_decisive_failures():
    result = reduce_tool_result(_pytest_output(), budget_chars=1000, artifact_id="a")
    assert "412 passed" in result.output
    assert "test_conflict" in result.output
    assert "expected conflict" in result.output


def test_tests_reducer_flags_zero_collected_as_not_passing():
    text = "collected 0 items\n\nno tests ran in 0.05s\n" + "x" * 5000
    result = reduce_tool_result(text, budget_chars=600)
    assert result.changed
    assert "0 tests collected" in result.output


def test_disclosure_reports_recovery_or_rerun():
    with_artifact = reduce_tool_result(_pytest_output(), budget_chars=1000, artifact_id="zzz")
    assert "artifact zzz" in with_artifact.output
    assert "artifact_read" in with_artifact.output
    without = reduce_tool_result(_pytest_output(), budget_chars=1000)
    assert without.recoverable is False
    assert "re-run the tool" in without.output


def test_small_input_is_unchanged():
    result = reduce_tool_result("tiny", budget_chars=1000)
    assert result.changed is False
    assert result.output == "tiny"


def test_json_reducer_outline_and_verbatim_head():
    document = {
        "items": [{"id": i, "name": f"n{i}"} for i in range(5000)],
        "big_number": 123456789012345678901234567890,
        "meta": {"a": 1, "b": 2},
    }
    text = json.dumps(document)
    result = reduce_tool_result(text, budget_chars=1500, artifact_id="j1")
    assert result.changed and result.method == "json"
    assert "5000" in result.output
    assert "items: list[5000]" in result.output
    # The head stays verbatim: exact lexical spellings (big integers) survive.
    head_index = result.output.index("first chars (verbatim):")
    assert text[:50] in result.output[head_index:]


def test_unparseable_payloads_fall_back_to_marked_head_tail():
    huge_invalid = "{" + "nope" * 5000
    result = reduce_tool_result(huge_invalid, budget_chars=800, artifact_id="a")
    assert result.changed and result.method == "head_tail"
    # Both ends are always kept and the omission seam is visible (the
    # disclosure is appended after the body).
    assert result.output.startswith(huge_invalid[:20])
    assert huge_invalid[-20:] in result.output
    assert "chars omitted" in result.output
    assert "nope" in result.output[:40] and "nope" in result.output[-200:]


def test_head_tail_keeps_the_verdict_tail_even_at_the_wire_budget():
    """The tail must never be dropped by the packer (reviewer probe)."""
    text = "HEAD_MARKER " + "m" * 5000 + " TAIL_MARKER"
    for budget in (300, 700, 1000, 4000):
        result = reduce_tool_result(text, budget_chars=budget, artifact_id="h")
        assert result.changed
        assert "HEAD_MARKER" in result.output, budget
        assert "TAIL_MARKER" in result.output, budget
        assert len(result.output) <= budget, budget


def test_lone_exit_code_line_is_not_a_test_run():
    text = "\n".join(f"building step {i}" for i in range(200)) + "\nexit code: 1"
    result = reduce_tool_result(text, budget_chars=800, artifact_id="e")
    assert result.changed
    assert result.method != "tests"


def test_csv_reducer_reports_rows_and_numeric_bounds():
    rows = ["value,name"] + [f"{i},{'n' * 50}" for i in range(400)]
    text = "\n".join(rows)
    result = reduce_tool_result(text, budget_chars=900, artifact_id="c1")
    assert result.changed and result.method == "csv"
    assert "400 rows" in result.output
    assert "min=0" in result.output and "max=399" in result.output


def test_diff_reducer_counts_changes():
    lines = ["diff --git a/x.py b/x.py", "--- a/x.py", "+++ b/x.py", "@@ -1,3 +1,4 @@"]
    lines += [f"+added line {i}" for i in range(200)]
    lines += [f"-removed line {i}" for i in range(50)]
    lines += [f" context {i}" for i in range(300)]
    result = reduce_tool_result("\n".join(lines), budget_chars=900, artifact_id="d1")
    assert result.changed and result.method == "diff"
    assert "+200" in result.output and "-50" in result.output
    assert "hunks" in result.output


def test_logs_reducer_groups_repetition_and_surfaces_errors():
    lines = []
    for i in range(120):
        lines.append(f"2026-10-16 10:00:{i % 60:02d} INFO request id={i} ok")
    lines.append("2026-10-16 10:01:00 ERROR boom id=999")
    for i in range(60):
        lines.append(f"2026-10-16 10:02:{i % 60:02d} INFO request id={i} ok")
    result = reduce_tool_result("\n".join(lines), budget_chars=1000, artifact_id="l1")
    assert result.changed and result.method == "logs"
    assert "distinct patterns" in result.output
    assert "ERROR boom" in result.output


def test_code_reducer_keeps_structural_lines():
    lines = ["import os", "", "def alpha():"] + [f"    x = {i}" for i in range(40)]
    for i in range(16):
        lines += [f"def helper_{i}(value):", f"    return value + {i}"]
    lines += ["class Beta:", "    def gamma(self):"] + [f"    y = {i}" for i in range(40)]
    text = "\n".join(lines)
    result = reduce_tool_result(text, budget_chars=900, artifact_id="k1")
    assert result.changed and result.method == "code"
    assert "def alpha" in result.output and "class Beta" in result.output


def test_budget_never_exceeded_across_detectors():
    samples = {
        "tests": _pytest_output(),
        "json": json.dumps({"k": ["v" * 100] * 500}),
        "logs": "\n".join(f"2026-10-16 10:00:00 INFO item {i}" for i in range(400)),
        "plain": "q" * 50000,
    }
    for name, text in samples.items():
        result = reduce_tool_result(text, budget_chars=700, artifact_id="t")
        if result.changed:
            assert len(result.output) <= 700, (name, len(result.output))


def test_determinism():
    text = _pytest_output()
    first = reduce_tool_result(text, budget_chars=1000, artifact_id="a")
    second = reduce_tool_result(text, budget_chars=1000, artifact_id="a")
    assert first == second
