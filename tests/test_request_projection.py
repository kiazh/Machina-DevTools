"""B1 — shared budgeted request projector, shadow mode (spec §14, §17-B1).

Acceptance-criteria mapping (tdd-guide spec-first workflow):
  AC-14.4/§17-B1  grouping, pins, deterministic selection, safe refusal
  C01  several tool rounds: older eligible bodies projected, recent group kept
  C02  oversized latest group: deterministic reduction or explicit refusal
  C03  protected constraints: exact requirements/negations survive
  C04  parallel tool batch: all ids/results valid and ordered
  C05  native signed blocks: continuation stays valid or the projection refuses
  C06  artifact write fails/full: no unrecoverable lossy publication
  C12  pins alone exceed capacity: honest failure, no silent truncation
  C15  multiple projection passes: bounded termination, no retry storm
  C16  no optimization enabled: baseline-compatible fallback

Shadow mode only: the projector never mutates its inputs and makes no
provider/DB calls — the counting and artifact lookups are injected, and both
live-loop call sites are untouched (wiring is B2). Fixtures live in
``tests/data/request_projection/corpus.json``.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from machina.core.request_projection import (
    MAX_FIT_PASSES,
    ProjectionInput,
    ProjectionMessage,
    ProjectionPins,
    project_request,
    provider_messages_to_units,
    shadow_report,
)

REPO = Path(os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina")
CORPUS_PATH = Path(__file__).resolve().parent / "data" / "request_projection" / "corpus.json"
MODULE_PATH = REPO / "machina" / "core" / "request_projection.py"


# --- deterministic test doubles -------------------------------------------------


def _wire_chars(message) -> int:
    total = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or ():
        total += len(json.dumps(call, sort_keys=True, ensure_ascii=False))
    return total


def counting_basis(messages) -> int:
    """Mirrors the locally-counted shape of prompt.request_context_tokens."""
    return 16 + sum(8 + _wire_chars(message) // 4 for message in messages)


class Counting:
    """Injected counter that records how many fit passes were run."""

    def __init__(self, basis=counting_basis):
        self.basis = basis
        self.calls = 0

    def __call__(self, messages) -> int:
        self.calls += 1
        return self.basis(messages)

    def __eq__(self, other):
        return isinstance(other, Counting) and self.basis == other.basis


def verified_lookup(handles, *, complete=True, available=True):
    """Artifact lookup reporting verified recoverable originals for *handles*."""
    views = {
        handle: {
            "available": available,
            "complete": complete,
            "handle": handle,
            "revision": "rev-1",
            "summary_view": f"typed summary for {handle}",
        }
        for handle in handles
    }
    return lambda handle: views.get(handle)


def call(call_id: str, path: str = "src/x.py") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
    }


# --- corpus loading -------------------------------------------------------------


def _case(name: str) -> dict:
    corpus = json.loads(CORPUS_PATH.read_text())
    return corpus["cases"][name]


def _pins(case: dict) -> ProjectionPins:
    raw = case.get("pins") or {}
    return ProjectionPins(
        objective=tuple(raw.get("objective") or ()),
        constraints=tuple(raw.get("constraints") or ()),
        read_only_scope=tuple(raw.get("read_only_scope") or ()),
        core_instructions=tuple(raw.get("core_instructions") or ()),
        blockers=tuple(raw.get("blockers") or ()),
        task_state=tuple(raw.get("task_state") or ()),
        provider_required_blocks=tuple(raw.get("provider_required_blocks") or ()),
    )


def _units(case: dict) -> tuple[ProjectionMessage, ...]:
    units = []
    for message in case["messages"]:
        native = message.get("native_blocks")
        units.append(
            ProjectionMessage(
                unit_id=message["unit_id"],
                role=message["role"],
                content=message.get("content", ""),
                tool_calls=tuple(message.get("tool_calls") or ()),
                tool_call_id=message.get("tool_call_id"),
                protected=bool(message.get("protected", False)),
                kind=message.get("kind", "history"),
                artifact_handle=message.get("artifact_handle"),
                priority=int(message.get("priority", 0)),
                native_blocks=native,
                native_key="provider_content" if native is not None else None,
            )
        )
    return tuple(units)


def _input(
    case: dict,
    *,
    counter=None,
    lookup=None,
    budget=None,
    headroom=None,
    messages=None,
    min_recent=None,
) -> ProjectionInput:
    return ProjectionInput(
        messages=messages if messages is not None else _units(case),
        pins=_pins(case),
        budget_tokens=case["budget_tokens"] if budget is None else budget,
        count_tokens=counter or Counting(),
        artifacts=lookup,
        safety_headroom_tokens=(case["safety_headroom_tokens"] if headroom is None else headroom),
        min_recent_groups=case["min_recent_groups"] if min_recent is None else min_recent,
    )


def _decisions(result) -> dict[str, str]:
    return {decision.unit_id: decision.action for decision in result.decisions}


def _reasons(result) -> dict[str, str]:
    return {decision.unit_id: decision.reason for decision in result.decisions}


# --- C16: baseline-compatible fallback -------------------------------------------


def _inline_units() -> tuple[ProjectionMessage, ...]:
    return (
        ProjectionMessage(unit_id="a", role="user", content="hello"),
        ProjectionMessage(unit_id="b", role="assistant", content="", tool_calls=(call("c1"),)),
        ProjectionMessage(unit_id="c", role="tool", content="ok", tool_call_id="c1"),
    )


def test_c16_nothing_over_budget_keeps_everything_verbatim():
    units = _inline_units()
    plan = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "src/x.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]
    result = project_request(
        ProjectionInput(
            messages=units,
            budget_tokens=counting_basis(plan) + 100,
            count_tokens=Counting(),
        )
    )
    assert result.refusal is None
    assert result.retained == ("a", "b", "c")
    assert result.projected == ()
    assert result.omitted == ()
    assert result.passes == 1
    assert result.pinned_survived is True
    assert [dict(message) for message in result.planned_messages] == plan
    assert result.estimated_tokens == counting_basis(plan)


# --- C01: several tool rounds -----------------------------------------------------


def test_c01_older_rounds_projected_while_latest_group_is_preserved():
    case = _case("multi_round")
    lookup = verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"])
    result = _input(case, lookup=lookup)
    result = project_request(result)

    assert result.refusal is None
    effective = case["budget_tokens"] - case["safety_headroom_tokens"]
    assert result.estimated_tokens <= effective  # a sendable plan always fits
    actions = _decisions(result)
    # Latest coherent group survives exactly.
    assert actions["m10"] == "retained"
    assert actions["m11"] == "retained"
    latest = next(m for m in result.planned_messages if m.get("tool_call_id") == "call_5")
    assert latest["content"] == "latest verdict: 3 passed, 0 failed; module_5.py imports cleanly"
    # At least one older body was externalized to a verified view.
    projected = {unit.unit_id: unit for unit in result.projected}
    assert projected, actions
    for unit_id, unit in projected.items():
        assert unit_id in {"m3", "m5", "m7", "m9"}  # tool results of the older rounds
        assert unit.artifact_handle.startswith("art-r")
        assert unit.completeness == "complete"
        assert "artifact_read pages the exact original" in unit.view
    # A projected unit never disappears from the plan: it is re-rendered as the view.
    planned_views = [
        m for m in result.planned_messages if "[machina view:" in str(m.get("content", ""))
    ]
    assert len(planned_views) == len(projected)
    # Every unit decision is accounted for.
    assert set(actions) == {message["unit_id"] for message in case["messages"]}


def test_c01_older_group_span_is_atomic_no_torn_batches():
    case = _case("multi_round")
    result = project_request(
        _input(case, lookup=verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"]))
    )
    actions = _decisions(result)
    for index in (1, 2, 3, 4):
        assistant_id, tool_id = f"m{index * 2}", f"m{index * 2 + 1}"
        assert (actions[assistant_id] == "omitted") == (actions[tool_id] == "omitted")
        if actions[assistant_id] != "omitted":
            # The exact executable call survives whenever its round survives.
            planned = next(m for m in result.planned_messages if "tool_calls" in m)
            assert assistant_id in result.retained or assistant_id in {
                unit.unit_id for unit in result.projected
            }
            assert planned["tool_calls"][0]["id"].startswith("call_")


def test_c01_projected_views_survive_the_stale_condensation_exemption():
    from machina.core.prompt import _REDUCED_VIEW_TAIL_RE

    case = _case("multi_round")
    result = project_request(
        _input(case, lookup=verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"]))
    )
    assert result.projected, "fixture must exercise view replacement"
    for unit in result.projected:
        assert _REDUCED_VIEW_TAIL_RE.search(unit.view), unit.unit_id


# --- C02: oversized latest group ---------------------------------------------------


def test_c02_oversized_latest_group_reduced_when_views_are_verified():
    case = _case("oversized_latest")
    result = project_request(_input(case, lookup=verified_lookup(["art-b1", "art-b2"])))
    assert result.refusal is None
    effective = case["budget_tokens"] - case["safety_headroom_tokens"]
    assert result.estimated_tokens <= effective
    actions = _decisions(result)
    assert actions["m1"] == "retained"
    assert actions["m2"] == "projected" and actions["m3"] == "projected"
    assistant = next(m for m in result.planned_messages if "tool_calls" in m)
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_b1", "call_b2"]
    tool_messages = [m for m in result.planned_messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_b1", "call_b2"]
    assert "art-b1" in tool_messages[0]["content"] and "art-b2" in tool_messages[1]["content"]
    assert result.pinned_survived is True


def test_c02_oversized_latest_group_refuses_when_views_are_unverified():
    case = _case("oversized_latest")
    result = project_request(_input(case, lookup=None))
    assert result.refusal is not None
    assert result.refusal.code == "floor_exceeded_capacity"
    assert result.refusal.required_tokens > result.refusal.budget_tokens
    assert result.refusal.safe_options
    assert result.pinned_survived is False
    assert result.planned_messages == ()
    assert {unit.unit_id for unit in result.omitted} == {"m0", "m1", "m2", "m3"}
    assert all(unit.reason == "capacity_refusal" for unit in result.omitted)


# --- C03: pinned constraints --------------------------------------------------------


def test_c03_pinned_constraints_survive_exactly():
    case = _case("multi_round")
    result = project_request(
        _input(case, lookup=verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"]))
    )
    assert result.pinned_survived is True
    expected = [
        ("objective", case["pins"]["objective"][0]),
        ("constraints", case["pins"]["constraints"][0]),
        ("constraints", case["pins"]["constraints"][1]),
        ("read_only_scope", case["pins"]["read_only_scope"][0]),
    ]
    assert [(pin.kind, pin.text) for pin in result.retained_pins] == expected
    planned_pins = [dict(m) for m in result.planned_messages[: len(expected)]]
    assert planned_pins == [{"role": "system", "content": text} for _, text in expected]


def test_c12_pins_alone_over_budget_refuse_structurally():
    big = "constraint: " + "x" * 4000
    projection_input = ProjectionInput(
        messages=(),
        pins=ProjectionPins(constraints=(big,)),
        budget_tokens=100,
        count_tokens=Counting(),
        safety_headroom_tokens=10,
    )
    result = project_request(projection_input)
    assert result.refusal is not None
    assert result.refusal.code == "pins_exceed_capacity"
    assert result.refusal.budget_tokens == 90  # effective budget after headroom
    assert result.refusal.required_tokens > 90
    assert result.refusal.safe_options
    assert result.pinned_survived is False
    assert result.planned_messages == ()
    assert result.retained == () and result.projected == () and result.omitted == ()
    assert _decisions(result) == {}

    report = shadow_report(projection_input)
    assert report["would_fit"] is False
    assert report["refusal"]["code"] == "pins_exceed_capacity"
    assert report["planned_messages"] == 0


def test_min_recent_groups_floor_protects_the_configured_recent_set():
    case = _case("multi_round")
    lookup = verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"])
    wide = project_request(_input(case, lookup=lookup, min_recent=2, budget=430, headroom=0))
    assert wide.refusal is None
    actions = _decisions(wide)
    assert actions["m8"] != "omitted" and actions["m9"] != "omitted"
    assert actions["m10"] == "retained" and actions["m11"] == "retained"
    assert (actions["m8"] == "omitted") == (actions["m9"] == "omitted")
    # Without verified views the wider floor cannot fit: refuse, never shrink silently.
    refused = project_request(_input(case, lookup=None, min_recent=2, budget=430, headroom=0))
    assert refused.refusal is not None
    assert refused.refusal.code == "floor_exceeded_capacity"


# --- C04: parallel tool batch --------------------------------------------------------


def test_c04_parallel_batch_keeps_all_ids_and_results_ordered():
    case = _case("parallel_batch")
    lookup = verified_lookup(["art-old", "art-p1", "art-p2", "art-p3"])
    result = project_request(_input(case, lookup=lookup))
    assert result.refusal is None
    actions = _decisions(result)
    # No batch member may be torn out of its round.
    assert "omitted" not in {actions["m3"], actions["m4"], actions["m5"], actions["m6"]}
    assistant = next(
        m
        for m in result.planned_messages
        if any(tc["id"] == "call_p3" for tc in m.get("tool_calls") or ())
    )
    assert [tc["id"] for tc in assistant["tool_calls"]] == [
        "call_p1",
        "call_p2",
        "call_p3",
    ]
    tool_index = result.planned_messages.index(assistant) + 1
    batch = result.planned_messages[tool_index : tool_index + 3]
    assert [m["role"] for m in batch] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in batch] == ["call_p1", "call_p2", "call_p3"]
    assert all(
        m["content"]
        in (
            next(u for u in case["messages"] if u["unit_id"] == uid)["content"]
            for uid in ("m4", "m5", "m6")
        )
        or "[machina view:" in m["content"]
        for m in batch
    )


def test_retrieved_material_is_optional_and_ordered_by_explicit_priority():
    def retrieved(unit_id, priority, filler):
        return ProjectionMessage(
            unit_id=unit_id,
            role="system",
            content=f"{unit_id} " + filler * 200,
            kind="retrieved",
            priority=priority,
        )

    user = ProjectionMessage(unit_id="u", role="user", content="go")
    low = retrieved("r-low", 5, "x")
    high = retrieved("r-high", 1, "y")
    # Input order is low-then-high; explicit priority decides (lower wins).
    result = project_request(
        ProjectionInput(messages=(user, low, high), budget_tokens=100, count_tokens=Counting())
    )
    assert result.refusal is None
    assert _decisions(result)["r-high"] == "retained"
    assert _decisions(result)["r-low"] == "omitted"
    assert _reasons(result)["r-low"] == "retrieved_over_budget"
    assert "r-high" in str(result.planned_messages)

    # Equal priorities keep input order (stable, no reordering of the plan).
    first = retrieved("r-first", 3, "a")
    second = retrieved("r-second", 3, "b")
    tied = project_request(
        ProjectionInput(messages=(user, first, second), budget_tokens=100, count_tokens=Counting())
    )
    assert _decisions(tied)["r-first"] == "retained"
    assert _decisions(tied)["r-second"] == "omitted"


# --- C05: native signed blocks -------------------------------------------------------


def test_c05_protected_signed_blocks_survive_verbatim():
    case = _case("protected_native")
    result = project_request(_input(case))
    assert result.refusal is None
    actions = _decisions(result)
    assert actions["m1"] == "retained" and actions["m2"] == "retained"
    planned = [dict(m) for m in result.planned_messages]
    assistant = next(m for m in planned if m.get("role") == "assistant" and "provider_content" in m)
    native = case["messages"][1]["native_blocks"]
    assert assistant["provider_content"] == native
    assert assistant["tool_calls"] == case["messages"][1]["tool_calls"]
    tool = next(m for m in planned if m.get("role") == "tool")
    assert tool["content"] == "signed result: 1 passed"
    assert all(unit.unit_id not in {"m1", "m2"} for unit in result.projected)


def test_c05_protected_blocks_alone_over_budget_refuse():
    case = _case("protected_native")
    result = project_request(_input(case, budget=100, headroom=0))
    assert result.refusal is not None
    assert result.refusal.code == "protected_exceed_capacity"
    assert result.pinned_survived is False
    assert result.planned_messages == ()


# --- C06: no unrecoverable lossy publication -----------------------------------------


def test_artifact_unavailable_keeps_original_and_never_fakes_a_view():
    case = _case("multi_round")
    generous = project_request(_input(case, lookup=None, budget=5000, headroom=0))
    assert generous.projected == ()
    assert generous.refusal is None
    planned = {
        m.get("tool_call_id"): m for m in generous.planned_messages if m.get("role") == "tool"
    }
    for message in case["messages"]:
        if message["role"] == "tool":
            assert planned[message["tool_call_id"]]["content"] == message["content"]

    tight = project_request(_input(case, lookup=None))
    assert tight.projected == ()
    assert tight.refusal is None  # the floor (pins + latest group) fits
    omitted = {unit.unit_id for unit in tight.omitted}
    assert omitted  # older rounds are dropped explicitly, not rewritten
    assert all("[machina view:" not in str(m.get("content", "")) for m in tight.planned_messages)
    for decision in tight.decisions:
        assert decision.reason in {
            "protected",
            "required",
            "latest_group",
            "recent_group",
            "optional_fit",
            "older_group_over_budget",
            "retrieved_over_budget",
        }


def test_artifact_incomplete_blocks_view_replacement():
    case = _case("multi_round")
    handles = ["art-r1", "art-r2", "art-r3", "art-r4"]
    result = project_request(_input(case, lookup=verified_lookup(handles, complete=False)))
    assert result.projected == ()
    result = project_request(_input(case, lookup=verified_lookup(handles, available=False)))
    assert result.projected == ()


def test_lookup_error_degrades_to_unavailable():
    case = _case("multi_round")

    def broken(handle):
        raise RuntimeError("artifact store down")

    result = project_request(_input(case, lookup=broken))
    assert result.projected == ()
    assert result.refusal is None  # floor still fits; older rounds are omitted


# --- determinism and bounds ----------------------------------------------------------


def test_determinism_same_input_yields_byte_identical_result():
    case = _case("multi_round")
    lookup = verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"])
    first = project_request(_input(case, lookup=lookup))
    second = project_request(_input(case, lookup=lookup))
    assert first == second
    assert first.retained == second.retained
    assert first.estimated_tokens == second.estimated_tokens
    assert first.input_revision == second.input_revision

    def report():
        return json.dumps(
            shadow_report(_input(case, lookup=lookup)), sort_keys=True, ensure_ascii=False
        )

    assert report() == report()


def test_bounded_passes_no_retry_storm():
    messages = [ProjectionMessage(unit_id="u0", role="user", content="go")]
    handles = []
    for index in range(24):
        handle = f"art-{index:02d}"
        handles.append(handle)
        messages.append(
            ProjectionMessage(
                unit_id=f"a{index:02d}",
                role="assistant",
                content="",
                tool_calls=(call(f"c{index:02d}"),),
            )
        )
        messages.append(
            ProjectionMessage(
                unit_id=f"t{index:02d}",
                role="tool",
                content=f"result {index} " + "y" * 1200,
                tool_call_id=f"c{index:02d}",
                artifact_handle=handle,
            )
        )
    counter = Counting()
    result = project_request(
        ProjectionInput(
            messages=tuple(messages),
            pins=ProjectionPins(objective=("bounded",)),
            budget_tokens=250,
            count_tokens=counter,
            artifacts=verified_lookup(handles),
            min_recent_groups=1,
        )
    )
    assert counter.calls == result.passes <= MAX_FIT_PASSES
    assert result.refusal is None  # the newest group fits
    assert result.estimated_tokens <= 250  # effective budget, no headroom
    assert result.passes > 1  # it really had to search


def test_pass_limit_backstop_refuses_without_a_retry_storm(monkeypatch):
    from machina.core import request_projection as module

    monkeypatch.setattr(module, "MAX_FIT_PASSES", 2)
    case = _case("multi_round")
    counter = Counting()
    result = project_request(
        _input(
            case,
            counter=counter,
            lookup=verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"]),
        )
    )
    assert result.refusal is not None
    assert result.refusal.code == "pass_limit_exceeded"
    assert result.passes == 2
    assert counter.calls == 2
    assert result.planned_messages == ()
    assert result.refusal.safe_options


def test_inputs_are_not_mutated_by_projection_or_report():
    case = _case("multi_round")
    lookup = verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"])
    projection_input = _input(case, lookup=lookup)
    before = copy.deepcopy(projection_input)
    project_request(projection_input)
    shadow_report(projection_input)
    assert projection_input == before
    # Copy-on-write: planning must not alias the caller's nested call dicts.
    planned_calls = [
        m["tool_calls"]
        for m in project_request(projection_input).planned_messages
        if "tool_calls" in m
    ]
    for calls in planned_calls:
        for nested in calls:
            assert nested is not case["messages"][2]["tool_calls"][0]


# --- shadow report -------------------------------------------------------------------


def test_shadow_report_is_plain_data_and_compares_baseline():
    case = _case("multi_round")
    lookup = verified_lookup(["art-r1", "art-r2", "art-r3", "art-r4"])
    projection_input = _input(case, lookup=lookup)
    baseline = [{"role": "user", "content": "legacy pre-projection payload"}]
    report = shadow_report(projection_input, baseline=baseline)
    json.dumps(report, sort_keys=True)  # JSON-serializable, no provider objects
    result = project_request(projection_input)
    assert report["shadow"] is True
    assert report["would_fit"] is True
    assert report["pinned_survived"] is True
    assert report["estimated_tokens"] == result.estimated_tokens
    assert report["passes"] == result.passes
    assert report["input_revision"] == result.input_revision
    assert report["retained"] == list(result.retained)
    assert report["baseline"]["messages"] == 1
    assert report["baseline"]["sha256"] != report["planned_sha256"]


# --- input contract validation ---------------------------------------------------------


def test_validation_rejects_duplicate_ids_and_missing_counter():
    units = (
        ProjectionMessage(unit_id="x", role="user", content="a"),
        ProjectionMessage(unit_id="x", role="user", content="b"),
    )
    with pytest.raises(ValueError):
        project_request(ProjectionInput(messages=units, budget_tokens=10, count_tokens=Counting()))
    with pytest.raises(ValueError):
        project_request(
            ProjectionInput(
                messages=(ProjectionMessage(unit_id="y", role="user", content="a"),),
                budget_tokens=10,
            )
        )


def test_validation_rejects_a_tool_result_outside_its_batch_call_ids():
    units = (
        ProjectionMessage(
            unit_id="b",
            role="assistant",
            content="",
            tool_calls=(call("c1"),),
        ),
        ProjectionMessage(unit_id="t", role="tool", content="stray", tool_call_id="c9"),
    )
    with pytest.raises(ValueError):
        project_request(ProjectionInput(messages=units, budget_tokens=100, count_tokens=Counting()))


def test_provider_messages_to_units_preserves_canonical_shape():
    raw = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("c1")],
            "provider_content": [{"type": "thinking", "signature": "sig"}],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]
    units = provider_messages_to_units(
        raw,
        protected_indexes={1},
        artifact_handles={2: "art-1"},
        kinds={2: "history"},
        priorities={2: 3},
        native_key="provider_content",
    )
    assert [unit.unit_id for unit in units] == ["m0", "m1", "m2"]
    assert units[1].protected is True and units[1].unit_id == "m1"
    assert units[1].native_blocks == [{"type": "thinking", "signature": "sig"}]
    assert units[2].artifact_handle == "art-1" and units[2].priority == 3
    result = project_request(
        ProjectionInput(messages=units, budget_tokens=1000, count_tokens=Counting())
    )
    assert [dict(m) for m in result.planned_messages] == raw


# --- no provider surface (static + import isolation) -----------------------------------


def test_module_imports_stdlib_only_and_never_machina_internals():
    banned = {
        "random",
        "time",
        "uuid",
        "secrets",
        "socket",
        "subprocess",
        "os",
        "asyncio",
        "threading",
    }
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top in sys.stdlib_module_names, f"non-stdlib import: {name}"
            assert top not in banned, f"non-deterministic/IO import: {name}"


def test_import_subprocess_loads_zero_provider_modules():
    code = (
        "import sys\n"
        "import machina.core.request_projection\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(REPO),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(proc.stdout.split())
    banned = sorted(
        name
        for name in loaded
        if name.startswith(("machina.core.provider", "machina.core.agent", "machina.core.prompt"))
        or name.split(".")[0] in {"anthropic", "openai", "httpx", "litellm", "ollama", "cohere"}
    )
    assert banned == [], banned
