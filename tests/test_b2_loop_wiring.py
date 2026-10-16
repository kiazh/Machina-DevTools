"""B2 — wire both live loops to the shared request projector (spec §14, §17-B2).

Coverage map (AC-14.5/§17-B2, constraints C02/C05/C10/C12/C16):

- C10: ``run()`` and ``run_stream()`` call ONE shared seam
  (``machina.core.request_gate.evaluate_request``); captured provider payloads
  and gate decisions are identical between the loops.
- §14.5: a multi-round overflowing fixture fits after projection; the latest
  coherent group survives verbatim and pins/constraints survive exactly.
- Refusal: zero provider calls and the same structured capacity failure the
  legacy guard emits, in both loops, with the audit event.
- C16: flag off -> shadow only; outgoing provider payloads are byte-identical
  to a pre-B2 passthrough run (captured-payload comparison).
- B1 seam contract: ``provider_messages_to_units`` + awaited artifact lookups
  prefetched before the pure projector; reprojection mutates nothing and
  replays no effects.

The loops are driven through a fake provider that captures the exact outgoing
payloads; session/context plumbing is mocked (same harness shape as
``test_te_views_integration``), so no database or network is required.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from machina.core.agent import BaseReActAgent, ReActAgent
from machina.core.config import DEFAULTS, Config, config, validate_value
from machina.core.models import LLMResponse, Session, StreamEvent
from machina.core.request_gate import RequestGate, evaluate_request
from machina.tools.base import ToolRegistry

REPO = Path(os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina")
CORPUS = Path(__file__).resolve().parent / "data" / "request_projection" / "corpus.json"

GOAL = "Ship the B2 loop wiring safely."
CONSTRAINT = "Do not edit ui/."
FILLER = "deterministic round filler line %03d for the B2 loop wiring fixture"
ROUNDS = 4


# ─── fake provider: captures the exact outgoing payloads ─────────────────────


class _ScriptedProvider:
    """Deterministic provider; every call snapshots its kwargs."""

    model = "fake-b2"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    async def stream_complete(self, **kwargs):
        response = await self.complete(**kwargs)
        if response.content:
            yield StreamEvent(type="text", content=response.content)
        yield StreamEvent(type="done", response=response)


class _RejectingProvider:
    """Fails the test if the seam ever tries to call a provider itself."""

    model = "reject"

    async def complete(self, **kwargs):
        raise AssertionError("provider must not be called during projection")

    async def stream_complete(self, **kwargs):
        raise AssertionError("provider must not be called during projection")
        yield  # pragma: no cover - generator protocol


def _round_response(index: int) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake-b2",
        usage={"total_tokens": 1},
        tool_calls=[
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "session_recall",
                    "arguments": json.dumps({"query": f"round {index}"}),
                },
            }
        ],
    )


def _final_response() -> LLMResponse:
    return LLMResponse(content="all rounds complete", model="fake-b2", usage={"total_tokens": 1})


def _script() -> list[LLMResponse]:
    return [_round_response(i) for i in range(1, ROUNDS + 1)] + [_final_response()]


# ─── fake artifact store: verified views + read-only lookup ──────────────────


class _VerifiedStore:
    """Retains stored content, answers the projector's lookup contract."""

    def __init__(self) -> None:
        self.stored: dict[str, str] = {}
        self.lookups: list[str] = []
        self.store_calls = 0

    async def store(self, kind, content, **kwargs):
        self.store_calls += 1
        # Deterministic handle: identical content -> identical handle, so two
        # scripted runs produce byte-identical payloads.
        handle = "hand-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        self.stored[handle] = content
        return {"artifact_id": handle, "kind": kind, "content_bytes": len(content)}

    async def lookup(self, handle, *, session_id, agent_id):
        self.lookups.append(str(handle))
        if str(handle) not in self.stored:
            return None
        return {
            "available": True,
            "complete": True,
            "handle": str(handle),
            "revision": f"rev-{handle}",
            "summary_view": f"verified summary for {handle}",
        }


class _VerifiedLookupOnly:
    """Verified lookup for handles it was never asked to retain (corpus ids)."""

    def __init__(self) -> None:
        self.lookups: list[str] = []

    async def lookup(self, handle, *, session_id, agent_id):
        self.lookups.append(str(handle))
        return {
            "available": True,
            "complete": True,
            "handle": str(handle),
            "revision": f"rev-{handle}",
            "summary_view": f"verified summary for {handle}",
        }


# ─── loop harness (mocked session/context, stub recall tool) ─────────────────


def _stub_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(
        name="session_recall",
        description="stub recall for the B2 fixture",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    async def session_recall(query: str = "") -> str:
        lines = [FILLER % i for i in range(48)]
        return f"{query}: " + "\n".join(lines)

    return reg


def _wire_session(monkeypatch, tmp_path, *, goal: str | None = None, state: dict | None = None):
    monkeypatch.setenv("MACHINA_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setattr(config, "workspace_root", str(tmp_path / "work"))
    monkeypatch.setattr(config, "execution_mode", "ask")
    # Small output reservation: the fixture's overflow then comes from history.
    monkeypatch.setattr(config, "max_tokens", 64)
    for field in ("memory_enabled", "rag_enabled", "personality_growth_enabled"):
        monkeypatch.setattr(config, field, False)
    session = Session(
        id=uuid.uuid4(), context_budget=8000, goal=goal, state=dict(state or {})
    )
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    monkeypatch.setattr("machina.core.agent.session_manager.update_activity", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("machina.core.agent.context_manager.add_chunks_batch", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(BaseReActAgent, "_reserve_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(BaseReActAgent, "_finish_usage", AsyncMock())
    return session


def _count(messages, tool_defs, provider) -> int:
    from machina.core.prompt import _request_context_tokens

    return _request_context_tokens(messages, tool_defs, provider)


def _corpus_messages(name: str = "multi_round") -> list[dict]:
    """Provider-shaped messages from the B1 corpus, with live-shaped handles."""
    case = json.loads(CORPUS.read_text())["cases"][name]
    messages: list[dict] = []
    for message in case["messages"]:
        converted = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            converted["tool_calls"] = copy.deepcopy(message["tool_calls"])
        if message.get("tool_call_id"):
            converted["tool_call_id"] = message["tool_call_id"]
        if message.get("artifact_handle"):
            converted["content"] += (
                f"\n[evidence retained as artifact {message['artifact_handle']} "
                f"({len(converted['content'])} chars); artifact_read pages the exact original]"
            )
        messages.append(converted)
    return messages


def _freeze_budget(monkeypatch, value: int) -> None:
    import machina.core.request_gate as gate_module

    monkeypatch.setattr(
        gate_module, "_effective_turn_budget", lambda session, provider: value
    )


def _tool_result(messages: list[dict], call_id: str) -> dict:
    return next(m for m in messages if m.get("tool_call_id") == call_id)


# ─── config registration (A4 lesson: typed field is mandatory for save()) ────


def test_request_projection_flag_is_registered_typed_and_savable(tmp_path):
    assert "request_projection_enabled" in DEFAULTS
    assert DEFAULTS["request_projection_enabled"] is True
    assert Config().request_projection_enabled is True
    assert config.get("request_projection_enabled") is True
    assert validate_value("request_projection_enabled", "on") is True
    assert validate_value("request_projection_enabled", "off") is False
    with pytest.raises(ValueError):
        validate_value("request_projection_enabled", "maybe")

    config.set("request_projection_enabled", True)
    assert config.get("request_projection_enabled") is True
    target = tmp_path / "config.yaml"
    config.save(target)  # must not raise (typed field present)
    payload = yaml.safe_load(target.read_text())
    assert payload["request_projection_enabled"] is True


# ─── C16: flag off -> shadow only, payloads byte-identical to pre-B2 ─────────


@pytest.mark.asyncio
async def test_flag_off_logs_bounded_shadow_and_payloads_stay_byte_identical(
    tmp_path, monkeypatch, caplog
):
    session = _wire_session(monkeypatch, tmp_path, goal=GOAL, state={"constraints": [CONSTRAINT]})
    monkeypatch.setattr("machina.core.agent.registry", _stub_registry())
    store = _VerifiedStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    monkeypatch.setattr("machina.core.request_gate.artifact_store", store)
    # Rollout gate: the production default is now ON, so this shadow-parity
    # test states the flag it exercises explicitly.
    config.set("request_projection_enabled", False)

    caplog.set_level(logging.INFO, logger="agent.request_projection.shadow")
    shadow_provider = _ScriptedProvider(_script())
    shadow_response = await ReActAgent(provider=shadow_provider).run(
        session.id, "go ahead", verbose=False
    )
    assert shadow_response.outcome == "completed"
    assert len(shadow_provider.calls) == ROUNDS + 1

    # The shadow report ran and logged bounded, redacted fields only.
    records = [r for r in caplog.records if r.name == "agent.request_projection.shadow"]
    assert len(records) == ROUNDS + 1
    for record in records:
        message = record.getMessage()
        assert "would_fit=" in message
        assert "estimated_tokens=" in message
        assert "basis=" in message
    assert "refusal=none" in caplog.text
    assert GOAL not in caplog.text
    assert CONSTRAINT not in caplog.text
    assert "deterministic round filler" not in caplog.text

    # Flag off: no pins were injected and the payloads are the unprojected list.
    assert all(
        call["messages"][0]["role"] == "user" for call in shadow_provider.calls
    )

    # Pre-B2 reference: the seam bypassed entirely (old guard-free passthrough).
    async def _passthrough(messages, **kwargs):
        return RequestGate(messages=messages, effective_budget=10**9, request_tokens=0)

    monkeypatch.setattr("machina.core.agent.evaluate_request", _passthrough)
    pre_provider = _ScriptedProvider(_script())
    pre_response = await ReActAgent(provider=pre_provider).run(
        session.id, "go ahead", verbose=False
    )
    assert pre_response.content == shadow_response.content
    assert len(pre_provider.calls) == ROUNDS + 1

    def _dumped(calls: list[dict]) -> str:
        return json.dumps(calls, sort_keys=True, default=repr)

    assert _dumped(shadow_provider.calls) == _dumped(pre_provider.calls)
    assert shadow_provider.calls == pre_provider.calls


# ─── pins: existing state only, never invented (§14.3) ─────────────────────


def test_pins_come_from_existing_state_and_never_invent_it():
    from machina.core.request_gate import _collect_pins

    empty = Session(id=uuid.uuid4())
    assert _collect_pins(empty, "go ahead").items() == ()

    session = Session(
        id=uuid.uuid4(),
        goal=GOAL,
        state={
            "constraints": [CONSTRAINT],
            "active_class": "implement",
            "active_families": ["coding"],
            "stagnation_stop": {"reason": "stagnant", "stop_text": "stopped: no new coverage"},
        },
    )
    pins = [(pin.kind, pin.text) for pin in _collect_pins(session, "fix it, do not edit anything").items()]
    assert ("objective", GOAL) in pins
    assert ("constraints", CONSTRAINT) in pins  # durable task-state constraint, exact
    assert ("constraints", "no-edit") in pins  # detected from the current request
    assert ("read_only_scope", "operation_scope=read-only") in pins
    assert ("blockers", "stopped: no new coverage") in pins  # durable failed-attempt record
    assert any(kind == "task_state" and text.startswith("task_class=") for kind, text in pins)
    # No task-state keys -> no invented task-state pin (covered by the empty session).


# ─── §14.5: wired multi-round overflow fits; pins + recent group exact ───────


@pytest.mark.asyncio
async def test_wired_overflow_fits_with_pins_and_recent_group_in_both_loops(
    tmp_path, monkeypatch
):
    session = _wire_session(
        monkeypatch,
        tmp_path,
        goal=GOAL,
        state={
            "constraints": [CONSTRAINT],
            "active_class": "implement",
            "active_families": ["coding"],
        },
    )
    monkeypatch.setattr("machina.core.agent.registry", _stub_registry())
    store = _VerifiedStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    monkeypatch.setattr("machina.core.request_gate.artifact_store", store)

    # Probe (flag off, huge budget): capture the full unprojected final request.
    import machina.core.request_gate as gate_module

    budget = {"value": 10**9}
    monkeypatch.setattr(
        gate_module, "_effective_turn_budget", lambda session, provider: budget["value"]
    )
    probe_provider = _ScriptedProvider(_script())
    probe = await ReActAgent(provider=probe_provider).run(session.id, "go ahead", verbose=False)
    assert probe.outcome == "completed"
    assert len(probe_provider.calls) == ROUNDS + 1
    full_messages = probe_provider.calls[-1]["messages"]
    tool_defs = probe_provider.calls[-1]["tools"]
    full_tokens = _count(full_messages, tool_defs, probe_provider)

    # Wired run: the same script with a budget the unprojected list overflows.
    budget["value"] = full_tokens - 300
    config.set("request_projection_enabled", True)
    wired_provider = _ScriptedProvider(_script())
    wired = await ReActAgent(provider=wired_provider).run(session.id, "go ahead", verbose=False)
    assert wired.outcome == "completed"
    assert len(wired_provider.calls) == ROUNDS + 1

    sent = wired_provider.calls[-1]["messages"]
    sent_tokens = _count(sent, wired_provider.calls[-1]["tools"], wired_provider)
    assert full_tokens > budget["value"] >= sent_tokens
    assert sent_tokens < full_tokens - 200  # the projection actually reduced

    # Pins are first, unevaluated exact strings (objective then constraints).
    assert sent[0] == {"role": "system", "content": GOAL}
    assert sent[1] == {"role": "system", "content": CONSTRAINT}
    assert any(m["content"].startswith("task_class=") for m in sent)

    # The latest coherent group survives verbatim (plain content, not a view).
    latest = [m for m in sent if m.get("tool_call_id") == f"call_{ROUNDS}"]
    original_latest = [m for m in full_messages if m.get("tool_call_id") == f"call_{ROUNDS}"]
    assert len(latest) == 1 and len(original_latest) == 1
    assert latest[0]["content"] == original_latest[0]["content"]
    assert "[machina view:" not in latest[0]["content"]

    # Older bulky rounds are replaced by verified recoverable views.
    views = [m for m in sent if "[machina view:" in str(m.get("content", ""))]
    assert views, "older rounds must be projected to verified views"
    assert all("verified summary for hand-" in m["content"] for m in views)
    view_calls = {m["tool_call_id"] for m in views}
    assert view_calls and view_calls <= {
        f"call_{i}" for i in range(1, ROUNDS)
    }
    # No torn batches: every kept assistant call keeps its result in order.
    for index in range(1, ROUNDS + 1):
        call_id = f"call_{index}"
        assistant = [
            m
            for m in sent
            if any(tc.get("id") == call_id for tc in m.get("tool_calls") or ())
        ]
        result = [m for m in sent if m.get("tool_call_id") == call_id]
        assert len(assistant) == len(result)
        if result:
            assert assistant[0]["tool_calls"][0]["id"] == call_id

    # Streaming loop: same policy -> identical final gate decision.
    stream_provider = _ScriptedProvider(_script())
    events = [
        event
        async for event in ReActAgent(provider=stream_provider).run_stream(
            session.id, "go ahead", verbose=False
        )
    ]
    done = [event for event in events if event.type == "done"]
    assert done[-1].response.content == wired.content
    assert len(stream_provider.calls) == ROUNDS + 1
    assert stream_provider.calls[-1]["messages"] == sent


# ─── refusal: zero provider calls + structured failure in both loops ────────


@pytest.mark.asyncio
async def test_wired_refusal_is_structured_with_no_provider_call_in_both_loops(
    tmp_path, monkeypatch
):
    session = _wire_session(monkeypatch, tmp_path, goal=GOAL, state={"constraints": [CONSTRAINT]})
    monkeypatch.setattr("machina.core.agent.registry", _stub_registry())
    _freeze_budget(monkeypatch, 1)
    config.set("request_projection_enabled", True)

    audits: list[tuple] = []
    monkeypatch.setattr(
        "machina.core.request_gate.audit_log", lambda *args, **kwargs: audits.append((args, kwargs))
    )

    run_provider = _ScriptedProvider(_script())
    response = await ReActAgent(provider=run_provider).run(session.id, "go ahead", verbose=False)
    assert run_provider.calls == []
    assert response.outcome == "failed"
    assert "Pinned content alone exceeds the effective request budget." in response.content
    assert "Safe options:" in response.content
    assert "larger context window" in response.content
    assert len(audits) == 2
    events = [args[0] for args, _kwargs in audits]
    assert events.count("agent_request_context_exceeded") == 1
    assert events.count("request_projection_decision") == 1
    _event, kwargs = next(a for a in audits if a[0] == ("agent_request_context_exceeded",))
    assert kwargs["refusal"] == "pins_exceed_capacity"
    assert kwargs["context_window"] == 1
    assert kwargs["request_tokens"] > 1
    # §14.7: the same refusal is observable as a bounded decision record.
    _event, decision = next(a for a in audits if a[0] == ("request_projection_decision",))
    assert decision["mode"] == "wired"
    assert decision["would_fit"] is False
    assert decision["refusal"] == "pins_exceed_capacity"
    assert decision["degraded"] is False
    assert decision["effective_budget_tokens"] == 1
    assert decision["request_tokens"] > 1

    stream_provider = _ScriptedProvider(_script())
    events = [
        event
        async for event in ReActAgent(provider=stream_provider).run_stream(
            session.id, "go ahead", verbose=False
        )
    ]
    assert stream_provider.calls == []
    done = [event for event in events if event.type == "done"]
    assert done and done[-1].response == response  # identical gates, same shape


# ─── seam-level purity: reprojection mutates nothing, replays no effects ────


@pytest.mark.asyncio
async def test_reprojection_is_pure_and_prefetches_lookups_before_projecting(
    tmp_path, monkeypatch
):
    messages = _corpus_messages("multi_round")
    before = copy.deepcopy(messages)
    store = _VerifiedLookupOnly()
    monkeypatch.setattr("machina.core.request_gate.artifact_store", store)
    full = _count(messages, [], _RejectingProvider())
    _freeze_budget(monkeypatch, full - 100)
    config.set("request_projection_enabled", True)

    kwargs = {
        "session": Session(id=uuid.uuid4(), goal=GOAL, state={"constraints": [CONSTRAINT]}),
        "provider": _RejectingProvider(),
        "tool_defs": [],
        "session_id": uuid.uuid4(),
        "agent_id": "harness",
        "user_message": "go ahead",
    }
    first = await evaluate_request(messages, **kwargs)
    second = await evaluate_request(messages, **kwargs)

    assert first.failure is None and first.projected is True
    assert first.messages == second.messages  # deterministic reprojection
    assert messages == before  # no mutation of the canonical request
    assert first.messages[0] == {"role": "system", "content": GOAL}
    assert first.messages[1] == {"role": "system", "content": CONSTRAINT}
    # Prefetch happened once per unique handle per call (read-only, idempotent).
    assert store.lookups.count("art-r1") == 2
    assert len(store.lookups) == 8
    # The newest group survives verbatim; older rounds became verified views.
    newest = [m for m in first.messages if m.get("tool_call_id") == "call_5"]
    assert len(newest) == 1
    assert newest[0]["content"] == messages[-1]["content"]
    assert any("[machina view:" in str(m.get("content", "")) for m in first.messages)
    assert first.request_tokens <= first.effective_budget


# ─── §14.3: instruction text is never dropped or demoted ────────────────────


@pytest.mark.asyncio
async def test_instruction_survives_projection_byte_for_byte_and_is_not_demoted(
    tmp_path, monkeypatch
):
    """A projected request keeps its instruction message byte-for-byte.

    The current turn's instruction is the first ``required`` unit. Even when
    the payload is over budget and older units are projected to verified views,
    the instruction must survive verbatim and render *ahead* of derived
    scope/state pins (read-only scope here) — projection can never drop or
    demote instruction text behind non-governing pins (§14.3).
    """
    messages = _corpus_messages("multi_round")
    instruction = messages[0]  # the current turn's instruction (required unit)
    assert instruction["role"] == "system" and "tool_calls" not in instruction

    monkeypatch.setattr("machina.core.request_gate.artifact_store", _VerifiedLookupOnly())
    full = _count(messages, [], _RejectingProvider())
    _freeze_budget(monkeypatch, full - 100)  # force a reduction
    config.set("request_projection_enabled", True)

    # This request carries a derived (non-governing) read-only-scope pin.
    gate = await evaluate_request(
        messages,
        session=Session(id=uuid.uuid4()),
        provider=_RejectingProvider(),
        tool_defs=[],
        session_id=uuid.uuid4(),
        agent_id="harness",
        user_message="Where is the persona stored?",
    )

    assert gate.failure is None and gate.projected is True
    # Instruction is byte-for-byte and first; the derived pin follows it.
    assert gate.messages[0] == instruction
    assert dict(instruction) in [dict(m) for m in gate.messages]
    assert gate.messages[1] == {"role": "system", "content": "operation_scope=read-only"}
    # It is a real projection, not a passthrough: older rounds became views.
    assert any("[machina view:" in str(m.get("content", "")) for m in gate.messages)


@pytest.mark.asyncio
async def test_seam_keeps_provider_native_blocks_protected(tmp_path, monkeypatch):
    native = [{"type": "thinking", "thinking": "signed", "signature": "sig-b2"}]
    big = FILLER % 0 + "\n" + "\n".join(FILLER % i for i in range(40))
    handle = "11111111-2222-3333-4444-555555555555"
    messages = [
        {"role": "user", "content": "continue the signed loop"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_n1",
                    "type": "function",
                    "function": {"name": "session_recall", "arguments": "{}"},
                }
            ],
            "provider_content": native,
        },
        {
            "role": "tool",
            "content": big + f"\n[evidence retained as artifact {handle} (retained)]",
            "tool_call_id": "call_n1",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_n2",
                    "type": "function",
                    "function": {"name": "session_recall", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "latest verdict: 2 passed", "tool_call_id": "call_n2"},
    ]
    before = copy.deepcopy(messages)
    store = _VerifiedLookupOnly()
    monkeypatch.setattr("machina.core.request_gate.artifact_store", store)
    full = _count(messages, [], _RejectingProvider())
    _freeze_budget(monkeypatch, full - 60)
    config.set("request_projection_enabled", True)

    gate = await evaluate_request(
        messages,
        session=Session(id=uuid.uuid4(), goal=GOAL),
        provider=_RejectingProvider(),
        tool_defs=[],
        session_id=uuid.uuid4(),
        agent_id="harness",
        user_message="continue",
    )
    assert gate.failure is None and gate.projected is True
    assert messages == before
    assistant = next(
        m for m in gate.messages if any(tc.get("id") == "call_n1" for tc in m.get("tool_calls") or ())
    )
    assert assistant["provider_content"] == native
    assert assistant["provider_content"] is not native  # deep-copied, never aliased
    projected = _tool_result(gate.messages, "call_n1")
    assert "verified summary for" in projected["content"]
    # The newest group survives.
    assert _tool_result(gate.messages, "call_n2")["content"] == "latest verdict: 2 passed"


# ─── C10: one shared seam, no per-loop drift ─────────────────────────────────


def test_both_loops_call_the_single_shared_projection_seam():
    import machina.core.agent as agent_mod

    for loop in (agent_mod.ReActAgent.run, agent_mod.ReActAgent.run_stream):
        source = inspect.getsource(loop)
        assert "evaluate_request(" in source
        # Neither loop carries its own projection policy or capacity math.
        assert "project_request(" not in source
        assert "shadow_report(" not in source
        assert "_effective_turn_budget(" not in source
        assert "_request_context_tokens(" not in source

    agent_source = (REPO / "machina" / "core" / "agent.py").read_text(encoding="utf-8")
    assert agent_source.count("evaluate_request(") == 2
