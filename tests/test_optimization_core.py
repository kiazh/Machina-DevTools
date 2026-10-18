"""Optimization policy, capability, and canonicalization contracts."""

from __future__ import annotations

import uuid

import pytest

from machina.core.agent_factory import resolve_provider_model
from machina.core.config import config, validate_value
from machina.core.models import LLMResponse
from machina.core.optimization.cache_context import CacheContext
from machina.core.optimization.canonicalizer import (
    canonicalize,
    scope_fingerprint,
    tool_schema_hash,
)
from machina.core.optimization.capabilities import resolve_capabilities
from machina.core.optimization.policy import (
    CircuitBreaker,
    OptimizationMode,
    Reason,
    effective_mode,
    evaluate_semantic_cache,
    prompt_cache_warming_enabled,
)
from machina.core.provider.prompt_cache import (
    _ANTHROPIC_EXPLICIT,
    apply_anthropic_breakpoints,
)
from machina.core.task_descriptor import TASK_CLASSES, build_task_descriptor


def test_optimization_defaults_are_off_and_independent():
    assert config.get("optimization_semantic_cache_mode") == "off"
    assert config.get("optimization_warmers_prompt_cache_mode") == "off"
    assert config.get("optimization_compression_mode") == "off"
    assert config.get("optimization_emergency_disabled") is False
    assert validate_value("optimization_semantic_cache_mode", "shadow") == "shadow"
    assert validate_value("optimization_compression_mode", "wired") == "wired"
    with pytest.raises(ValueError, match="off\\|shadow\\|wired"):
        validate_value("optimization_semantic_cache_mode", "on")


def test_emergency_switch_overrides_every_mode(monkeypatch):
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    monkeypatch.setattr(config, "optimization_emergency_disabled", True)
    assert effective_mode("semantic_cache") is OptimizationMode.OFF
    assert prompt_cache_warming_enabled() is False


def test_prompt_cache_mode_preserves_legacy_flag(monkeypatch):
    monkeypatch.setattr(config, "optimization_warmers_prompt_cache_mode", "off")
    monkeypatch.setattr(config, "prompt_cache_enabled", True)
    assert prompt_cache_warming_enabled() is True
    monkeypatch.setattr(config, "optimization_warmers_prompt_cache_mode", "shadow")
    assert prompt_cache_warming_enabled() is False
    monkeypatch.setattr(config, "optimization_warmers_prompt_cache_mode", "wired")
    assert prompt_cache_warming_enabled() is True


def test_wired_prompt_cache_enforces_minimum_prefix(monkeypatch):
    monkeypatch.setattr(config, "optimization_warmers_prompt_cache_mode", "wired")
    monkeypatch.setattr(config, "optimization_warmers_min_prefix_tokens", 4096)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    assert apply_anthropic_breakpoints(
        payload,
        system="short system",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        profile=_ANTHROPIC_EXPLICIT,
    ) == (False, False, False)
    monkeypatch.setattr(config, "optimization_warmers_min_prefix_tokens", 0)
    assert (
        apply_anthropic_breakpoints(
            payload,
            system="short system",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            profile=_ANTHROPIC_EXPLICIT,
        )[0]
        is True
    )


def test_capability_profile_uses_verified_anthropic_string_route():
    profile = resolve_capabilities("anthropic", "claude-sonnet-4")
    assert profile.provider == "anthropic"
    assert profile.prompt_cache.mode == "explicit"
    assert profile.cache_ttl_seconds == 300
    assert profile.context_verified is True
    # No built-in price snapshot for this model: paid warmers must stay off.
    assert profile.pricing_verified is False
    assert profile.verified_complete is False
    assert profile.supports_paid_warmers is False


def test_capability_unknown_model_fails_closed():
    profile = resolve_capabilities("custom", "totally-unknown-model")
    assert profile.prompt_cache.mode == "none"
    assert profile.verified_complete is False


def test_scope_fingerprint_is_session_scoped_and_stable():
    session = uuid.uuid4()
    base = CacheContext.for_auxiliary(
        session_id=session,
        agent_id="harness",
        provider="deepseek",
        model="deepseek-chat",
        purpose="compression_summary",
        messages=[{"role": "user", "content": "Summarize this"}],
        tools=[],
    )
    same = CacheContext.for_auxiliary(
        session_id=session,
        agent_id="harness",
        provider="deepseek",
        model="deepseek-chat",
        purpose="compression_summary",
        messages=[{"role": "user", "content": "Summarize this"}],
        tools=[],
    )
    other = CacheContext.for_auxiliary(
        session_id=uuid.uuid4(),
        agent_id="harness",
        provider="deepseek",
        model="deepseek-chat",
        purpose="compression_summary",
        messages=[{"role": "user", "content": "Summarize this"}],
        tools=[],
    )
    first = scope_fingerprint(base, embedding_space="local:test:1536", schema_hash="")
    second = scope_fingerprint(same, embedding_space="local:test:1536", schema_hash="")
    third = scope_fingerprint(other, embedding_space="local:test:1536", schema_hash="")
    assert first == second
    assert first != third


def test_canonicalizer_ignores_wire_metadata_and_tool_markers():
    messages = [
        {"role": "system", "content": "Rules", "timestamp": "2026-01-01"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello", "cache_control": {}}],
        },
    ]
    first = canonicalize(messages)
    second = canonicalize(
        [
            {"role": "system", "content": "Rules", "request_id": "different"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"x": 1}}
                ],
            },
        ]
    )
    assert first.request_hash == second.request_hash
    assert "cache_control" not in first.text


def test_tool_schema_hash_changes_with_schema():
    class Tool:
        def __init__(self, name, parameters):
            self.name = name
            self.description = ""
            self.parameters = parameters

    left = tool_schema_hash([Tool("read", {"type": "object"})])
    right = tool_schema_hash([Tool("read", {"type": "object", "required": ["path"]})])
    assert left != right


def _eligible_context(**overrides):
    values = {
        "session_id": uuid.uuid4(),
        "agent_id": "harness",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "purpose": "compression_summary",
        "messages": [{"role": "user", "content": "Summarize this short text"}],
        "tools": [],
        "temperature": 0.0,
    }
    values.update(overrides)
    return CacheContext.for_auxiliary(**values)


def test_semantic_cache_eligibility_matrix(monkeypatch):
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "wired")
    canonical = canonicalize([{"role": "user", "content": "Summarize this short text"}])
    allowed = evaluate_semantic_cache(
        _eligible_context(), canonical, tools_present=False
    )
    assert allowed.allowed is True

    def reason(context, *, tools=False, text="Summarize this short text"):
        result = evaluate_semantic_cache(
            context,
            canonicalize([{"role": "user", "content": text}]),
            tools_present=tools,
        )
        return result.reasons

    assert Reason.TOOL_REQUEST in reason(_eligible_context(), tools=True)
    assert Reason.TEMPERATURE_TOO_HIGH in reason(_eligible_context(temperature=0.8))
    assert Reason.FRESHNESS_SENSITIVE in reason(
        _eligible_context(), text="What is the latest news right now?"
    )
    assert Reason.SECRET_DETECTED in reason(
        _eligible_context(), text="Use sk-abcdefghijklmnopqrstuvwxyz12345"
    )
    assert Reason.PII_DETECTED in reason(
        _eligible_context(), text="Contact someone@example.com about this"
    )
    assert Reason.CODE_CONTENT in reason(
        _eligible_context(), text="Review ```python\nprint(1)\n```"
    )
    assert Reason.REQUEST_TOO_LONG in reason(_eligible_context(), text="x" * 2100)


def test_shadow_mode_computes_but_never_serves(monkeypatch):
    monkeypatch.setattr(config, "optimization_semantic_cache_mode", "shadow")
    canonical = canonicalize([{"role": "user", "content": "Summarize this short text"}])
    decision = evaluate_semantic_cache(
        _eligible_context(), canonical, tools_present=False
    )
    assert decision.allowed is True
    assert decision.should_compute is True
    assert decision.should_serve is False


def test_circuit_breaker_opens_and_recovers(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    assert breaker.allow("x") is True
    breaker.record_failure("x")
    assert breaker.allow("x") is True
    breaker.record_failure("x")
    assert breaker.allow("x") is False
    breaker.record_success("x")
    assert breaker.allow("x") is True


def test_response_cache_source_post_init_keeps_legacy_boolean():
    response = LLMResponse(content="x", model="m", cache_hit=True)
    assert response.cache_source == "exact"
    semantic = LLMResponse(content="x", model="m", cache_source="semantic")
    assert semantic.cache_hit is True


def test_task_descriptor_never_selects_a_model(monkeypatch):
    """All deterministic task classes route to the same explicit model."""
    monkeypatch.setattr(config, "provider", "openai")
    monkeypatch.setattr(config, "model", "gpt-4.1")
    for task_class in TASK_CLASSES:
        descriptor = build_task_descriptor(
            f"Please handle a {task_class} request", task_state={}, goal=None
        )
        exported = descriptor.to_dict()
        assert "model" not in exported
        assert "provider" not in exported
        assert resolve_provider_model() == ("openai", "gpt-4.1")
