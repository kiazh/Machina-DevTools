"""Provider/model resolution keeps model provenance aligned with the selected provider."""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core import agent_factory
from machina.core.agent_def import AgentDef, agent_registry
from machina.core.config import config
from machina.core.models import Session


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.setattr(
        config,
        "get",
        lambda key: {
            "provider": "openrouter",
            "model": "openrouter/free",
            "context_budget": 8000,
        }.get(key),
    )
    monkeypatch.setattr(agent_factory, "_memory_enabled", lambda: False)
    monkeypatch.setattr(agent_factory, "_shared_rag_pipeline", AsyncMock(return_value=None))
    monkeypatch.setattr("machina.core.agent.ReActAgent", lambda **kwargs: SimpleNamespace(**kwargs))


@pytest.mark.parametrize(
    (
        "session_provider",
        "session_model",
        "definition_provider",
        "definition_model",
        "override",
        "expected",
    ),
    [
        ("openrouter", "openrouter/free", None, None, "anthropic", ("anthropic", None)),
        (
            "openrouter",
            "openrouter/free",
            "anthropic",
            "claude",
            None,
            ("openrouter", "openrouter/free"),
        ),
        (None, None, "anthropic", "claude", None, ("anthropic", "claude")),
        ("anthropic", None, "openrouter", "openrouter/free", None, ("anthropic", None)),
        (
            "openrouter",
            "openrouter/free",
            "anthropic",
            "claude",
            "anthropic",
            ("anthropic", "claude"),
        ),
    ],
)
async def test_factory_only_inherits_models_for_the_winning_provider(
    routing,
    monkeypatch,
    session_provider,
    session_model,
    definition_provider,
    definition_model,
    override,
    expected,
):
    definition = AgentDef("harness", provider=definition_provider, model=definition_model)
    monkeypatch.setattr(agent_registry, "get", AsyncMock(return_value=definition))
    captured = []

    def provider(**kwargs):
        captured.append((kwargs["provider"], kwargs["model"]))
        return SimpleNamespace(model="test-model")

    monkeypatch.setattr("machina.core.provider.get_provider", provider)
    session = Session(id=uuid.uuid4(), provider=session_provider, model=session_model)
    await agent_factory.build_agent_for_session(session, provider_override=override)
    assert captured == [expected]
    from machina import services

    assert await services._effective_session_route(session, provider=override) == expected


@pytest.mark.parametrize("transport", ["http", "gateway"])
async def test_new_session_applies_definition_defaults_before_global_pins(
    routing, monkeypatch, transport
):
    from machina.api.routes.models import CreateSessionRequest
    from machina.api.routes.sessions import create_session
    from machina.core.session import session_manager
    from machina.gateway.server import Gateway

    monkeypatch.setattr(
        agent_registry,
        "get",
        AsyncMock(return_value=AgentDef("harness", provider="anthropic", model="claude")),
    )
    create = AsyncMock(return_value=Session(id=uuid.uuid4()))
    monkeypatch.setattr(session_manager, "create", create)
    monkeypatch.setattr("machina.gateway.features.mode.grant_full_default", AsyncMock())
    if transport == "http":
        await create_session(CreateSessionRequest())
    else:
        gateway = SimpleNamespace(
            require_db=lambda: None,
            _config_lock=asyncio.Lock(),
            provider="openrouter",
            model="openrouter/free",
        )
        await Gateway._session_create(gateway, {})
    assert create.await_args.kwargs["provider"] == "anthropic"
    assert create.await_args.kwargs["model"] == "claude"


def test_route_default_and_explicit_model_provenance(routing):
    resolve = agent_factory.resolve_provider_model
    assert resolve(default_provider="anthropic") == ("anthropic", None)
    assert resolve(provider_override="anthropic", model_override="explicit") == (
        "anthropic",
        "explicit",
    )
    assert resolve(
        definition=AgentDef("harness", model="global-family"), provider_override="anthropic"
    ) == ("anthropic", None)
