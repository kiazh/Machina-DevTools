"""models.dev snapshot and provider listing hygiene."""

from __future__ import annotations

from machina.core.provider.models_dev import provider_models


def test_snapshot_has_current_provider_models_without_duplicate_ids():
    for provider in ("openai", "anthropic", "google", "deepseek", "xai", "groq", "mistral", "together", "openrouter"):
        rows = provider_models(provider)
        assert rows, provider
        ids = [row["id"] for row in rows]
        assert len(ids) == len(set(ids)), provider
        assert all(row["displayName"] and row["contextWindow"] for row in rows), provider


def test_grok_subscription_has_multiple_current_models():
    ids = {row["id"] for row in provider_models("grok")}
    assert {"grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1"} <= ids
