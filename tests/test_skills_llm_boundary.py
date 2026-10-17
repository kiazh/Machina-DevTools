"""The query-time skill picker never generates metadata (product boundary).

Metadata generation (the only LLM use on the skills path) happens at install /
learn / enrich time. Search and automatic selection are the deterministic
picker and must keep working with generation hard-failing: a provider outage
can never make the catalogue silently empty or change rankings.
"""

from __future__ import annotations


def test_search_and_automatic_never_call_metadata_generation(monkeypatch):
    from machina.skills import enrich, runtime
    from machina.skills.registry import SkillRegistry

    def fail_sync(*args, **kwargs):
        raise AssertionError("warm search must never generate skill metadata")

    async def fail_async(*args, **kwargs):
        raise AssertionError("warm search must never generate skill metadata")

    monkeypatch.setattr(enrich, "ensure_skill_metadata_sync", fail_sync)
    monkeypatch.setattr(enrich, "ensure_skill_metadata", fail_async)

    registry = SkillRegistry()
    registry.load_all()

    result = runtime.search_skills(registry, "make this website", limit=10)
    assert result.candidates
    assert result.candidates[0].name == "web-frontend"

    automatic = runtime.automatic_skills(registry, "make this website")
    assert automatic
    assert automatic[0].name == "web-frontend"

    # The routing regression pair (worker retry vs provider retry) also runs
    # generation-free on the real catalogue.
    worker = [skill.name for skill in runtime.automatic_skills(registry, "add retry logic")]
    provider = [
        skill.name
        for skill in runtime.automatic_skills(registry, "retries for the openrouter provider")
    ]
    assert "background-jobs" in worker[:3]
    assert provider[0] == "llm-provider", provider[:3]
