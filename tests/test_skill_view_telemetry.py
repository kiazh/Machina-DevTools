"""Combined #21: gateway skill previews record views in skill telemetry."""

from __future__ import annotations

from types import SimpleNamespace

from machina.gateway.features import skills as gateway_skills
from machina.skills import registry as registry_module
from machina.skills.registry import SkillRegistry


async def test_show_and_section_record_a_skill_view(tmp_path, monkeypatch):
    registry = SkillRegistry(tmp_path / "skills")
    registry.create_skill("demo-view", "Deploy steps", "Steps to deploy safely.")
    monkeypatch.setattr(registry_module, "skill_registry", registry)
    monkeypatch.setattr(registry, "get_skill_content", lambda _name: "Steps to deploy safely.")
    gateway = SimpleNamespace(require_db=lambda: None)

    await gateway_skills.skills_show(gateway, {"name": "demo-view"})
    assert registry.get("demo-view").view_count == 1

    await gateway_skills.skills_section(
        gateway, {"skillId": "demo-view", "offset": 0, "budget": 20}
    )
    assert registry.get("demo-view").view_count == 2
