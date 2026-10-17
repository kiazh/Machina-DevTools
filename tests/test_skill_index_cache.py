"""Combined #52: the disabled-skill index variant is cached by signature."""

from __future__ import annotations

from machina.skills import runtime
from machina.skills.registry import SkillRegistry


def test_include_excluded_index_is_cached(tmp_path):
    registry = SkillRegistry(tmp_path / "skills")
    registry.create_skill("demo-index", "Deploy steps", "Steps to deploy safely.")
    registry.load_all()
    runtime._INDEXES.clear()

    runtime.search_skills(registry, "deploy", include_excluded=True)
    assert True in runtime._INDEXES
    assert runtime._INDEXES[True][1] is not None
