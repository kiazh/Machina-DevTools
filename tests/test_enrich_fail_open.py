"""Combined #49: enrichment persistence failures stay fail-open."""

from __future__ import annotations

import pytest

from machina.skills.enrich import ensure_skill_metadata
from machina.skills.metadata import SkillMetadata


async def test_metadata_persistence_value_error_is_fail_open(monkeypatch):
    import machina.skills.enrich as enrich

    skill = type(
        "SkillDouble",
        (),
        {"name": "demo", "content": "Run the steps.", "skill_metadata": None},
    )()
    recorded: list[tuple[str, Exception]] = []

    async def _generate(*args, **kwargs):
        return SkillMetadata(skill_id="demo")

    def _persist(*args, **kwargs):
        raise ValueError("invalid front matter")

    registry = type("RegistryDouble", (), {"persist_skill_metadata": _persist})()
    monkeypatch.setattr(enrich, "generate_skill_metadata", _generate)
    monkeypatch.setattr(enrich, "record_failure", lambda op, exc: recorded.append((op, exc)))

    assert await ensure_skill_metadata(skill, agent_seam=None, registry=registry) is False
    assert recorded and recorded[-1][0] == "skills.enrich.persist"
    assert isinstance(recorded[-1][1], ValueError)
