"""A4 fix campaign — skills slice regression tests.

Covers:
- P0-8: the per-registry ``SkillIndex`` cache signature ignored skill content
  and revision metadata, so an out-of-band SKILL.md body edit (same name,
  description, triggers, enabled) left ``search_skills`` serving stale
  ``content_revision`` values from the old ``Skill`` objects.
- MCH-013: ``SkillRegistry.delete_skill`` popped the live entry BEFORE the
  persistent removal was validated/completed, so a refused deletion left the
  registry loaded-but-deleted until the next ``load_all``.
- Dead-code debloat pins (review/skills.md P1-1..P1-8) so they cannot
  silently resurrect.

These tests are self-contained (tmp skills dir, no DB, no network).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from machina.skills import metadata as metadata_module
from machina.skills import repo_install as repo_install_module
from machina.skills import runtime as skills_runtime
from machina.skills import search as search_module
from machina.skills.registry import SkillRegistry
from machina.skills.repo_install import RepoInstallError

SKILL_MD = """---
name: alpha
description: A test skill for caching behaviour
triggers:
  - test
version: 1.0.0
enabled: true
---

{body}
"""


def _write_skill(
    root: Path,
    name: str,
    *,
    body: str = "Original body one.",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(SKILL_MD.format(body=body), encoding="utf-8")
    return directory


# ─── P0-8: stale SkillIndex after an out-of-band body edit ───────────────────


def test_body_edit_bumps_search_candidate_revision(tmp_path):
    """Editing a skill body in place must move the next search result's revision.

    The registry reload replaces the ``Skill`` objects (documented behavior,
    ``test_out_of_band_edit_is_still_picked_up``); the cached index must not
    keep serving the old objects when the routing fields are all unchanged.
    """
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry(skills_dir=tmp_path)
    registry.load_all()

    first = skills_runtime.search_skills(registry, "test skill caching behaviour")
    assert first.candidates, "the query must retrieve the skill"
    assert first.candidates[0].name == "alpha"
    live_revision = search_module.content_revision(registry.get("alpha"))
    assert first.candidates[0].content_revision == live_revision
    assert first.catalog_revision == search_module.catalog_revision(
        registry.list_skills()
    )

    # Out-of-band edit: body changes, name/description/triggers/enabled do not.
    _write_skill(tmp_path, "alpha", body="Edited body with brand-new guidance.")
    registry.load_all()
    live_revision_after = search_module.content_revision(registry.get("alpha"))
    assert live_revision_after != live_revision

    second = skills_runtime.search_skills(registry, "test skill caching behaviour")
    assert second.candidates[0].content_revision == live_revision_after, (
        "search results must carry the fresh content revision after a body edit"
    )
    assert second.catalog_revision == search_module.catalog_revision(
        registry.list_skills()
    )


def test_body_edit_bumps_search_candidate_revision_without_reload_call(tmp_path):
    """``search_skills`` itself calls ``load_all``; the same freshness contract
    must hold through that path alone."""
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry(skills_dir=tmp_path)
    skills_runtime.search_skills(registry, "test skill caching behaviour")

    _write_skill(tmp_path, "alpha", body="Second edit body.")
    # No explicit load_all here; search_skills re-validates the registry.
    result = skills_runtime.search_skills(registry, "test skill caching behaviour")
    live = search_module.content_revision(registry.get("alpha"))
    assert result.candidates[0].content_revision == live


# ─── MCH-013: delete ordering (persistent first, then live) ──────────────────


def test_failed_manifest_removal_keeps_the_live_entry(tmp_path):
    """A refused deletion must keep both persistent state and the live entry."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD.format(body="body"), encoding="utf-8")
    (skill_dir / ".install-manifest.json").write_text("{not json", encoding="utf-8")
    registry = SkillRegistry(skills_dir=skills_dir)
    registry.load_all()

    with pytest.raises(RepoInstallError):
        registry.delete_skill("alpha")

    # The mutation-to-failure trace (MCH-013): nothing is removed anywhere.
    assert registry.get("alpha") is not None
    assert [s.name for s in registry.list_skills()] == ["alpha"]
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / ".install-manifest.json").is_file()


def test_ownership_refusal_keeps_the_live_entry(tmp_path):
    """A manifest that cannot prove ownership refuses loudly and keeps the
    skill loaded (the audit's concrete event ordering)."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD.format(body="body"), encoding="utf-8")
    manifest = {
        "source": "https://github.com/o/r",
        "revision": "rev",
        "digest": "0" * 64,
        "files": {"beta/SKILL.md": "0" * 64},
    }
    (skill_dir / ".install-manifest.json").write_text(
        json_dumps(manifest), encoding="utf-8"
    )
    registry = SkillRegistry(skills_dir=skills_dir)
    registry.load_all()

    with pytest.raises(RepoInstallError) as error:
        registry.delete_skill("alpha")
    assert "refusing to delete" in str(error.value)
    assert registry.get("alpha") is not None
    assert (skill_dir / "SKILL.md").is_file()


def test_unlink_failure_keeps_the_live_entry(tmp_path):
    """A plain skill whose file cannot be removed: the error surfaces and the
    live entry stays (persistent removal first, then the live pop)."""
    skills_dir = tmp_path / "skills"
    skill_dir = _write_skill(skills_dir, "alpha")
    registry = SkillRegistry(skills_dir=skills_dir)
    registry.load_all()

    # Real-filesystem failure injection: the skill directory loses its write
    # permission, so unlinking SKILL.md fails with PermissionError on POSIX.
    skill_dir.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            registry.delete_skill("alpha")
        assert registry.get("alpha") is not None
        assert [s.name for s in registry.list_skills()] == ["alpha"]
        assert (skill_dir / "SKILL.md").is_file()
    finally:
        skill_dir.chmod(0o755)


def test_manifest_owned_delete_success_removes_live_and_persistent(tmp_path):
    """Success removes both representations exactly once."""
    skills_dir = tmp_path / "skills"
    skill_dir = _write_skill(skills_dir, "alpha")
    manifest = {
        "source": "https://github.com/o/r",
        "revision": "rev",
        "digest": "0" * 64,
        "files": {"alpha/SKILL.md": "0" * 64},
    }
    (skill_dir / ".install-manifest.json").write_text(
        json_dumps(manifest), encoding="utf-8"
    )
    registry = SkillRegistry(skills_dir=skills_dir)
    registry.load_all()

    assert registry.delete_skill("alpha") is True
    # No load_all in between: the live entry must be gone from memory too.
    assert registry.get("alpha") is None
    assert registry.list_skills() == []
    assert not skill_dir.exists()


def test_plain_skill_delete_success_removes_live_and_persistent(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "alpha")
    registry = SkillRegistry(skills_dir=skills_dir)
    registry.load_all()

    assert registry.delete_skill("alpha") is True
    assert registry.get("alpha") is None
    assert registry.list_skills() == []
    assert not (skills_dir / "alpha").exists()


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


# ─── dead-code debloat pins (review/skills.md P1) ────────────────────────────


def test_dead_skill_aliases_and_constants_removed():
    import machina.plugins.base as plugins_base
    from machina.soulspec import adapters as soulspec_adapters

    # P1-1: _SPECIALTIES alias had no readers (only comments mention it).
    assert not hasattr(skills_runtime, "_SPECIALTIES")
    # P1-5: dead module constant.
    assert not hasattr(repo_install_module, "_SKILL_GLOB")
    # P1-6 is intentionally NOT removed: the legacy-shim regex still has a
    # live DevTools regression test asserting the guard
    # (test_repo_install.py::test_registry_ignores_legacy_incomplete_install_dir).
    # P1-7: the sentinel import is no longer guarded by a dead ImportError
    # fallback (schema.py defines it unconditionally).
    assert "except ImportError" not in inspect.getsource(soulspec_adapters)
    # P1-8: the unreferenced Plugin base-class surface is gone.
    assert "class Plugin" not in inspect.getsource(plugins_base)


def test_dead_function_parameters_removed():
    # P1-2: _alias_list never used its warn parameter.
    assert "warn" not in inspect.signature(metadata_module._alias_list).parameters
    # P1-4: _verify_installed never used its safe parameter (private).
    assert "safe" not in inspect.signature(
        repo_install_module._verify_installed
    ).parameters
