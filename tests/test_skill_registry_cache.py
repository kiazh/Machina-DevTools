"""Regression tests for SkillRegistry caching.

``load_all()`` is invoked on every skill tool call (``skill_list``,
``skill_read``) and by every CLI/gateway skill handler. Re-reading the whole
skills directory plus telemetry.json each time was pure waste within a turn,
so the parsed result is cached and invalidated on mutation.
"""

from __future__ import annotations

from pathlib import Path

from machina.skills.registry import SkillRegistry

SKILL_MD = """---
name: {name}
description: A test skill for caching behaviour
triggers:
  - test
version: 1.0.0
enabled: true
schema_version: 2
skill_id: {name}
---

Body for {name}.
"""


def _write_skill(root: Path, name: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")


def _parse_count(monkeypatch) -> list[int]:
    """Count SkillParser.parse calls."""
    from machina.skills import registry as reg

    calls: list[int] = []
    original = reg.SkillParser.parse

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(reg.SkillParser, "parse", staticmethod(counted))
    return calls


def test_load_all_is_cached(tmp_path, monkeypatch):
    """Repeated load_all() calls with no on-disk change must not re-parse."""
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    registry = SkillRegistry(skills_dir=tmp_path)
    calls = _parse_count(monkeypatch)

    registry.load_all()
    assert len(calls) == 2, "first load should parse both skills"

    registry.load_all()
    registry.load_all()
    assert len(calls) == 2, "unchanged loads must not re-parse"

    assert len(registry.list_skills()) == 2


def test_out_of_band_edit_is_still_picked_up(tmp_path, monkeypatch):
    """Reload semantics are preserved: a changed SKILL.md invalidates the cache."""
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry(skills_dir=tmp_path)
    calls = _parse_count(monkeypatch)

    registry.load_all()
    assert len(calls) == 1

    # Modify the file on disk, the way a curator or a human edit would.
    path = tmp_path / "alpha" / "SKILL.md"
    path.write_text(SKILL_MD.format(name="alpha") + "\nNew guidance.\n", encoding="utf-8")
    registry.load_all()
    assert len(calls) == 2, "a changed file must be re-parsed"
    assert "New guidance." in registry.get("alpha").content


def test_refresh_true_forces_a_reload(tmp_path, monkeypatch):
    registry = SkillRegistry(skills_dir=tmp_path)
    _write_skill(tmp_path, "alpha")
    calls = _parse_count(monkeypatch)

    registry.load_all()
    assert len(calls) == 1

    registry.load_all(refresh=True)
    assert len(calls) == 2


def test_create_skill_invalidates_cache(tmp_path, monkeypatch):
    registry = SkillRegistry(skills_dir=tmp_path)
    _write_skill(tmp_path, "alpha")
    registry.load_all()
    assert {s.name for s in registry.list_skills()} == {"alpha"}

    registry.create_skill("delta", "A newly created skill", "Step one.\nStep two.\n")

    # Without invalidation the stale cache would hide the new skill.
    registry.load_all()
    assert {s.name for s in registry.list_skills()} == {"alpha", "delta"}


def test_delete_skill_invalidates_cache(tmp_path):
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    registry = SkillRegistry(skills_dir=tmp_path)
    registry.load_all()
    assert {s.name for s in registry.list_skills()} == {"alpha", "beta"}

    assert registry.delete_skill("beta") is True

    registry.load_all()
    assert {s.name for s in registry.list_skills()} == {"alpha"}


def test_record_use_invalidates_cache(tmp_path):
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry(skills_dir=tmp_path)
    registry.load_all()

    registry.record_use("alpha")

    registry.load_all()
    assert registry.get("alpha").usage_count == 1


def test_missing_directory_loads_empty_and_is_cached(tmp_path, monkeypatch):
    registry = SkillRegistry(skills_dir=tmp_path / "does-not-exist")
    calls = _parse_count(monkeypatch)

    registry.load_all()
    registry.load_all()
    assert calls == []
    assert registry.list_skills() == []


def test_default_skill_usage_persists_with_a_new_local_directory(tmp_path, monkeypatch):
    """A fresh installation must not lose builtin telemetry to a missing directory."""
    local = tmp_path / "new" / "skills"
    monkeypatch.setenv("MACHINA_SKILLS_DIR", str(local))
    registry = SkillRegistry()
    registry.load_all()
    registry.record_use("ecc-python-patterns")
    assert (local / "telemetry.json").is_file()
    restored = SkillRegistry()
    restored.load_all()
    assert restored.get("ecc-python-patterns").usage_count == 1


def test_telemetry_save_does_not_mask_out_of_band_instruction_edit(tmp_path):
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry(tmp_path)
    registry.load_all()
    path = tmp_path / "alpha" / "SKILL.md"
    path.write_text(SKILL_MD.format(name="alpha") + "\nNew instruction.\n")
    registry.record_use("alpha")
    registry.load_all()
    assert "New instruction." in registry.get_skill_content("alpha")


def test_warm_search_does_not_rehash_every_body(tmp_path, monkeypatch):
    """A6 ledger: warm search must not pay content_revision for every skill.

    The per-registry revision cache is keyed by (path, mtime_ns, size, version,
    metadata object); a warm call is a cache hit, while an out-of-band body edit
    still moves the revision (P0-8 freshness).
    """
    from machina.skills import runtime
    from machina.skills import search as search_module

    for name in ("alpha", "beta", "gamma"):
        _write_skill(tmp_path, name)
    registry = SkillRegistry(skills_dir=tmp_path)

    calls: list[str] = []
    original = search_module.content_revision

    def counted(skill):
        calls.append(skill.name)
        return original(skill)

    monkeypatch.setattr(search_module, "content_revision", counted)
    first = runtime.search_skills(registry, "caching behaviour")
    assert type(first).__name__ == "SearchResult"
    cold_calls = len(calls)
    assert cold_calls >= 3  # first search builds the revision identity

    calls.clear()
    runtime.search_skills(registry, "caching behaviour")
    assert calls == [], f"warm search re-hashed {calls}"

    # An out-of-band body edit with a new mtime recomputes that skill's revision.
    path = tmp_path / "alpha" / "SKILL.md"
    path.write_text(SKILL_MD.format(name="alpha") + "\nEdited body.\n", encoding="utf-8")
    calls.clear()
    runtime.search_skills(registry, "caching behaviour")
    assert calls == ["alpha"]
