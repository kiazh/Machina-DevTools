"""A4 — revision identities + generation-guarded publication (spec §11).

RED-first coverage at real consumer boundaries:

* ``machina.skills.revisions`` — content / catalog / telemetry / ranking-policy
  / task-state identities as first-class values, and ``identity_for`` mapping
  a cache to exactly the identities that affect it;
* the §11 separation matrix: telemetry activity NEVER invalidates
  content-derived data (the section index and plan caches keep hitting), while
  a content edit moves the content/catalog identities and invalidates them;
* generation-guarded publication: a compute that races an invalidation
  (``clear_section_cache`` / ``RevisionCache.invalidate`` / a source edit) must
  not publish a stale value into the cache — the next compute republishes;
* the fingerprint policy: stat size/mtime signatures are a fast pre-filter
  only; section reads are bound to exact content fingerprints;
* the A3 rollback switch ``skills_disclosure_planner`` is registered in the
  config DEFAULTS table and its ON/OFF paths are both exercised;
* the new modules import no provider/embedding/network code and start no
  threads (AST scan + fresh-import subprocess).

Measured where cheap: repeated telemetry writes must not grow section parse
counts or trigger skill re-parses.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina")

GUIDE = (
    "Intro before any heading.\n\n"
    "## Run tests\n\n"
    "Run the focused suite first, then the full suite.\n\n"
    "## Roll back\n\n"
    "Restore the previous release if checks fail.\n"
)

DEPLOY_CONTENT = (
    "DEPLOY PREAMBLE MARKER: keep one rollback path.\n\n"
    "## Preflight checks\n\n"
    "Check credentials and backups before touching production.\n\n"
    "## Deploy the service\n\n"
    "Run scripts/deploy.sh only after preflight passes.\n\n"
    "## Roll back\n\n"
    "Restore the previous release immediately if health checks fail.\n"
)


def _registry(tmp_path):
    from machina.skills.registry import SkillRegistry

    skills = SkillRegistry(tmp_path)
    skills.create_skill("guide", "A deployment guide", GUIDE, triggers=["guide"])
    skills.load_all()
    return skills


def _deploy_registry(tmp_path):
    from machina.skills.metadata import SectionMetadata, SkillMetadata
    from machina.skills.registry import SkillRegistry

    skills = SkillRegistry(tmp_path)
    skills.create_skill(
        "deploy-guide",
        "Deploy services safely with a rollback path",
        DEPLOY_CONTENT,
        triggers=["deploy"],
    )
    skills.load_all()
    skills.get("deploy-guide").skill_metadata = SkillMetadata(
        skill_id="deploy-guide",
        sections=(
            SectionMetadata(id="preflight", heading="Preflight checks", stage="plan"),
            SectionMetadata(id="deploy-steps", heading="Deploy the service", stage="deliver"),
            SectionMetadata(
                id="rollback",
                heading="Roll back",
                stage="verify",
                requires_sections=("deploy-steps",),
            ),
        ),
    )
    return skills


# --- identity values and feature-area mapping -------------------------------


def test_revision_identities_are_distinct_first_class_values(tmp_path):
    from machina.skills import revisions
    from machina.skills.search import POLICY_REVISION

    registry = _registry(tmp_path)
    skill = registry.get("guide")

    per_skill = revisions.RevisionRegistry.for_skill(skill)
    assert per_skill.content == revisions.content_revision(skill)
    assert len(per_skill.content) == 16
    assert per_skill.catalog == revisions.catalog_revision([skill])
    assert per_skill.ranking_policy == POLICY_REVISION
    assert per_skill.task_state == "0"
    assert per_skill.identity_for(revisions.SECTION_INDEX) == (per_skill.content,)

    catalog = revisions.RevisionRegistry.for_skills([skill], task_revision=3)
    assert catalog.task_state == "3"
    assert catalog.telemetry == per_skill.telemetry
    assert catalog.content and catalog.catalog
    assert revisions.task_state_revision({"revision": "7"}) == 7
    assert revisions.task_state_revision(None) == 0
    assert revisions.task_state_revision({"revision": -3}) == 0
    assert revisions.task_state_revision({"revision": "nonsense"}) == 0


def test_identity_for_exposes_exactly_the_feature_identities(tmp_path):
    from machina.skills import revisions

    registry = _registry(tmp_path)
    snapshot = revisions.RevisionRegistry.for_skills(registry.list_skills(), task_revision=7)

    assert snapshot.identity_for(revisions.SECTION_INDEX) == (snapshot.content,)
    assert snapshot.identity_for(revisions.SECTION_BINDINGS) == (snapshot.content,)
    assert snapshot.identity_for(revisions.SEARCH_RESULTS) == (
        snapshot.catalog,
        snapshot.ranking_policy,
    )
    assert snapshot.identity_for(revisions.DISCLOSURE_PLAN) == (
        snapshot.catalog,
        snapshot.ranking_policy,
        snapshot.task_state,
    )
    assert snapshot.identity_for(revisions.TELEMETRY_READS) == (snapshot.telemetry,)
    assert snapshot.identity_for(revisions.ADMINISTRATIVE_READS) == (snapshot.catalog,)
    assert set(snapshot.as_dict()) == set(revisions.IDENTITY_NAMES)
    with pytest.raises(ValueError):
        snapshot.identity_for("not_a_feature_area")


def test_telemetry_bump_changes_only_the_telemetry_identity(tmp_path):
    from machina.skills import revisions

    registry = _registry(tmp_path)
    before = registry.revisions()

    registry.record_use("guide")
    registry.record_view("guide")
    after = registry.revisions()

    assert after.telemetry != before.telemetry
    assert after.content == before.content
    assert after.catalog == before.catalog
    assert after.identity_for(revisions.DISCLOSURE_PLAN) == before.identity_for(
        revisions.DISCLOSURE_PLAN
    )
    assert after.identity_for(revisions.SECTION_INDEX) == before.identity_for(
        revisions.SECTION_INDEX
    )


def test_enabled_toggle_moves_catalog_identity_not_content(tmp_path):
    registry = _registry(tmp_path)
    before = registry.revisions()

    registry.set_enabled("guide", False)
    after = registry.revisions()

    assert after.catalog != before.catalog
    assert after.content == before.content


def test_content_edit_moves_content_and_catalog_identities(tmp_path):
    registry = _registry(tmp_path)
    before = registry.revisions()

    registry.update_skill("guide", content=GUIDE + "\n## Extra\n\nMore guidance.\n")
    after = registry.revisions()

    assert after.content != before.content
    assert after.catalog != before.catalog
    assert after.telemetry == before.telemetry


def test_task_state_revision_mirrors_the_planned_step_reader():
    from machina.skills import revisions
    from machina.skills.planned_steps import task_revision_from

    for raw in (None, {}, {"revision": 4}, {"revision": "5"}, {"revision": -2}, {"revision": "x"}):
        assert revisions.task_state_revision(raw) == task_revision_from(raw)


# --- telemetry vs content invalidation matrix -------------------------------


def test_telemetry_bumps_keep_section_and_plan_caches_hot(tmp_path):
    from machina.skills import revisions
    from machina.skills.sections import (
        clear_section_cache,
        index_sections,
        section_cache_generation,
    )

    registry = _registry(tmp_path)
    clear_section_cache()
    content = registry.get_skill_content("guide")

    first = index_sections(content)
    generation = section_cache_generation()

    plan_cache = revisions.RevisionCache(lambda: registry.revisions(), max_entries=4)
    plan_calls: list[int] = []

    def compute_plan() -> str:
        plan_calls.append(1)
        return f"plan-{len(plan_calls)}"

    key = ("stage", "verify", 1200)
    assert plan_cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, compute_plan) == "plan-1"

    for _ in range(10):
        registry.record_use("guide")
        registry.record_view("guide")

    # Telemetry is activity: every content-derived cache still hits.
    assert index_sections(content) is first
    assert section_cache_generation() == generation
    assert plan_cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, compute_plan) == "plan-1"
    assert len(plan_calls) == 1
    assert registry.get("guide").usage_count == 10


def test_content_edit_invalidates_section_and_plan_caches(tmp_path):
    from machina.skills import revisions
    from machina.skills.sections import clear_section_cache, index_sections

    registry = _registry(tmp_path)
    clear_section_cache()
    content = registry.get_skill_content("guide")

    first = index_sections(content)
    plan_cache = revisions.RevisionCache(lambda: registry.revisions(), max_entries=4)
    plan_calls: list[int] = []

    def compute_plan() -> str:
        plan_calls.append(1)
        return f"plan-{len(plan_calls)}"

    key = ("stage", "verify", 1200)
    assert plan_cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, compute_plan) == "plan-1"

    registry.update_skill("guide", content=content + "\n## Extra\n\nMore guidance.\n")
    edited = registry.get_skill_content("guide")

    second = index_sections(edited)
    assert second is not first
    assert second.revision != first.revision
    assert index_sections(edited) is second  # the new revision is cached again
    assert plan_cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, compute_plan) == "plan-2"
    assert len(plan_calls) == 2


# --- generation-guarded publication -----------------------------------------


def test_guarded_publish_withholds_a_stale_value():
    from machina.skills.revisions import GenerationCounter, guarded_publish

    generation = GenerationCounter()
    stored: list[str] = []

    def racy_compute() -> str:
        value = "first"
        generation.bump()  # source invalidated mid-compute
        return value

    assert guarded_publish(generation, racy_compute, stored.append) == "first"
    assert stored == []  # the stale result was NOT published
    assert guarded_publish(generation, lambda: "second", stored.append) == "second"
    assert stored == ["second"]  # the next compute republishes


def test_index_sections_withholds_a_stale_parse_after_mid_compute_clear(monkeypatch):
    from machina.skills import sections

    sections.clear_section_cache()
    calls: list[str] = []
    real_parse = sections._parse_index

    def racy_parse(content: str, revision: str):
        calls.append(revision)
        index = real_parse(content, revision)
        if len(calls) == 1:
            # An invalidation lands while this parse is still in flight.
            sections.clear_section_cache()
        return index

    monkeypatch.setattr(sections, "_parse_index", racy_parse)

    first = sections.index_sections(GUIDE)
    assert first.revision == sections.content_fingerprint(GUIDE)
    assert len(calls) == 1

    # Nothing was published, so the next call recomputes (a fresh object).
    second = sections.index_sections(GUIDE)
    assert len(calls) == 2
    assert second is not first
    assert second.revision == first.revision

    # The republished index is now a cache hit.
    third = sections.index_sections(GUIDE)
    assert third is second
    assert len(calls) == 2


def test_revision_cache_withholds_a_stale_compute():
    from machina.skills import revisions

    snapshots = [
        revisions.RevisionRegistry(content="c1", catalog="k1", telemetry="t1"),
    ]
    cache = revisions.RevisionCache(lambda: snapshots[-1], max_entries=4)
    calls: list[int] = []

    def racy_compute() -> str:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            cache.invalidate()  # explicit invalidation lands mid-compute
        return f"value-{len(calls)}"

    assert cache.get_or_compute(revisions.SEARCH_RESULTS, (), racy_compute) == "value-1"
    assert calls == [1]
    assert cache.get_or_compute(revisions.SEARCH_RESULTS, (), racy_compute) == "value-2"
    assert calls == [1, 2]
    assert cache.get_or_compute(revisions.SEARCH_RESULTS, (), racy_compute) == "value-2"
    assert calls == [1, 2]  # published by the second compute, then a hit


def test_revision_cache_does_not_publish_across_a_content_edit(tmp_path):
    from machina.skills import revisions

    registry = _registry(tmp_path)
    cache = revisions.RevisionCache(lambda: registry.revisions(), max_entries=4)
    calls: list[int] = []

    def racy_compute() -> str:
        calls.append(1)
        if len(calls) == 1:
            registry.update_skill("guide", content=GUIDE + "\n## Changed\n\nNew text.\n")
        return f"plan-{len(calls)}"

    key = ("stage", "verify", 1200)
    assert cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, racy_compute) == "plan-1"
    # The identity moved while computing: that value must not have been stored.
    assert cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, racy_compute) == "plan-2"
    assert len(calls) == 2
    assert cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, racy_compute) == "plan-2"
    assert len(calls) == 2


def test_revision_cache_isolates_feature_areas(tmp_path):
    from machina.skills import revisions

    registry = _registry(tmp_path)
    cache = revisions.RevisionCache(lambda: registry.revisions(), max_entries=8)
    calls: list[str] = []

    def compute() -> str:
        calls.append("x")
        return f"v{len(calls)}"

    key = ("same-key",)
    first = cache.get_or_compute(revisions.SEARCH_RESULTS, key, compute)
    # A different feature area with the same key is a different cache entry.
    second = cache.get_or_compute(revisions.DISCLOSURE_PLAN, key, compute)
    assert first == "v1" and second == "v2"
    assert cache.get_or_compute(revisions.SEARCH_RESULTS, key, compute) == "v1"
    assert len(calls) == 2


# --- registry / runtime plumbing --------------------------------------------


def test_registry_and_runtime_expose_revision_snapshots(tmp_path):
    from machina.skills import revisions
    from machina.skills.runtime import revisions_for

    registry = _registry(tmp_path)
    snapshot = registry.revisions()
    assert isinstance(snapshot, revisions.RevisionRegistry)
    assert snapshot.catalog == revisions.catalog_revision(registry.list_skills())

    with_task = revisions_for(registry, task_state={"revision": 5})
    assert with_task.task_state == "5"
    assert with_task.catalog == snapshot.catalog
    assert with_task.telemetry == snapshot.telemetry

    without_task = revisions_for(registry)
    assert without_task.task_state == "0"


# --- config registration + rollback switch -----------------------------------


def test_config_registers_disclosure_planner_switch_and_rollback(tmp_path, monkeypatch):
    from machina.core.config import DEFAULTS, get_config
    from machina.skills.planned_steps import StageState
    from machina.skills.runtime import automatic_skill_context

    assert DEFAULTS.get("skills_disclosure_planner") is True

    skills = _deploy_registry(tmp_path)
    config = get_config()
    monkeypatch.delenv("MACHINA_SKILLS_DISCLOSURE_PLANNER", raising=False)
    try:
        assert config.get("skills_disclosure_planner") is True
        planned = automatic_skill_context(
            "help me deploy",
            skills=skills,
            budget=1200,
            stage_state=StageState("verify", source="user"),
            can_read=False,
            can_search=False,
        )
        assert "### deploy-guide \u00b7 Roll back" in planned

        config.set("skills_disclosure_planner", False)
        assert config.get("skills_disclosure_planner") is False
        legacy = automatic_skill_context(
            "help me deploy",
            skills=skills,
            budget=1200,
            stage_state=StageState("verify", source="user"),
            can_read=False,
            can_search=False,
        )
        assert "### deploy-guide \u00b7 " not in legacy
        assert "DEPLOY PREAMBLE MARKER" in legacy
    finally:
        config.set("skills_disclosure_planner", True)


def test_env_override_still_reads_as_boolean(tmp_path, monkeypatch):
    from machina.core.config import get_config

    monkeypatch.setenv("MACHINA_SKILLS_DISCLOSURE_PLANNER", "0")
    assert get_config().get("skills_disclosure_planner") is False
    monkeypatch.setenv("MACHINA_SKILLS_DISCLOSURE_PLANNER", "true")
    assert get_config().get("skills_disclosure_planner") is True
    monkeypatch.setenv("MACHINA_SKILLS_DISCLOSURE_PLANNER", "nonsense")
    with pytest.raises(ValueError):
        get_config().get("skills_disclosure_planner")


# --- cheap measurement: telemetry writes don't grow parse counts -------------


def test_telemetry_writes_and_reloads_do_not_grow_parse_counts(tmp_path, monkeypatch):
    from machina.skills import registry as registry_module
    from machina.skills import sections

    registry = _registry(tmp_path)
    content = registry.get_skill_content("guide")
    sections.clear_section_cache()

    section_parses: list[str] = []
    real_index_parse = sections._parse_index

    def counting_index_parse(text: str, revision: str):
        section_parses.append(revision)
        return real_index_parse(text, revision)

    skill_parses: list[int] = []
    real_skill_parse = registry_module.SkillParser.parse

    def counting_skill_parse(*args, **kwargs):
        skill_parses.append(1)
        return real_skill_parse(*args, **kwargs)

    monkeypatch.setattr(sections, "_parse_index", counting_index_parse)
    monkeypatch.setattr(registry_module.SkillParser, "parse", staticmethod(counting_skill_parse))

    first = sections.index_sections(content)
    registry.load_all()
    baseline_sections = len(section_parses)
    baseline_skills = len(skill_parses)
    assert baseline_sections == 1

    for _ in range(25):
        registry.record_use("guide")
        registry.load_all()
        assert sections.index_sections(content) is first

    assert len(section_parses) == baseline_sections
    assert len(skill_parses) == baseline_skills
    assert registry.get("guide").usage_count == 25


# --- no provider / embedding / network / threads -----------------------------

BANNED_PARTS = {
    "provider",
    "llm",
    "openai",
    "anthropic",
    "litellm",
    "ollama",
    "cohere",
    "gemini",
    "mistral",
    "embed",
}


def _scanned_modules() -> tuple[str, ...]:
    return (
        "skills/revisions.py",
        "skills/events.py",
        "skills/sections.py",
        "skills/runtime.py",
        "skills/registry.py",
    )


def test_shipped_sources_import_no_provider_or_embedding_code():
    for name in _scanned_modules():
        path = REPO / "machina" / name
        assert path.is_file(), name
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for dotted in names:
                parts = dotted.lower().split(".")
                assert not any(part in BANNED_PARTS for part in parts), f"{name}: {dotted}"
        if name in ("skills/revisions.py", "skills/events.py"):
            assert "__import__" not in text, name
            assert "subprocess" not in text, name
            assert "socket" not in text, name


def test_fresh_import_pulls_no_provider_embedding_or_threads():
    script = textwrap.dedent(
        """
        import threading
        before = threading.active_count()
        import machina.skills.events  # noqa: F401
        import machina.skills.revisions  # noqa: F401
        import machina.skills.sections  # noqa: F401
        import machina.skills.runtime  # noqa: F401
        after = threading.active_count()
        assert after <= before, (before, after)
        banned = %r
        pulled = sorted(
            name
            for name in __import__("sys").modules
            if name.startswith("machina.")
            and any(part in banned for part in name.lower().split("."))
        )
        assert not pulled, pulled
        print("ok")
        """
    ) % (sorted(BANNED_PARTS),)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
