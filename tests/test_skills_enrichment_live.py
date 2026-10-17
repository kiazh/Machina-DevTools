"""A1 live enrichment — the bundled ECC routing sidecar is authored, active, not stale.

The sidecar (``machina/skills/enrichment/<revision>.json``, spec §7.3) is the
only place local routing metadata may enrich vendored ECC instructions without
editing them. These tests run against the **real** registry, bundle, runtime,
and default sidecar directory — no fixture catalog — and assert the shipped
enrichment is:

* authored against the current manifest revision and within the closed
  schema-v2 vocabularies;
* actually attached to the loaded bundled skills (``origin="sidecar"``) with
  section headings that anchor to the real instruction content;
* free of a ``stale-enrichment`` diagnostic for those skills;
* visible to ``search_skills`` as matched task-family / capability / alias
  signals;
* selected as stage sections by the automatic disclosure path instead of the
  pre-A3 whole-body fallback; and
* safely ignored (baseline fallback + ``stale-enrichment``) when a sidecar
  carries the wrong revision.
"""

from __future__ import annotations

import json
from pathlib import Path

from machina.core.assembler import get_token_count
from machina.skills.bundle import PIN, ECCBundle
from machina.skills.metadata import (
    CAPABILITIES,
    CODE_STALE_ENRICHMENT,
    STAGES,
    TASK_FAMILIES,
    bind_section_anchors,
    default_enrichment_dir,
)
from machina.skills.planned_steps import normalize_stage_state, plan_disclosure
from machina.skills.registry import SkillRegistry
from machina.skills.runtime import automatic_skill_context, search_skills

#: The routing anchors the workflow pipeline selects by name; enrichment must
#: cover all of them or the automatic path keeps its whole-body fallback.
WORKFLOW_ANCHORS = (
    "ecc-search-first",
    "ecc-tdd-workflow",
    "ecc-verification-loop",
    "ecc-security-review",
)

#: One concrete, high-value request plus the honest stage fact that makes the
#: implement-stage sections selectable (spec §10.2).
REQUEST = "Implement a FastAPI endpoint for user signup and add pytest tests"
STAGE_STATE = {"stage": "implement", "stage_source": "user"}


def _sidecar_path() -> Path:
    return default_enrichment_dir() / f"{PIN}.json"


def _sidecar() -> dict:
    return json.loads(_sidecar_path().read_text(encoding="utf-8"))


def _authored() -> dict[str, dict]:
    return _sidecar()["skills"]


def _real_registry(*, enrichment_dir: Path | None = None) -> SkillRegistry:
    registry = SkillRegistry(enrichment_dir=enrichment_dir)
    registry.load_all()
    return registry


def _plan(registry: SkillRegistry):
    from machina.skills.workflow import plan_workflow

    workflow = plan_workflow(REQUEST, skills=registry, task_state=STAGE_STATE)
    disclosure = plan_disclosure(
        workflow.steps,
        skill_for=registry.get,
        content_for=registry.get_skill_content,
        budget=1200,
        stage_state=normalize_stage_state(STAGE_STATE),
        task_revision=workflow.task_revision,
        catalog_revision=workflow.catalog_revision,
        token_count=get_token_count,
        token_basis="assembler.get_token_count",
    )
    return workflow, disclosure


def test_shipped_sidecar_matches_the_current_manifest_revision():
    bundle = ECCBundle()
    assert bundle.manifest["revision"] == PIN
    assert _sidecar_path().is_file()

    payload = _sidecar()
    assert payload["schema_version"] == 1
    assert payload["revision"] == PIN
    assert isinstance(payload["skills"], dict)

    authored = payload["skills"]
    assert len(authored) >= 10, "the enrichment must cover a meaningful subset"
    assert set(WORKFLOW_ANCHORS) <= set(authored)

    bundled_ids = {entry["name"] for entry in bundle.manifest["skills"]}
    assert set(authored) <= bundled_ids, "enrichment may only name bundled skills"


def test_authored_values_use_only_the_closed_vocabularies():
    for skill_id, entry in _authored().items():
        assert set(entry["task_families"]) <= TASK_FAMILIES, skill_id
        assert set(entry["capabilities"]) <= CAPABILITIES, skill_id
        for section in entry.get("sections", []):
            stage = section.get("stage")
            assert stage is None or stage in STAGES, (skill_id, section)
            for required in section.get("requires_sections", []):
                declared = {item["id"] for item in entry["sections"]}
                assert required in declared, (skill_id, section["id"], required)


def test_authored_sections_anchor_to_real_bundle_headings():
    bundle = ECCBundle()
    paths = {entry["name"]: entry["path"] for entry in bundle.manifest["skills"]}
    registry = _real_registry()
    for skill_id, entry in _authored().items():
        if not entry.get("sections"):
            continue
        content = bundle.read_resource(paths[skill_id]).split("---", 2)[2].strip()
        metadata = registry.get(skill_id).skill_metadata
        assert metadata is not None, skill_id
        bound, anchor_diags = bind_section_anchors(metadata, content)
        assert not [d for d in anchor_diags if d.code == "section-missing-anchor"], skill_id
        ambiguous = {d.code for d in anchor_diags}
        for section in bound.sections:
            if section.anchor is None:
                # combined #48: duplicate headings are deliberately left
                # unbound rather than silently pointing at the first match.
                assert "section-duplicate-heading" in ambiguous, (skill_id, section.id)
                continue
            heading = content.splitlines()[section.anchor.line - 1]
            assert section.heading.casefold() in heading.casefold(), (skill_id, section.id)


def test_enriched_skills_are_live_and_not_stale():
    registry = _real_registry()
    for skill_id in WORKFLOW_ANCHORS:
        skill = registry.get(skill_id)
        assert skill is not None and skill.source_type == "builtin-ecc"
        metadata = skill.skill_metadata
        assert metadata is not None and metadata.origin == "sidecar"
        assert metadata.task_families and metadata.sections

    codes = {diagnostic.code for diagnostic in registry.metadata_diagnostics()}
    assert CODE_STALE_ENRICHMENT not in codes
    assert not [d for d in registry.metadata_diagnostics() if d.severity == "error"]


def test_search_exposes_authored_families_capabilities_and_aliases():
    registry = _real_registry()

    migration = search_skills(registry, "plan a database migration for postgres", limit=20)
    by_id = {candidate.skill_id: candidate for candidate in migration.candidates}
    assert "ecc-database-migrations" in by_id, sorted(by_id)
    candidate = by_id["ecc-database-migrations"]
    assert {"database", "migration"} <= set(candidate.matched_capabilities)
    # The declared task family ("backend-api" -> implementing) is a scored
    # signal, not just lexical overlap.
    assert "task_family" in candidate.score_components
    assert "families implementing" in candidate.explanation

    alias = search_skills(registry, "search-first", limit=30)
    aliased = [c for c in alias.candidates if c.skill_id == "ecc-search-first"]
    assert aliased and "search-first" in aliased[0].matched_phrases


def test_automatic_disclosure_selects_authored_stage_sections(tmp_path):
    # Fail-closed policy: a skill with no metadata at all is unusable, so the
    # "before" catalog models the pre-section state as a sidecar with metadata
    # but NO authored sections (whole-body fallback). A sidecar (not an
    # in-memory attach) survives the telemetry-triggered reload that follows
    # another automatic context in this test.
    payload = _sidecar()
    for entry in payload["skills"].values():
        entry["sections"] = []
    sections_less = tmp_path / "metadata-no-sections"
    sections_less.mkdir()
    (sections_less / f"{PIN}.json").write_text(json.dumps(payload), encoding="utf-8")
    before = _real_registry(enrichment_dir=sections_less)
    after = _real_registry()

    _, plan_before = _plan(before)
    _, plan_after = _plan(after)

    authored_sections = {
        (skill_id, section["id"])
        for skill_id, entry in _authored().items()
        for section in entry.get("sections", [])
    }

    # Before: no sidecar-declared sections -> the ECC whole-body fallback, the
    # implement anchor is not disclosed at all. Native skills may now carry
    # their own authored metadata, so scope this assertion to the sidecar
    # entries under test rather than requiring an empty plan.
    chosen_before = {(entry.skill_id, entry.section_id) for entry in plan_before.chosen}
    assert not (chosen_before & authored_sections), chosen_before
    assert plan_before.whole_body

    # After: at least one authored stage section is selected whole.
    chosen = {(entry.skill_id, entry.section_id) for entry in plan_after.chosen}
    assert chosen & authored_sections, chosen
    assert all(entry.estimate_tokens > 0 for entry in plan_after.chosen)

    after_context = automatic_skill_context(REQUEST, skills=after, task_state=STAGE_STATE)
    before_context = automatic_skill_context(REQUEST, skills=before, task_state=STAGE_STATE)
    assert get_token_count(after_context) <= 1200
    assert "## Automatic skills" in after_context
    assert "### ecc-python-testing \u00b7 Fixtures" in after_context
    assert "### ecc-python-testing \u00b7 Fixtures" not in before_context


def test_wrong_revision_sidecar_is_stale_and_falls_back(tmp_path):
    payload = _sidecar()
    payload["revision"] = "0" * 40
    sidecar_dir = tmp_path / "wrong-revision"
    sidecar_dir.mkdir()
    (sidecar_dir / f"{PIN}.json").write_text(json.dumps(payload), encoding="utf-8")

    registry = _real_registry(enrichment_dir=sidecar_dir)
    for skill_id in WORKFLOW_ANCHORS:
        skill = registry.get(skill_id)
        assert skill is not None and skill.source_type == "builtin-ecc"
        assert skill.skill_metadata is None, skill_id

    stale = [d for d in registry.metadata_diagnostics() if d.code == CODE_STALE_ENRICHMENT]
    assert stale and PIN in stale[0].message
