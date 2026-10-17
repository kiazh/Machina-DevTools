"""A3 — one authoritative planned-step order + stage-honest disclosure (spec §10).

RED-first, consumer-boundary tests for:

* ``machina.skills.workflow.plan_workflow`` — display order, selection order,
  and disclosure order must all derive from one ``PlannedStep`` sequence
  (spec §10.1; A0 observed ``selected`` order != ``stages`` order);
* ``machina.skills.planned_steps.plan_disclosure`` — the §10.3 disclosure
  contract: revisions, prerequisites-or-explicit-inability, whole-or-omitted
  mandatory sections, stable locators, honest token basis;
* ``machina.skills.runtime.automatic_skill_context`` — stage-aware disclosure
  behind ``skills_disclosure_planner`` (default ON, rollback switch), with
  byte parity for skills that have no metadata/sections.

Acceptance mapping: D01 (order coherence), D02 (prerequisite in-or-explicit),
D06 (tiny allowance never claims load), §10.2 (no stage inferred from a
sentence), §10.5 (later sections stay discoverable), §10.3 (contract fields).

Every path under test is deterministic and stdlib-only: no provider, embedding,
network, or subprocess call is possible (the suite patches/scans for them).
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from machina.skills.metadata import SectionMetadata, SkillMetadata
from machina.skills.registry import SkillRegistry

REPO = Path(
    os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina"
)

#: Fixture body: a staged deployment guide. Stage labels use the closed
#: metadata vocabulary (plan/deliver/verify); "Safety notes" is deliberately
#: untagged so it is phase-neutral bootstrap material.
DEPLOY_CONTENT = (
    "DEPLOY PREAMBLE: keep one rollback path and never rush a release.\n\n"
    "## Safety notes\n\n"
    "Only deploy from a clean tree; never force-push. If unsure, stop and ask.\n\n"
    "## Preflight checks\n\n"
    "Verify credentials, backups, and free disk space before touching production. "
    "Record the current release ID and the database migration state so a rollback "
    "has an exact target.\n\n"
    "## Deploy the service\n\n"
    "Run scripts/deploy.sh only after preflight passes. Watch the logs until the "
    "health check reports green, then confirm the version endpoint reports the new "
    "build. Keep the previous release ID handy.\n\n"
    "## Roll back\n\n"
    "Restore the previous release immediately if any health check fails or the "
    "error rate exceeds the baseline. Do not debug forward on a broken production "
    "release; roll back first and investigate afterwards.\n"
)

#: Budget windows (tokens) measured against the 361-token automatic header in
#: this checkout: OK fits the whole guide; TIGHT keeps the header + locators
#: but no staged section; MID fits the deliver-stage section + locators.
OK_BUDGET = 1200
TIGHT_BUDGET = 460
MID_BUDGET = 480

# Byte-parity locks captured from the pre-A3 implementation (working tree
# b7bd6b6) for skills without metadata/sections: the whole-body excerpt path
# must stay unchanged, including its continuation offsets.
PARITY_SINGLE_BUDGET = 600
# Captured from the pre-A3 implementation working tree (b7bd6b6, 2026-10-16):
#   automatic_skill_context("help me deploy", skills=<single v1 "deploy">, budget=600)
PARITY_SINGLE_SHA256 = "328e411d9be132316b65bf638606f75716f0161f037c7c85dfdd66d75b016e97"
#   automatic_skill_context("use alpha and beta tooling", skills=<two v1 skills>, budget=1200)
# Re-captured under the canonical selector (A2c): the excerpt mechanism is
# byte-identical (forcing the canonical order back to alpha-first on the same
# contents reproduces the pre-A3 5d776514… hash exactly — see the test's
# comment), but the canonical ranking adds a phrase signal so "beta tooling"
# outranks "alpha tooling" and the two blocks swap. The pin moves with the
# ranking the planner now shares with human search.
PARITY_TWO_SKILL_SHA256 = "879c7cacb251fdfed70ee3146c3ca8f4ab85b743af972a851bdba074935e75f7"


def _give_minimal_metadata(skills: SkillRegistry, *names: str) -> None:
    """Fail-closed policy: usable fixtures need schema-v2 metadata.

    In-memory on purpose: the stat-only registry cache keeps it across loads,
    and these tests exercise disclosure mechanics, not loading.
    """
    skills.load_all()
    for name in names:
        skill = skills.get(name)
        if skill is not None and skill.skill_metadata is None:
            skill.skill_metadata = SkillMetadata(skill_id=name)


def _deploy_registry(tmp_path) -> SkillRegistry:
    """Local registry with one staged skill; ECC bundle excluded (tmp dir)."""
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
        task_families=("deployment",),
        sections=(
            SectionMetadata(id="safety", heading="Safety notes"),
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


def _coding_registry(tmp_path) -> SkillRegistry:
    """Four v1 skills that activate the coding pipeline with a stage mismatch.

    Insertion order is domain -> inspect (the domain anchor is appended
    first), while the pipeline/stage order is inspect -> domain. Before A3 the
    disclosure used the insertion order and the display used the stage order.
    """
    skills = SkillRegistry(tmp_path)
    skills.create_skill("ecc-python-patterns", "Python patterns and idioms", "PATTERNS-BODY")
    skills.create_skill(
        "ecc-search-first", "Inspect existing code before writing new code", "SEARCH-FIRST-BODY"
    )
    skills.create_skill("ecc-tdd-workflow", "Implement features with TDD", "TDD-BODY")
    skills.create_skill(
        "ecc-verification-loop", "Verify results before claiming success", "VERIFY-BODY"
    )
    _give_minimal_metadata(
        skills,
        "ecc-python-patterns",
        "ecc-search-first",
        "ecc-tdd-workflow",
        "ecc-verification-loop",
    )
    return skills


def _whole_body_registry(tmp_path, *, with_emoji: bool = True):
    """One v1 skill without metadata; content exercises a token-cut boundary."""
    skills = SkillRegistry(tmp_path)
    if with_emoji:
        content = "Intro line.\n\n" + "\u00df" * 60 + "\nTAIL MARKER\n" + "z" * 2000
    else:
        content = "Short deployment notes. " * 40
    skills.create_skill("deploy", "Deploy steps safely", content, triggers=["deploy"])
    _give_minimal_metadata(skills, "deploy")
    return skills, content


def _two_whole_body_registry(tmp_path):
    skills = SkillRegistry(tmp_path)
    skills.create_skill(
        "alpha-tools", "Alpha tooling notes", "ALPHA BODY " * 20, triggers=["alpha"]
    )
    skills.create_skill("beta-tools", "Beta tooling notes", "BETA BODY " * 20, triggers=["beta"])
    _give_minimal_metadata(skills, "alpha-tools", "beta-tools")
    return skills


def _plan_disclosure(query: str, skills: SkillRegistry, **kwargs):
    from machina.skills.planned_steps import plan_disclosure
    from machina.skills.workflow import plan_workflow

    workflow = plan_workflow(query, skills=skills, task_state=kwargs.pop("task_state", None))
    kwargs.setdefault("catalog_revision", workflow.catalog_revision)
    return plan_disclosure(
        workflow.steps,
        skill_for=skills.get,
        content_for=skills.get_skill_content,
        **kwargs,
    )


# --- D01: one authoritative order ---------------------------------------------


def test_d01_first_disclosed_body_is_the_first_authoritative_step(tmp_path):
    """RED (pre-A3): the rendered stage list sorted `inspect and plan` ahead of
    `domain` while selection and disclosure led with the domain anchor, so the
    plan and the excerpts disagreed. Both now derive from one order."""
    from machina.skills.runtime import automatic_skill_context
    from machina.skills.workflow import plan_workflow

    skills = _coding_registry(tmp_path)
    plan = plan_workflow("Implement a Python API", skills=skills)
    text = automatic_skill_context("Implement a Python API", skills=skills, budget=1200)
    assert "### ecc-search-first" in text and "### ecc-python-patterns" in text
    first_stage_skill = plan.stages[0][1]
    positions = {
        skill_id: text.index(f"### {skill_id}")
        for skill_id in ("ecc-python-patterns", "ecc-search-first")
    }
    # The first displayed step is the first disclosed body: display, selection
    # and disclosure can no longer disagree.
    assert positions[first_stage_skill] == min(positions.values())
    # The task-specific anchor leads; the generic pipeline follows.
    assert text.index("### ecc-python-patterns") < text.index("### ecc-search-first")


def test_d01_steps_are_the_single_source_for_selection_display_and_disclosure(tmp_path):
    from machina.skills.workflow import plan_workflow

    skills = _coding_registry(tmp_path)
    plan = plan_workflow("Implement a Python API", skills=skills)
    assert [step.skill_id for step in plan.steps] == [
        "ecc-python-patterns",
        "ecc-search-first",
        "ecc-tdd-workflow",
        "ecc-verification-loop",
    ]
    # Display pairs derive from the same steps, in the same order.
    assert [(step.label, step.skill_id) for step in plan.steps] == plan.stages
    # The legacy `skills` view is the same sequence (no second ordering).
    assert [skill.name for skill in plan.skills] == [step.skill_id for step in plan.steps]
    assert len({step.step_id for step in plan.steps}) == len(plan.steps)


def test_steps_carry_stable_ids_stages_provenance_estimates_and_prerequisites(tmp_path):
    from machina.skills.workflow import plan_workflow

    skills = _coding_registry(tmp_path)
    plan = plan_workflow("Implement a Python API", skills=skills)
    domain = plan.steps[0]
    assert domain.step_id == "domain:ecc-python-patterns"
    assert domain.label == "domain"
    assert domain.stage == ""  # the anchor position makes no phase claim
    assert domain.provenance == "policy"
    assert domain.estimate_tokens == max(1, len("PATTERNS-BODY") // 4)
    assert domain.prerequisites == ()
    inspect = next(step for step in plan.steps if step.label == "inspect and plan")
    assert inspect.step_id == "inspect-and-plan:ecc-search-first"
    assert inspect.stage == "inspect"
    verify = next(step for step in plan.steps if step.label == "verify")
    assert verify.stage == "verify"
    assert verify.prerequisites == ("implement-and-test:ecc-tdd-workflow",)


def test_explicit_request_carries_user_provenance(tmp_path):
    from machina.skills.workflow import plan_workflow

    skills = SkillRegistry(tmp_path)
    skills.create_skill("team-workflow", "Follow team Python conventions", "TEAM RULE", triggers=["python"])
    _give_minimal_metadata(skills, "team-workflow")
    plan = plan_workflow("Implement a Python API", skills=skills)
    requested = next(step for step in plan.steps if step.label == "requested workflow")
    assert requested.provenance == "user"
    assert requested.skill_id == "team-workflow"


# --- §10.2: stage source honesty ----------------------------------------------


def test_stage_state_selects_current_stage_sections_and_prerequisites(tmp_path):
    from machina.skills.planned_steps import StageState
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=OK_BUDGET,
        stage_state=StageState("verify", source="user"),
    )
    assert "Current stage (user): verify" in text
    assert "### deploy-guide \u00b7 Deploy the service" in text
    assert "### deploy-guide \u00b7 Roll back" in text
    # The prerequisite is loaded before the section that requires it.
    assert text.index("Deploy the service") < text.index("Roll back")
    # The plan-stage section is not current and must not be inlined.
    assert "Verify credentials" not in text


def test_without_stage_state_only_bootstrap_loads_and_later_sections_are_locators(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context("help me deploy", skills=skills, budget=OK_BUDGET)
    assert "Current stage" not in text
    # Bootstrap: the preamble and the untagged (phase-neutral) section.
    assert "DEPLOY PREAMBLE" in text
    assert "Only deploy from a clean tree" in text
    # Later/phase-scoped sections are not claimed as loaded...
    assert "Run scripts/deploy.sh" not in text
    assert "Restore the previous release" not in text
    assert "Verify credentials" not in text
    assert "### deploy-guide \u00b7 Deploy the service" not in text
    # ...but they stay discoverable through stable locators.
    assert "Not loaded" in text
    assert "Deploy the service" in text
    assert "Roll back" in text


def test_sentence_phase_words_do_not_create_a_stage(tmp_path):
    """'now testing' in a sentence is never a reliable phase source (§10.2)."""
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy now, testing afterwards", skills=skills, budget=OK_BUDGET
    )
    assert "Current stage" not in text
    assert "Restore the previous release" not in text
    assert "### deploy-guide \u00b7 Roll back" not in text


def test_observed_stage_state_marks_the_matching_step(tmp_path):
    from machina.skills.workflow import plan_workflow

    skills = _coding_registry(tmp_path)
    plan = plan_workflow(
        "Implement a Python API",
        skills=skills,
        task_state={"stage": "verify", "stage_source": "observed", "revision": 4},
    )
    verify = next(step for step in plan.steps if step.label == "verify")
    assert verify.provenance == "observed"
    assert all(step.provenance == "policy" for step in plan.steps if step.label != "verify")


def test_invalid_stage_values_are_never_silently_accepted():
    from machina.skills.planned_steps import StageState, normalize_stage_state

    assert normalize_stage_state(None) is None
    assert normalize_stage_state({"stage": "not-a-stage"}) is None
    assert normalize_stage_state({"stage": "verify"}) == StageState("verify", source="user")
    assert normalize_stage_state({"stage": "verify", "stage_source": "observed"}) == StageState(
        "verify", source="observed"
    )
    with pytest.raises(ValueError):
        StageState("verify", source="wishful-thinking")


# --- D02: prerequisite in-or-explicit -----------------------------------------


def test_d02_required_prerequisite_is_loaded_whole(tmp_path):
    from machina.skills.planned_steps import StageState

    skills = _deploy_registry(tmp_path)
    plan = _plan_disclosure(
        "help me deploy",
        skills,
        budget=OK_BUDGET,
        stage_state=StageState("verify", source="user"),
        task_revision=7,
        catalog_revision="cat-1",
    )
    ids = [chosen.section_id for chosen in plan.chosen]
    assert ids[:2] == ["deploy-steps", "rollback"]
    assert plan.required_prerequisites[0].section_id == "deploy-steps"
    assert plan.unresolved_prerequisites == ()
    assert plan.missing_context == ()
    chosen = {item.section_id: item for item in plan.chosen}
    assert chosen["rollback"].required_by == ()
    assert chosen["deploy-steps"].required_by == ("rollback",)


def test_d02_missing_prerequisite_is_explicit_when_it_cannot_fit(tmp_path):
    from machina.skills.planned_steps import StageState

    skills = _deploy_registry(tmp_path)
    plan = _plan_disclosure(
        "help me deploy",
        skills,
        budget=10,
        stage_state=StageState("verify", source="user"),
    )
    assert plan.chosen == ()
    reasons = {(item.section_id, item.reason) for item in plan.unresolved_prerequisites}
    assert ("deploy-steps", "budget") in reasons
    assert any("rollback" in item for item in plan.missing_context)
    assert any(
        omitted.locator.section_id == "rollback" and omitted.reason == "budget"
        for omitted in plan.omitted
    )


def test_d02_mandatory_section_is_never_fragmented(tmp_path):
    """A section that will not fit whole is omitted whole + located."""
    from machina.skills.planned_steps import StageState
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=MID_BUDGET,
        stage_state=StageState("deliver", source="user"),
    )
    assert "Run scripts/deploy.sh" in text  # the deliver section fits whole
    # The verify section does not fit and is not present as a fragment.
    assert "Restore the previous" not in text
    assert "Restore the previous release immediately if any health check fails" not in text
    assert "Not loaded" in text


# --- D06: tiny allowance, no false "loaded" claim ------------------------------


def test_d06_tight_budget_never_claims_sections_were_loaded(tmp_path):
    from machina.skills.planned_steps import StageState
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=TIGHT_BUDGET,
        stage_state=StageState("verify", source="user"),
    )
    assert text
    assert "Run scripts/deploy.sh" not in text
    assert "Restore the previous release" not in text
    assert "### deploy-guide \u00b7 Deploy the service" not in text
    assert "### deploy-guide \u00b7 Roll back" not in text
    assert "Current stage (user): verify" in text
    assert "Not loaded" in text


def test_tiny_budget_without_stage_state_still_lists_locators(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context("help me deploy", skills=skills, budget=TIGHT_BUDGET)
    assert "Run scripts/deploy.sh" not in text
    assert "Not loaded" in text


# --- §10.3 disclosure contract --------------------------------------------------


def test_disclosure_plan_exposes_revisioned_contract_fields(tmp_path):
    from machina.skills.planned_steps import StageState
    from machina.skills.search import content_revision
    from machina.skills.workflow import plan_workflow

    skills = _deploy_registry(tmp_path)
    workflow = plan_workflow("help me deploy", skills=skills)
    plan = _plan_disclosure(
        "help me deploy",
        skills,
        budget=OK_BUDGET,
        stage_state=StageState("verify", source="user", revision=3),
        task_revision=7,
    )
    assert plan.task_revision == 7
    assert plan.stage_revision == 3
    assert plan.stage_source == "user"
    assert plan.current_stage == "verify"
    assert plan.catalog_revision == workflow.catalog_revision
    chosen = plan.chosen[0]
    assert chosen.skill_id == "deploy-guide"
    assert chosen.skill_revision == content_revision(skills.get("deploy-guide"))
    assert chosen.content_revision  # exact body fingerprint of the parsed text
    assert chosen.text.startswith("## Deploy the service")
    assert chosen.estimate_tokens >= max(1, len(chosen.text) // 4)
    assert plan.token_basis == "char/4"
    assert plan.included_tokens == sum(item.estimate_tokens for item in plan.chosen)
    later = {item.locator.section_id: item.reason for item in plan.omitted}
    assert later.get("preflight") == "later-stage"
    assert plan.to_dict()["current_stage"] == "verify"


def test_f2_included_tokens_counts_whole_body_fallback_blocks(tmp_path):
    """F2 follow-up: a metadata-less disclosure reports the blocks it loads.

    Before the fix ``included_tokens`` summed chosen sections only, so the
    legacy whole-body fallback (every skill in the current catalog) reported
    0 while the assembled context still carried the excerpt. Each whole-body
    unit is now measured as the assembled block —
    ``"\\n### {skill_id}\\n{body}\\n"`` (the shape
    ``runtime._planned_disclosure`` emits) — under the plan's own token
    counter, and ``included_tokens`` sums chosen sections plus whole-body
    units.
    """
    import json

    from machina.core.assembler import get_token_count
    from machina.skills.planned_steps import plan_disclosure
    from machina.skills.runtime import automatic_skill_context
    from machina.skills.workflow import plan_workflow

    skills, _raw = _whole_body_registry(tmp_path, with_emoji=False)
    content = skills.get_skill_content("deploy")
    block = f"\n### deploy\n{content}\n"
    text = automatic_skill_context("help me deploy", skills=skills, budget=OK_BUDGET)
    assert text.endswith(block), "the whole-body unit fits this budget uncut"

    workflow = plan_workflow("help me deploy", skills=skills)
    plan = plan_disclosure(
        workflow.steps,
        skill_for=skills.get,
        content_for=skills.get_skill_content,
        budget=OK_BUDGET,
        catalog_revision=workflow.catalog_revision,
        token_count=get_token_count,
        token_basis="assembler.get_token_count",
    )
    assert plan.chosen == ()
    assert [unit.skill_id for unit in plan.whole_body] == ["deploy"]
    # The estimate equals the assembled whole-body block under the same
    # counter: > 0, and the reported basis is the assembler's own.
    assert plan.included_tokens == get_token_count(block) > 0
    assert plan.whole_body[0].estimate_tokens == plan.included_tokens
    assert plan.token_basis == "assembler.get_token_count"
    json.dumps(plan.to_dict())


def test_task_constraints_render_before_optional_sections(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=OK_BUDGET,
        task_state={
            "stage": "verify",
            "stage_source": "user",
            "revision": 5,
            "constraints": ["read-only", "no-edit"],
        },
    )
    assert "read-only" in text and "no-edit" in text
    assert text.index("no-edit") < text.index("### deploy-guide \u00b7 Deploy the service")


def test_duplicate_heading_binding_is_deterministic_and_surfaced(tmp_path):
    from machina.skills.metadata import CODE_SECTION_DUPLICATE_HEADING
    from machina.skills.planned_steps import StageState

    skills = SkillRegistry(tmp_path)
    content = DEPLOY_CONTENT.replace(
        "## Deploy the service",
        "## Deploy the service\n\nfirst occurrence body\n\n## Deploy the service",
        1,
    )
    skills.create_skill("deploy-guide", "Deploy services safely", content, triggers=["deploy"])
    skills.load_all()
    skills.get("deploy-guide").skill_metadata = SkillMetadata(
        skill_id="deploy-guide",
        sections=(
            SectionMetadata(id="deploy-steps", heading="Deploy the service", stage="deliver"),
        ),
    )
    first = _plan_disclosure(
        "help me deploy", skills, budget=OK_BUDGET, stage_state=StageState("deliver", "user")
    )
    second = _plan_disclosure(
        "help me deploy", skills, budget=OK_BUDGET, stage_state=StageState("deliver", "user")
    )
    assert first.to_dict() == second.to_dict()
    assert any(d.code == CODE_SECTION_DUPLICATE_HEADING for d in first.diagnostics)
    # Declared sections bind to the first matching heading (A1 rule).
    assert first.chosen[0].text.startswith("## Deploy the service\n\nfirst occurrence body")


# --- parity: no metadata/sections -> pre-A3 whole-body output ------------------


def test_parity_no_metadata_single_skill_matches_pre_a3_output(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills, content = _whole_body_registry(tmp_path)
    text = automatic_skill_context("help me deploy", skills=skills, budget=PARITY_SINGLE_BUDGET)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PARITY_SINGLE_SHA256
    assert "### deploy\n" in text
    assert "\u00b7" not in text  # whole-body path never emits section markers
    assert "offset=" in text  # a token cut keeps its legacy continuation
    assert "/" in text


def test_parity_no_metadata_two_skill_plan_matches_pre_a3_output(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills = _two_whole_body_registry(tmp_path)
    text = automatic_skill_context("use alpha and beta tooling", skills=skills, budget=1200)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PARITY_TWO_SKILL_SHA256
    # Canonical order: the two-skill contents are byte-identical to the pre-A3
    # mechanism (reordering them on the same contents reproduces the old hash),
    # but canonical ranking scores "beta tooling" above "alpha tooling".
    assert text.index("### beta-tools") < text.index("### alpha-tools")


def test_parity_conversation_and_nonsense_return_empty(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills, _ = _whole_body_registry(tmp_path)
    assert automatic_skill_context("hello!", skills=skills) == ""
    assert automatic_skill_context("blorple snazzlewicket", skills=skills) == ""


def test_parity_restricted_context_keeps_its_legacy_phrasing(tmp_path):
    from machina.skills.runtime import automatic_skill_context

    skills, _ = _whole_body_registry(tmp_path)
    text = automatic_skill_context(
        "help me deploy", skills=skills, budget=600, can_read=False, can_search=False
    )
    assert "full instructions at" in text
    assert "skill_read" not in text


# --- rollback switch ------------------------------------------------------------


def test_rollback_switch_restores_legacy_disclosure(tmp_path, monkeypatch):
    from machina.skills.planned_steps import StageState
    from machina.skills.runtime import automatic_skill_context

    skills = _deploy_registry(tmp_path)
    monkeypatch.setenv("MACHINA_SKILLS_DISCLOSURE_PLANNER", "0")
    legacy = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=OK_BUDGET,
        stage_state=StageState("verify", source="user"),
    )
    assert "### deploy-guide \u00b7 " not in legacy
    assert "Run scripts/deploy.sh" in legacy  # whole-body excerpt, pre-A3 path
    monkeypatch.delenv("MACHINA_SKILLS_DISCLOSURE_PLANNER")
    planned = automatic_skill_context(
        "help me deploy",
        skills=skills,
        budget=OK_BUDGET,
        stage_state=StageState("verify", source="user"),
    )
    assert "### deploy-guide \u00b7 Deploy the service" in planned


# --- determinism ----------------------------------------------------------------


def test_plan_and_disclosure_are_deterministic(tmp_path):
    from machina.skills.planned_steps import StageState
    from machina.skills.workflow import plan_workflow

    skills = _deploy_registry(tmp_path)
    first = plan_workflow("help me deploy", skills=skills)
    second = plan_workflow("help me deploy", skills=skills)
    assert first.steps == second.steps
    assert first.catalog_revision == second.catalog_revision
    plan_a = _plan_disclosure(
        "help me deploy", skills, budget=800, stage_state=StageState("verify", source="user")
    )
    plan_b = _plan_disclosure(
        "help me deploy", skills, budget=800, stage_state=StageState("verify", source="user")
    )
    assert plan_a.to_dict() == plan_b.to_dict()
    assert plan_a.catalog_revision == second.catalog_revision


# --- no provider / embedding / external calls ----------------------------------


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


def test_shipped_source_imports_no_provider_or_embedding_code():
    for name in (
        "skills/planned_steps.py",
        "skills/sections.py",
        "skills/workflow.py",
        "skills/runtime.py",
    ):
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
        assert "__import__" not in text, name
        assert "subprocess" not in text, name


def test_fresh_import_pulls_no_provider_or_embedding_modules():
    script = textwrap.dedent(
        """
        import machina.skills.planned_steps as planned_steps  # noqa: F401
        import machina.skills.sections as sections  # noqa: F401
        import machina.skills.workflow as workflow  # noqa: F401
        import machina.skills.runtime as runtime  # noqa: F401
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
        check=False,
    )
    assert proc.returncode == 0, f"fresh import failed:\\n{proc.stderr}"
    assert proc.stdout.strip() == "ok"
