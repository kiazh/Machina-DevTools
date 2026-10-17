"""Deterministic skill search (spec §9): typed contract, applicability, explanations.

RED-first consumer-boundary tests for ``machina.skills.runtime.search_skills``
and the typed ``SearchResult`` contract in ``machina.skills.search``.

Policy under test (all deterministic, no model/embedding call possible):

* scores are RANKING SIGNALS with named components — never calibrated
  probabilities;
* applicability (applicable/conditional/excluded) is decided before ordering,
  and generic applicable skills are not crowded out by conditional technical
  ones;
* automatic search excludes disabled skills entirely; the browsing path
  (``include_excluded=True``) retains them with an ``excluded`` flag;
* no-match never enumerates the catalog nor forces a top pick;
* catalog growth (unrelated skills, duplicated generic wording) must not
  displace a multi-signal top candidate.
"""

from __future__ import annotations

import socket

import pytest

from machina.skills.registry import Skill, SkillRegistry
from machina.skills.runtime import automatic_skills, search_skills
from machina.skills.search import (
    APPLICABLE,
    CONDITIONAL,
    EXCLUDED,
    POLICY_REVISION,
    QUERY_DESCRIPTOR_REVISION,
    SearchResult,
)


class _Registry(SkillRegistry):
    """Test registry whose created skills carry minimal schema-v2 metadata.

    Fail-closed policy: ``create_skill`` writes a v1 document, which is
    intentionally unusable until metadata generation runs. Tests that are not
    about that policy persist a minimal valid v2 document so search/runtime
    behavior under test is unchanged.
    """

    def create_skill(self, *args, **kwargs):
        from machina.skills.metadata import SkillMetadata

        skill = super().create_skill(*args, **kwargs)
        self.load_all()
        current = self.get(skill.name)
        self.persist_skill_metadata(current, SkillMetadata(skill_id=current.name))
        return skill


def _registry(tmp_path, name: str = "skills") -> SkillRegistry:
    return _Registry(tmp_path / name)


def _persist_minimal_metadata(reg: SkillRegistry, name: str) -> None:
    """Fail-closed policy: every test skill needs schema-v2 metadata to be usable."""
    from machina.skills.metadata import SkillMetadata

    reg.load_all()
    skill = reg.get(name)
    if skill is not None and skill.skill_metadata is None:
        reg.persist_skill_metadata(skill, SkillMetadata(skill_id=skill.name))


def _skill(
    reg: SkillRegistry,
    name: str,
    description: str,
    *,
    triggers: list[str] | None = None,
    content: str = "Steps",
) -> Skill:
    skill = reg.create_skill(name, description, content, triggers=triggers or [])
    _persist_minimal_metadata(reg, name)
    return skill


def _attach_metadata(reg: SkillRegistry, metadata_by_name: dict) -> None:
    """Prime the parse cache, then attach metadata to the live objects.

    ``create_skill`` invalidates the registry's parse cache, so the next
    ``load_all`` would re-parse from disk and replace the live objects. Priming
    first makes the search's internal ``load_all`` a no-op, and the in-memory
    metadata survives — the same shape a parsed-metadata source will have.
    """
    reg.load_all()
    for name, metadata in metadata_by_name.items():
        reg.get(name).metadata = metadata


def _attach_skill_metadata(reg: SkillRegistry, metadata_by_name: dict) -> None:
    """Attach validated schema-v2 metadata objects to the live skills."""
    reg.load_all()
    for name, metadata in metadata_by_name.items():
        reg.get(name).skill_metadata = metadata


def _ids(result: SearchResult) -> list[str]:
    return [candidate.skill_id for candidate in result.candidates]


def _by_id(result: SearchResult) -> dict:
    return {candidate.skill_id: candidate for candidate in result.candidates}


def _filler(reg: SkillRegistry, index: int) -> None:
    name = f"lathe-{index:02d}"
    reg.create_skill(name, f"Lathe calibration notes for wooden bowls {index}", "S")
    _persist_minimal_metadata(reg, name)


def test_search_result_contract_and_ranking_signal_semantics(tmp_path):
    reg = _registry(tmp_path)
    _skill(
        reg,
        "python-testing",
        "Run pytest suites for python testing",
        triggers=["pytest"],
    )
    result = search_skills(reg, "run pytest for python testing")

    assert isinstance(result, SearchResult)
    assert result.query_descriptor_revision == QUERY_DESCRIPTOR_REVISION
    assert result.policy_revision == POLICY_REVISION
    assert isinstance(result.catalog_revision, str) and result.catalog_revision
    assert isinstance(result.candidates, list) and result.candidates
    assert result.next_cursor is None
    assert isinstance(result.diagnostics, dict)

    candidate = result.candidates[0]
    assert candidate.skill_id == "python-testing"
    assert candidate.name == "python-testing"
    assert isinstance(candidate.content_revision, str) and candidate.content_revision
    assert isinstance(candidate.score, float)
    assert candidate.score_components
    assert all(isinstance(value, float) for value in candidate.score_components.values())
    # The score is derived, not invented: it is exactly the sum of the named
    # components (a bounded preference delta is itself a component).
    assert candidate.score == pytest.approx(sum(candidate.score_components.values()), abs=1e-9)
    assert isinstance(candidate.matched_phrases, tuple)
    assert "pytest" in candidate.matched_phrases
    assert isinstance(candidate.matched_capabilities, tuple)
    assert candidate.applicability == APPLICABLE
    assert isinstance(candidate.reasons, tuple)
    assert isinstance(candidate.workflow_stages, tuple)
    assert isinstance(candidate.explanation, str)
    # Scores are ranking signals, never calibrated probabilities.
    assert "ranking signal" in candidate.explanation
    assert "not a calibrated probability" in candidate.explanation

    # The catalog revision is deterministic and changes when content changes.
    same = search_skills(reg, "completely different wording")
    assert same.catalog_revision == result.catalog_revision
    reg.update_skill("python-testing", content="Updated steps for testing.")
    changed = search_skills(reg, "run pytest for python testing")
    assert changed.catalog_revision != result.catalog_revision
    assert changed.candidates[0].content_revision != candidate.content_revision


def test_deterministic_ties_prefer_source_trust_then_casefolded_name(tmp_path):
    reg = _registry(tmp_path, "ties")
    reg.create_skill("zebra-tool", "Widget tuning shared guidance", "S", source_type="local")
    reg.create_skill("apple-tool", "Widget tuning shared guidance", "S", source_type="learned")
    _filler(reg, 0)
    _filler(reg, 1)

    result = search_skills(reg, "widget tuning shared guidance")
    # Identical score: source precedence (local, curated) beats the
    # case-insensitive name order the same-score pair would otherwise use.
    assert _ids(result) == ["zebra-tool", "apple-tool"]
    assert search_skills(reg, "widget tuning shared guidance").candidates == result.candidates

    names = _registry(tmp_path, "names")
    names.create_skill("beta-helper", "Shared widget guidance", "S")
    names.create_skill("alpha-helper", "Shared widget guidance", "S")
    _filler(names, 0)
    _filler(names, 1)
    # Insertion order must not leak: equal scores resolve by name.
    assert _ids(search_skills(names, "shared widget guidance")) == ["alpha-helper", "beta-helper"]


def test_disabled_skills_are_excluded_automatically_and_flagged_when_browsing(tmp_path):
    reg = _registry(tmp_path)
    reg.create_skill("team-deploy", "Deploy services carefully", "S", triggers=["deploy"])
    reg.create_skill("old-deploy", "Deploy services with legacy steps", "S", triggers=["deploy"])
    reg.set_enabled("old-deploy", False)

    automatic = search_skills(reg, "deploy the service")
    assert "old-deploy" not in _ids(automatic)
    assert all(candidate.applicability == APPLICABLE for candidate in automatic.candidates)

    browsed = search_skills(reg, "deploy the service", include_excluded=True)
    candidate = _by_id(browsed)["old-deploy"]
    assert candidate.applicability == EXCLUDED
    assert any("disabled" in reason for reason in candidate.reasons)
    assert browsed.diagnostics["excluded"] >= 1


def test_stack_unknown_is_conditional_and_missing_tool_is_excluded(tmp_path):
    reg = _registry(tmp_path)
    reg.create_skill("rust-async", "Async service refactoring patterns for rust backends", "S")
    _filler(reg, 0)

    result = search_skills(reg, "refactor async services")
    candidate = result.candidates[0]
    assert candidate.skill_id == "rust-async"
    assert candidate.applicability == CONDITIONAL
    assert any("rust" in reason for reason in candidate.reasons)
    assert result.diagnostics["classification"] == "conditional_only"
    assert not any(c.applicability == APPLICABLE for c in result.candidates)

    # Explicitly naming the skill preserves the user's selection.
    explicit = search_skills(reg, "use rust-async to refactor services")
    assert explicit.candidates[0].applicability == APPLICABLE
    assert explicit.candidates[0].reasons == ()

    facts = _registry(tmp_path, "tools")
    _skill(facts, "live-facts", "Fetch current facts from the web", triggers=["current facts"])
    _attach_metadata(facts, {"live-facts": {"requires_tools": ["web_search"]}})
    blocked = search_skills(facts, "current facts about rust", available_tools={"read_file"})
    blocked_candidate = blocked.candidates[0]
    assert blocked_candidate.applicability == EXCLUDED
    assert any("web_search" in reason for reason in blocked_candidate.reasons)
    assert blocked.diagnostics["classification"] == "missing_capability"

    unknown = search_skills(facts, "current facts about rust")
    assert unknown.candidates[0].applicability == CONDITIONAL
    assert any("unknown" in reason for reason in unknown.candidates[0].reasons)


def test_conditional_technical_skill_does_not_crowd_out_generic_applicable_skill(tmp_path):
    reg = _registry(tmp_path)
    _skill(
        reg,
        "code-quality",
        "Improve code quality and readability for the team",
        triggers=["code quality"],
    )
    _skill(
        reg, "rust-async", "Improve code quality for async rust services", triggers=["code quality"]
    )
    _attach_metadata(reg, {"rust-async": {"stack": ["rust"]}})

    result = search_skills(reg, "improve code quality on the login screen")
    by_id = _by_id(result)
    assert by_id["rust-async"].applicability == CONDITIONAL
    assert any("rust" in reason for reason in by_id["rust-async"].reasons)
    assert by_id["code-quality"].applicability == APPLICABLE
    # Both are trigger matches; the conditional specialist must not outrank
    # the applicable generic skill, whatever the lexical score says.
    assert result.candidates[0].skill_id == "code-quality"
    assert result.diagnostics["classification"] == "ok"


def test_task_family_signal_from_plain_phrasing_and_declared_families(tmp_path):
    reg = _registry(tmp_path)
    reg.create_skill("incident-triage", "Triage failures and diagnose crashes in services", "S")
    _filler(reg, 0)

    result = search_skills(reg, "diagnose why the app keeps crashing")
    candidate = _by_id(result)["incident-triage"]
    assert candidate.score_components.get("task_family", 0.0) > 0.0
    assert "debug" in candidate.workflow_stages
    assert candidate.applicability == APPLICABLE

    declared = _registry(tmp_path, "declared")
    _skill(declared, "ops-runbook", "Run operation checklists")
    _filler(declared, 0)
    _attach_metadata(declared, {"ops-runbook": {"task_families": ["debugging"]}})
    # Declared families admit on their own (no lexical overlap with the query).
    declared_result = search_skills(declared, "it keeps crashing")
    family_candidate = _by_id(declared_result)["ops-runbook"]
    assert family_candidate.score_components["task_family"] > 0.0
    assert "debug" in family_candidate.workflow_stages


def test_catalog_growth_and_generic_word_duplication_keep_top_candidate(tmp_path):
    """Adding 20 unrelated skills, then 25 duplicating a generic word.

    Documented catalog-dependent change: the rider ``misc-notes`` is admitted
    in the 2-skill baseline (its only overlap word is present in 1/2 docs) and
    is gated out once 26/47 docs share that generic word. The multi-signal top
    candidate (boundary trigger + rare terms) is unaffected in both stages; the
    result never enumerates the catalog.
    """
    reg = _registry(tmp_path)
    reg.create_skill(
        "payments-service", "Refund and retry payment APIs", "S", triggers=["payments"]
    )
    reg.create_skill("misc-notes", "General handling notes for bowls", "S")

    baseline = search_skills(reg, "add refund handling for payments", limit=50)
    assert baseline.candidates[0].skill_id == "payments-service"
    assert "misc-notes" in _ids(baseline)

    for index in range(20):
        reg.create_skill(f"unrelated-{index:02d}", f"Lathe calibration log {index}", "S")
    grown = search_skills(reg, "add refund handling for payments", limit=50)
    assert grown.candidates[0].skill_id == "payments-service"
    assert "misc-notes" in _ids(grown)

    for index in range(25):
        reg.create_skill(f"duplicator-{index:02d}", f"Handling wooden bowls safely {index}", "S")
    duplicated = search_skills(reg, "add refund handling for payments", limit=50)
    assert duplicated.candidates[0].skill_id == "payments-service"
    assert "misc-notes" not in _ids(duplicated)
    assert duplicated.diagnostics["catalog_size"] == 47
    assert len(duplicated.candidates) < duplicated.diagnostics["catalog_size"]


def test_all_excluded_candidates_never_force_a_top_pick(tmp_path):
    reg = _registry(tmp_path)
    _skill(reg, "prod-gate", "Release gating guidance", triggers=["production releases"])
    _attach_metadata(reg, {"prod-gate": {"not_for": ["production releases"]}})
    result = search_skills(reg, "run production releases")
    assert result.candidates
    assert all(candidate.applicability == EXCLUDED for candidate in result.candidates)
    assert any("production releases" in reason for reason in result.candidates[0].reasons)
    assert result.diagnostics["classification"] == "all_excluded"


def test_empty_catalog_and_no_match_never_enumerate_the_catalog(tmp_path):
    empty = search_skills(_registry(tmp_path, "none"), "anything at all")
    assert empty.candidates == []
    assert empty.diagnostics["classification"] == "invalid_catalog"

    reg = _registry(tmp_path, "catalog")
    for index in range(30):
        reg.create_skill(f"catalog-{index:02d}", f"Topic number {index} reference", "S")
    no_match = search_skills(reg, "kwyjibo zzzqx")
    assert no_match.candidates == []
    assert no_match.diagnostics["classification"] == "no_lexical_evidence"
    assert no_match.diagnostics["catalog_size"] == 30
    assert no_match.next_cursor is None

    blank = search_skills(reg, "")
    assert blank.candidates == []
    assert blank.diagnostics["classification"] == "no_lexical_evidence"


def test_cursor_pages_deterministically(tmp_path):
    reg = _registry(tmp_path)
    for index in range(5):
        reg.create_skill(f"widget-tool-{index}", "Widget alignment guidance", "S")
    for index in range(15):
        _filler(reg, index)

    full = search_skills(reg, "widget alignment guidance", limit=10)
    assert _ids(full) == [f"widget-tool-{index}" for index in range(5)]
    assert full.next_cursor is None

    first = search_skills(reg, "widget alignment guidance", limit=2)
    assert len(first.candidates) == 2 and first.next_cursor
    second = search_skills(reg, "widget alignment guidance", limit=2, cursor=first.next_cursor)
    assert len(second.candidates) == 2 and second.next_cursor
    third = search_skills(reg, "widget alignment guidance", limit=2, cursor=second.next_cursor)
    assert len(third.candidates) == 1 and third.next_cursor is None
    assert _ids(first) + _ids(second) + _ids(third) == _ids(full)


def test_bounded_weights_never_override_eligibility(tmp_path):
    reg = _registry(tmp_path)
    reg.create_skill("rust-async", "Async service refactoring patterns for rust backends", "S")
    _filler(reg, 0)
    plain = search_skills(reg, "refactor async services")
    weighted = search_skills(reg, "refactor async services", weights={"rust-async": 50.0})

    candidate = weighted.candidates[0]
    assert candidate.applicability == CONDITIONAL  # eligibility ignores weights
    assert candidate.score_components["preference"] > 0.0
    assert candidate.score == pytest.approx(plain.candidates[0].score * 1.2)

    reg.create_skill("old-deploy", "Deploy services", "S", triggers=["deploy"])
    reg.set_enabled("old-deploy", False)
    browsed = search_skills(
        reg, "deploy the service", include_excluded=True, weights={"old-deploy": 50.0}
    )
    assert _by_id(browsed)["old-deploy"].applicability == EXCLUDED


def test_alias_normalization_preserves_cpp_csharp_dotnet(tmp_path):
    reg = _registry(tmp_path)
    reg.create_skill("cpp-export", "C++ export bindings", "S", triggers=["c++"])
    reg.create_skill("csharp-export", "C# export bindings", "S", triggers=["c#"])
    reg.create_skill("dotnet-export", ".NET export bindings", "S", triggers=[".net"])

    for query, expected, phrase in (
        ("fix the c++ export", "cpp-export", "c++"),
        ("fix the c# export", "csharp-export", "c#"),
        ("fix the .net export", "dotnet-export", ".net"),
    ):
        result = search_skills(reg, query)
        assert result.candidates[0].skill_id == expected
        assert phrase in result.candidates[0].matched_phrases
        assert result.candidates[0].applicability == APPLICABLE

    # Boundary-aware triggers: "c++builder" is not a c++ mention.
    assert search_skills(reg, "use c++builder syntax").candidates == []


def test_typo_tolerance_is_off_by_default_and_bounded_when_enabled(tmp_path):
    reg = _registry(tmp_path)
    _skill(reg, "python-patterns", "Python patterns and idioms for services", triggers=["python"])
    _skill(reg, "registry-notes", "Registry maintenance notes")

    off = search_skills(reg, "pyton idiums")
    assert off.diagnostics["corrections"] == []
    assert off.candidates == []

    on = search_skills(reg, "pyton idiums", typo_tolerance=True)
    corrections = dict(on.diagnostics["corrections"])
    assert corrections["pyton"] == "python"
    assert corrections["idiums"] == "idioms"
    assert "python-patterns" in _ids(on)

    # Paths and id-like tokens are never fuzzy-corrected, even when the
    # correction target exists in the vocabulary.
    guarded = search_skills(reg, "update user_regsitry /etc/regsitry now", typo_tolerance=True)
    assert "regsitry" not in dict(guarded.diagnostics["corrections"])


def test_metadata_is_consumed_defensively_around_the_v1_baseline(tmp_path):
    reg = _registry(tmp_path)
    _skill(reg, "plain-helper", "Widget guidance for the team")
    _skill(reg, "string-stack", "Widget guidance for the team")
    _skill(reg, "weird-helper", "Widget guidance for the team")
    _skill(reg, "none-helper", "Widget guidance for the team")
    for index in range(4):
        _filler(reg, index)
    _attach_metadata(
        reg,
        {
            "string-stack": {"stack": "rust"},
            "weird-helper": {
                "stack": 42,
                "requires_tools": None,
                "task_families": 42,
                "not_for": {"no", "sets"},
            },
            "none-helper": None,
        },
    )

    result = search_skills(reg, "widget guidance for the team")
    by_id = _by_id(result)
    assert set(by_id) == {"plain-helper", "string-stack", "weird-helper", "none-helper"}
    # A scalar string is tolerated as a one-item stack; junk types fall back to
    # the v1 baseline (no requirements) instead of crashing the search.
    assert by_id["plain-helper"].applicability == APPLICABLE
    assert by_id["weird-helper"].applicability == APPLICABLE
    assert by_id["none-helper"].applicability == APPLICABLE
    assert by_id["string-stack"].applicability == CONDITIONAL


def test_typed_schema_v2_metadata_drives_signals_and_applicability(tmp_path):
    """A1's validated SkillMetadata is consumed without a hard import.

    ``machina.skills.search`` reads ``skill.skill_metadata`` via getattr, so
    the v1 baseline keeps working when the attribute is absent/None.
    """
    from machina.skills.metadata import SectionMetadata, SkillMetadata, StackMetadata

    reg = _registry(tmp_path)
    _skill(reg, "typed-ship", "Ship services", triggers=["ship"])
    _skill(reg, "typed-rust", "Async service patterns", triggers=["async patterns"])
    _skill(reg, "typed-facts", "Fetch live values", triggers=["live values"])
    _skill(reg, "typed-gate", "Release gating", triggers=["gating"])
    _skill(reg, "typed-align", "Widget alignment guidance")
    _skill(reg, "typed-cred", "Production deployment steps", triggers=["deployment steps"])
    _filler(reg, 0)
    _attach_skill_metadata(
        reg,
        {
            "typed-ship": SkillMetadata(
                task_families=("deployment",),
                sections=(SectionMetadata(id="deliver-step", heading="Deliver", stage="deliver"),),
            ),
            "typed-rust": SkillMetadata(stack=StackMetadata(languages=("Rust",))),
            "typed-facts": SkillMetadata(required_tools=("web_search",)),
            "typed-gate": SkillMetadata(exclusions=("release gating",)),
            "typed-align": SkillMetadata(aliases=("widget-align",)),
            "typed-cred": SkillMetadata(preconditions=("production credentials",)),
        },
    )

    ship = _by_id(search_skills(reg, "ship the service"))["typed-ship"]
    assert ship.applicability == APPLICABLE
    assert ship.score_components["task_family"] > 0.0
    assert "deliver" in ship.workflow_stages

    rust = _by_id(search_skills(reg, "refactor async patterns"))["typed-rust"]
    assert rust.applicability == CONDITIONAL
    assert any("rust" in reason for reason in rust.reasons)

    facts = _by_id(search_skills(reg, "live values now", available_tools={"read_file"}))[
        "typed-facts"
    ]
    assert facts.applicability == EXCLUDED
    assert any("web_search" in reason for reason in facts.reasons)

    gate = _by_id(search_skills(reg, "release gating please"))["typed-gate"]
    assert gate.applicability == EXCLUDED
    assert any("release gating" in reason for reason in gate.reasons)

    align = _by_id(search_skills(reg, "use widget-align for the layout"))["typed-align"]
    assert align.applicability == APPLICABLE
    assert align.score_components["exact_name"] > 0.0
    assert "widget-align" in align.matched_phrases

    cred = _by_id(search_skills(reg, "deployment steps"))["typed-cred"]
    assert cred.applicability == CONDITIONAL
    assert any("production credentials" in reason for reason in cred.reasons)


def test_schema_v2_frontmatter_flows_through_the_registry_into_search(tmp_path):
    root = tmp_path / "catalog"
    skill_dir = root / "typed-ship"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: typed-ship\n"
        "description: Ship services through gated releases\n"
        "triggers: [ship]\n"
        "schema_version: 2\n"
        "skill_id: typed-ship\n"
        "task_families: [deployment]\n"
        "stack:\n"
        "  languages: [rust]\n"
        "  frameworks: []\n"
        "required_tools: [web_search]\n"
        "sections:\n"
        "  - id: deliver-step\n"
        "    heading: Deliver\n"
        "    stage: deliver\n"
        "---\n\n## Deliver\n\nDeliver safely.\n",
        encoding="utf-8",
    )
    reg = SkillRegistry(root)
    auto = search_skills(reg, "ship the service")
    candidate = auto.candidates[0]
    # Unknown tool availability and an unconfirmed stack keep it conditional.
    assert candidate.applicability == CONDITIONAL
    assert any("web_search" in reason for reason in candidate.reasons)
    assert any("rust" in reason for reason in candidate.reasons)
    assert candidate.score_components["task_family"] > 0.0
    assert "deliver" in candidate.workflow_stages

    blocked = search_skills(reg, "ship the service", available_tools={"read_file"})
    assert blocked.candidates[0].applicability == EXCLUDED
    assert blocked.diagnostics["classification"] == "missing_capability"


def test_score_components_and_explanations_trace_actual_decisions(tmp_path):
    reg = _registry(tmp_path)
    _skill(reg, "python-testing", "Testing guidance for python services", triggers=["pytest"])
    result = search_skills(reg, "run pytest for python testing")
    candidate = result.candidates[0]

    for component in ("exact_name", "trigger", "lexical", "phrase", "name_overlap"):
        assert component in candidate.score_components, component
        assert candidate.score_components[component] > 0.0, component
    assert "exact name" in candidate.explanation
    assert "trigger" in candidate.explanation
    assert "lexical" in candidate.explanation

    spelled_out = search_skills(reg, "use python-testing on this task")
    top = spelled_out.candidates[0]
    assert top.score_components["exact_name"] > 0.0
    assert "exact name" in top.explanation

    # No trigger fired in an unrelated-but-explicit query naming the skill.
    assert "trigger" not in top.score_components


def test_no_model_or_embedding_calls_on_the_search_path(tmp_path, monkeypatch):
    from machina.core import provider as provider_module

    def forbidden(*args, **kwargs):
        raise AssertionError("skill search must not call a model or embedding provider")

    monkeypatch.setattr(provider_module, "get_provider", forbidden, raising=False)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    reg = _registry(tmp_path)
    reg.create_skill("deploy", "Deploy services with care", "S", triggers=["deploy"])
    result = search_skills(reg, "deploy the service")
    assert result.candidates and result.candidates[0].skill_id == "deploy"


def test_automatic_skills_contract_excludes_disabled(tmp_path):
    """The automatic consumer returns plain ranked skills with disabled gone.

    The legacy ``rank_skills`` selector was deleted (A2c); automatic selection,
    the human browse path and the ``skills.ports`` dispatch all read
    ``search_skills``, so this is the only ranked-skill-list contract left.
    """
    reg = _registry(tmp_path)
    reg.create_skill("team-deploy", "Deploy services carefully", "S", triggers=["deploy"])
    reg.create_skill("old-deploy", "Deploy services with legacy steps", "S", triggers=["deploy"])
    reg.set_enabled("old-deploy", False)

    ranked = automatic_skills(reg, "deploy payments")
    assert isinstance(ranked, list)
    assert all(isinstance(skill, Skill) for skill in ranked)
    assert [skill.name for skill in ranked] == ["team-deploy"]


# ─── vague web-build tolerance (deterministic catalogue, no model call) ──────


@pytest.mark.parametrize(
    "query",
    [
        "make this website",
        "build a landing page for my business",
        "make a web app",
    ],
)
def test_vague_web_build_queries_reach_web_frontend(query):
    """A vague build request must find the frontend skill deterministically."""
    registry = SkillRegistry()
    result = search_skills(registry, query, limit=10)
    ids = _ids(result)
    assert "web-frontend" in ids[:3], ids

    automatic = automatic_skills(registry, query)
    assert "web-frontend" in [skill.name for skill in automatic][:3]


def test_website_query_surfaces_the_related_skill_cluster():
    registry = SkillRegistry()
    ids = _ids(search_skills(registry, "make this website", limit=10))
    assert "web-frontend" in ids[:3]
    assert "data-visualization" in ids
    assert "docs-communication" in ids


def test_dashboard_query_leads_with_data_visualization():
    registry = SkillRegistry()
    ids = _ids(search_skills(registry, "create a dashboard", limit=10))
    assert ids[0] == "data-visualization", ids
