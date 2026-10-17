"""A3 — parsed sections, deterministic anchors, version-bound reads (spec §10.4).

RED-first tests for ``machina.skills.sections``:

* headings are parsed once per content revision and return the same index;
* anchors are deterministic and duplicate headings get stable suffixes;
* declared sections bind through ``metadata.bind_section_anchors`` against the
  exact parsed revision; unsupported/stale reads are diagnostics, never
  silent fallbacks;
* required-section edges form a deterministic acyclic graph (cycle rejection);
* D05: a continuation against edited content returns stale-version instead of
  a different passage at the same offset, while the legacy character-offset
  tool read keeps its exact meaning.
"""

from __future__ import annotations

import inspect

import pytest

from machina.skills.metadata import (
    CODE_DEPENDENCY_UNKNOWN,
    CODE_SECTION_DUPLICATE_HEADING,
    CODE_SECTION_MISSING_ANCHOR,
    SectionMetadata,
    SkillMetadata,
)

GUIDE = (
    "Intro paragraph before any heading.\n\n"
    "# Deploy guide\n\n"
    "Run the guide top to bottom.\n\n"
    "## Run tests\n\n"
    "First run the focused suite, then the full suite.\n\n"
    "## Run tests\n\n"
    "The repeated heading must get its own stable anchor.\n\n"
    "## Roll back\n\n"
    "Restore the previous release if checks fail.\n"
)


def _guide_metadata() -> SkillMetadata:
    return SkillMetadata(
        skill_id="guide",
        sections=(
            SectionMetadata(id="tests", heading="Run tests", stage="verify"),
            SectionMetadata(id="rollback", heading="Roll back", stage="verify"),
        ),
    )


# --- parsing + anchors ---------------------------------------------------------


def test_index_is_parsed_once_per_content_revision():
    from machina.skills.sections import clear_section_cache, content_fingerprint, index_sections

    clear_section_cache()
    first = index_sections(GUIDE)
    second = index_sections(GUIDE)
    assert first is second
    assert first.revision == content_fingerprint(GUIDE)
    edited = index_sections(GUIDE + "\n## Extra\n\nMore.\n")
    assert edited.revision != first.revision
    clear_section_cache()
    fresh = index_sections(GUIDE)
    assert fresh is not first
    assert fresh.revision == first.revision
    assert [(s.section_id, s.start, s.end) for s in fresh.sections] == [
        (s.section_id, s.start, s.end) for s in first.sections
    ]


def test_declared_section_binding_is_reused_within_a_revision():
    from machina.skills.sections import bound_sections, clear_section_cache

    clear_section_cache()
    metadata = _guide_metadata()
    first, diagnostics = bound_sections(metadata, GUIDE)
    second, diagnostics_again = bound_sections(metadata, GUIDE)
    assert second is first  # one A1 bind per content revision + metadata identity
    assert diagnostics_again == diagnostics
    edited, _ = bound_sections(metadata, GUIDE + "\n## Extra\n\nx\n")
    assert edited is not first


def test_duplicate_headings_get_deterministic_suffixes():
    from machina.skills.sections import index_sections

    index = index_sections(GUIDE)
    ids = [section.section_id for section in index.sections]
    assert ids == ["deploy-guide", "run-tests", "run-tests-2", "roll-back"]
    # Slices tile the body without overlaps and stay addressable by anchor.
    for section in index.sections:
        assert GUIDE[section.start : section.end] == section.text
    assert index.find("run-tests").text != index.find("run-tests-2").text
    assert any(d.code == CODE_SECTION_DUPLICATE_HEADING for d in index.diagnostics)
    assert index.preamble.startswith("Intro paragraph")


def test_read_section_returns_exact_slices_and_continuations():
    from machina.skills.sections import read_section

    full = read_section(GUIDE, "run-tests-2")
    assert not full.stale
    assert full.text.startswith("## Run tests")
    assert GUIDE[full.start : full.end] == full.text
    assert full.continuation is None

    pieces = []
    offset = 0
    while True:
        page = read_section(GUIDE, "run-tests-2", offset=offset, limit=10)
        pieces.append(page.text)
        if page.continuation is None:
            break
        offset = page.continuation
    assert "".join(pieces) == full.text


def test_unicode_continuation_tiles_without_gaps_or_repeats():
    from machina.skills.sections import read_section

    content = "## Emoji\n\n" + "\U0001f600" * 12 + "\nend\n"
    full = read_section(content, "emoji")
    pieces = []
    offset = 0
    while True:
        page = read_section(content, "emoji", offset=offset, limit=5)
        pieces.append(page.text)
        if page.continuation is None:
            break
        offset = page.continuation
    assert "".join(pieces) == full.text


# --- D05: stale revisions ------------------------------------------------------


def test_d05_edited_content_returns_stale_version_not_wrong_passage():
    from machina.skills.sections import (
        CODE_SECTION_STALE,
        content_fingerprint,
        read_section,
    )

    first = read_section(GUIDE, "run-tests", limit=20)
    assert not first.stale
    assert first.text == GUIDE[first.start : first.start + 20]
    edited = GUIDE.replace("Intro paragraph before any heading.", "A much longer edited introduction.")
    assert content_fingerprint(edited) != first.content_revision

    continuation = read_section(
        edited,
        "run-tests",
        expected_revision=first.content_revision,
        offset=20,
    )
    assert continuation.stale is True
    assert continuation.text == ""
    assert continuation.content_revision == content_fingerprint(edited)
    assert any(d.code == CODE_SECTION_STALE for d in continuation.diagnostics)
    # The same continuation works once the caller re-binds to the new revision.
    fresh = read_section(
        edited,
        "run-tests",
        expected_revision=continuation.content_revision,
        offset=20,
    )
    assert not fresh.stale
    assert fresh.text


def test_d05_registry_edit_between_pages_returns_stale(tmp_path):
    """Real consumer path: an edited SKILL.md invalidates a continuation."""
    from machina.skills.registry import SkillRegistry
    from machina.skills.sections import read_skill_section

    skills = SkillRegistry(tmp_path)
    skills.create_skill("guide", "A guide", GUIDE, triggers=["guide"])
    skills.load_all()
    skills.get("guide").skill_metadata = _guide_metadata()
    first = read_skill_section(
        skills.get("guide"), "tests", content=skills.get_skill_content("guide"), limit=20
    )
    assert not first.stale
    edited = GUIDE.replace(
        "Intro paragraph before any heading.", "A much longer edited introduction."
    )
    skills.update_skill("guide", content=edited)
    continuation = read_skill_section(
        skills.get("guide"),
        "tests",
        content=skills.get_skill_content("guide"),
        expected_revision=first.content_revision,
        offset=20,
    )
    assert continuation.stale is True
    assert continuation.text == ""
    assert continuation.content_revision != first.content_revision


# --- declared sections + diagnostics -------------------------------------------


def test_declared_sections_bind_via_metadata_anchor_binding():
    from machina.skills.sections import read_section

    result = read_section(GUIDE, "rollback", metadata=_guide_metadata())
    assert result.bound is True
    assert result.section_id == "rollback"
    assert result.heading == "Roll back"
    assert result.text.startswith("## Roll back")

    # combined #48: a duplicate heading is ambiguous, so it is not bound.
    duplicate = read_section(GUIDE, "tests", metadata=_guide_metadata())
    assert duplicate.bound is False


def test_missing_anchor_is_an_explicit_diagnostic():
    from machina.skills.sections import CODE_SECTION_UNKNOWN, read_section

    metadata = SkillMetadata(
        skill_id="guide",
        sections=(SectionMetadata(id="missing", heading="No Such Heading", stage="verify"),),
    )
    result = read_section(GUIDE, "missing", metadata=metadata)
    assert result.text == ""
    assert result.bound is False
    assert any(d.code == CODE_SECTION_MISSING_ANCHOR for d in result.diagnostics)
    unknown = read_section(GUIDE, "also-missing", metadata=metadata)
    assert unknown.text == ""
    assert any(d.code == CODE_SECTION_UNKNOWN for d in unknown.diagnostics)


def test_undeclared_anchor_is_served_with_a_diagnostic_not_silently():
    from machina.skills.metadata import CODE_SECTION_MISSING_ANCHOR
    from machina.skills.sections import CODE_SECTION_UNDECLARED, read_section

    metadata = SkillMetadata(
        skill_id="guide",
        sections=(SectionMetadata(id="missing", heading="No Such Heading", stage="verify"),),
    )
    result = read_section(GUIDE, "roll-back", metadata=metadata)
    assert result.text.startswith("## Roll back")
    assert result.bound is False
    assert any(d.code == CODE_SECTION_UNDECLARED for d in result.diagnostics)
    assert not any(d.code == CODE_SECTION_MISSING_ANCHOR for d in result.diagnostics)


def test_read_skill_section_uses_the_skill_object():
    from machina.skills.sections import read_skill_section

    class _Skill:
        name = "guide"
        content = GUIDE
        skill_metadata = _guide_metadata()

    result = read_skill_section(_Skill(), "rollback")
    assert result.skill_id == "guide"
    assert result.bound is True
    assert result.text.startswith("## Roll back")

    # Explicit bodies (e.g. builtin bundle instructions kept outside
    # ``skill.content``) are addressed the same way via ``content=``.
    loaded = read_skill_section(_Skill(), "rollback", content=GUIDE + "\n## Later\n\nx\n")
    assert loaded.section_id == "rollback"
    assert loaded.text.startswith("## Roll back")
    assert "## Later" not in loaded.text


# --- requires_sections graph ----------------------------------------------------


def _graph_metadata() -> SkillMetadata:
    return SkillMetadata(
        skill_id="guide",
        sections=(
            SectionMetadata(id="deploy", heading="Deploy the service", stage="deliver"),
            SectionMetadata(id="warnings", heading="Warnings", stage="deliver"),
            SectionMetadata(
                id="rollback",
                heading="Roll back",
                stage="verify",
                requires_sections=("deploy", "warnings"),
            ),
            SectionMetadata(
                id="verify-app", heading="Verify app", stage="verify", requires_sections=("deploy",)
            ),
        ),
    )


def test_required_section_closure_is_deterministic_and_prerequisites_first():
    from machina.skills.sections import build_section_graph

    graph, diagnostics = build_section_graph(_graph_metadata())
    assert diagnostics == ()
    assert graph.closure("rollback") == ("deploy", "warnings")
    assert graph.closure("verify-app") == ("deploy",)
    assert graph.closure("deploy") == ()
    assert graph.is_acyclic() is True


def test_unknown_dependency_targets_are_dropped_with_a_diagnostic():
    from machina.skills.sections import build_section_graph

    metadata = SkillMetadata(
        skill_id="guide",
        sections=(
            SectionMetadata(id="a", heading="A", requires_sections=("ghost",)),
            SectionMetadata(id="b", heading="B"),
        ),
    )
    graph, diagnostics = build_section_graph(metadata)
    assert graph.closure("a") == ()
    assert any(d.code == CODE_DEPENDENCY_UNKNOWN for d in diagnostics)


def test_requires_section_cycles_are_rejected():
    from machina.skills.sections import SectionCycleError, build_section_graph

    metadata = SkillMetadata(
        skill_id="guide",
        sections=(
            SectionMetadata(id="a", heading="A", requires_sections=("b",)),
            SectionMetadata(id="b", heading="B", requires_sections=("a",)),
        ),
    )
    with pytest.raises(SectionCycleError):
        build_section_graph(metadata)


def test_disclosure_degrades_cleanly_on_a_cyclic_graph(tmp_path):
    from machina.skills.metadata import CODE_DEPENDENCY_CYCLE
    from machina.skills.planned_steps import StageState, plan_disclosure
    from machina.skills.registry import SkillRegistry

    skills = SkillRegistry(tmp_path)
    skills.create_skill(
        "guide", "Deploy guide", "## A\n\nbody a\n\n## B\n\nbody b\n", triggers=["deploy"]
    )
    skills.load_all()
    skills.get("guide").skill_metadata = SkillMetadata(
        skill_id="guide",
        sections=(
            SectionMetadata(id="a", heading="A", stage="deliver", requires_sections=("b",)),
            SectionMetadata(id="b", heading="B", stage="deliver", requires_sections=("a",)),
        ),
    )
    from machina.skills.workflow import plan_workflow

    workflow = plan_workflow("help me deploy", skills=skills)
    plan = plan_disclosure(
        workflow.steps,
        skill_for=skills.get,
        content_for=skills.get_skill_content,
        budget=400,
        stage_state=StageState("deliver", source="user"),
    )
    assert plan.chosen == ()
    assert any(d.code == CODE_DEPENDENCY_CYCLE for d in plan.diagnostics)
    assert plan.unresolved_prerequisites


# --- legacy character-offset reads stay unchanged ------------------------------


def test_legacy_skill_read_tool_offsets_are_untouched():
    from machina.tools.skills import skill_read

    params = list(inspect.signature(skill_read).parameters)
    assert params[:4] == ["skill_name", "offset", "limit", "query"]
    assert inspect.signature(skill_read).parameters["limit"].default == 800
