"""A2c — consumer wiring: automatic selection + browse on canonical ``search_skills``.

Closes the A6 checklist-1 gap: the automatic selection path
(``plan_workflow`` / ``automatic_skill_context``) and the ``skill_list`` browse
tool now consume the canonical typed search instead of the deleted legacy
``rank_skills`` selector. This file pins:

* the automatic path ranks through ``search_skills`` and the legacy selector is
  gone (mutation-style observation);
* the three legacy quirks are either ported with a test (the ``testing``-name
  preference) or documented-removed with the reason they are no longer needed
  (the ``ecc-x-api`` mention gate; the absolute ``idf >= 1.6`` gate, replaced by
  the catalog-relative ratio gate);
* the A6 human/automatic parity requirement on the real catalog;
* ``skill_list`` keeps its documented response shape while browsing canonically.

No provider, embedding or network call is made by any test here.
"""

from __future__ import annotations

import pytest

from machina.skills.registry import SkillRegistry
from machina.skills.runtime import automatic_skills, search_skills
from machina.skills.search import APPLICABLE, EXCLUDED


class _Registry(SkillRegistry):
    """Fail-closed policy: created test skills get minimal schema-v2 metadata."""

    def create_skill(self, *args, **kwargs):
        from machina.skills.metadata import SkillMetadata

        skill = super().create_skill(*args, **kwargs)
        self.load_all()
        current = self.get(skill.name)
        self.persist_skill_metadata(current, SkillMetadata(skill_id=current.name))
        return skill


def _registry(tmp_path, name: str = "skills") -> SkillRegistry:
    return _Registry(tmp_path / name)


def _real_registry():
    from machina.skills.registry import skill_registry

    skill_registry.load_all()
    return skill_registry


def _ids(result) -> list[str]:
    return [candidate.skill_id for candidate in result.candidates]


def _filler(reg: SkillRegistry, index: int) -> None:
    reg.create_skill(f"lathe-{index:02d}", f"Lathe calibration notes {index}", "S")


# ─── mutation-style: automatic path uses canonical search, not the legacy index ──


def test_automatic_path_ranks_through_canonical_search_and_legacy_index_is_gone(
    tmp_path, monkeypatch
):
    from machina.skills import runtime
    from machina.skills.workflow import plan_workflow

    calls: list[str] = []
    real_search = runtime.search_skills

    def spy(skills, query, **kwargs):
        calls.append(query)
        return real_search(skills, query, **kwargs)

    monkeypatch.setattr(runtime, "search_skills", spy)

    reg = _Registry(tmp_path)
    reg.create_skill(
        "deploy", "Deploy services carefully", "Steps", triggers=["deploy"]
    )

    plan = plan_workflow("deploy the service", skills=reg)
    assert plan.skills, plan.stages
    runtime.automatic_skill_context("help me deploy", skills=reg)

    assert calls, "the automatic path must rank through search_skills"
    assert "deploy the service" in calls
    # The legacy selector is deleted, not merely unused.
    assert not hasattr(runtime, "rank_skills")
    assert not hasattr(runtime.SkillIndex, "rank")


def test_ports_seam_dispatches_to_the_canonical_selector():
    from machina.skills import ports

    assert hasattr(ports, "automatic_skills")
    assert not hasattr(ports, "rank_skills")


# ─── legacy quirk (a): the `ecc-x-api` mention gate is retired ────────────────


def test_no_name_based_mention_gate_for_x_api(tmp_path):
    """A skill named ``ecc-x-api`` is selected on its merits, not gated.

    The retired legacy selector dropped ``ecc-x-api`` unless the query mentioned
    ``x``/``twitter``/``tweets``. No such gate exists on the canonical path, so
    an otherwise-relevant match (its own trigger) is selectable either way.
    """
    reg = _registry(tmp_path, "xapi")
    reg.create_skill(
        "ecc-x-api",
        "Post and read social updates through the API",
        "S",
        triggers=["social api"],
    )
    _filler(reg, 0)
    _filler(reg, 1)

    without_mention = automatic_skills(reg, "use the social api for updates")
    with_mention = automatic_skills(reg, "post tweets through the social api")

    assert [s.name for s in without_mention][:1] == ["ecc-x-api"]
    assert [s.name for s in with_mention][:1] == ["ecc-x-api"]


# ─── legacy quirk (b): the "testing"-name preference is ported ────────────────


def test_testing_named_skill_is_only_a_specialist_when_testing_is_requested(tmp_path):
    reg = _registry(tmp_path, "testing-pref")
    reg.create_skill("alpha-testing", "Widget build guidance for services", "S")
    reg.create_skill("zeta-generic", "Widget build guidance for services", "S")
    for index in range(3):
        _filler(reg, index)

    build = automatic_skills(reg, "widget build guidance services")
    assert [s.name for s in build][:2] == ["zeta-generic", "alpha-testing"], build

    requested = automatic_skills(reg, "widget testing guidance services")
    assert [s.name for s in requested][:2] == ["alpha-testing", "zeta-generic"], (
        requested
    )


# ─── legacy quirk (c): absolute idf gate retired for the ratio gate ───────────


def test_retrieval_gate_is_catalog_relative_not_an_absolute_idf_threshold(tmp_path):
    """The old ``idf(word) >= 1.6`` gate is replaced by ``df/N <= 0.5``.

    With four documents a word in two of them has idf ≈ 0.69 (< 1.6) yet is
    still distinguishing (ratio 0.5); the canonical path admits it. The old
    absolute threshold would have dropped the same candidate, which is why it
    needed no port (spec §9.2: the gate is catalog-dependent by construction).
    """
    reg = _registry(tmp_path, "ratio-admit")
    reg.create_skill("widget-helper", "Widget alignment guidance", "S")
    reg.create_skill("widget-notes", "Widget alignment guidance", "S")
    _filler(reg, 0)
    _filler(reg, 1)

    admitted = search_skills(reg, "widget alignment guidance")
    assert {"widget-helper", "widget-notes"} <= set(_ids(admitted))
    assert all(c.applicability == APPLICABLE for c in admitted.candidates)

    # A word carried by 3 of 4 documents is not distinguishing; with no other
    # signal the candidate is not retrieved (never a forced top pick).
    boundary = _registry(tmp_path, "ratio-boundary")
    boundary.create_skill("common-one", "Common release notes", "S")
    boundary.create_skill("common-two", "Common release notes", "S")
    boundary.create_skill("common-three", "Common release notes", "S")
    boundary.create_skill("lathe-four", "Lathe calibration notes", "S")

    assert search_skills(boundary, "common").candidates == []


# ─── A6 parity: human browse vs automatic differ only by scope ────────────────


PARITY_QUERIES = (
    "run the python test suite",
    "deploy the service",
    "review this landing page copy",
)


def test_real_catalog_browse_and_automatic_differ_only_by_documented_scope():
    reg = _real_registry()
    catalog = reg.list_skills()
    disabled = {skill.name for skill in catalog if not skill.enabled}

    for query in PARITY_QUERIES:
        automatic = search_skills(reg, query, limit=50)
        browse = search_skills(reg, query, limit=50, include_excluded=True)

        # Same engine, same revisions for the ranking policy and descriptor.
        assert automatic.policy_revision == browse.policy_revision, query
        assert (
            automatic.query_descriptor_revision == browse.query_descriptor_revision
        ), query

        if not disabled:
            # No disabled member: the scopes are byte-identical.
            assert automatic.catalog_revision == browse.catalog_revision, query
            assert automatic.diagnostics == browse.diagnostics, query
            assert [
                (c.skill_id, c.score, c.applicability, c.content_revision)
                for c in automatic.candidates
            ] == [
                (c.skill_id, c.score, c.applicability, c.content_revision)
                for c in browse.candidates
            ], query
        else:
            # The only difference is the disabled member itself: automatic drops
            # it, browse keeps it flagged excluded; enabled order/applicability
            # are identical.
            browsed_enabled = [
                c for c in browse.candidates if c.applicability != EXCLUDED
            ]
            assert _ids(automatic) == [c.skill_id for c in browsed_enabled], query
            disabled_browse = [
                c
                for c in browse.candidates
                if c.skill_id in {n.casefold() for n in disabled}
            ]
            assert disabled_browse and all(
                c.applicability == EXCLUDED for c in disabled_browse
            )


# ─── skill_list browse rides the canonical contract, shape unchanged ──────────


@pytest.mark.asyncio
async def test_skill_list_query_browse_is_canonical_and_excludes_disabled(
    tmp_path, monkeypatch
):
    import machina.skills.registry as skills_module
    from machina.tools.base import registry as tool_registry
    from machina.tools.skills import skill_list

    reg = _Registry(tmp_path)
    reg.create_skill(
        "team-deploy", "Deploy services carefully", "S", triggers=["deploy"]
    )
    reg.create_skill(
        "old-deploy", "Deploy services with legacy steps", "S", triggers=["deploy"]
    )
    reg.set_enabled("old-deploy", False)
    monkeypatch.setattr(skills_module, "skill_registry", reg)

    out = skill_list(query="deploy the service")
    assert "team-deploy" in out
    assert "old-deploy" not in out

    # The page order is exactly the canonical candidate order (first four).
    expected = _ids(search_skills(reg, "deploy the service"))[:4]
    listed = [line.split(":", 1)[0] for line in out.splitlines() if ":" in line]
    assert listed == expected

    # The empty-query browse path still lists enabled skills in stable order.
    browse = skill_list(query="")
    assert "team-deploy" in browse and "old-deploy" not in browse
    assert "Next offset" not in browse  # one enabled skill, one page

    # Documented paging shape survives on the canonical query path.
    for index in range(6):
        reg.create_skill(
            f"deploy-ext-{index}", f"Deploy extension {index}", "S", triggers=["deploy"]
        )
    first = skill_list(query="deploy")
    assert first.splitlines()[-1].startswith("Next offset: ")
    second = skill_list(query="deploy", offset=4)
    assert first.splitlines()[0] != second.splitlines()[0]

    # Typed parameters are unchanged (R4-090 pin lives in test_r4_090_tools).
    tool = tool_registry.get_tool("skill_list")
    assert set(tool.parameters["properties"]) == {"query", "offset"}
    assert tool.parameters["properties"]["offset"]["default"] == 0
