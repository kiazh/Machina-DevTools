"""A5 backend gateway contracts (spec §12.1/§12.2): bounded skill search,
call-free route previews, and version-bound section reads.

Handler-level tests with a minimal fake gateway exposing only what the feature
module touches: ``require_db`` (guarded and recorded), ``_write`` (frame
capture — the preview must emit none) and ``add_method``. Params arrive as
untrusted JSON, so validation, closed vocabularies, bounds and the
disabled-skill semantics are asserted at this boundary.

The preview's call-free guarantees are pinned with an import tripwire on
``machina.core.provider`` (fresh imports, cached module entry, and package
attribute), a usage-count snapshot (no ``record_use``, no turn, no approval
mutation) and wire-shape assertions (everything JSON-ready).
"""

from __future__ import annotations

import json
import sys

import pytest

from machina.gateway import features
from machina.gateway.errors import INVALID_PARAMS, NOT_FOUND, RpcError
from machina.gateway.features import skills as feature

SKILLS_GATEWAY_METHODS = {"skills.search", "skills.preview_route", "skills.section"}

GUIDE_CONTENT = (
    "Intro paragraph before any heading.\n\n"
    "# Deploy guide\n\n"
    "Run the guide top to bottom.\n\n"
    "## Run tests\n\n"
    "First run the focused suite, then the full suite.\n\n"
    "## Roll back\n\n"
    "Restore the previous release if checks fail.\n"
)


class FakeGateway:
    """Minimal gateway surface for handler tests."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.methods: dict = {}
        self.guards = 0

    def require_db(self) -> None:
        self.guards += 1

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)

    def add_method(self, name, handler) -> None:
        self.methods[name] = handler


def _registry(tmp_path):
    from machina.skills.registry import SkillRegistry

    reg = SkillRegistry(tmp_path / "skills")
    reg.create_skill(
        "python-testing",
        "Run pytest suites and coverage checks for python projects",
        "# Python testing\n\nRun pytest.\n",
        triggers=["pytest", "coverage"],
        source_type="learned",
    )
    reg.create_skill(
        "deploy-guide",
        "Deploy releases and roll back safely",
        GUIDE_CONTENT,
        triggers=["deploy"],
        source_type="local",
    )
    reg.create_skill(
        "release-runbook",
        "Release runbook: stage, deploy and verify a rollout",
        "# Release runbook\n\nStage the release.\n",
        triggers=["deploy"],
        source_type="local",
    )
    reg.load_all()
    return reg


def _attach_metadata(reg, name: str, metadata) -> None:
    """Attach validated schema-v2 metadata to the live skill object.

    ``_registry`` is monkeypatched in these tests, so no reload re-parses the
    on-disk documents and the attached metadata is what handlers see.
    """
    reg.load_all()
    reg.get(name).skill_metadata = metadata


def _candidate_ids(payload: dict) -> list[str]:
    return [candidate["skillId"] for candidate in payload["candidates"]]


class _ProviderTripwire:
    """Any attribute access or call is a test failure."""

    def __getattr__(self, name: str):
        raise AssertionError(
            f"provider module touched during call-free preview: {name}"
        )

    def __call__(self, *args, **kwargs):
        raise AssertionError("provider called during call-free preview")


class _ImportTripwire:
    """Meta-path finder that fails the test on a forbidden fresh import."""

    def __init__(self, *prefixes: str) -> None:
        self.prefixes = prefixes

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes):
            raise AssertionError(
                f"forbidden import during call-free preview: {fullname}"
            )


# ─── registration + backward compatibility ───────────────────────────────────


def test_skills_gateway_methods_registered():
    assert SKILLS_GATEWAY_METHODS <= set(features.METHODS)
    gw = FakeGateway()
    features.register(gw)
    assert SKILLS_GATEWAY_METHODS <= set(gw.methods)
    # The pre-existing surface is untouched.
    assert "skills.list" in features.METHODS and "skills.show" in features.METHODS


async def test_skills_list_response_shape_unchanged(tmp_path, monkeypatch):
    """U07 backward compat: no new field may leak into skills.list."""
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    result = await feature.skills_list(gw, {})

    assert result == {
        "skills": [
            {
                "name": "deploy-guide",
                "description": "Deploy releases and roll back safely",
                "triggers": ["deploy"],
                "version": "1.0.0",
                "sourceType": "local",
                "usageCount": 0,
                "enabled": True,
            },
            {
                "name": "python-testing",
                "description": "Run pytest suites and coverage checks for python projects",
                "triggers": ["pytest", "coverage"],
                "version": "1.0.0",
                "sourceType": "learned",
                "usageCount": 0,
                "enabled": True,
            },
            {
                "name": "release-runbook",
                "description": "Release runbook: stage, deploy and verify a rollout",
                "triggers": ["deploy"],
                "version": "1.0.0",
                "sourceType": "local",
                "usageCount": 0,
                "enabled": True,
            },
        ]
    }
    assert gw.guards == 1


# ─── skills.search ───────────────────────────────────────────────────────────


async def test_search_response_contract_and_source_trust(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    result = await feature.skills_search(gw, {"query": "deploy releases"})

    assert set(result) == {
        "query",
        "filters",
        "limit",
        "browse",
        "candidates",
        "catalogRevision",
        "queryDescriptorRevision",
        "policyRevision",
        "nextCursor",
        "diagnostics",
    }
    assert result["query"] == "deploy releases"
    assert result["limit"] == feature.SKILLS_SEARCH_DEFAULT_LIMIT
    assert result["browse"] is False
    assert result["nextCursor"] is None
    assert result["policyRevision"] == "skills-search-v1"
    assert result["catalogRevision"]
    assert result["diagnostics"]["classification"] in {
        "ok",
        "conditional_only",
        "missing_capability",
        "all_excluded",
        "no_lexical_evidence",
        "invalid_catalog",
    }

    top = result["candidates"][0]
    assert set(top) == {
        "skillId",
        "name",
        "contentRevision",
        "score",
        "scoreComponents",
        "matchedPhrases",
        "matchedCapabilities",
        "applicability",
        "reasons",
        "workflowStages",
        "explanation",
        "sourceType",
        "trust",
        "enabled",
    }
    assert top["skillId"] == "deploy-guide"
    assert top["name"] == "deploy-guide"
    assert top["applicability"] == "applicable" and top["reasons"] == []
    assert top["sourceType"] == "local" and top["trust"] == "local"
    assert top["enabled"] is True
    assert top["contentRevision"]
    assert isinstance(top["score"], float)
    json.dumps(result)
    assert gw.guards == 1


async def test_search_cursor_pages_deterministically(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    first = await feature.skills_search(gw, {"query": "deploy releases", "limit": 1})
    assert len(first["candidates"]) == 1
    assert first["nextCursor"] is not None
    assert first["nextCursor"].startswith("skills-search-v1:")

    second = await feature.skills_search(
        gw, {"query": "deploy releases", "limit": 1, "cursor": first["nextCursor"]}
    )
    assert second["nextCursor"] is None
    assert [c["skillId"] for c in second["candidates"]] != _candidate_ids(first)

    combined = _candidate_ids(first) + _candidate_ids(second)
    assert sorted(combined) == ["deploy-guide", "release-runbook"]


async def test_search_invalid_params_fail_closed(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    bad_params = (
        {"query": 5},
        {"limit": 0},
        {"limit": feature.SKILLS_SEARCH_MAX_LIMIT + 1},
        {"limit": "10"},
        {"limit": True},
        {"cursor": 12},
        {"cursor": "not-a-cursor"},
        {"filters": "nope"},
        {"filters": {"unknown": True}},
        {"filters": {"task_family": "not-a-family"}},
        {"filters": {"task_family": 7}},
        {"filters": {"capability": 9}},
        {"filters": {"source": "downloads"}},
        {"filters": {"source": ["local", 5]}},
        {"filters": {"source": ["local"] * (feature.SKILLS_SEARCH_MAX_SOURCES + 1)}},
        {"filters": {"enabled": "maybe"}},
        {"filters": {"include_excluded": "sometimes"}},
    )
    for params in bad_params:
        with pytest.raises(RpcError) as error:
            await feature.skills_search(gw, params)
        assert error.value.code == INVALID_PARAMS, params

    ok = await feature.skills_search(gw, {"limit": feature.SKILLS_SEARCH_MAX_LIMIT})
    assert ok["limit"] == feature.SKILLS_SEARCH_MAX_LIMIT


async def test_search_disabled_skills_only_on_explicit_browse(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    reg.set_enabled("deploy-guide", False)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    automatic = await feature.skills_search(gw, {"query": "deploy releases"})
    assert "deploy-guide" not in _candidate_ids(automatic)
    assert automatic["browse"] is False

    browse = await feature.skills_search(
        gw, {"query": "deploy releases", "filters": {"include_excluded": True}}
    )
    row = next(c for c in browse["candidates"] if c["skillId"] == "deploy-guide")
    assert row["applicability"] == "excluded"
    assert any(reason.startswith("disabled:") for reason in row["reasons"])
    assert row["enabled"] is False
    assert browse["browse"] is True

    disabled_only = await feature.skills_search(
        gw, {"query": "deploy releases", "filters": {"enabled": False}}
    )
    assert _candidate_ids(disabled_only) == ["deploy-guide"]
    assert disabled_only["browse"] is True

    enabled_only = await feature.skills_search(
        gw,
        {
            "query": "deploy releases",
            "filters": {"enabled": True, "include_excluded": True},
        },
    )
    assert "deploy-guide" not in _candidate_ids(enabled_only)
    assert _candidate_ids(enabled_only) == ["release-runbook"]
    assert enabled_only["filters"]["include_excluded"] is True


async def test_search_catalog_filters_use_declared_metadata(tmp_path, monkeypatch):
    from machina.skills.metadata import SkillMetadata

    reg = _registry(tmp_path)
    _attach_metadata(
        reg,
        "python-testing",
        SkillMetadata(
            skill_id="python-testing",
            task_families=("testing",),
            capabilities=("coverage",),
        ),
    )
    _attach_metadata(
        reg,
        "deploy-guide",
        SkillMetadata(skill_id="deploy-guide", task_families=("deployment",)),
    )
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    family = await feature.skills_search(
        gw, {"query": "pytest suites", "filters": {"task_family": "testing"}}
    )
    assert _candidate_ids(family) == ["python-testing"]
    assert family["filters"]["task_family"] == "testing"

    # Declared schema-v2 family values canonicalize through the search aliases
    # ("deployment" -> "deploying").
    alias = await feature.skills_search(
        gw, {"query": "deploy releases", "filters": {"task_family": "deployment"}}
    )
    assert "deploy-guide" in _candidate_ids(alias)

    capability = await feature.skills_search(
        gw, {"query": "pytest suites", "filters": {"capability": "coverage"}}
    )
    assert _candidate_ids(capability) == ["python-testing"]

    miss = await feature.skills_search(
        gw, {"query": "pytest suites", "filters": {"capability": "deployment"}}
    )
    assert _candidate_ids(miss) == []

    source_miss = await feature.skills_search(
        gw, {"query": "pytest suites", "filters": {"source": "local"}}
    )
    assert _candidate_ids(source_miss) == []

    source_hit = await feature.skills_search(
        gw, {"query": "pytest suites", "filters": {"source": ["learned"]}}
    )
    assert _candidate_ids(source_hit) == ["python-testing"]
    assert source_hit["filters"]["source"] == ["learned"]


# ─── skills.preview_route ────────────────────────────────────────────────────


async def test_preview_route_is_call_free_and_never_touches_provider(
    tmp_path, monkeypatch
):
    import machina.core as machina_core

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    trap = _ProviderTripwire()
    monkeypatch.setitem(sys.modules, "machina.core.provider", trap)
    monkeypatch.setattr(machina_core, "provider", trap, raising=False)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [_ImportTripwire("machina.core.provider"), *sys.meta_path],
    )

    before = sum(skill.usage_count for skill in reg.list_skills())
    result = await feature.skills_preview_route(
        gw, {"request": "deploy the release", "context": {"active_class": "implement"}}
    )
    after = sum(skill.usage_count for skill in reg.list_skills())

    assert gw.frames == []
    assert after == before
    assert set(result) == {
        "descriptor",
        "reasons",
        "selection",
        "disclosure",
        "budget",
        "tokenBasis",
    }
    assert result["descriptor"]["task_class"] == "implement"
    assert isinstance(result["reasons"], list) and result["reasons"]
    assert set(result["selection"]) == {
        "kind",
        "skills",
        "steps",
        "catalogRevision",
        "taskRevision",
    }
    assert result["selection"]["kind"] in {"coding", "research", "task", "conversation"}
    assert isinstance(result["selection"]["steps"], list)
    assert set(result["disclosure"]) == {
        "task_revision",
        "stage_revision",
        "catalog_revision",
        "stage_source",
        "current_stage",
        "chosen",
        "whole_body",
        "required_prerequisites",
        "omitted",
        "unresolved_prerequisites",
        "missing_context",
        "task_constraints",
        "included_tokens",
        "token_basis",
        "diagnostics",
    }
    assert result["disclosure"]["included_tokens"] >= 0
    assert result["budget"] == feature.SKILLS_PREVIEW_BUDGET
    assert result["tokenBasis"] == result["disclosure"]["token_basis"]
    json.dumps(result)
    assert gw.guards == 1


async def test_preview_route_stage_state_selects_stage_sections(tmp_path, monkeypatch):
    from machina.skills.metadata import SectionMetadata, SkillMetadata

    reg = _registry(tmp_path)
    _attach_metadata(
        reg,
        "deploy-guide",
        SkillMetadata(
            skill_id="deploy-guide",
            task_families=("deployment",),
            sections=(
                SectionMetadata(id="tests", heading="Run tests", stage="verify"),
                SectionMetadata(id="rollback", heading="Roll back", stage="verify"),
            ),
        ),
    )
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    result = await feature.skills_preview_route(
        gw,
        {
            "request": "deploy guide run the release",
            "context": {"stage": "verify", "stage_source": "user", "revision": 2},
        },
    )

    assert result["descriptor"]["task_state_revision"] == 3
    assert result["selection"]["taskRevision"] == 3
    plan = result["disclosure"]
    assert plan["current_stage"] == "verify" and plan["stage_source"] == "user"
    assert plan["chosen"], "the verify-stage section must be disclosed for stage verify"
    headings = {entry["heading"] for entry in plan["chosen"]}
    assert "Run tests" in headings


async def test_preview_route_context_is_bounded_and_typed(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    bad_contexts = (
        "nope",
        {"unknown_field": 1},
        {"active_class": 5},
        {"stack": 5},
        {"stack": {"not-a-category": ["python"]}},
        {"stack": ["python", 5]},
        {"constraints": [5]},
        {"constraints": ["x"] * (feature.SKILLS_PREVIEW_MAX_CONSTRAINTS + 1)},
        {"capabilities": ["x"] * 20},
        {"revision": -1},
        {"revision": True},
        {"stage_revision": "1"},
    )
    for context in bad_contexts:
        with pytest.raises(RpcError) as error:
            await feature.skills_preview_route(
                gw, {"request": "deploy the release", "context": context}
            )
        assert error.value.code == INVALID_PARAMS, context

    with pytest.raises(RpcError) as missing:
        await feature.skills_preview_route(gw, {"context": {}})
    assert missing.value.code == INVALID_PARAMS

    ok = await feature.skills_preview_route(
        gw,
        {
            "request": "deploy the release",
            "context": {
                "stack": {"languages": ["python"]},
                "constraints": ["read-only"],
                "capabilities": ["deployment"],
                "revision": 0,
            },
        },
    )
    assert ok["descriptor"]["operation_scope"] == "read-only"


# ─── skills.section ──────────────────────────────────────────────────────────


async def test_section_returns_exact_versioned_passage_and_continuation(
    tmp_path, monkeypatch
):
    from machina.skills.sections import content_fingerprint, index_sections

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    body = reg.get_skill_content("deploy-guide")
    section_text = index_sections(body).find("run-tests").text
    revision = content_fingerprint(body)

    first = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "run-tests", "budget": 20}
    )
    assert first["sectionId"] == "run-tests"
    assert first["contentRevision"] == revision
    assert first["text"] == section_text[:20]
    assert first["sectionTotal"] == len(section_text)
    assert first["total"] == len(body)
    assert first["continuation"] == 20
    assert first["stale"] is False
    assert first["budget"] == 20

    second = await feature.skills_section(
        gw,
        {
            "skillId": "deploy-guide",
            "section": "run-tests",
            "contentRevision": revision,
            "offset": first["continuation"],
            "budget": 20,
        },
    )
    assert second["text"] == section_text[20:40]
    assert first["text"] + second["text"] == section_text[:40]
    json.dumps(first)

    whole = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "run-tests"}
    )
    assert whole["text"] == section_text
    assert whole["continuation"] is None
    assert gw.guards == 3


async def test_section_declared_binding(tmp_path, monkeypatch):
    from machina.skills.metadata import SectionMetadata, SkillMetadata
    from machina.skills.sections import content_fingerprint

    reg = _registry(tmp_path)
    _attach_metadata(
        reg,
        "deploy-guide",
        SkillMetadata(
            skill_id="deploy-guide",
            sections=(
                SectionMetadata(id="tests", heading="Run tests", stage="verify"),
            ),
        ),
    )
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    result = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "tests", "budget": 8000}
    )
    assert result["sectionId"] == "tests"
    assert result["anchorId"] == "run-tests"
    assert result["bound"] is True
    assert result["heading"] == "Run tests"
    assert (
        "Run pytest" in result["text"]
        or "First run the focused suite" in result["text"]
    )
    assert "Roll back" not in result["text"]
    assert result["contentRevision"] == content_fingerprint(
        reg.get_skill_content("deploy-guide")
    )


async def test_section_stale_revision_never_serves_a_different_passage(
    tmp_path, monkeypatch
):
    from machina.skills.sections import content_fingerprint

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    revision = content_fingerprint(reg.get_skill_content("deploy-guide"))
    stale = await feature.skills_section(
        gw,
        {
            "skillId": "deploy-guide",
            "section": "run-tests",
            "contentRevision": "0000000000000000",
        },
    )
    assert stale["stale"] is True
    assert stale["text"] == ""
    assert stale["continuation"] is None
    assert stale["contentRevision"] == revision
    assert any(d["code"] == "section-stale-content" for d in stale["diagnostics"])

    stale_offset = await feature.skills_section(
        gw,
        {
            "skillId": "deploy-guide",
            "offset": 10,
            "contentRevision": "ffffffffffffffff",
        },
    )
    assert stale_offset["stale"] is True and stale_offset["text"] == ""


async def test_section_offset_mode_and_unknown_sections(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    offset_read = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "offset": 4, "budget": 10}
    )
    body = reg.get_skill_content("deploy-guide")
    assert offset_read["sectionId"] == ""
    assert offset_read["text"] == body[4:14]
    assert offset_read["continuation"] == 14
    assert offset_read["stale"] is False

    unknown = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "nope"}
    )
    assert unknown["stale"] is False
    assert unknown["text"] == ""
    assert unknown["sectionId"] == "nope"
    assert any(d["code"] == "section-unknown" for d in unknown["diagnostics"])

    ranged = await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "run-tests", "offset": 99999}
    )
    assert ranged["stale"] is False and ranged["text"] == ""
    assert any(d["code"] == "section-continuation-range" for d in ranged["diagnostics"])


async def test_section_validates_params_and_skill_existence(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    with pytest.raises(RpcError) as missing_skill:
        await feature.skills_section(gw, {"section": "run-tests"})
    assert missing_skill.value.code == INVALID_PARAMS

    with pytest.raises(RpcError) as no_address:
        await feature.skills_section(gw, {"skillId": "deploy-guide"})
    assert no_address.value.code == INVALID_PARAMS

    # Registry lookups are case-insensitive; search returns casefolded ids.
    mixed_case = await feature.skills_section(
        gw, {"skillId": "Deploy-Guide", "section": "run-tests"}
    )
    assert mixed_case["skillId"] == "deploy-guide"

    with pytest.raises(RpcError) as unknown_skill:
        await feature.skills_section(gw, {"skillId": "no-such-skill", "section": "x"})
    assert unknown_skill.value.code == NOT_FOUND

    for params in (
        {"skillId": "deploy-guide", "section": "run-tests", "budget": 0},
        {"skillId": "deploy-guide", "section": "run-tests", "budget": 8001},
        {"skillId": "deploy-guide", "section": "run-tests", "budget": "big"},
        {"skillId": "deploy-guide", "offset": -3},
        {"skillId": "deploy-guide", "offset": "3"},
        {"skillId": "deploy-guide", "section": "run-tests", "contentRevision": 5},
    ):
        with pytest.raises(RpcError) as error:
            await feature.skills_section(gw, params)
        assert error.value.code == INVALID_PARAMS, params


# ─── browse/read telemetry (A4-optional, never an operational success) ───────


class _ImportBlocker:
    """Meta-path finder that raises ImportError for blocked module prefixes."""

    def __init__(self, *prefixes: str) -> None:
        self.prefixes = prefixes

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes):
            raise ImportError(f"blocked for test: {fullname}")


async def test_browse_activity_falls_back_to_local_counter(tmp_path, monkeypatch):
    import machina.skills as skills_pkg

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    monkeypatch.delattr(skills_pkg, "events", raising=False)
    monkeypatch.delitem(sys.modules, "machina.skills.events", raising=False)
    monkeypatch.setattr(
        sys, "meta_path", [_ImportBlocker("machina.skills.events"), *sys.meta_path]
    )
    feature._BROWSE_ACTIVITY.update({"search": 0, "preview_route": 0, "section": 0})

    await feature.skills_search(gw, {"query": "deploy releases"})
    await feature.skills_section(
        gw, {"skillId": "deploy-guide", "section": "run-tests"}
    )

    assert feature._BROWSE_ACTIVITY["search"] == 1
    assert feature._BROWSE_ACTIVITY["section"] == 1
    # Browse activity only: no operational-success counter exists here.
    assert not any(
        "success" in key or "completed" in key or "task" in key
        for key in feature._BROWSE_ACTIVITY
    )


async def test_browse_activity_prefers_a4_events_recorder(tmp_path, monkeypatch):
    from types import ModuleType

    import machina.skills as skills_pkg

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()

    calls: list[tuple[str, dict]] = []
    fake_events = ModuleType("machina.skills.events")

    def record_event(kind, **fields):
        calls.append((kind, fields))

    fake_events.record_event = record_event  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "machina.skills.events", fake_events)
    monkeypatch.setattr(skills_pkg, "events", fake_events, raising=False)
    feature._BROWSE_ACTIVITY.update({"search": 0, "preview_route": 0, "section": 0})

    await feature.skills_search(gw, {"query": "deploy releases"})

    assert calls and calls[-1][0] == "search"
    assert calls[-1][1]["browse"] is False
    assert feature._BROWSE_ACTIVITY["search"] == 0, (
        "A4 recorder present: no local double count"
    )


async def test_preview_and_section_record_no_activation_events(tmp_path, monkeypatch):
    """A4's closed activation vocabulary stays untouched by administrative reads."""
    from machina.skills import events as skill_events

    reg = _registry(tmp_path)
    monkeypatch.setattr(feature, "_registry", lambda: reg)
    gw = FakeGateway()
    skill_events.clear()
    try:
        await feature.skills_preview_route(gw, {"request": "deploy the release"})
        await feature.skills_section(
            gw, {"skillId": "deploy-guide", "section": "run-tests"}
        )

        assert skill_events.snapshot() == []
        assert skill_events.counts() == dict.fromkeys(skill_events.EVENT_NAMES, 0)
    finally:
        skill_events.clear()


# ─── native skill_read: section+revision, legacy offsets byte-identical ──────


def _tool_registry(tmp_path, monkeypatch):
    import machina.skills.registry as skills_module

    reg = _registry(tmp_path)
    monkeypatch.setattr(skills_module, "skill_registry", reg)
    return reg


def test_native_skill_read_legacy_offsets_byte_identical(tmp_path, monkeypatch):
    from machina.tools.skills import skill_read

    reg = _tool_registry(tmp_path, monkeypatch)
    body = reg.get_skill_content("deploy-guide")
    total = len(body)

    assert skill_read("deploy-guide", offset=4, limit=10) == (
        f"Skill deploy-guide [4:14/{total}]:\n{body[4:14]}"
    )
    assert (
        skill_read("deploy-guide") == f"Skill deploy-guide [0:{total}/{total}]:\n{body}"
    )
    # revision only matters in section mode; without section the read is legacy.
    assert skill_read("deploy-guide", offset=4, limit=10, revision="whatever") == (
        f"Skill deploy-guide [4:14/{total}]:\n{body[4:14]}"
    )
    query_read = skill_read("deploy-guide", query="focused suite")
    assert "focused suite" in query_read and query_read.startswith(
        "Skill deploy-guide ["
    )

    assert skill_read("no-such-skill") == "Error: skill not found or disabled"
    assert (
        skill_read("deploy-guide", offset=99999)
        == "Error: offset is outside the skill content"
    )
    assert (
        skill_read("deploy-guide", limit=0) == "Error: limit must be between 1 and 8000"
    )


def test_native_skill_read_section_exact_continuation_and_stale(tmp_path, monkeypatch):
    from machina.skills.sections import content_fingerprint, index_sections
    from machina.tools.skills import skill_read

    reg = _tool_registry(tmp_path, monkeypatch)
    body = reg.get_skill_content("deploy-guide")
    section_text = index_sections(body).find("run-tests").text
    revision = content_fingerprint(body)
    marker = f"revision {revision}:\n"

    whole = skill_read("deploy-guide", section="run-tests")
    assert whole.split(marker, 1)[1] == section_text
    assert whole.startswith("Skill deploy-guide section run-tests [0:")

    first = skill_read("deploy-guide", section="run-tests", limit=12)
    served = first.split(marker, 1)[1].split("\n[Section continues", 1)[0]
    assert served == section_text[:12]
    assert first.endswith(
        "[Section continues; continue skill_read with "
        f"section='run-tests', revision={revision!r}, offset=12.]"
    )

    resumed = skill_read(
        "deploy-guide", section="run-tests", revision=revision, offset=12, limit=12
    )
    resumed_text = resumed.split(marker, 1)[1].split("\n[Section continues", 1)[0]
    assert resumed_text == section_text[12:24]

    stale = skill_read("deploy-guide", section="run-tests", revision="0000000000000000")
    assert stale.startswith("Error:")
    assert "Re-read the section" in stale
    assert section_text[:24] not in stale

    unknown = skill_read("deploy-guide", section="no-such-section")
    assert unknown.startswith("Error:")
    assert "no-such-section" in unknown

    assert skill_read("deploy-guide", section="run-tests", limit=0) == (
        "Error: limit must be between 1 and 8000"
    )
