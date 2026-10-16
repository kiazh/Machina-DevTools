"""A6 — Track A integration and release evidence (spec §17 A6, §15, §18, §19).

Held-out routing/ranking cases, the call-free audit for the whole deterministic
routing path, human/automatic parity, a live in-process gateway contract run
over the real JSON-RPC serialization boundary, and the §15 performance/cost
ledger. Every assertion here runs the ACTUAL pipeline
(``build_task_descriptor`` → ``search_skills`` / ``plan_workflow`` →
``automatic_skill_context``) or the real gateway handlers — no provider,
embedding, or network call is made by any test in this file (the DB-backed
gateway test is the sole exception: it connects to the disposable test
database named by ``MACHINA_TEST_DATABASE_URL``).

The A6 follow-up (2026-10-16) fixed the four recorded held-out misses
(H01/H02/H04/H11) through narrow shared rules; they now run as real assertions
under their unchanged frozen gates, and ``RECORDED_MISSES`` stays empty until a
future capture records a new miss. Thresholds and gates were NOT tuned to make
any phrase pass. The original capture, including the misses, is reproduced in
``docs/verification/DETERMINISTIC_SKILLS_A6_EVIDENCE.md``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(
    os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina"
)

TEST_DSN = os.environ.get("MACHINA_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not TEST_DSN, reason="MACHINA_TEST_DATABASE_URL not set")

PYTHON_STATE = {
    "active_class": "implement",
    "active_families": ["coding"],
    "stack": {"languages": ["python"]},
    "revision": 3,
}
REACT_STATE = {
    "active_class": "implement",
    "active_families": ["coding"],
    "stack": {"frameworks": ["react"]},
    "revision": 2,
}


# ─── shared harness ───────────────────────────────────────────────────────────


def _real_registry():
    from machina.skills.registry import skill_registry

    skill_registry.load_all()
    return skill_registry


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test the moment a socket or subprocess call happens."""

    def boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("network/subprocess call attempted by the deterministic path")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(os, "system", boom)
    monkeypatch.setattr(os, "popen", boom)


def _run_pipeline(query: str, **kwargs) -> dict:
    """Descriptor + search + workflow + automatic disclosure for one query."""
    from machina.core.task_descriptor import build_task_descriptor
    from machina.skills import runtime
    from machina.skills.workflow import plan_workflow

    registry = _real_registry()
    descriptor = build_task_descriptor(query, **kwargs)
    search = runtime.search_skills(registry, query, limit=10)
    workflow = plan_workflow(query, skills=registry, **kwargs)
    context = runtime.automatic_skill_context(query, skills=registry, **kwargs)
    return {
        "descriptor": descriptor,
        "search": search,
        "workflow": workflow,
        "context": context,
    }


# ─── §18 held-out routing/ranking cases ───────────────────────────────────────


#: Cases are paraphrases of the S01–S16 requirements and fresh negatives that
#: are NOT copied from the spec's table. Each gate names the required result;
#: ``observed`` fields document what the actual pipeline returns at the
#: inspected commit.
HELD_OUT = [
    {
        "id": "H01",
        "gate": "action + web noun paraphrase is web-build with unresolved stack",
        "query": "spin up a marketing site",
        "class_in": {"implement"},
        "families_include": ("web-build",),
        "families_exclude": ("coding",),
        "scope": "mutating",
        "unresolved_includes": ("stack",),
        "capabilities_exclude": ("database", "payments", "deployment"),
    },
    {
        "id": "H02",
        "gate": "homepage is a web-building noun",
        "query": "create a simple homepage for my band",
        "class_in": {"implement"},
        "families_include": ("web-build",),
        "families_exclude": ("coding",),
        "scope": "mutating",
        "unresolved_includes": ("stack",),
        "capabilities_exclude": ("database", "payments", "deployment"),
    },
    {
        "id": "H03",
        "gate": "concept question about a website stays an explanation",
        "query": "what even is a website",
        "class_in": {"explain"},
        "families_exclude": ("web-build", "coding"),
        "scope": "read-only",
        "context_empty": True,
    },
    {
        "id": "H04",
        "gate": "negated build + explain request is not build authorization",
        "query": "don't build anything, just explain how sites work",
        "class_in": {"explain", "ambiguous", "conversational"},
        "families_exclude": ("web-build",),
        "scope_in": {"read-only", "unknown"},
        "context_empty": True,
    },
    {
        "id": "H05",
        "gate": "asking how a site gets made is not a build request",
        "query": "walk me through how a band site gets made",
        "class_in": {"explain", "ambiguous"},
        "families_exclude": ("web-build",),
        "context_empty": True,
    },
    {
        "id": "H06",
        "gate": "financial portfolio is not a portfolio website",
        "query": "is my stock portfolio diversified",
        "class_in": {"explain"},
        "families_exclude": ("web-build", "coding"),
        "scope": "read-only",
        "context_empty": True,
    },
    {
        "id": "H07",
        "gate": "read-only review constraints survive with a web noun present",
        "query": "review my landing page copy, don't change anything",
        "class_in": {"review-readonly"},
        "families_exclude": ("web-build",),
        "constraints_include": ("read-only", "no-edit"),
        "scope": "read-only",
        "context_empty": True,
    },
    {
        "id": "H08",
        "gate": "quoted website request inside a routing question is not permission",
        "query": 'if someone says "build me a website", which skill family handles it?',
        "class_in": {"explain", "ambiguous"},
        "families_exclude": ("web-build",),
        "scope_in": {"read-only", "unknown"},
        "context_absent": "web-build",
    },
    {
        "id": "H09",
        "gate": "follow-up testing request continues the active Python task",
        "query": "add unit tests",
        "kwargs": {"task_state": PYTHON_STATE},
        "class_in": {"implement"},
        "families_include": ("coding", "continuation"),
        "inherited_stack": {"languages": ("python",)},
        "effective_stack": {"languages": ("python",)},
        "capabilities_include": ("testing",),
        "unresolved_excludes": ("stack",),
        "workflow_kind": "coding",
        "workflow_skills_include": ("ecc-tdd-workflow", "testing-qa"),
    },
    {
        "id": "H10",
        "gate": "same request with no active state is not a continuation",
        "query": "add unit tests",
        "class_in": {"implement"},
        "families_include": ("coding",),
        "families_exclude": ("continuation",),
        "inherited_stack": {},
        "effective_stack": {},
        "unresolved_includes": ("stack",),
        "capabilities_include": ("testing",),
    },
    {
        "id": "H11",
        "gate": "explicit new framework supersedes inherited React",
        "query": "switch the frontend to Svelte",
        "kwargs": {"task_state": REACT_STATE},
        "class_in": {"implement"},
        "explicit_stack": {"frameworks": ("svelte",)},
        "inherited_stack": {"frameworks": ("react",)},
        "effective_stack": {"frameworks": ("svelte",)},
        "stack_provenance": "known",
    },
    {
        "id": "H12",
        "gate": "bare English 'go' is a topic, never a language workflow",
        "query": "go",
        "class_in": {"ambiguous"},
        "families_exclude": ("coding",),
        "explicit_stack": {},
        "inherited_stack": {},
        "context_empty": True,
    },
    {
        "id": "H13",
        "gate": "bare English 'rust' is a topic, never a language workflow",
        "query": "rust",
        "class_in": {"ambiguous"},
        "families_exclude": ("coding",),
        "explicit_stack": {},
        "inherited_stack": {},
        "context_empty": True,
    },
    {
        "id": "H14",
        "gate": "bare English 'swift' is a topic, never a language workflow",
        "query": "swift",
        "class_in": {"ambiguous"},
        "families_exclude": ("coding",),
        "explicit_stack": {},
        "inherited_stack": {},
        "context_empty": True,
    },
    {
        "id": "H15",
        "gate": "deploy request gains the deployment capability without backend inference",
        "query": "deploy my portfolio site to vercel",
        "class_in": {"implement"},
        "capabilities_include": ("deployment",),
        "capabilities_exclude": ("database", "payments"),
        "scope": "mutating",
        "context_contains": "deployment-devops",
    },
    {
        "id": "H16",
        "gate": "landing-page paraphrase is web-build without backend capabilities",
        "query": "craft a landing page for the launch",
        "class_in": {"implement"},
        "families_include": ("web-build",),
        "scope": "mutating",
        "unresolved_includes": ("stack",),
        "capabilities_exclude": ("database", "payments", "deployment"),
    },
    {
        "id": "H17",
        "gate": "portfolio-page paraphrase is web-build without backend capabilities",
        "query": "set up a portfolio page for my photography",
        "class_in": {"implement"},
        "families_include": ("web-build",),
        "scope": "mutating",
        "unresolved_includes": ("stack",),
        "capabilities_exclude": ("database", "payments", "deployment"),
    },
]

#: Recorded held-out misses. Empty after the A6 follow-up (2026-10-16): the
#: four frozen-gate misses H01/H02 (web-noun vocabulary), H04 (negated build)
#: and H11 (Svelte vocabulary) were fixed through narrow shared rules and now
#: run as real assertions; the frozen gates in HELD_OUT above are unchanged
#: and no threshold was tuned. The mechanism stays for future captures.
RECORDED_MISSES = {}


def _param(case):
    marks = ()
    if case["id"] in RECORDED_MISSES:
        marks = (pytest.mark.xfail(reason=RECORDED_MISSES[case["id"]], strict=False),)
    return pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize("case", [_param(case) for case in HELD_OUT])
def test_a6_held_out_pipeline(case, no_network):
    triple = _run_pipeline(case["query"], **case.get("kwargs", {}))
    descriptor = triple["descriptor"]
    workflow = triple["workflow"]
    context = triple["context"]

    assert descriptor.task_class in case["class_in"], descriptor.to_dict()
    families = set(descriptor.task_families)
    for family in case.get("families_include", ()):
        assert family in families, descriptor.to_dict()
    for family in case.get("families_exclude", ()):
        assert family not in families, descriptor.to_dict()

    if "scope" in case:
        assert descriptor.operation_scope == case["scope"], descriptor.to_dict()
    if "scope_in" in case:
        assert descriptor.operation_scope in case["scope_in"], descriptor.to_dict()
    if "explicit_stack" in case:
        assert dict(descriptor.explicit_stack) == case["explicit_stack"], descriptor.to_dict()
    if "inherited_stack" in case:
        assert dict(descriptor.inherited_stack) == case["inherited_stack"], descriptor.to_dict()
    if "effective_stack" in case:
        assert dict(descriptor.effective_stack()) == case["effective_stack"], descriptor.to_dict()
    if "stack_provenance" in case:
        assert descriptor.stack_provenance() == case["stack_provenance"], descriptor.to_dict()
    for constraint in case.get("constraints_include", ()):
        assert constraint in descriptor.constraints, descriptor.to_dict()
    for capability in case.get("capabilities_include", ()):
        assert capability in descriptor.requested_capabilities, descriptor.to_dict()
    for capability in case.get("capabilities_exclude", ()):
        assert capability not in descriptor.requested_capabilities, descriptor.to_dict()
    for field in case.get("unresolved_includes", ()):
        assert field in descriptor.unresolved_fields, descriptor.to_dict()
    for field in case.get("unresolved_excludes", ()):
        assert field not in descriptor.unresolved_fields, descriptor.to_dict()

    if "workflow_kind" in case:
        assert workflow.kind == case["workflow_kind"], (workflow.kind, workflow.stages)
    for skill_id in case.get("workflow_skills_include", ()):
        assert skill_id in {s.name for s in workflow.skills}, workflow.stages

    if case.get("context_empty"):
        assert context == "", context[:200]
    if "context_contains" in case:
        assert case["context_contains"] in context, context[:200]
    if "context_absent" in case:
        assert case["context_absent"] not in context, context[:200]

    # Every held-out case runs on the search path too and must never claim a
    # calibrated probability or enumerate the catalog on a no-match.
    assert triple["search"].policy_revision == "skills-search-v1"
    if triple["search"].diagnostics["classification"] == "no_lexical_evidence":
        assert triple["search"].candidates == []


def test_a6_held_out_table_is_complete_and_not_copied_from_the_spec():
    assert len(HELD_OUT) >= 12
    queries = {case["query"] for case in HELD_OUT}
    assert len(queries) >= 12
    spec_table = {
        "make me a website",
        "build a landing page",
        "what is a website?",
        "explain how someone would build a website",
        "review this website, do not edit",
        "build a portfolio website",
        "compare my stock portfolio",
        "switch the implementation to Vue",
        "go ahead",
        "hello",
        "thanks",
    }
    assert not (queries & spec_table), "held-out phrases must not reuse the spec table"
    assert set(RECORDED_MISSES) <= {case["id"] for case in HELD_OUT}


# ─── §15/§19 call-free audit: no provider/embedding/network on the routing path ─

#: Shipped modules that make up the deterministic routing path. ``registry`` is
#: included because runtime/search read the catalog through it; the gateway
#: handler is included because ``skills.preview_route`` is the preview path.
ROUTER_SOURCES = (
    "machina/core/task_descriptor.py",
    "machina/core/intent.py",
    "machina/skills/runtime.py",
    "machina/skills/search.py",
    "machina/skills/workflow.py",
    "machina/skills/planned_steps.py",
    "machina/skills/sections.py",
    "machina/skills/metadata.py",
    "machina/skills/revisions.py",
    "machina/skills/registry.py",
    "machina/gateway/features/skills.py",
)

ROUTER_MODULES = tuple(
    path.removesuffix(".py").replace("/", ".") for path in ROUTER_SOURCES
)

BANNED_IMPORT_ROOTS = (
    "machina.core.provider",
    "machina.core.agent",
    "machina.rag",
    "openai",
    "anthropic",
    "cohere",
    "mistralai",
    "litellm",
    "ollama",
    "sentence_transformers",
    "fastembed",
    "model2vec",
    "onnxruntime",
    "torch",
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
    "socket",
    "subprocess",
)

#: Banned module roots among *freshly imported* modules.
BANNED_FRESH_MODULE_ROOTS = (
    "openai",
    "anthropic",
    "cohere",
    "mistralai",
    "litellm",
    "ollama",
    "sentence_transformers",
    "fastembed",
    "model2vec",
    "onnxruntime",
    "torch",
)

#: Deterministic core modules whose fresh import must be strictly clean.
CLEAN_MODULES = tuple(
    module for module in ROUTER_MODULES if module != "machina.gateway.features.skills"
)


def test_a6_routing_sources_import_no_provider_embedding_or_network_module():
    for relative in ROUTER_SOURCES:
        path = REPO / relative
        assert path.is_file(), relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=relative)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for dotted in names:
                assert not any(
                    dotted == banned or dotted.startswith(banned + ".")
                    for banned in BANNED_IMPORT_ROOTS
                ), f"{relative}: banned import {dotted!r}"
        assert "__import__" not in text, relative
        assert "os.system" not in text, relative
        assert "os.popen" not in text, relative


def test_a6_fresh_import_of_the_routing_path_pulls_no_provider_or_embedding_modules():
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        clean_modules = %(clean)r
        gateway_module = %(gateway)r
        banned_roots = %(roots)r
        banned_prefixes = (
            "machina.core.provider",
            "machina.rag",
            "machina.core.agent",
            "machina.core.images",
            "machina.connectors",
        )

        def banned(names):
            return sorted(
                name
                for name in names
                if name.split(".")[0] in banned_roots
                or any(name == prefix or name.startswith(prefix + ".") for prefix in banned_prefixes)
            )

        for name in clean_modules:
            importlib.import_module(name)
        strict = banned(sys.modules)
        assert not strict, strict

        # The deterministic path runs with those modules absent.
        from machina.core.task_descriptor import build_task_descriptor
        from machina.skills.registry import skill_registry
        from machina.skills.runtime import automatic_skill_context, search_skills

        skill_registry.load_all()
        descriptor = build_task_descriptor("make me a website")
        assert "web-build" in descriptor.task_families
        result = search_skills(skill_registry, "add tests", limit=5)
        assert result.policy_revision
        automatic_skill_context("add tests", skills=skill_registry)

        # The pre-existing gateway package init (gateway/__init__ -> server ->
        # provider wiring) already loads provider/httpx; snapshot it, then prove
        # the skills handler import itself adds no provider/embedding/network
        # module of its own.
        importlib.import_module("machina.gateway.server")
        before = set(sys.modules)
        importlib.import_module(gateway_module)
        delta = sorted(set(sys.modules) - before)
        added = banned(delta)
        assert not added, added
        assert "httpx" not in delta and "machina.core.provider" not in delta
        print(json.dumps({"ok": True, "clean": len(clean_modules), "delta": len(delta)}))
        """
    ) % {
        "clean": list(CLEAN_MODULES),
        "gateway": "machina.gateway.features.skills",
        "roots": list(BANNED_FRESH_MODULE_ROOTS),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"fresh import failed:\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip())
    assert payload["ok"] is True
    assert payload["clean"] == len(CLEAN_MODULES)


def test_a6_full_routing_pipeline_makes_zero_network_or_subprocess_calls(no_network):
    from machina.skills import runtime
    from machina.skills.workflow import plan_workflow

    registry = _real_registry()
    for phrase, kwargs in (
        ("make me a website", {}),
        ("build a landing page", {}),
        ("what is a website?", {}),
        ("add tests", {"task_state": PYTHON_STATE}),
        ("switch the frontend to Svelte", {"task_state": REACT_STATE}),
        ("go ahead", {}),
        ("hi", {}),
    ):
        triple = _run_pipeline(phrase, **kwargs)
        assert triple["descriptor"].explain()
        # Search twice: cold index build and warm cache hit.
        runtime.search_skills(registry, phrase, limit=5)
        runtime.search_skills(registry, phrase, limit=5)
        plan_workflow(phrase, skills=registry, **kwargs)
        runtime.automatic_skill_context(phrase, skills=registry, **kwargs)


def test_a6_preview_and_search_handlers_never_touch_a_provider(monkeypatch):
    """The gateway preview/browse/section handlers stay call-free under the guard.

    Sync on purpose: the event loop is created BEFORE the socket patch so the
    loop's own self-pipe cannot trip the tripwire; the handlers themselves run
    with every socket and subprocess entry point failing.
    """
    from machina.gateway.features import skills as feature

    class FakeGateway:
        def __init__(self) -> None:
            self.guards = 0

        def require_db(self) -> None:
            self.guards += 1

    class ProviderTripwire:
        def __getattr__(self, name):
            raise AssertionError(f"provider module touched by a call-free handler: {name}")

        def __call__(self, *args, **kwargs):
            raise AssertionError("provider called by a call-free handler")

    def boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("network/subprocess call attempted by a call-free handler")

    gw = FakeGateway()

    async def body() -> None:
        search = await feature.skills_search(
            gw, {"query": "testing", "filters": {"include_excluded": True}, "limit": 5}
        )
        assert search["browse"] is True
        assert search["candidates"], "the real catalog must return testing candidates"

        candidate = search["candidates"][0]
        # The shown search revision is the canonical join identity: the first
        # offset read after search is fresh (F1 follow-up).
        section = await feature.skills_section(
            gw,
            {
                "skillId": candidate["skillId"],
                "contentRevision": candidate["contentRevision"],
                "offset": 0,
                "budget": 200,
            },
        )
        assert section["stale"] is False
        assert section["text"]

        preview = await feature.skills_preview_route(gw, {"request": "make me a website"})
        assert preview["descriptor"]["task_families"] == ["web-build"]
        assert preview["budget"] == 1200
        assert gw.guards == 3, "every handler keeps its database guard"

    loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(socket, "socket", boom)
        monkeypatch.setattr(socket, "create_connection", boom)
        monkeypatch.setattr(socket, "getaddrinfo", boom)
        monkeypatch.setattr(subprocess, "run", boom)
        monkeypatch.setattr(subprocess, "Popen", boom)
        monkeypatch.setattr(os, "system", boom)
        monkeypatch.setattr(os, "popen", boom)
        monkeypatch.setitem(sys.modules, "machina.core.provider", ProviderTripwire())
        monkeypatch.setattr(feature, "_registry", _real_registry)
        loop.run_until_complete(body())
    finally:
        loop.close()


# ─── §17 A6 parity: automatic disclosure scope vs browsing scope ──────────────


PARITY_QUERIES = (
    "make me a website",
    "add tests",
    "testing",
    "deploy",
    "website",
    "go ahead",
)


def test_a6_automatic_and_browse_search_agree_on_the_real_catalog():
    from machina.skills.runtime import search_skills

    registry = _real_registry()
    automatic_total = registry.list_skills()
    assert all(skill.enabled for skill in automatic_total), (
        "parity is only identical while every catalog skill is enabled; a disabled "
        "skill shows up as the documented browse-only difference (next test)"
    )
    for query in PARITY_QUERIES:
        automatic = search_skills(registry, query, limit=20)
        browse = search_skills(registry, query, limit=20, include_excluded=True)
        assert automatic.catalog_revision == browse.catalog_revision, query
        assert automatic.diagnostics == browse.diagnostics, query
        assert automatic.next_cursor == browse.next_cursor, query
        assert [
            (c.skill_id, c.score, c.applicability, c.content_revision)
            for c in automatic.candidates
        ] == [
            (c.skill_id, c.score, c.applicability, c.content_revision)
            for c in browse.candidates
        ], query


def test_a6_disabled_skill_is_the_only_documented_scope_difference(tmp_path):
    from machina.skills.registry import SkillRegistry
    from machina.skills.runtime import search_skills
    from machina.skills.search import EXCLUDED, catalog_revision

    registry = SkillRegistry(tmp_path / "skills")
    registry.create_skill("team-deploy", "Deploy services carefully", "S", triggers=["deploy"])
    registry.create_skill("old-deploy", "Deploy services with legacy steps", "S", triggers=["deploy"])
    registry.set_enabled("old-deploy", False)

    automatic = search_skills(registry, "deploy the service", limit=20)
    browse = search_skills(registry, "deploy the service", limit=20, include_excluded=True)

    automatic_ids = [c.skill_id for c in automatic.candidates]
    assert "old-deploy" not in automatic_ids
    browsed = {c.skill_id: c for c in browse.candidates}
    assert browsed["old-deploy"].applicability == EXCLUDED
    assert any("disabled" in reason for reason in browsed["old-deploy"].reasons)

    # Candidate set, order, and applicability are identical apart from the
    # disabled entry. Scores are lexical (BM25-style IDF) and the spec §9.2
    # documents that they are catalog-dependent, so including the disabled
    # document in the browse index shifts scores but must not reorder or
    # re-classify the enabled candidates.
    assert automatic_ids == [
        c.skill_id for c in browse.candidates if c.skill_id != "old-deploy"
    ]
    enabled_browse = {
        c.skill_id: c for c in browse.candidates if c.skill_id != "old-deploy"
    }
    for candidate in automatic.candidates:
        counterpart = enabled_browse[candidate.skill_id]
        assert counterpart.applicability == candidate.applicability
        assert abs(counterpart.score - candidate.score) < 2.0, (
            candidate.skill_id,
            candidate.score,
            counterpart.score,
        )

    # Revisions are computed over each documented effective scope: automatic =
    # enabled catalog, browse = full catalog (the disabled member is the delta).
    registry.load_all()
    all_skills = registry.list_skills()
    assert automatic.catalog_revision == catalog_revision(
        [skill for skill in all_skills if skill.enabled]
    )
    assert browse.catalog_revision == catalog_revision(all_skills)
    assert automatic.catalog_revision != browse.catalog_revision


async def test_a6_gateway_browse_matches_the_runtime_browse_scope(monkeypatch):
    """The human-facing `skills.search` browse path equals `search_skills(include_excluded)`."""
    from machina.gateway.features import skills as feature
    from machina.skills.runtime import search_skills

    class FakeGateway:
        def require_db(self) -> None:
            pass

    registry = _real_registry()
    monkeypatch.setattr(feature, "_registry", lambda: registry)
    gw = FakeGateway()
    for query in PARITY_QUERIES:
        payload = await feature.skills_search(
            gw,
            {"query": query, "filters": {"include_excluded": True}, "limit": 20},
        )
        runtime_result = search_skills(registry, query, limit=20, include_excluded=True)
        assert payload["browse"] is True
        assert payload["catalogRevision"] == runtime_result.catalog_revision, query
        assert payload["nextCursor"] == runtime_result.next_cursor, query
        assert [
            (c["skillId"], c["score"], c["applicability"], c["contentRevision"])
            for c in payload["candidates"]
        ] == [
            (c.skill_id, c.score, c.applicability, c.content_revision)
            for c in runtime_result.candidates
        ], query
        # The automatic scope differs only for disabled skills — none here.
        assert payload["filters"]["include_excluded"] is True
        assert payload["filters"]["enabled"] is None


# ─── §19.4 live gateway over the real serialization boundary ──────────────────


class LiveGatewayHarness:
    """In-process Gateway over a real test DB; every call crosses JSON-RPC."""

    def __init__(self) -> None:
        from machina.gateway.server import Gateway

        self.frames: list[dict] = []
        self.gateway = Gateway(self.frames.append)
        self._next_id = 0

    async def call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        rid = self._next_id
        frame = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            frame["params"] = params
        await self.gateway.handle_line(json.dumps(frame))
        return next(f for f in self.frames if f.get("id") == rid and "method" not in f)

    def method_calls(self, method: str) -> list[dict]:
        return [f for f in self.frames if f.get("method") == method]


@needs_db
async def test_a6_live_gateway_skills_contracts_over_json_rpc():
    from machina.gateway.errors import INVALID_PARAMS
    from machina.skills.registry import skill_registry
    from machina.skills.sections import index_sections

    harness = LiveGatewayHarness()
    try:
        init = await harness.call("initialize")
        assert "result" in init, init
        assert init["result"]["version"]

        search = await harness.call(
            "skills.search",
            {"query": "testing", "filters": {"include_excluded": True}, "limit": 5},
        )
        assert "result" in search, search
        payload = search["result"]
        assert payload["browse"] is True
        assert payload["candidates"], payload
        assert payload["catalogRevision"], payload
        assert payload["policyRevision"] == "skills-search-v1"
        assert payload["diagnostics"]["catalog_size"] >= 70, payload["diagnostics"]

        # Closed filter vocabulary fails closed over the wire.
        bad = await harness.call("skills.search", {"query": "x", "filters": {"nope": 1}})
        assert bad["error"]["code"] == INVALID_PARAMS

        # Preview never starts a turn and returns the deterministic plan.
        preview = await harness.call("skills.preview_route", {"request": "make me a website"})
        assert "result" in preview, preview
        plan = preview["result"]
        assert plan["descriptor"]["task_families"] == ["web-build"]
        assert plan["descriptor"]["unresolved_fields"] == ["stack"]
        assert plan["budget"] == 1200
        assert plan["tokenBasis"]
        assert not harness.method_calls("prompt.submit"), "preview must not start a turn"

        # Version-bound section read for a real catalog entry: the address is
        # derived with the same section module the handler uses, then read over
        # the wire; the payload tiles the section text exactly. The revision is
        # the canonical skill content revision — the same identity the search
        # candidate above exposes (F1 follow-up) — so the shown revision joins.
        from machina.skills.sections import skill_content_revision

        candidate = payload["candidates"][0]
        skill = skill_registry.get(candidate["skillId"])
        assert skill is not None, candidate
        content = skill_registry.get_skill_content(skill.name)
        revision = skill_content_revision(skill)
        assert revision == candidate["contentRevision"], candidate
        first = index_sections(content).sections[0]
        section = await harness.call(
            "skills.section",
            {
                "skillId": candidate["skillId"],
                "contentRevision": revision,
                "section": first.section_id,
                "budget": 120,
            },
        )
        assert "result" in section, section
        read = section["result"]
        assert read["stale"] is False
        assert read["text"], read
        assert read["contentRevision"] == revision
        section_text = content[read["start"] : read["end"]]
        assert section_text.startswith(read["text"])
        if read["continuation"] is not None:
            continuation = await harness.call(
                "skills.section",
                {
                    "skillId": candidate["skillId"],
                    "contentRevision": revision,
                    "section": first.section_id,
                    "offset": read["continuation"],
                    "budget": 120,
                },
            )
            assert "result" in continuation, continuation
            second = continuation["result"]
            assert second["stale"] is False
            combined = read["text"] + second["text"]
            assert combined == section_text[: len(combined)], "pages tile the section"

        # A stale revision returns the structured stale response, never a
        # different passage at the same address.
        stale = await harness.call(
            "skills.section",
            {
                "skillId": candidate["skillId"],
                "contentRevision": "deadbeefdeadbeef",
                "section": first.section_id,
                "budget": 120,
            },
        )
        stale_read = stale["result"]
        assert stale_read["stale"] is True
        assert stale_read["text"] == ""
        assert any(
            diagnostic["code"] == "section-stale-content"
            for diagnostic in stale_read["diagnostics"]
        ), stale_read

        # Unknown section: explicit diagnostic, empty text — never a silent
        # neighbouring passage.
        unknown = await harness.call(
            "skills.section",
            {
                "skillId": candidate["skillId"],
                "contentRevision": revision,
                "section": "no-such-section-xyz",
                "budget": 120,
            },
        )
        unknown_read = unknown["result"]
        assert unknown_read["text"] == ""
        assert any(
            diagnostic["code"] == "section-unknown"
            for diagnostic in unknown_read["diagnostics"]
        ), unknown_read

        # Every response frame stays JSON-serializable (the real wire shape).
        for frame in harness.frames:
            json.dumps(frame)
    finally:
        await harness.gateway.close()


#: F1 follow-up (resolved): ``skills.search`` and ``skills.section`` share one
#: canonical skill content revision (``machina.skills.search.content_revision``
#: — version + served content bytes + bundle identity + typed metadata digest),
#: so a revision shown by search is fresh on the first read. The earlier
#: mismatch (search ``cae54bcc…`` vs section fingerprint ``59235bb3…`` for
#: ``testing-qa``) is recorded in the findings register of
#: ``docs/verification/DETERMINISTIC_SKILLS_A6_EVIDENCE.md``.


@needs_db
async def test_a6_live_gateway_search_revision_joins_section_read():
    from machina.skills import revisions
    from machina.skills.registry import skill_registry
    from machina.skills.sections import index_sections, skill_content_revision

    harness = LiveGatewayHarness()
    try:
        assert "result" in await harness.call("initialize")
        search = await harness.call(
            "skills.search",
            {"query": "testing", "filters": {"include_excluded": True}, "limit": 5},
        )
        candidates = search["result"]["candidates"]
        assert candidates, search
        joined: list[tuple[str, str]] = []
        for candidate in candidates:
            skill = skill_registry.get(candidate["name"])
            content = skill_registry.get_skill_content(candidate["name"])
            if skill is None or not content:
                continue
            parsed = index_sections(content).sections
            if not parsed:
                continue
            # One canonical identity on all three sides, at the same load.
            assert candidate["contentRevision"] == skill_content_revision(skill)
            assert (
                revisions.RevisionRegistry.for_skill(skill).content
                == candidate["contentRevision"]
            )
            section = await harness.call(
                "skills.section",
                {
                    "skillId": candidate["skillId"],
                    "contentRevision": candidate["contentRevision"],
                    "section": parsed[0].section_id,
                    "budget": 120,
                },
            )
            read = section["result"]
            assert read["stale"] is False, (
                f"shown revision {candidate['contentRevision']} was rejected as stale"
            )
            assert read["contentRevision"] == candidate["contentRevision"], read
            assert read["text"], read
            joined.append((candidate["skillId"], parsed[0].section_id))
        # The real page joins local and bundled skills on one revision identity.
        assert len(joined) >= 2, joined

        # A genuinely unknown revision still returns the structured stale
        # response with empty text — the join does not weaken stale detection.
        skill_id, section_id = joined[0]
        unknown = await harness.call(
            "skills.section",
            {
                "skillId": skill_id,
                "contentRevision": "0000000000000000",
                "section": section_id,
                "budget": 120,
            },
        )
        unknown_read = unknown["result"]
        assert unknown_read["stale"] is True
        assert unknown_read["text"] == ""
        assert any(
            diagnostic["code"] == "section-stale-content"
            for diagnostic in unknown_read["diagnostics"]
        ), unknown_read
    finally:
        await harness.gateway.close()


# ─── §15 performance/cost ledger ──────────────────────────────────────────────


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


def _measure(callable_, iterations: int) -> dict:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callable_()
        samples.append((time.perf_counter_ns() - started) / 1000)  # µs
    return {"p50_us": _pct(samples, 50), "p95_us": _pct(samples, 95), "n": iterations}


LEDGER_CORPUS = (
    "make me a website",
    "build a landing page",
    "what is a website?",
    "add tests",
    "compare my stock portfolio",
    "go ahead",
)


def test_a6_performance_and_cost_ledger():
    from machina.core.assembler import get_token_count
    from machina.core.task_descriptor import build_task_descriptor
    from machina.skills import runtime
    from machina.skills.planned_steps import normalize_stage_state, plan_disclosure
    from machina.skills.sections import (
        clear_section_cache,
        content_fingerprint,
        index_sections,
    )
    from machina.skills.workflow import plan_workflow

    registry = _real_registry()
    registry.load_all()

    lines: list[str] = []
    lines.append("| metric | p50 | p95 | n |")
    lines.append("|---|---|---|---|")

    # 1. descriptor
    descriptor = _measure(
        lambda: [build_task_descriptor(query) for query in LEDGER_CORPUS], 50
    )
    lines.append(
        f"| build_task_descriptor (6-query batch) | {descriptor['p50_us']:.1f} µs | "
        f"{descriptor['p95_us']:.1f} µs | {descriptor['n']} |"
    )
    assert descriptor["p95_us"] < 50_000

    # 2. warm search (shared per-turn cache)
    for query in LEDGER_CORPUS:
        runtime.search_skills(registry, query, limit=10)
    warm = _measure(lambda: runtime.search_skills(registry, "make me a website", limit=10), 200)
    lines.append(
        f"| search_skills warm (cache hit) | {warm['p50_us']:.1f} µs | "
        f"{warm['p95_us']:.1f} µs | {warm['n']} |"
    )
    assert warm["p95_us"] < 50_000

    # 3. cold index rebuild (cache cleared per iteration)
    cold_samples = []
    for _ in range(8):
        runtime._INDEXES.clear()
        started = time.perf_counter_ns()
        runtime.search_skills(registry, "make me a website", limit=10)
        cold_samples.append((time.perf_counter_ns() - started) / 1000)
    lines.append(
        f"| search_skills cold (index rebuild) | {_pct(cold_samples, 50):.1f} µs | "
        f"{_pct(cold_samples, 95):.1f} µs | 8 |"
    )

    # 4. disclosure planning over the authoritative steps
    workflow = plan_workflow("add tests", skills=registry, task_state=PYTHON_STATE)
    state = normalize_stage_state(PYTHON_STATE)

    def disclosure():
        return plan_disclosure(
            workflow.steps,
            skill_for=registry.get,
            content_for=registry.get_skill_content,
            budget=1200,
            stage_state=state,
            task_revision=workflow.task_revision,
            catalog_revision=workflow.catalog_revision,
        )

    planned = _measure(disclosure, 200)
    lines.append(
        f"| plan_disclosure (add tests, 4 steps) | {planned['p50_us']:.1f} µs | "
        f"{planned['p95_us']:.1f} µs | {planned['n']} |"
    )
    assert planned["p95_us"] < 100_000

    # 5. section parse (miss) and per-revision cache hit
    section_skill = next(
        skill
        for skill in registry.list_skills()
        if skill.name == "skills-system" and registry.get_skill_content(skill.name)
    )
    section_content = registry.get_skill_content(section_skill.name)
    clear_section_cache()
    miss_samples = []
    for _ in range(24):
        clear_section_cache()
        started = time.perf_counter_ns()
        index_sections(section_content)
        miss_samples.append((time.perf_counter_ns() - started) / 1000)
    index_sections(section_content)  # ensure cached
    hit_samples = []
    for _ in range(2000):
        started = time.perf_counter_ns()
        index_sections(section_content)
        hit_samples.append((time.perf_counter_ns() - started) / 1000)
    lines.append(
        f"| section parse cold (miss, {len(section_content)} B) | {_pct(miss_samples, 50):.1f} µs | "
        f"{_pct(miss_samples, 95):.1f} µs | 24 |"
    )
    lines.append(
        f"| section index hit (same revision) | {_pct(hit_samples, 50):.1f} µs | "
        f"{_pct(hit_samples, 95):.1f} µs | 2000 |"
    )
    # A4 pinned a 0.55 µs hit on the revision-cache path; this is the same
    # predicate A6 measures (content fingerprint + dict lookup). Record, don't tune.
    assert _pct(hit_samples, 95) < 50

    # 6. tokens loaded per acceptance case (disclosure included_tokens + basis).
    # F2 follow-up (fixed): plan_disclosure.included_tokens sums every disclosed
    # block — chosen sections plus whole-body fallback units — measured with
    # the plan's own token counter, so a metadata-less catalog no longer reads
    # 0 while the assembled context carries tokens. The ledger keeps the
    # independent assembler measurement side by side with the plan estimate.
    lines.append("")
    lines.append(
        "| case | context chars | tokens (assembler) | plan included_tokens "
        "(chosen+whole-body) | whole-body est tokens | basis | steps |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for query, kwargs in (
        ("make me a website", {}),
        ("build a landing page", {}),
        ("what is a website?", {}),
        ("add tests", {"task_state": PYTHON_STATE}),
        ("compare my stock portfolio", {}),
        ("go ahead", {}),
    ):
        context = runtime.automatic_skill_context(query, skills=registry, **kwargs)
        flow = plan_workflow(query, skills=registry, **kwargs)
        disclosure = plan_disclosure(
            flow.steps,
            skill_for=registry.get,
            content_for=registry.get_skill_content,
            budget=1200,
            stage_state=normalize_stage_state(kwargs.get("task_state")),
            task_revision=flow.task_revision,
            catalog_revision=flow.catalog_revision,
        )
        whole_body_tokens = sum(item.estimate_tokens for item in disclosure.whole_body)
        chosen_tokens = sum(item.estimate_tokens for item in disclosure.chosen)
        assert disclosure.included_tokens == chosen_tokens + whole_body_tokens, query
        if disclosure.whole_body:
            assert disclosure.included_tokens > 0, query
        json.dumps(disclosure.to_dict())
        lines.append(
            f"| {query!r} | {len(context)} | {get_token_count(context)} | "
            f"{disclosure.included_tokens} | {whole_body_tokens} | "
            f"{disclosure.token_basis} | {len(flow.steps)} |"
        )

    # 7. parse/stat counts for a warm load: no media churn, no re-parse.
    import pathlib

    from machina.skills.registry import SkillParser

    parse_calls = {"count": 0}
    original_parse = SkillParser.parse

    def counting_parse(self, *args, **kwargs):
        parse_calls["count"] += 1
        return original_parse(self, *args, **kwargs)

    stat_calls = {"count": 0}
    original_stat = pathlib.Path.stat

    def counting_stat(self, *args, **kwargs):
        stat_calls["count"] += 1
        return original_stat(self, *args, **kwargs)

    monkey_parse = counting_parse
    SkillParser.parse = monkey_parse
    pathlib.Path.stat = counting_stat
    try:
        registry.load_all()
    finally:
        SkillParser.parse = original_parse
        pathlib.Path.stat = original_stat
    lines.append("")
    lines.append(
        f"warm load_all: {parse_calls['count']} content parses, "
        f"{stat_calls['count']} Path.stat calls; "
        f"fingerprint {content_fingerprint(section_content)} "
        f"({len(section_content)} B section skill); routing corpus is text-only "
        "(no media/attachment module is imported by the clean routing modules — "
        "pinned by the fresh-import audit)"
    )

    report = "\n".join(lines)
    assert "build_task_descriptor" in report
    assert "section index hit" in report
    print(report)
