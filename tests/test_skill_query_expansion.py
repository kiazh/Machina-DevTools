"""One-call LLM query expansion into deterministic catalog vocabulary.

Product contract (fix campaign, query-expansion lane):

* a turn makes exactly ONE bounded model call to split the request into the
  closed v2 routing vocabulary (task families, capabilities, keywords/stack);
* the deterministic, non-LLM picker stays the only selector: the expansion is
  merged into the query text and handed to ``automatic_skills``;
* any provider error, timeout, or junk answer is a fail-open empty expansion;
* the expansion is turn-scoped (ContextVar), cached by normalized query, and
  never leaks raw query text to instrumentation.
"""

from __future__ import annotations

import asyncio

import pytest

import machina.skills.query_expansion as qe
from machina.skills.query_expansion import (
    Expansion,
    build_expansion_prompt,
    clear_cache,
    current_expansion,
    expand_query,
    merge_into_query,
    prepare_turn,
)
from machina.skills.registry import SkillRegistry
from machina.skills.runtime import automatic_skill_context, automatic_skills


@pytest.fixture(autouse=True)
def _isolate_turn_state():
    clear_cache()
    qe._CURRENT.set(None)
    yield
    clear_cache()
    qe._CURRENT.set(None)


def _seam(payload, calls: list[str] | None = None):
    async def _call(prompt: str) -> str:
        if calls is not None:
            calls.append(prompt)
        return payload

    return _call


_WEB_ANSWER = (
    "task_families: [web-build]\n"
    "capabilities: [layout, responsive-ui]\n"
    "keywords: [website, landing page, html, css]\n"
    "stack_languages: [html, css]\n"
    "stack_frameworks: []\n"
)


def test_build_expansion_prompt_is_bounded_and_catalog_scoped():
    prompt = build_expansion_prompt("make me a website")
    assert "catalog router" in prompt.casefold()
    assert "Return ONLY" in prompt
    assert "web-build" in prompt and "backend-api" in prompt
    assert "layout" in prompt and "responsive-ui" in prompt
    assert "Do not answer the task" in prompt or "do not answer" in prompt.casefold()
    assert "make me a website" in prompt


@pytest.mark.asyncio
async def test_prepare_turn_routes_website_to_web_frontend():
    expansion = await prepare_turn("make me a website", seam=_seam(_WEB_ANSWER))
    assert expansion.source == "llm"
    assert expansion.task_families == ("web-build",)
    assert set(expansion.capabilities) == {"layout", "responsive-ui"}
    assert current_expansion("make me a website") == expansion

    skills = SkillRegistry()
    ranked = [s.name for s in automatic_skills(skills, "make me a website", expansion=expansion)]
    assert "web-frontend" in ranked[:3], ranked[:5]

    context = automatic_skill_context("make me a website", skills=skills)
    assert "web-frontend" in context


@pytest.mark.asyncio
async def test_exactly_one_seam_call_for_repeated_query_and_none_from_search():
    calls: list[str] = []
    first = await prepare_turn("make me a unique website", seam=_seam(_WEB_ANSWER, calls))
    second = await prepare_turn("make me a unique website", seam=_seam(_WEB_ANSWER, calls))
    assert first.source == "llm"
    assert second.source == "cache"
    assert first.keywords == second.keywords
    assert first.task_families == second.task_families
    assert first.capabilities == second.capabilities
    assert len(calls) == 1

    skills = SkillRegistry()
    automatic_skills(
        skills,
        "make me a unique website",
        expansion=current_expansion("make me a unique website"),
    )
    automatic_skills(skills, "make me a unique website")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_seam_failure_timeout_and_junk_are_fail_open():
    async def boom(_prompt: str) -> str:
        raise RuntimeError("provider unavailable")

    async def slow(_prompt: str) -> str:
        await asyncio.sleep(1)
        return _WEB_ANSWER

    for query, seam in (
        ("fail-open raising query", boom),
        ("fail-open junk query", _seam("I would be happy to help with that!")),
        ("fail-open list query", _seam("- just\n- a\n- list\n")),
    ):
        expansion = await expand_query(query, seam=seam)
        assert expansion.source == "fallback"
        assert expansion.keywords == ()
        assert expansion.task_families == ()

    timed_out = await expand_query("fail-open timeout query", seam=slow, timeout=0.01)
    assert timed_out.source == "fallback"

    # Deterministic search still works with the empty fallback.
    skills = SkillRegistry()
    assert automatic_skills(skills, "add retry logic to the worker")


@pytest.mark.asyncio
async def test_vocabulary_is_closed_keywords_bounded_deduped_and_redacted():
    answer = (
        "task_families: [web-build, marketing, web-build]\n"
        "capabilities: [layout, nonsense, responsive-ui]\n"
        "keywords:\n"
        "  - website\n"
        "  - Website\n"
        "  - landing page\n"
        "  - landing page\n"
        "  - " + ("x" * 120) + "\n"
        "  - sk-abcdefghijklmnopqrstuvwxyz012345\n"
        "  - html\n"
        "  - css\n"
        "  - javascript\n"
        "  - seo\n"
        "  - forms\n"
        "  - accessibility\n"
        "  - performance\n"
        "  - browser\n"
        "  - homepage\n"
        "  - responsive\n"
        "stack_languages: [html, css]\n"
        "stack_frameworks: [react]\n"
    )
    expansion = await expand_query("unique closed vocabulary query", seam=_seam(answer))
    assert expansion.source == "llm"
    assert expansion.task_families == ("web-build",)
    assert expansion.capabilities == ("layout", "responsive-ui")
    assert len(expansion.keywords) <= 12
    lowered = [keyword.casefold() for keyword in expansion.keywords]
    assert len(lowered) == len(set(lowered))
    assert all(len(keyword) <= 60 for keyword in expansion.keywords)
    assert not any("sk-abcdefghijklmnop" in keyword for keyword in expansion.keywords)
    assert "javascript" in expansion.keywords
    assert expansion.stack_languages == ("html", "css")
    assert expansion.stack_frameworks == ("react",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["build a landing page for my business", "make a web app"],
)
async def test_expanded_routing_reaches_web_frontend(query):
    expansion = await prepare_turn(query, seam=_seam(_WEB_ANSWER))
    skills = SkillRegistry()
    ranked = [s.name for s in automatic_skills(skills, query, expansion=expansion)]
    assert "web-frontend" in ranked[:3], ranked[:5]


@pytest.mark.asyncio
async def test_expansion_adds_hidden_intent_terms_for_opaque_request():
    answer = (
        "task_families: [web-build]\n"
        "capabilities: [layout]\n"
        "keywords: [website, landing page, html, css]\n"
    )
    query = "put together a brochure presence for my bakery"
    expansion = await prepare_turn(query, seam=_seam(answer))
    merged = merge_into_query(query, expansion)
    assert "website" in merged
    assert "web-build" in merged
    assert "layout" in merged

    ranked = [s.name for s in automatic_skills(SkillRegistry(), query, expansion=expansion)]
    assert "web-frontend" in ranked[:3], ranked[:5]


def test_deterministic_routing_is_unchanged_without_expansion():
    skills = SkillRegistry()
    worker = [s.name for s in automatic_skills(skills, "add retry logic to the worker")]
    assert worker[0] == "background-jobs", worker[:3]
    provider = [
        s.name for s in automatic_skills(skills, "configure retries for the openrouter provider")
    ]
    assert provider[0] == "llm-provider", provider[:3]

    # A turn-scoped expansion for another query never changes these results.
    assert current_expansion("add retry logic to the worker").source == "fallback"


def test_agent_prepare_context_awaits_prepare_turn_once_before_prompt():
    """Wiring pin: the turn path makes one expansion call before the prompt.

    The expansion mechanics are covered above; this pins the single production
    call site (a second call would double the per-turn model spend).
    """
    import inspect

    from machina.core.agent import BaseReActAgent

    source = inspect.getsource(BaseReActAgent._prepare_context)
    marker = "await query_expansion.prepare_turn(user_message)"
    assert source.count(marker) == 1
    assert source.index(marker) < source.index("system_prompt = build_system_prompt(")


def test_exact_name_match_dominates_with_expansion_and_is_deterministic():
    from machina.skills.runtime import search_skills

    skills = SkillRegistry()
    expansion = Expansion(keywords=("website", "landing page"), task_families=("web-build",))
    ranked = [s.name for s in automatic_skills(skills, "web-frontend", expansion=expansion)]
    assert ranked[0] == "web-frontend", ranked[:3]
    assert ranked == [
        s.name for s in automatic_skills(skills, "web-frontend", expansion=expansion)
    ]
    first = search_skills(skills, "web-frontend", expansion=expansion)
    second = search_skills(skills, "web-frontend", expansion=expansion)
    assert [c.skill_id for c in first.candidates] == [c.skill_id for c in second.candidates]


def test_merge_into_query_is_sanitized_and_bounded():
    expansion = Expansion(
        keywords=("website", "landing page", "x" * 500),
        task_families=("web-build",),
        capabilities=("layout",),
        source="cache",
    )
    merged = merge_into_query("make me a website", expansion)
    assert "web-build" in merged
    assert "layout" in merged
    assert merged.startswith("make me a website")
    assert len(merged) <= 600
    assert merge_into_query("hello", Expansion()) == "hello"


@pytest.mark.asyncio
async def test_cache_hits_are_reported_without_raw_query(monkeypatch):
    events: list[tuple[str, dict]] = []

    def _capture(event: str, **fields):
        events.append((event, fields))

    monkeypatch.setattr(qe, "log_event", _capture)
    calls: list[str] = []
    await expand_query("secret-ish query for cache", seam=_seam(_WEB_ANSWER, calls))
    await expand_query("secret-ish query for cache", seam=_seam(_WEB_ANSWER, calls))
    assert len(calls) == 1
    names = [event for event, _ in events]
    assert "skills.expand.finish" in names
    assert "skills.expand.cache_hit" in names
    for _event, fields in events:
        assert "secret-ish query" not in str(fields)
        assert "query" not in fields
