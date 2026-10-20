"""A2a — shared deterministic TaskDescriptor (spec §8.1–8.4).

Acceptance-criteria mapping (tdd-guide spec-first workflow):
  AC-8.1  one normalized descriptor over the existing intent layer
  AC-8.2  broad task coverage without blanket keyword routing
  AC-8.3  quoted-text conservatism + sticky constraints (not a security boundary)
  AC-8.4  bounded project facts; missing stack is a normal unresolved field

Every descriptor build must stay free of provider calls, embeddings, network
I/O, and package-script execution; the suite patches those entry points to
fail and scans the shipped source for banned imports.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(
    os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina"
)

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
BACKEND_CAPABILITIES = {"database", "payments", "deployment"}


def _descriptor(query, **kwargs):
    from machina.core.task_descriptor import build_task_descriptor

    return build_task_descriptor(query, **kwargs)


# --- AC-8.1: one normalized contract ------------------------------------------


def test_descriptor_exposes_the_shared_contract_fields():
    import dataclasses

    from machina.core.task_descriptor import TaskDescriptor

    assert {f.name for f in dataclasses.fields(TaskDescriptor)} == {
        "task_class",
        "task_families",
        "requested_capabilities",
        "explicit_stack",
        "project_stack",
        "inherited_stack",
        "operation_scope",
        "constraints",
        "unresolved_fields",
        "evidence",
        "task_state_revision",
    }


def test_task_class_reuses_the_existing_intent_vocabulary():
    from machina.core.intent import TASK_CLASSES

    for query in [
        "make me a website",
        "hi",
        "Where is the persona stored?",
        "continue",
    ]:
        assert _descriptor(query).task_class in TASK_CLASSES


def test_descriptor_is_deterministic_and_serializable():
    first = _descriptor("Build a landing page")
    second = _descriptor("Build a landing page")
    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict()))["task_families"] == ["web-build"]


# --- AC-8.2: required results table (spec §8.2) --------------------------------


def test_make_me_a_website_is_web_build_with_unresolved_stack():
    d = _descriptor("make me a website")
    assert d.task_class == "implement"
    assert d.task_families == ("web-build",)
    assert d.stack_provenance() == "unknown"
    assert "stack" in d.unresolved_fields
    assert d.operation_scope == "mutating"
    assert not set(d.requested_capabilities) & BACKEND_CAPABILITIES


def test_landing_page_does_not_imply_database_payment_or_deployment():
    d = _descriptor("build a landing page")
    assert d.task_class == "implement"
    assert d.task_families == ("web-build",)
    assert not set(d.requested_capabilities) & BACKEND_CAPABILITIES
    assert not {"database", "payments", "deployment"} & set(d.task_families)


def test_portfolio_website_is_web_build_but_stock_portfolio_is_not():
    built = _descriptor("build a portfolio website")
    assert built.task_families == ("web-build",)

    compared = _descriptor("compare my stock portfolio")
    assert "web-build" not in compared.task_families
    assert compared.task_class == "ambiguous"
    assert compared.task_families == ()
    assert compared.operation_scope == "unknown"


def test_website_concept_question_stays_explanation():
    d = _descriptor("what is a website?")
    assert d.task_class == "explain"
    assert d.task_families == ()
    assert d.operation_scope == "read-only"
    assert "web-build" not in d.task_families


def test_explain_how_someone_would_build_a_website_stays_explanation():
    d = _descriptor("explain how someone would build a website")
    assert d.task_class == "explain"
    assert d.task_families == ()


def test_marketing_site_paraphrase_is_web_build_with_unresolved_stack():
    d = _descriptor("spin up a marketing site")
    assert d.task_class == "implement"
    assert d.task_families == ("web-build",)
    assert d.stack_provenance() == "unknown"
    assert "stack" in d.unresolved_fields
    assert d.operation_scope == "mutating"
    assert not set(d.requested_capabilities) & BACKEND_CAPABILITIES


def test_homepage_paraphrase_is_web_build_with_unresolved_stack():
    d = _descriptor("create a simple homepage for my band")
    assert d.task_class == "implement"
    assert d.task_families == ("web-build",)
    assert d.stack_provenance() == "unknown"
    assert "stack" in d.unresolved_fields
    assert d.operation_scope == "mutating"
    assert not set(d.requested_capabilities) & BACKEND_CAPABILITIES


def test_qualified_web_nouns_stay_phrase_level_and_bare_site_still_never_matches():
    from machina.core.intent import web_build_request

    assert web_build_request("spin up a marketing site")
    assert web_build_request("create a simple homepage for my band")
    # The web-noun vocabulary stays a closed phrase list (spec §8.2): bare
    # "site" alone is not a web-build noun, and the familiar negatives keep
    # their observed decisions.
    assert not web_build_request("build a site")
    assert not web_build_request("compare my stock portfolio")
    assert not web_build_request("walk me through how a band site gets made")


def test_readonly_website_review_preserves_the_constraint():
    d = _descriptor("review this website, do not edit")
    assert d.task_class == "review-readonly"
    assert d.operation_scope == "read-only"
    assert "read-only" in d.constraints
    assert "no-edit" in d.constraints


def test_add_tests_after_active_python_state_is_a_continuation():
    d = _descriptor("add tests", task_state=PYTHON_STATE)
    assert d.task_class == "implement"
    assert d.task_families == ("coding", "continuation")
    assert "testing" in d.requested_capabilities
    assert d.inherited_stack["languages"] == ("python",)
    assert d.effective_stack()["languages"] == ("python",)
    assert d.task_state_revision == 4
    assert "stack" not in d.unresolved_fields


def test_add_tests_without_state_is_not_a_continuation():
    d = _descriptor("add tests")
    assert d.task_class == "implement"
    assert "continuation" not in d.task_families
    assert d.inherited_stack == {}
    assert d.effective_stack() == {}
    assert d.task_state_revision == 0


def test_explicit_vue_switch_supersedes_inherited_react():
    d = _descriptor("switch the implementation to Vue", task_state=REACT_STATE)
    assert d.task_class == "implement"
    assert d.explicit_stack == {"frameworks": ("vue",)}
    assert d.inherited_stack == {"frameworks": ("react",)}
    assert d.effective_stack() == {"frameworks": ("vue",)}
    assert d.stack_provenance() == "known"


def test_explicit_svelte_switch_supersedes_inherited_react():
    d = _descriptor("switch the frontend to Svelte", task_state=REACT_STATE)
    assert d.task_class == "implement"
    assert d.explicit_stack == {"frameworks": ("svelte",)}
    assert d.inherited_stack == {"frameworks": ("react",)}
    assert d.effective_stack() == {"frameworks": ("svelte",)}
    assert d.stack_provenance() == "known"


def test_explicit_solid_switch_is_recognized_but_the_bare_token_does_not_route():
    switched = _descriptor("switch the frontend to Solid", task_state=REACT_STATE)
    assert switched.task_class == "implement"
    assert switched.explicit_stack == {"frameworks": ("solid",)}
    assert switched.effective_stack() == {"frameworks": ("solid",)}

    bare = _descriptor("solid")
    assert bare.task_class == "ambiguous"
    assert bare.task_families == ()
    assert bare.explicit_stack == {}


def test_bare_svelte_token_does_not_route_to_a_coding_workflow():
    d = _descriptor("svelte")
    assert d.task_class == "ambiguous"
    assert d.task_families == ()
    assert "coding" not in d.task_families


def test_explicit_stack_wins_over_recent_context_without_task_state():
    recent = [
        {"chunk_type": "user_message", "payload": {"content": "Build a React app"}}
    ]
    d = _descriptor("switch the implementation to Vue", recent=recent)
    assert d.explicit_stack == {"frameworks": ("vue",)}
    assert d.inherited_stack == {"frameworks": ("react",)}
    assert d.effective_stack() == {"frameworks": ("vue",)}


def test_go_ahead_without_state_is_ambiguous_and_invents_nothing():
    d = _descriptor("go ahead")
    assert d.task_class == "ambiguous"
    assert d.task_families == ()
    assert d.operation_scope == "unknown"
    assert d.explicit_stack == {}
    assert d.inherited_stack == {}
    assert d.task_state_revision == 0
    assert "task_family" in d.unresolved_fields


@pytest.mark.parametrize("query", ["go", "rust", "swift"])
def test_bare_ambiguous_technologies_do_not_produce_language_workflows(query):
    d = _descriptor(query)
    assert d.task_class == "ambiguous"
    assert d.task_families == ()
    assert d.explicit_stack == {}
    assert "coding" not in d.task_families


# --- AC-8.3: quoted text and sticky constraints --------------------------------


@pytest.mark.parametrize(
    "query",
    [
        'Does "make me a website" route to web-build?',
        "Is 'build a landing page' a web-build request?",
        "“make me a website” — what family is that?",
    ],
)
def test_quoted_web_request_is_not_permission(query):
    d = _descriptor(query)
    assert "web-build" not in d.task_families
    assert d.task_class in {"explain", "ambiguous"}


def test_readonly_constraint_survives_a_later_fix_match_in_one_query():
    d = _descriptor("review this website, do not edit, then fix the bug")
    assert d.task_class == "review-readonly"
    assert d.operation_scope == "read-only"
    assert {"read-only", "no-edit"} <= set(d.constraints)


def test_inherited_readonly_constraint_is_sticky_across_turns():
    state = {"active_class": "implement", "constraints": ["read-only"], "revision": 1}
    d = _descriptor("now fix the bug", task_state=state)
    assert d.task_class == "debug"
    assert "read-only" in d.constraints
    assert d.operation_scope == "read-only"


def test_negated_intent_with_want_to_is_a_constraint_not_permission():
    d = _descriptor("I don't want to change the website")
    assert "no-edit" in d.constraints
    assert d.operation_scope == "read-only"


def test_constraint_detection_is_scanned_once(monkeypatch):
    import machina.core.task_descriptor as module

    calls: list[str] = []
    real = module.detect_constraints

    def counting(query):
        calls.append(query)
        return real(query)

    monkeypatch.setattr(module, "detect_constraints", counting)
    state = {
        "active_class": "implement",
        "constraints": ["read-only", "no-edit", "no-deploy"],
        "revision": 1,
    }
    query = "don't install anything; keep it read-only"
    d = _descriptor(query, task_state=state)
    assert calls == [query]
    assert {"read-only", "no-edit", "no-deploy", "no-install"} <= set(d.constraints)


def test_negated_build_with_explain_clause_is_an_explanation_with_a_constraint():
    d = _descriptor("don't build anything, just explain how sites work")
    assert d.task_class == "explain"
    assert "web-build" not in d.task_families
    assert d.operation_scope == "read-only"
    assert "read-only" in d.constraints


@pytest.mark.parametrize(
    "query",
    [
        "don't build anything",
        "do not build anything yet",
        "no need to build a homepage",
    ],
)
def test_negated_build_is_a_constraint_not_a_build_request(query):
    d = _descriptor(query)
    assert d.task_class == "ambiguous"
    assert "web-build" not in d.task_families
    assert d.operation_scope == "read-only"
    assert "read-only" in d.constraints


def test_negated_follow_on_after_a_build_keeps_the_build():
    d = _descriptor("build a website but don't deploy yet")
    assert d.task_class == "implement"
    assert d.task_families == ("web-build",)
    assert "no-deploy" in d.constraints
    assert d.operation_scope == "mutating"


def test_explicit_release_allows_editing_again():
    state = {
        "active_class": "implement",
        "constraints": ["read-only", "no-edit"],
        "revision": 1,
    }
    d = _descriptor("you can edit now", task_state=state)
    assert "read-only" not in d.constraints
    assert "no-edit" not in d.constraints
    assert d.operation_scope == "mutating"


def test_constraints_are_not_a_security_boundary_documented():
    import machina.core.task_descriptor as module

    text = module.__doc__ or ""
    assert "security boundary" in text
    assert "permissions" in text


# --- AC-8.4: bounded project facts ---------------------------------------------


def test_project_stack_from_existing_package_json_path(tmp_path):
    manifest = tmp_path / "package.json"
    manifest.write_text(json.dumps({"dependencies": {"react": "^18", "vue": "^3"}}))
    d = _descriptor("build a web app", project_metadata=manifest)
    assert d.project_stack["frameworks"] == ("react", "vue")
    assert d.stack_provenance() == "known"
    assert "stack" not in d.unresolved_fields


def test_project_stack_from_already_known_mapping():
    d = _descriptor(
        "build a web app", project_metadata={"stack": {"languages": ["python"]}}
    )
    assert d.project_stack["languages"] == ("python",)


def test_package_json_scripts_are_never_executed(tmp_path, monkeypatch):
    manifest = tmp_path / "package.json"
    manifest.write_text(
        json.dumps(
            {
                "dependencies": {"vue": "^3"},
                "scripts": {"postinstall": "echo SHOULD-NEVER-RUN > ran.txt"},
            }
        )
    )

    def boom(*args, **kwargs):
        raise AssertionError("package script or subprocess execution attempted")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(os, "system", boom)
    monkeypatch.setattr(os, "popen", boom)
    d = _descriptor("build a web app", project_metadata=manifest)
    assert d.project_stack["frameworks"] == ("vue",)
    assert not (tmp_path / "ran.txt").exists()


def test_malformed_package_json_degrades_to_unknown(tmp_path):
    manifest = tmp_path / "package.json"
    manifest.write_text("{not json")
    d = _descriptor("build a web app", project_metadata=manifest)
    assert d.project_stack == {}
    assert d.stack_provenance() == "unknown"
    assert "stack" in d.unresolved_fields


def test_builder_does_not_scan_project_trees(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"vue": "^3"}}))
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "package.json").write_text(json.dumps({"dependencies": {"react": "^18"}}))

    def boom(*args, **kwargs):
        raise AssertionError("recursive scan attempted")

    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr(Path, "rglob", boom)
    d = _descriptor("build a web app", project_metadata=tmp_path)
    assert d.project_stack["frameworks"] == ("vue",)


def test_missing_stack_is_a_normal_unresolved_field():
    d = _descriptor("Fix this traceback in one file")
    assert d.task_class == "debug"
    assert d.stack_provenance() == "unknown"
    assert "stack" in d.unresolved_fields


def test_stack_provenance_tracks_known_inherited_and_unknown():
    assert _descriptor("build a web app").stack_provenance() == "unknown"
    recent = [
        {"chunk_type": "user_message", "payload": {"content": "Build a Go server"}}
    ]
    assert _descriptor("add tests", recent=recent).stack_provenance() == "inherited"
    assert _descriptor("Implement Python tooling").stack_provenance() == "known"


# --- revision chain -------------------------------------------------------------


def test_revision_increments_across_state_export_round_trip():
    first = _descriptor("add tests", task_state=PYTHON_STATE)
    assert first.task_state_revision == 4
    exported = first.export_state()
    assert exported["revision"] == 4
    second = _descriptor("add more tests", task_state=exported)
    assert second.task_state_revision == 5
    assert second.inherited_stack["languages"] == ("python",)


def test_revision_is_zero_without_any_task_state():
    assert _descriptor("Implement a Python API").task_state_revision == 0


# --- evidence / provenance -------------------------------------------------------


def test_evidence_records_source_for_each_inferred_fact():
    d = _descriptor("make me a website")
    web = [e for e in d.evidence_for("task_families") if e.value == "web-build"]
    assert web and web[0].source == "rule"
    classes = d.evidence_for("task_class")
    assert any(e.value == "implement" and e.source == "query" for e in classes)


def test_evidence_marks_state_and_project_sources():
    inherited = _descriptor("add tests", task_state=PYTHON_STATE)
    assert any(
        e.source == "task_state" for e in inherited.evidence_for("inherited_stack")
    )

    manifest = {"stack": {"frameworks": ["vue"]}}
    project = _descriptor("build a web app", project_metadata=manifest)
    assert any(
        e.source == "project_metadata" for e in project.evidence_for("project_stack")
    )


def test_effective_stack_precedence_explicit_over_project_over_inherited():
    d = _descriptor(
        "build a React app",
        project_metadata={"stack": {"languages": ["python"]}},
        recent=[
            {"chunk_type": "user_message", "payload": {"content": "Build a Go server"}}
        ],
    )
    assert d.explicit_stack == {"frameworks": ("react",)}
    assert d.project_stack == {"languages": ("python",)}
    assert d.inherited_stack == {"languages": ("go",)}
    effective = d.effective_stack()
    assert effective["frameworks"] == ("react",)
    assert effective["languages"] == ("python",)


# --- empty / malformed input ----------------------------------------------------


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_ambiguous_without_crashing(query):
    d = _descriptor(query)
    assert d.task_class == "ambiguous"
    assert d.task_families == ()
    assert d.operation_scope == "unknown"
    assert d.task_state_revision == 0


# --- no provider / embedding / external calls ------------------------------------


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
    for name in ("core/task_descriptor.py", "core/intent.py"):
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
                assert not any(part in BANNED_PARTS for part in parts), (
                    f"{name}: {dotted}"
                )
        assert "__import__" not in text, name
        assert "subprocess" not in text, name
        assert "os.system" not in text, name


def test_task_descriptor_adds_no_third_classifier():
    text = (REPO / "machina" / "core" / "task_descriptor.py").read_text()
    assert "re.compile" not in text
    assert "classify_task" in text  # reuses the intent layer


def test_fresh_import_pulls_no_provider_or_embedding_modules():
    script = textwrap.dedent(
        """
        import machina.core.intent  # noqa: F401
        import machina.core.task_descriptor as td
        banned = %r
        pulled = sorted(
            name
            for name in __import__("sys").modules
            if name.startswith("machina.")
            and any(part in banned for part in name.lower().split("."))
        )
        assert not pulled, pulled
        d = td.build_task_descriptor("make me a website")
        assert "web-build" in d.task_families
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


CORPUS = [
    "make me a website",
    "build a landing page",
    "what is a website?",
    "explain how someone would build a website",
    "review this website, do not edit",
    "compare my stock portfolio",
    "go ahead",
    "hi",
    "",
]


def test_build_and_preview_make_zero_external_calls(monkeypatch):
    from machina.core.task_descriptor import (
        build_task_descriptor,
        preview_task_descriptor,
    )

    def boom(*args, **kwargs):
        raise AssertionError("external call attempted from the deterministic path")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(os, "system", boom)
    monkeypatch.setattr(os, "popen", boom)

    for query in CORPUS:
        descriptor = build_task_descriptor(query)
        preview = preview_task_descriptor(query)
        assert preview["descriptor"]["task_class"] == descriptor.task_class
        assert descriptor.explain()


# --- compatibility with existing intent callers ----------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hi", "conversational"),
        ("hello there!", "conversational"),
        ("What is memory growth?", "explain"),
        ("Where is the persona stored?", "learn-directory"),
        ("dont implement changes just review", "review-readonly"),
        ("Do a Python code review", "review-readonly"),
        ("Fix this traceback in one file", "debug"),
        ("Implement user login in auth.py", "implement"),
        ("continue", "ambiguous"),
        ("just write the file", "ambiguous"),
        ("go ahead", "ambiguous"),
    ],
)
def test_classify_task_behavior_is_preserved(query, expected):
    from machina.core.intent import classify_task

    assert classify_task(query) == expected


def test_existing_intent_signatures_are_preserved():
    from machina.core.intent import TASK_CLASSES, classify_task, conversational

    assert list(inspect.signature(classify_task).parameters) == ["query"]
    assert list(inspect.signature(conversational).parameters) == ["query"]
    assert TASK_CLASSES == (
        "conversational",
        "explain",
        "learn-directory",
        "review-readonly",
        "debug",
        "implement",
        "ambiguous",
    )


def test_resolve_task_compatibility_is_unchanged():
    from machina.controller.triage import resolve_task

    assert resolve_task("continue", task_state=None) == "implement"
    assert resolve_task("continue", task_state=None, default=None) == "ambiguous"
    assert resolve_task("continue", task_state={"active_class": "debug"}) == "debug"
