"""Fail-closed metadata generation for every skill-creation path.

Product requirement: under the v2-only policy a skill without validated
schema-v2 metadata is unusable, so a freshly installed/learned skill must get
metadata generated exactly once, for every creation path, before the installer
reports success. A generation failure must be reported honestly (structured
``metadataGenerated: false`` / ``metadataError`` plus an operator retry path),
never as ordinary success.

Covered paths:

* ``SkillHub.install`` (hub)
* ``repo_install`` activation through ``machina.tools.repo``
* learned / auto-applied skills (``LearningReviewer._auto_publish``)
* gateway ``skills.learn``
* CLI ``learn`` and the explicit ``machina skills enrich`` retry command
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from machina.skills import enrich, repo_install
from machina.skills.registry import SkillHub, SkillMetadataUnavailable, SkillRegistry

runner = CliRunner()

V1_BODY = "Body line.\n\n## Steps\n\nDo the thing carefully.\n"

ALPHA_SKILL = """---
name: alpha
description: Alpha skill for fixtures
triggers: [alpha]
---
Help with alpha tasks.
"""


class FakeSeam:
    """Injectable model seam: counts calls and returns (or raises) a response."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if isinstance(self.response, BaseException):
            raise self.response
        return str(self.response)


class RecordingLog:
    """Capture the structured metadata-outcome records."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def outcomes(self, status: str) -> list[dict]:
        return [
            fields
            for event, fields in self.events
            if event == "skills.metadata_generation" and fields.get("status") == status
        ]


@pytest.fixture
def no_network(monkeypatch):
    """Tripwire: any real provider or network call fails the test loudly."""
    import socket

    import machina.core.provider as provider_mod
    import machina.core.usage as usage_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("real provider/network call attempted")

    monkeypatch.setattr(provider_mod, "get_provider", _boom)
    monkeypatch.setattr(usage_mod.usage_store, "complete_call", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    return _boom


@pytest.fixture
def recorded_log(monkeypatch):
    from machina.observability import instrumentation

    recorder = RecordingLog()
    monkeypatch.setattr(instrumentation, "log_event", recorder)
    return recorder


def _valid_yaml(name: str, *, sections: list[dict] | None = None) -> str:
    payload = {
        "schema_version": 2,
        "skill_id": name,
        "task_families": ["testing"],
        "capabilities": ["test-authoring"],
        "aliases": [],
        "stack": {"languages": [], "frameworks": [], "independent": True},
        "preconditions": [],
        "required_tools": [],
        "exclusions": [],
        "sections": sections
        if sections is not None
        else [
            {
                "id": "steps",
                "heading": "Steps",
                "stage": "implement",
                "requires_sections": [],
            }
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _publish_hub_entry(tmp_path: Path, *, name: str = "hub-skill") -> tuple[SkillHub, SkillRegistry]:
    """Publish a v2 hub entry (metadata travels with the hub file)."""
    from machina.skills.metadata import SkillMetadata

    registry = SkillRegistry(tmp_path / "skills")
    hub = SkillHub(registry, hub_dir=tmp_path / "hub")
    registry.create_skill(name, "Hub skill", V1_BODY, triggers=["hub"])
    registry.load_all()
    registry.persist_skill_metadata(
        registry.get(name),
        SkillMetadata(
            skill_id=name,
            task_families=("testing",),
            capabilities=("test-authoring",),
        ),
    )
    hub.publish(name)
    registry.delete_skill(name)
    return hub, registry


def _search_names(registry: SkillRegistry, query: str) -> list[str]:
    from machina.skills.runtime import search_skills

    return [candidate.name for candidate in search_skills(registry, query).candidates]


def _strip_hub_metadata(hub: SkillHub, name: str) -> None:
    """Turn a published entry into a legacy one (no v2 metadata payload)."""
    import json

    path = hub._hub_file(name)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("metadata", None)
    path.write_text(json.dumps(data), encoding="utf-8")


# ─── hub install ─────────────────────────────────────────────────────────────


def test_hub_install_preserves_published_metadata(tmp_path, monkeypatch, no_network):
    """A v2 hub entry travels with its metadata: install does NOT regenerate."""
    hub, _ = _publish_hub_entry(tmp_path)
    seam = FakeSeam(_valid_yaml("hub-skill"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    skill = hub.install("hub-skill")
    assert skill.skill_metadata is not None
    assert skill.skill_metadata.task_families == ("testing",)
    assert len(seam.calls) == 0

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    installed = fresh.get("hub-skill")
    assert installed.skill_metadata is not None
    assert "hub-skill" in _search_names(fresh, "test authoring")

    # A second load/install never regenerates: the persisted document wins.
    assert enrich.ensure_skill_metadata_sync(installed, agent_seam=seam, registry=fresh) is False
    assert len(seam.calls) == 0


def test_legacy_hub_entry_generates_metadata_once(tmp_path, monkeypatch, no_network):
    """Legacy hub entries (no metadata payload) get one generation call."""
    hub, _ = _publish_hub_entry(tmp_path)
    _strip_hub_metadata(hub, "hub-skill")
    seam = FakeSeam(_valid_yaml("hub-skill"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    skill = hub.install("hub-skill")
    assert skill.skill_metadata is not None
    assert len(seam.calls) == 1

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    installed = fresh.get("hub-skill")
    assert installed.skill_metadata is not None
    assert enrich.ensure_skill_metadata_sync(installed, agent_seam=seam, registry=fresh) is False
    assert len(seam.calls) == 1


def test_hub_install_reports_metadata_failure_and_stays_unusable(
    tmp_path, monkeypatch, no_network, recorded_log
):
    hub, _ = _publish_hub_entry(tmp_path)
    _strip_hub_metadata(hub, "hub-skill")
    seam = FakeSeam(RuntimeError("provider down"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    with pytest.raises(SkillMetadataUnavailable) as excinfo:
        hub.install("hub-skill")
    message = str(excinfo.value)
    assert "unusable" in message.lower()
    assert "machina skills enrich hub-skill" in message

    # The skill may remain on disk (retryable) but never becomes usable.
    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    installed = fresh.get("hub-skill")
    assert installed is not None
    assert installed.skill_metadata is None
    assert "hub-skill" not in _search_names(fresh, "test authoring")

    assert len((tmp_path / "skills" / "hub-skill" / "SKILL.md").read_text())
    failures = recorded_log.outcomes("failed")
    assert failures and failures[0]["skill"] == "hub-skill"
    assert failures[0]["path"].endswith("SKILL.md")
    assert failures[0]["error_type"]


# ─── repo / GitHub activation ────────────────────────────────────────────────


def _make_repo(root: Path, files: dict[str, str]) -> str:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _stage(tmp_path: Path, monkeypatch, *, suffix: str = "") -> tuple[dict, Path, Path]:
    root = tmp_path / f"repo{suffix}"
    root.mkdir()
    revision = _make_repo(root, {"skills/alpha/SKILL.md": ALPHA_SKILL})
    staging = tmp_path / f"staging{suffix}"
    monkeypatch.setattr(repo_install, "staging_root", lambda: staging)
    plan = repo_install.scan_repo(
        root, repo_install.parse_repo_url(f"https://github.com/o/r{suffix}"), revision
    )
    repo_install.write_staged_plan(plan, root)
    detail, found = repo_install.load_staged_plan(plan.digest)
    skills_dir = tmp_path / f"skills{suffix}"
    skills_dir.mkdir()
    return detail, found, skills_dir


def _repo_tool(monkeypatch, skills_dir: Path) -> tuple[object, SkillRegistry]:
    import machina.tools.repo as repo_tool

    registry = SkillRegistry(skills_dir)
    monkeypatch.setattr(repo_tool, "skill_registry", registry)
    return repo_tool, registry


def test_repo_activation_generates_metadata_once_and_refreshes_manifest(
    tmp_path, monkeypatch, no_network
):
    detail, staged, skills_dir = _stage(tmp_path, monkeypatch)
    repo_tool, _registry = _repo_tool(monkeypatch, skills_dir)
    seam = FakeSeam(_valid_yaml("alpha", sections=[]))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = repo_tool._activate_result(detail, staged, skills_dir)
    assert "metadataGenerated: true" in result

    fresh = SkillRegistry(skills_dir)
    fresh.load_all()
    assert fresh.get("alpha").skill_metadata is not None
    assert len(seam.calls) == 1

    # The generated front matter is acknowledged by the install manifest, or
    # every later reinstall would refuse the tree as diverged.
    manifest = json.loads((skills_dir / "alpha" / ".install-manifest.json").read_text())
    digest = hashlib.sha256((skills_dir / "alpha" / "SKILL.md").read_bytes()).hexdigest()
    assert manifest["files"]["alpha/SKILL.md"] == digest

    # Re-activating the exact install skips generation (metadata is present).
    again = repo_tool._activate_result(detail, staged, skills_dir)
    assert "metadataGenerated" in again
    assert len(seam.calls) == 1


def test_repo_activation_reports_metadata_failure_and_stays_unusable(
    tmp_path, monkeypatch, no_network, recorded_log
):
    detail, staged, skills_dir = _stage(tmp_path, monkeypatch)
    repo_tool, _registry = _repo_tool(monkeypatch, skills_dir)
    seam = FakeSeam(RuntimeError("provider down"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = repo_tool._activate_result(detail, staged, skills_dir)
    assert "metadataGenerated: false" in result
    assert "UNUSABLE" in result
    assert "machina skills enrich alpha" in result

    fresh = SkillRegistry(skills_dir)
    fresh.load_all()
    installed = fresh.get("alpha")
    assert installed is not None and installed.skill_metadata is None
    assert "alpha" not in _search_names(fresh, "test authoring")

    failures = recorded_log.outcomes("failed")
    assert failures and failures[0]["skill"] == "alpha"
    assert failures[0]["error_type"]


# ─── learned / auto-applied ─────────────────────────────────────────────────


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeConnection:
    def __init__(self, row: dict) -> None:
        self._row = row
        self.queries: list[str] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, query, *args):
        self.queries.append(query)
        return "OK"

    async def fetchval(self, query, *args):
        return 0

    async def fetchrow(self, query, *args):
        return self._row


class _FakeDb:
    def __init__(self, row: dict) -> None:
        self.conn = _FakeConnection(row)

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self.conn


def _review_row(name: str) -> dict:
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "session_id": "22222222-2222-2222-2222-222222222222",
        "agent_id": "agent-1",
        "status": "approved",
        "name": name,
        "description": "Learned skill",
        "triggers": json.dumps(["learned"]),
        "content": "Run the learned workflow steps carefully.",
        "reason": "auto-applied",
        "auto_applied": True,
    }


async def test_auto_applied_skill_gets_metadata_before_use(tmp_path, monkeypatch, no_network):
    import machina.skills.learning as learning_mod

    row = _review_row("learned-skill")
    fake_db = _FakeDb(row)
    monkeypatch.setattr(learning_mod, "db", fake_db)
    monkeypatch.setattr(learning_mod, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(learning_mod.config, "learning_review_auto_apply_per_day", 5)
    seam = FakeSeam(_valid_yaml("learned-skill"))
    registry = SkillRegistry(tmp_path / "skills")
    reviewer = learning_mod.LearningReviewer(registry=registry, agent_seam=seam)

    applied, metadata_generated, metadata_error = await reviewer._auto_publish(
        row["id"],
        "learned-skill",
        "Learned skill",
        ["learned"],
        "Run the learned workflow steps carefully.",
        "agent-1",
    )
    assert applied is not None
    assert metadata_generated is True
    assert metadata_error == ""
    assert len(seam.calls) == 1

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    assert fresh.get("learned-skill").skill_metadata is not None
    assert "learned-skill" in _search_names(fresh, "test authoring")


async def test_auto_applied_metadata_failure_keeps_skill_unusable(
    tmp_path, monkeypatch, no_network, recorded_log
):
    import machina.skills.learning as learning_mod

    row = _review_row("learned-bad")
    monkeypatch.setattr(learning_mod, "db", _FakeDb(row))
    monkeypatch.setattr(learning_mod, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(learning_mod.config, "learning_review_auto_apply_per_day", 5)
    seam = FakeSeam(RuntimeError("provider down"))
    registry = SkillRegistry(tmp_path / "skills")
    reviewer = learning_mod.LearningReviewer(registry=registry, agent_seam=seam)

    applied, metadata_generated, metadata_error = await reviewer._auto_publish(
        row["id"],
        "learned-bad",
        "Learned skill",
        ["learned"],
        "Run the learned workflow steps carefully.",
        "agent-1",
    )
    assert applied is not None
    assert metadata_generated is False
    assert metadata_error

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    unusable = fresh.get("learned-bad")
    assert unusable is not None and unusable.skill_metadata is None
    assert "learned-bad" not in _search_names(fresh, "test authoring")
    assert recorded_log.outcomes("failed")


# ─── gateway learn ──────────────────────────────────────────────────────────


class _FakeGateway:
    provider = None
    model = None

    def require_db(self) -> None:
        return None


async def test_gateway_learn_reports_metadata_generation(tmp_path, monkeypatch, no_network):
    import machina.gateway.features.skills as feature

    registry = SkillRegistry(tmp_path / "skills")
    created = registry.create_skill("gw-skill", "GW", V1_BODY, triggers=["gw"])
    monkeypatch.setattr(feature, "_registry", lambda: registry)
    monkeypatch.setattr("machina.services.learn_skill", lambda *a, **k: created)
    seam = FakeSeam(_valid_yaml("gw-skill"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = await feature.skills_learn(_FakeGateway(), {"source": "demo"})
    assert result["metadataGenerated"] is True
    assert result["usable"] is True
    assert result["metadataError"] is None
    assert len(seam.calls) == 1


async def test_gateway_learn_failure_is_structured_not_silent(
    tmp_path, monkeypatch, no_network, recorded_log
):
    import machina.gateway.features.skills as feature

    registry = SkillRegistry(tmp_path / "skills")
    created = registry.create_skill("gw-bad", "GW", V1_BODY, triggers=["gw"])
    monkeypatch.setattr(feature, "_registry", lambda: registry)
    monkeypatch.setattr("machina.services.learn_skill", lambda *a, **k: created)
    seam = FakeSeam(RuntimeError("provider down"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = await feature.skills_learn(_FakeGateway(), {"source": "demo"})
    assert result["metadataGenerated"] is False
    assert result["usable"] is False
    assert result["metadataError"]
    assert "machina skills enrich gw-bad" in result["message"]
    assert recorded_log.outcomes("failed")


# ─── CLI learn + skills enrich ──────────────────────────────────────────────


def _cli_registry(monkeypatch, tmp_path: Path) -> SkillRegistry:
    import machina.skills.registry as registry_mod

    registry = SkillRegistry(tmp_path / "skills")
    monkeypatch.setattr(registry_mod, "skill_registry", registry)
    return registry


def test_cli_learn_exits_nonzero_when_metadata_generation_fails(
    tmp_path, monkeypatch, no_network, recorded_log
):
    from machina import cli, services

    registry = _cli_registry(monkeypatch, tmp_path)
    created = registry.create_skill("cli-bad", "CLI", V1_BODY, triggers=["cli"])
    monkeypatch.setattr(services, "learn_skill", lambda *a, **k: created)
    seam = FakeSeam(RuntimeError("provider down"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = runner.invoke(cli.app, ["learn", "demo", "--name", "cli-bad"])
    assert result.exit_code == 1, result.output
    assert "UNUSABLE" in result.output
    normalized = " ".join(result.output.split())
    assert "machina skills enrich cli-bad" in normalized
    assert recorded_log.outcomes("failed")


def test_cli_learn_succeeds_only_when_metadata_is_present(tmp_path, monkeypatch, no_network):
    from machina import cli, services

    registry = _cli_registry(monkeypatch, tmp_path)
    created = registry.create_skill("cli-good", "CLI", V1_BODY, triggers=["cli"])
    monkeypatch.setattr(services, "learn_skill", lambda *a, **k: created)
    seam = FakeSeam(_valid_yaml("cli-good"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = runner.invoke(cli.app, ["learn", "demo", "--name", "cli-good"])
    assert result.exit_code == 0, result.output
    assert len(seam.calls) == 1
    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    assert fresh.get("cli-good").skill_metadata is not None


def test_cli_skills_enrich_backfills_and_is_idempotent(tmp_path, monkeypatch, no_network):
    from machina import cli

    registry = _cli_registry(monkeypatch, tmp_path)
    registry.create_skill("needs-meta", "Needs metadata", V1_BODY, triggers=["needs"])
    seam = FakeSeam(_valid_yaml("needs-meta"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = runner.invoke(cli.app, ["skills", "enrich", "needs-meta"])
    assert result.exit_code == 0, result.output
    assert "needs-meta" in result.output
    assert len(seam.calls) == 1
    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    assert fresh.get("needs-meta").skill_metadata is not None

    # Idempotent: valid metadata is never regenerated.
    again = runner.invoke(cli.app, ["skills", "enrich", "needs-meta"])
    assert again.exit_code == 0, again.output
    assert len(seam.calls) == 1


def test_cli_skills_enrich_all_reports_failures_and_exits_nonzero(
    tmp_path, monkeypatch, no_network, recorded_log
):
    from machina import cli

    registry = _cli_registry(monkeypatch, tmp_path)
    registry.create_skill("bad-one", "Bad", V1_BODY, triggers=["bad"])
    seam = FakeSeam(RuntimeError("provider down"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    result = runner.invoke(cli.app, ["skills", "enrich", "--all"])
    assert result.exit_code == 1, result.output
    assert "bad-one" in result.output
    assert "UNUSABLE" in result.output or "unusable" in result.output
    assert recorded_log.outcomes("failed")

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    assert fresh.get("bad-one").skill_metadata is None


def test_cli_skills_lists_without_a_subcommand(tmp_path, monkeypatch, no_network):
    from machina import cli

    _cli_registry(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["skills"])
    assert result.exit_code == 0, result.output
    assert "No skills found" in result.output
