"""Install-time generation of schema-v2 skill metadata, exactly once.

Consumer-boundary tests for ``machina.skills.enrich`` and the install/learn
hooks (gateway ``skills.learn``, ``SkillHub.install``, the ``repo_activate``
worker path). The contract under test:

* a skill with no metadata triggers ONE provider call, its validated metadata
  is persisted into the front matter, and the deterministic selector sees the
  generated families/capabilities;
* a skill that already carries metadata (authored or generated) triggers ZERO
  calls — generation never runs twice;
* invalid model output leaves the skill installed on its baseline path with a
  recorded diagnostic and no metadata written;
* the front-matter merge preserves the instruction body byte-for-byte;
* no test touches a real provider or the network (tripwire fixture).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import yaml

from machina.skills import enrich
from machina.skills.registry import SkillHub, SkillRegistry

V1_BODY = "Body line.\n\n## Steps\n\nDo the thing carefully.\n"

AUTHORED_V2 = """---
schema_version: 2
skill_id: authored-skill
name: authored-skill
description: Authored v2 metadata must be left alone
triggers:
  - authored
task_families: [documentation]
capabilities: [cli]
---
# Authored

Already labeled.
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


def _metadata_yaml(
    name: str,
    *,
    families: tuple[str, ...] = ("testing",),
    capabilities: tuple[str, ...] = ("test-authoring",),
    sections: list[dict] | None = None,
) -> str:
    payload = {
        "schema_version": 2,
        "skill_id": name,
        "task_families": list(families),
        "capabilities": list(capabilities),
        "aliases": [],
        "stack": {"languages": [], "frameworks": [], "independent": True},
        "preconditions": [],
        "required_tools": [],
        "exclusions": [],
        "sections": sections
        or [
            {
                "id": "steps",
                "heading": "Steps",
                "stage": "implement",
                "requires_sections": [],
            }
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _split_body(text: str) -> str:
    return text.split("---\n", 2)[2]


@pytest.fixture
def no_network(monkeypatch):
    """Tripwire: any real provider or network call fails the test loudly."""
    import machina.core.provider as provider_mod
    import machina.core.usage as usage_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("real provider/network call attempted")

    monkeypatch.setattr(provider_mod, "get_provider", _boom)
    monkeypatch.setattr(usage_mod.usage_store, "complete_call", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    return _boom


# ─── missing metadata: exactly one call, persisted, selector sees it ─────────


async def test_missing_metadata_generates_once_and_selector_sees_it(tmp_path, no_network):
    reg = SkillRegistry(tmp_path)
    reg.create_skill("sample-skill", "Sample", V1_BODY, triggers=["sample"])
    reg.load_all()
    skill = reg.get("sample-skill")
    assert skill.skill_metadata is None

    seam = FakeSeam(
        _metadata_yaml(
            "sample-skill",
            families=("testing",),
            capabilities=("test-authoring",),
        )
    )
    ok = await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reg)
    assert ok is True
    assert len(seam.calls) == 1

    fresh = SkillRegistry(tmp_path)
    fresh.load_all()
    meta = fresh.get("sample-skill").skill_metadata
    assert meta is not None
    assert meta.task_families == ("testing",)
    assert meta.capabilities == ("test-authoring",)
    assert meta.sections and meta.sections[0].anchor is not None  # bound to "Steps"
    # Provenance is recognized, not an extension warning.
    assert not [
        d for d in fresh.metadata_diagnostics() if d.code == "metadata-unsupported-field"
    ]

    # The deterministic selector now routes on the generated metadata.
    from machina.skills.runtime import search_skills

    result = search_skills(fresh, "test authoring")
    candidate = next(c for c in result.candidates if c.name == "sample-skill")
    assert "test-authoring" in candidate.matched_capabilities
    assert "verify" in candidate.workflow_stages


async def test_generate_returns_none_and_diagnostic_when_seam_raises(tmp_path, no_network):
    seam = FakeSeam(RuntimeError("provider down"))
    diagnostics: list = []
    metadata = await enrich.generate_skill_metadata(
        V1_BODY, skill_name="sample-skill", agent_seam=seam, diagnostics=diagnostics
    )
    assert metadata is None
    assert len(seam.calls) == 1
    assert diagnostics and diagnostics[0].severity == "error"


# ─── metadata already present: zero calls (authored OR generated) ────────────


def _write_raw(tmp_path, name: str, text: str) -> SkillRegistry:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    reg = SkillRegistry(tmp_path)
    reg.load_all()
    return reg


async def test_authored_metadata_skips_generation(tmp_path, no_network):
    reg = _write_raw(tmp_path, "authored-skill", AUTHORED_V2)
    skill = reg.get("authored-skill")
    assert skill.skill_metadata is not None
    seam = FakeSeam(_metadata_yaml("authored-skill"))
    assert await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reg) is False
    assert seam.calls == []


async def test_generated_metadata_is_never_regenerated(tmp_path, no_network):
    reg = SkillRegistry(tmp_path)
    reg.create_skill("sample-skill", "Sample", V1_BODY, triggers=["sample"])
    reg.load_all()
    seam = FakeSeam(_metadata_yaml("sample-skill"))
    assert await enrich.ensure_skill_metadata(
        reg.get("sample-skill"), agent_seam=seam, registry=reg
    )
    assert len(seam.calls) == 1

    # A fresh load (what a reinstall/re-activate sees) already carries metadata.
    reloaded = SkillRegistry(tmp_path)
    reloaded.load_all()
    skill = reloaded.get("sample-skill")
    assert skill.skill_metadata is not None
    assert (
        await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reloaded) is False
    )
    assert len(seam.calls) == 1


# ─── invalid model output: install continues, diagnostic recorded ────────────


async def test_invalid_model_output_keeps_skill_and_records_diagnostic(
    tmp_path, monkeypatch, no_network
):
    reg = SkillRegistry(tmp_path)
    reg.create_skill("sample-skill", "Sample", V1_BODY, triggers=["sample"])
    reg.load_all()
    skill = reg.get("sample-skill")
    path = skill.file_path

    recorded: list[str] = []
    monkeypatch.setattr(
        enrich, "record_failure", lambda boundary, error: recorded.append(boundary)
    )
    seam = FakeSeam("this is not a YAML mapping at all")
    assert await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reg) is False
    assert len(seam.calls) == 1
    assert recorded == ["skills.enrich.generate"]

    # No metadata written and the skill still loads (baseline behavior).
    assert "schema_version" not in Path(path).read_text(encoding="utf-8")
    again = SkillRegistry(tmp_path)
    again.load_all()
    assert again.get("sample-skill") is not None
    assert again.get("sample-skill").skill_metadata is None


def test_sync_bridge_refuses_inside_a_running_loop(tmp_path, monkeypatch, no_network):
    reg = SkillRegistry(tmp_path)
    reg.create_skill("sample-skill", "Sample", V1_BODY, triggers=["sample"])
    reg.load_all()
    seam = FakeSeam(_metadata_yaml("sample-skill"))

    recorded: list[str] = []
    monkeypatch.setattr(
        enrich, "record_failure", lambda boundary, error: recorded.append(boundary)
    )

    async def _inside_loop():
        return enrich.ensure_skill_metadata_sync(
            reg.get("sample-skill"), agent_seam=seam, registry=reg
        )

    import asyncio

    assert asyncio.run(_inside_loop()) is False
    assert seam.calls == []
    assert recorded == ["skills.enrich.sync_bridge"]


# ─── front-matter merge preserves the body byte-for-byte ─────────────────────


async def test_frontmatter_merge_preserves_body_and_original_keys(tmp_path, no_network):
    body = "Intro line.  \n\n\n## Steps\n\nKeep trailing spaces.   \n"
    original = (
        "---\n"
        "name: sample-skill\n"
        "description: Sample description\n"
        "triggers:\n"
        "  - sample\n"
        "version: 2.3.4\n"
        "source: https://example.test/sample.md\n"
        "source_type: url\n"
        "---\n" + body
    )
    reg = _write_raw(tmp_path, "sample-skill", original)
    skill = reg.get("sample-skill")
    assert _split_body(Path(skill.file_path).read_text(encoding="utf-8")) == body

    seam = FakeSeam(_metadata_yaml("sample-skill"))
    assert await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reg)

    updated = Path(skill.file_path).read_text(encoding="utf-8")
    assert _split_body(updated) == body  # byte-for-byte outside the metadata block
    frontmatter = yaml.safe_load(updated.split("---\n", 2)[1])
    assert frontmatter["schema_version"] == 2
    assert frontmatter["description"] == "Sample description"
    assert frontmatter["triggers"] == ["sample"]
    assert frontmatter["version"] == "2.3.4"
    assert frontmatter["source_type"] == "url"
    assert frontmatter["metadata_source"] == "generated"


async def test_persist_without_frontmatter_adds_valid_frontmatter(tmp_path, no_network):
    directory = tmp_path / "bare-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(V1_BODY, encoding="utf-8")
    reg = SkillRegistry(tmp_path)
    reg.load_all()
    skill = reg.get("bare-skill")
    seam = FakeSeam(_metadata_yaml("bare-skill"))
    assert await enrich.ensure_skill_metadata(skill, agent_seam=seam, registry=reg)
    fresh = SkillRegistry(tmp_path)
    fresh.load_all()
    meta = fresh.get("bare-skill").skill_metadata
    assert meta is not None and meta.task_families == ("testing",)
    assert _split_body(Path(skill.file_path).read_text(encoding="utf-8")) == V1_BODY


# ─── install/learn hooks ─────────────────────────────────────────────────────


class _FakeGateway:
    provider = None
    model = None

    def require_db(self) -> None:  # pragma: no cover - trivial
        return None


async def test_gateway_learn_hook_generates_metadata_once(tmp_path, monkeypatch, no_network):
    reg = SkillRegistry(tmp_path)
    created = reg.create_skill("gw-skill", "GW", "Body.\n", triggers=["gw"])
    seam = FakeSeam(_metadata_yaml("gw-skill"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)
    monkeypatch.setattr("machina.services.learn_skill", lambda *a, **k: created)
    import machina.gateway.features.skills as feature

    monkeypatch.setattr(feature, "_registry", lambda: reg)

    result = await feature.skills_learn(_FakeGateway(), {"source": "demo"})
    assert result["skill"]["name"] == "gw-skill"
    assert len(seam.calls) == 1

    fresh = SkillRegistry(tmp_path)
    fresh.load_all()
    assert fresh.get("gw-skill").skill_metadata is not None


def test_hub_install_hook_generates_metadata_once(tmp_path, monkeypatch, no_network):
    reg = SkillRegistry(tmp_path / "skills")
    hub = SkillHub(reg, hub_dir=tmp_path / "hub")
    reg.create_skill("hub-skill", "Hub", V1_BODY, triggers=["hub"])
    hub.publish("hub-skill")
    reg.delete_skill("hub-skill")

    seam = FakeSeam(_metadata_yaml("hub-skill"))
    monkeypatch.setattr(enrich, "default_agent_seam", lambda **_: seam)

    skill = hub.install("hub-skill")
    assert skill.name == "hub-skill"
    assert len(seam.calls) == 1

    fresh = SkillRegistry(tmp_path / "skills")
    fresh.load_all()
    installed = fresh.get("hub-skill")
    assert installed.skill_metadata is not None
    assert installed.skill_metadata.task_families == ("testing",)

    # A reinstall attempt is refused before any generation, and a re-run of the
    # ensure hook on the installed skill is a no-op: no second provider call.
    import asyncio

    with pytest.raises(ValueError):
        hub.install("hub-skill")
    assert (
        asyncio.run(
            enrich.ensure_skill_metadata(installed, agent_seam=seam, registry=fresh)
        )
        is False
    )
    assert len(seam.calls) == 1
