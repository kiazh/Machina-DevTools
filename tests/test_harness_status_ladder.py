"""Truncation-ladder ordering for ``harness_status`` (P0-19 sibling).

The ladder must shed duplicated/recoverable detail (the FEATURES map, persona
display config) before live learning state. The review found the opposite
order: oversized notes were emptied before the ~80%-of-payload FEATURES map,
so self-knowledge lost learning state first. These tests pin the pathological
payloads, not just a typical snapshot.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock

import pytest

from machina.core.models import Session


def _status_fixture(monkeypatch, harness, *, notes, rsi):
    sid = uuid.uuid4()
    owner = "ladder-owner"
    monkeypatch.setattr(harness, "active_agent_scope", AsyncMock(return_value=(owner, sid)))
    monkeypatch.setattr(
        harness.session_manager,
        "get",
        AsyncMock(return_value=Session(id=sid, agent_id=owner, execution_mode="ask")),
    )
    monkeypatch.setattr(
        harness.growth_store,
        "get",
        AsyncMock(return_value={"notes": notes, "interactions": 3, "feedback_count": 1}),
    )
    monkeypatch.setattr(harness.growth_store, "rsi_status", AsyncMock(return_value=rsi))
    monkeypatch.setattr(
        harness,
        "_persona_summary",
        AsyncMock(return_value={"available": True, "effective_style": "mentor"}),
    )
    monkeypatch.setattr(harness, "_installed_inventory", lambda: {"skills_by_source": {}})
    monkeypatch.setattr(harness, "_workspace_section", lambda: {"available": False})
    monkeypatch.setattr(harness.interop_mod, "state_summary", dict)
    monkeypatch.setattr(harness, "configuration", lambda session: {"execution_mode": "ask"})


@pytest.mark.asyncio
async def test_oversized_features_shrink_before_learning_notes(monkeypatch):
    from machina.tools import harness

    notes = {"style:tone": "Speak plainly", "lesson:tests": "Run focused tests first"}
    _status_fixture(
        monkeypatch,
        harness,
        notes=notes,
        rsi={"enabled": True, "revision": 3, "trial": {"id": "trial-1"}},
    )
    # The feature map is the expected ~80% of a healthy payload.
    monkeypatch.setattr(
        harness,
        "FEATURES",
        {f"feature_{i}": "x" * 256 for i in range(80)},
    )

    result = await harness.harness_status()
    payload = json.loads(result)

    assert len(result) <= harness._MAX_STATUS_BYTES
    assert payload["learning"]["notes"] == notes, "notes are live evidence and must survive"
    assert "notes_omitted" not in payload["learning"]
    assert payload["features"] == {}, "FEATURES must be the section that shrinks"
    assert "features_omitted" in payload
    # Pinned trial state is the last learning section to go and fits here.
    assert payload["learning"]["rsi"]["trial"] == {"id": "trial-1"}
    assert "trial_omitted" not in payload["learning"]["rsi"]


@pytest.mark.asyncio
async def test_persona_display_config_shrinks_before_learning_notes(monkeypatch):
    from machina.core.config import config
    from machina.tools import harness

    notes = {"style:tone": "Speak plainly"}
    _status_fixture(monkeypatch, harness, notes=notes, rsi={"enabled": True, "revision": 1})
    monkeypatch.setattr(harness, "FEATURES", {})
    # Recoverable display config (the /persona editor holds the real value).
    monkeypatch.setattr(
        config,
        "get",
        lambda key: "p" * 9000 if key == "persona_description" else None,
    )

    result = await harness.harness_status()
    payload = json.loads(result)

    assert len(result) <= harness._MAX_STATUS_BYTES
    assert payload["learning"]["notes"] == notes
    assert "notes_omitted" not in payload["learning"]
    assert payload["persona_description"] is None
    assert "persona_description_omitted" in payload
