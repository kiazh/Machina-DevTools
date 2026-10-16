"""Regression tests for the W2-E findings against machina/workspace/lsp.py.

1. typescript-language-server suppresses ALL diagnostics unless the client
   declares ``textDocument.publishDiagnostics``; the capability must be
   present in initialize params.
2. ``_finalize_stop`` must give a just-killed process group a bounded grace
   window to reap (wrapper launchers exit before their tsserver children), so
   an ordinary stop reports ``stopped`` instead of a spurious
   ``cleanup_pending`` — while still staying honest when members persist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from machina.workspace import lsp as lsp_mod
from machina.workspace.lsp import (
    STATE_STOPPED,
    STATE_STOPPING,
    STOP_OUTCOME_CLEANUP_PENDING,
    STOP_OUTCOME_STOPPED,
    LspConnection,
    LspProfile,
)


def _profile(tmp_path: Path) -> LspProfile:
    script = tmp_path / "server.py"
    script.write_text("import sys\nsys.stdin.read()\n", encoding="utf-8")
    return LspProfile(
        profile_id="python",
        language_id="python",
        executable=str(script),
        args=[],
    )


def test_client_capabilities_declare_publish_diagnostics() -> None:
    capabilities = lsp_mod._client_capabilities()
    text_document = capabilities["textDocument"]
    assert "publishDiagnostics" in text_document
    # typescript-language-server checks truthiness of the key, not its shape.
    assert text_document["publishDiagnostics"]["relatedInformation"] is True
    # The conservative omissions stay omitted.
    assert "dynamicRegistration" not in str(capabilities)
    assert "executeCommand" not in str(capabilities)


def test_finalize_stop_waits_out_a_reaping_group(tmp_path: Path, monkeypatch) -> None:
    """Members vanish inside the grace window -> clean stop, no pending."""
    connection = LspConnection(_profile(tmp_path), root=tmp_path, group_reap_grace=1.0)
    calls = {"count": 0}

    def members() -> bool:
        calls["count"] += 1
        return calls["count"] <= 2  # members twice, then reaped

    monkeypatch.setattr(connection, "_group_has_members", members)
    monkeypatch.setattr(connection, "_process_handle", lambda: None)
    monkeypatch.setattr(lsp_mod, "kill_tree", lambda *args, **kwargs: None)
    outcome = asyncio.run(connection._finalize_stop())
    assert outcome == STOP_OUTCOME_STOPPED
    assert connection.cleanup_pending is False
    assert connection.state == STATE_STOPPED
    assert calls["count"] >= 3  # swept, waited, then confirmed empty


def test_finalize_stop_stays_honest_when_members_persist(tmp_path: Path, monkeypatch) -> None:
    connection = LspConnection(_profile(tmp_path), root=tmp_path, group_reap_grace=0.2)
    monkeypatch.setattr(connection, "_group_has_members", lambda: True)
    monkeypatch.setattr(connection, "_process_handle", lambda: None)
    monkeypatch.setattr(lsp_mod, "kill_tree", lambda *args, **kwargs: None)
    outcome = asyncio.run(connection._finalize_stop())
    assert outcome == STOP_OUTCOME_CLEANUP_PENDING
    assert connection.cleanup_pending is True
    assert connection.state == STATE_STOPPING


def test_finalize_stop_with_zero_grace_keeps_previous_behavior(tmp_path: Path, monkeypatch) -> None:
    connection = LspConnection(_profile(tmp_path), root=tmp_path, group_reap_grace=0.0)
    monkeypatch.setattr(connection, "_group_has_members", lambda: True)
    monkeypatch.setattr(connection, "_process_handle", lambda: None)
    monkeypatch.setattr(lsp_mod, "kill_tree", lambda *args, **kwargs: None)
    outcome = asyncio.run(connection._finalize_stop())
    assert outcome == STOP_OUTCOME_CLEANUP_PENDING


@pytest.mark.parametrize("grace", [0.0, 2.0])
def test_finalize_stop_clean_path_untouched_by_grace(tmp_path: Path, monkeypatch, grace: float) -> None:
    """No group members at all -> straight to stopped, regardless of grace."""
    connection = LspConnection(_profile(tmp_path), root=tmp_path, group_reap_grace=grace)
    monkeypatch.setattr(connection, "_group_has_members", lambda: False)
    outcome = asyncio.run(connection._finalize_stop())
    assert outcome == STOP_OUTCOME_STOPPED
    assert connection.state == STATE_STOPPED
