"""Combined #39: an explicit CLI --mode overrides the environment shadow."""

from __future__ import annotations

import os

from machina.cli import _apply_cli_mode
from machina.core.config import config


def test_cli_mode_overrides_environment_and_config(monkeypatch):
    monkeypatch.setenv("MACHINA_EXECUTION_MODE", "full")
    assert _apply_cli_mode("ask") == "ask"
    assert os.environ["MACHINA_EXECUTION_MODE"] == "ask"
    assert config.get("execution_mode") == "ask"
