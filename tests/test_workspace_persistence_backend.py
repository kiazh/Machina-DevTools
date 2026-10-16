"""T6 workspace UI persistence backend tests (plan §21).

The UI persists versioned, non-sensitive layout metadata as one JSON string
under the additive ``workspace_ui_state`` config key. These tests pin the
declared default, the shared generic coercion path (``coerce_config_value``
-> ``validate_value``), the 64 KiB refusal, and that arbitrary (even
invalid-JSON) strings round-trip harmlessly through ``config.set``/``get`` and
the YAML config file.
"""

from __future__ import annotations

import json


def test_workspace_ui_state_default_is_a_blank_string():
    from machina.core.config import (
        DEFAULTS,
        SECRET_KEYS,
        WORKSPACE_UI_STATE_MAX_BYTES,
        Config,
        config,
    )

    assert DEFAULTS["workspace_ui_state"] == ""
    assert Config().workspace_ui_state == ""
    assert config.get("workspace_ui_state") == ""
    assert "workspace_ui_state" not in SECRET_KEYS
    assert WORKSPACE_UI_STATE_MAX_BYTES == 64 * 1024


def test_workspace_ui_state_generic_coercion_accepts_a_plain_string():
    """The shared coercion path needs no key-specific branch for strings."""
    from machina.gateway.features import coerce_config_value

    payload = json.dumps(
        {
            "version": 1,
            "byRoot": {
                "/repo": {
                    "version": 1,
                    "requestedMode": "split",
                    "splitRatio": 0.5,
                    "explorerRequested": False,
                    "openUris": [],
                }
            },
        }
    )
    assert coerce_config_value("workspace_ui_state", payload) == payload


def test_workspace_ui_state_config_set_get_roundtrip():
    from machina.core.config import config

    payload = json.dumps(
        {
            "version": 1,
            "byRoot": {
                "/repo": {"version": 1, "openUris": ["file:///repo/a.ts"]},
            },
        }
    )
    config.set("workspace_ui_state", payload)
    assert config.workspace_ui_state == payload
    assert config.get("workspace_ui_state") == payload


def test_workspace_ui_state_invalid_json_is_stored_harmlessly():
    from machina.core.config import config

    # The backend never parses the value; malformed JSON is a string like any
    # other and must not break set/get (the UI treats it as corrupt state and
    # falls back to safe defaults without blocking chat).
    config.set("workspace_ui_state", "not json {")
    assert config.get("workspace_ui_state") == "not json {"
    config.set("workspace_ui_state", "")
    assert config.get("workspace_ui_state") == ""


def test_workspace_ui_state_size_bound_is_enforced_with_a_refusal():
    import pytest

    from machina.core.config import (
        WORKSPACE_UI_STATE_MAX_BYTES,
        config,
        validate_value,
    )
    from machina.gateway.errors import INVALID_PARAMS, RpcError
    from machina.gateway.features import coerce_config_value

    at_limit = "x" * WORKSPACE_UI_STATE_MAX_BYTES
    assert validate_value("workspace_ui_state", at_limit) == at_limit

    over = "x" * (WORKSPACE_UI_STATE_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="workspace_ui_state"):
        validate_value("workspace_ui_state", over)
    with pytest.raises(ValueError, match="workspace_ui_state"):
        config.set("workspace_ui_state", over)
    with pytest.raises(RpcError) as error:
        coerce_config_value("workspace_ui_state", over)
    assert error.value.code == INVALID_PARAMS
    assert "workspace_ui_state" in error.value.message

    # A refused set mutates nothing (AH-AUDIT-042 shared refusal contract).
    config.set("workspace_ui_state", "kept")
    with pytest.raises(ValueError, match="workspace_ui_state"):
        config.set("workspace_ui_state", over)
    assert config.get("workspace_ui_state") == "kept"

    # The bound counts UTF-8 bytes, not characters: 2-byte characters hit the
    # limit at half the character count and one more is refused.
    multibyte = "é" * (WORKSPACE_UI_STATE_MAX_BYTES // 2)
    assert len(multibyte.encode("utf-8")) == WORKSPACE_UI_STATE_MAX_BYTES
    assert validate_value("workspace_ui_state", multibyte) == multibyte
    with pytest.raises(ValueError, match="workspace_ui_state"):
        validate_value("workspace_ui_state", multibyte + "é")


def test_workspace_ui_state_rejects_non_strings():
    import pytest

    from machina.core.config import validate_value

    for value in ({"version": 1}, 5, None, ["x"], True):
        with pytest.raises(ValueError, match="workspace_ui_state"):
            validate_value("workspace_ui_state", value)


def test_workspace_ui_state_survives_a_config_file_roundtrip(tmp_path):
    from machina.core.config import Config

    payload = json.dumps(
        {
            "version": 1,
            "byRoot": {
                "/repo": {
                    "version": 1,
                    "requestedMode": "editor",
                    "splitRatio": 0.6,
                    "explorerRequested": True,
                    "openUris": ["file:///repo/a.ts"],
                }
            },
        }
    )
    path = tmp_path / "config.yaml"
    fresh = Config()
    fresh.workspace_ui_state = payload
    fresh.save(path)
    loaded = Config.load(path)
    assert loaded.workspace_ui_state == payload
    assert loaded.get("workspace_ui_state") == payload
    # A hand-edited oversized value is ignored on load, never activated.
    over = "x" * (64 * 1024 + 1)
    path.write_text(f"workspace_ui_state: {over}\n", encoding="utf-8")
    guarded = Config.load(path)
    assert guarded.workspace_ui_state == ""
