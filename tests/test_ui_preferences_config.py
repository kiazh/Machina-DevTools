"""Global UI preferences config key (closing pass pre-work).

`ui_preferences` stores one versioned JSON string of editor preferences
(syntax highlighting, soft wrap, tab width) — never source text or secrets.
The 16 KiB bound is enforced in ``validate_value`` so ``config.set`` and every
transport coercion share one refusal (AH-AUDIT-042 contract).
"""

from __future__ import annotations

import json

import pytest

from machina.core.config import (
    DEFAULTS,
    UI_PREFERENCES_MAX_BYTES,
    Config,
    validate_value,
)


def test_default_is_a_bounded_string() -> None:
    assert DEFAULTS["ui_preferences"] == ""
    assert UI_PREFERENCES_MAX_BYTES == 16 * 1024
    assert Config().ui_preferences == ""


def test_roundtrip_json_passes_validation() -> None:
    value = json.dumps({"version": 1, "syntax": True, "softWrap": False, "tabWidth": 4})
    assert validate_value("ui_preferences", value) == value


def test_non_string_refused() -> None:
    with pytest.raises(ValueError, match="ui_preferences"):
        validate_value("ui_preferences", {"version": 1})


def test_oversized_refused_with_the_key_named() -> None:
    oversized = "x" * (UI_PREFERENCES_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="ui_preferences"):
        validate_value("ui_preferences", oversized)
    # Exactly at the bound passes.
    at_bound = "x" * UI_PREFERENCES_MAX_BYTES
    assert validate_value("ui_preferences", at_bound) == at_bound
