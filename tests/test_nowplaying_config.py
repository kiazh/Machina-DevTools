"""Pinned now-playing bar config key (`nowplaying_enabled`).

The TUI shows what is playing on this computer through the gateway's
nowplaying.get detection; the backend stores only the bar's on/off switch.
"""

from __future__ import annotations

import pytest

from machina.core.config import DEFAULTS, Config, validate_value


def test_default_is_enabled() -> None:
    assert DEFAULTS["nowplaying_enabled"] is True
    assert Config().nowplaying_enabled is True


def test_coerces_booleans_and_refuses_garbage() -> None:
    assert validate_value("nowplaying_enabled", "off") is False
    assert validate_value("nowplaying_enabled", "1") is True
    with pytest.raises(ValueError, match="nowplaying_enabled"):
        validate_value("nowplaying_enabled", "banana")


def test_local_player_keys_are_gone() -> None:
    for retired in ("music_enabled", "music_dir", "music_autoplay"):
        assert retired not in DEFAULTS, f"{retired} was retired with the local player"
