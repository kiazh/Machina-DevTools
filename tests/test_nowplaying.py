"""Now-playing detection: layered sources, honest states, command routing.

The macOS-specific layers are exercised through monkeypatched runners (no
real osascript/MediaRemote calls), so parsing, fallback order, permission
handling, caching, and the gateway handlers are pinned deterministically.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from machina.core import nowplaying

SEP = nowplaying._LINE


@pytest.fixture(autouse=True)
def _reset_cache():
    nowplaying._cache = None
    nowplaying._cache_at = 0.0
    yield
    nowplaying._cache = None
    nowplaying._cache_at = 0.0


def _no_media_remote(monkeypatch):
    monkeypatch.setattr(nowplaying, "_media_remote_probe", lambda: None)


def _no_cli(monkeypatch):
    monkeypatch.setattr(nowplaying, "_cli_binary", lambda: None)


def _only_app_running(monkeypatch, names: set[str]):
    monkeypatch.setattr(nowplaying, "_app_running", lambda name: name in names)


# ─── layer order and parsing ────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_mediaremote_reading_wins_and_maps_state(monkeypatch):
    monkeypatch.setattr(
        nowplaying,
        "_media_remote_probe",
        lambda: {
            "info": {
                "title": "Song",
                "artist": "Artist",
                "album": "Album",
                "duration": 210.0,
                "elapsed": 12.0,
                "rate": 1.0,
            },
            "playing": True,
            "pid": 4242,
        },
    )
    monkeypatch.setattr(nowplaying, "_process_name", lambda pid: "Google Chrome")
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is True
    assert result["state"] == "playing"
    assert result["title"] == "Song"
    assert result["source"] == "Google Chrome"
    assert result["duration"] == 210.0


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_mediaremote_empty_falls_through_to_spotify(monkeypatch):
    # Empty MediaRemote reading (macOS gates it): continue to the app layer.
    monkeypatch.setattr(
        nowplaying, "_media_remote_probe", lambda: {"info": None, "playing": False, "pid": None}
    )
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Spotify"})

    def fake_osascript(script: str, timeout: float = 3.0):
        assert "Spotify" in script
        return (
            0,
            SEP.join(["playing", "Night Drive", "Aurora", "All My Demons", "220000", "12.5"]),
            "",
        )

    monkeypatch.setattr(nowplaying, "_osascript", fake_osascript)
    result = nowplaying.detect_now_playing(cache=False)
    assert result["state"] == "playing"
    assert result["title"] == "Night Drive"
    assert result["artist"] == "Aurora"
    assert result["duration"] == 220.0, "Spotify milliseconds normalize to seconds"
    assert result["elapsed"] == 12.5
    assert result["source"] == "Spotify"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_music_app_duration_is_already_seconds(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Music"})
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (
            (0, SEP.join(["paused", "Track", "Band", "Record", "180", "3"]), "")
            if "Music" in script
            else (0, "NOT_RUNNING", "")
        ),
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["state"] == "paused"
    assert result["duration"] == 180.0
    assert result["source"] == "Music"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_front_youtube_tab_is_displayed_with_unknown_state(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Google Chrome"})
    called: list[str] = []

    def fake_osascript(script: str, timeout: float = 3.0):
        called.append(script)
        if "Google Chrome" not in script:
            return 0, "NOT_RUNNING", ""
        return (
            0,
            f"Lo-fi beats to relax to - YouTube{SEP}https://www.youtube.com/watch?v=abc",
            "",
        )

    monkeypatch.setattr(nowplaying, "_osascript", fake_osascript)
    result = nowplaying.detect_now_playing(cache=False)
    assert result["title"] == "Lo-fi beats to relax to"
    assert result["state"] == "unknown", "browser playback state is never guessed"
    assert result["source"] == "Google Chrome (front tab)"
    assert result["duration"] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_youtube_music_title_splits_artist(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Google Chrome"})
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (
            0,
            f"Song • Artist - YouTube Music{SEP}https://music.youtube.com/watch?v=x",
            "",
        )
        if "Google Chrome" in script
        else (0, "NOT_RUNNING", ""),
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["title"] == "Song"
    assert result["artist"] == "Artist"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_non_youtube_tab_is_not_reported(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Google Chrome"})
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (
            (0, f"Some article{SEP}https://example.com/post", "")
            if "Google Chrome" in script
            else (0, "NOT_RUNNING", "")
        ),
    )
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is True, "layers work; nothing qualifying is playing"
    assert result["title"] is None
    assert result["error"] is None


# ─── honest failure and permission states ───────────────────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_nothing_playing_is_available_without_a_title(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, set())
    result = nowplaying.detect_now_playing(cache=False)
    assert result == {
        "available": True,
        "state": None,
        "title": None,
        "artist": None,
        "album": None,
        "source": None,
        "elapsed": None,
        "duration": None,
        "error": None,
    }


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_permission_timeout_names_the_app_and_never_fabricates(monkeypatch):
    _no_media_remote(monkeypatch)
    _no_cli(monkeypatch)
    _only_app_running(monkeypatch, {"Spotify"})
    monkeypatch.setattr(nowplaying, "_osascript", lambda script, timeout=3.0: (124, "", "timeout"))
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is False
    assert result["title"] is None
    assert "blocked automation for Spotify" in (result["error"] or "")
    assert "Automation" in (result["error"] or "")


def test_non_macos_is_unavailable_not_empty(monkeypatch):
    monkeypatch.setattr(nowplaying, "sys", type("S", (), {"platform": "linux"}))
    result = nowplaying.detect_now_playing(cache=False)
    assert result["available"] is False
    assert "requires macOS" in (result["error"] or "")


def test_permission_results_cache_longer_than_readings(monkeypatch):
    calls = {"n": 0}

    def fake_detect():
        calls["n"] += 1
        return {
            "available": False,
            "state": None,
            "title": None,
            "artist": None,
            "album": None,
            "source": None,
            "elapsed": None,
            "duration": None,
            "error": "macOS blocked automation for Spotify",
        }

    monkeypatch.setattr(nowplaying, "_detect", fake_detect)
    nowplaying.detect_now_playing()
    nowplaying.detect_now_playing()
    assert calls["n"] == 1, "the denial is cached, not re-probed every poll"
    assert nowplaying._cache_ttl(nowplaying._cache) > nowplaying._CACHE_TTL_S


# ─── transport commands ─────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_toggle_routes_to_spotify_applescript(monkeypatch):
    monkeypatch.setattr(
        nowplaying,
        "detect_now_playing",
        lambda **kwargs: {"source": "Spotify", "title": "x"},
    )
    scripts: list[str] = []
    monkeypatch.setattr(
        nowplaying,
        "_osascript",
        lambda script, timeout=3.0: (scripts.append(script), (0, "", ""))[1],
    )
    result = nowplaying.send_now_playing_command("toggle")
    assert result["ok"] is True
    assert scripts and "playpause" in scripts[0] and "Spotify" in scripts[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_toggle_on_a_browser_source_reports_honestly(monkeypatch):
    monkeypatch.setattr(
        nowplaying,
        "detect_now_playing",
        lambda **kwargs: {"source": "Google Chrome (front tab)", "title": "x"},
    )
    _no_cli(monkeypatch)
    monkeypatch.setattr(nowplaying, "_load_media_remote", lambda: None)
    result = nowplaying.send_now_playing_command("toggle")
    assert result["ok"] is False
    assert "does not accept transport commands" in (result["error"] or "")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only now-playing behavior")
def test_unknown_action_is_refused(monkeypatch):
    result = nowplaying.send_now_playing_command("rewind")
    assert result["ok"] is False
    assert "unknown action" in (result["error"] or "")


# ─── gateway handlers ───────────────────────────────────────────────────────


def test_gateway_get_returns_the_reading(monkeypatch):
    from machina.gateway.features.nowplaying import nowplaying_get

    monkeypatch.setattr(
        nowplaying,
        "detect_now_playing",
        lambda **kwargs: {"available": True, "title": "Song", "state": "playing"},
    )
    result = asyncio.run(nowplaying_get(None, {}))
    assert result["title"] == "Song"


def test_gateway_command_validates_actions(monkeypatch):
    from machina.gateway.features.nowplaying import nowplaying_command

    result = asyncio.run(nowplaying_command(None, {"action": "rewind"}))
    assert result == {"ok": False, "source": None, "error": "unknown action 'rewind'"}
    sent: list[str] = []
    monkeypatch.setattr(
        nowplaying,
        "send_now_playing_command",
        lambda action: (sent.append(action), {"ok": True})[1],
    )
    result = asyncio.run(nowplaying_command(None, {"action": "Toggle"}))
    assert result == {"ok": True}
    assert sent == ["toggle"]


def test_stale_mediaremote_callbacks_are_ignored():
    """A late reply from a previous poll must not contaminate the next one."""
    with nowplaying._mr_state_lock:
        nowplaying._mr_last.clear()
        nowplaying._mr_last["info"] = (0.0, {"title": "old"})
        nowplaying._mr_last["playing"] = (0.5, True)
    info, playing, pid = nowplaying._mr_read(10.0)
    assert info is None and playing is None and pid is None

    with nowplaying._mr_state_lock:
        nowplaying._mr_last["info"] = (12.0, {"title": "new"})
        nowplaying._mr_last["playing"] = (11.0, True)
    info, playing, pid = nowplaying._mr_read(10.0)
    assert info == {"title": "new"}
    assert playing is True
    assert pid is None

