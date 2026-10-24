"""Docker database lifecycle leases."""

from __future__ import annotations

from types import SimpleNamespace

from machina.db import lifecycle


def test_managed_database_stops_after_last_client(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@127.0.0.1:5432/machina")
    monkeypatch.setenv("MACHINA_DB_MANAGE", "1")
    monkeypatch.setenv("MACHINA_DB_STATE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)

    first = lifecycle.acquire()
    second = lifecycle.acquire()
    assert first is not None and second is not None
    assert any(call[:2] == ["docker", "start"] for call in calls)

    lifecycle.release(first)
    assert not any(call[:2] == ["docker", "stop"] for call in calls)

    lifecycle.release(second)
    assert any(call[:2] == ["docker", "stop"] for call in calls)


def test_lifecycle_ignores_non_loopback_database(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@db.example.com:5432/machina")
    monkeypatch.setenv("MACHINA_DB_MANAGE", "1")
    monkeypatch.setenv("MACHINA_DB_STATE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/docker")
    assert lifecycle.acquire() is None


def test_acquire_creates_container_and_persists_credentials(monkeypatch, tmp_path):
    """A fresh ``machina`` starts a database even if no container exists yet."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    env_file = tmp_path / ".env"
    monkeypatch.setenv("MACHINA_ENV_FILE", str(env_file))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@127.0.0.1:5432/machina")
    monkeypatch.setenv("MACHINA_DB_MANAGE", "1")
    monkeypatch.setenv("MACHINA_DB_STATE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    monkeypatch.setattr(lifecycle, "_initialize_schema", lambda url: True)

    lease = lifecycle.acquire()
    assert lease is not None
    run_calls = [call for call in calls if call[:2] == ["docker", "run"]]
    assert run_calls
    assert "--restart" in run_calls[0] and "unless-stopped" in run_calls[0]
    assert "DATABASE_URL=postgresql://postgres:" in env_file.read_text(encoding="utf-8")
    assert "127.0.0.1:5432/machina" in env_file.read_text(encoding="utf-8")

    lifecycle.release(lease)
