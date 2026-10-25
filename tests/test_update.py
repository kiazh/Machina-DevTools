"""Interactive checkout update check."""

from __future__ import annotations

from types import SimpleNamespace

from machina import update


def _run_result(returncode: int = 0, stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_version_tuple_compares_numeric_parts():
    assert update._version_tuple("1.2.8") < update._version_tuple("1.2.9")
    assert update._version_tuple("1.2.9") == update._version_tuple("1.2.9-dev")
    assert update._version_tuple("2.0") > update._version_tuple("1.99.99")


def test_auto_update_disabled_never_touches_git(monkeypatch):
    monkeypatch.setenv("MACHINA_AUTO_UPDATE", "0")
    monkeypatch.setattr(update, "_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    update.auto_update()


def test_auto_update_fast_forwards_clean_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINA_AUTO_UPDATE", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(update.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(update, "_repo_root", lambda: tmp_path)
    calls: list[list[str]] = []
    execs: list[list[str]] = []

    def fake_run(args, *, cwd, timeout):
        calls.append(args)
        if args[:2] == ["git", "fetch"]:
            return _run_result()
        if args[:2] == ["git", "show"]:
            return _run_result(stdout='[project]\nversion = "9.9.9"\n')
        if args[:2] == ["git", "status"]:
            return _run_result()
        if args[:2] == ["git", "pull"]:
            return _run_result()
        if args[0] == "git" and "diff" in args:
            return _run_result(stdout="pyproject.toml\n")
        if args[:3] == [update.sys.executable, "-m", "pip"]:
            return _run_result()
        raise AssertionError(args)

    monkeypatch.setattr(update, "_run", fake_run)
    monkeypatch.setattr(update.os, "execv", lambda executable, argv: execs.append(argv))

    update.auto_update()

    assert any(call[:2] == ["git", "pull"] for call in calls)
    assert any(call[:2] == ["git", "show"] for call in calls)
    assert execs


def test_auto_update_skips_dirty_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINA_AUTO_UPDATE", "1")
    monkeypatch.setattr(update.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(update, "_repo_root", lambda: tmp_path)
    calls: list[list[str]] = []

    def fake_run(args, *, cwd, timeout):
        calls.append(args)
        if args[:2] == ["git", "fetch"]:
            return _run_result()
        if args[:2] == ["git", "show"]:
            return _run_result(stdout='[project]\nversion = "9.9.9"\n')
        if args[:2] == ["git", "status"]:
            return _run_result(stdout=" M local.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(update, "_run", fake_run)
    update.auto_update()
    assert not any(call[:2] == ["git", "pull"] for call in calls)
