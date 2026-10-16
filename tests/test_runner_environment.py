"""The canonical runner owns interpreter selection and disposable user state."""

import os
import subprocess
from pathlib import Path


def _run_with_interpreter_probe(tmp_path):
    selected = tmp_path / "selected-python"
    selected.write_text(
        '#!/bin/sh\n'
        'printf "HELPER=%s\\n" "${VENV_PY:-}"\n'
        'printf "PATH_FIRST=%s\\n" "${PATH%%:*}"\n'
        'printf "HOME=%s\\n" "$HOME"\n'
        'printf "APP_HOME=%s\\n" "${MACHINA_HOME:-}"\n'
        'printf "ENV_FILE=%s\\n" "${MACHINA_ENV_FILE:-}"\n'
        '[ -f "${MACHINA_ENV_FILE:-/nonexistent}" ] && printf "ENV_READY=yes\\n"\n'
        'exit 0\n'
    )
    selected.chmod(0o700)
    env = os.environ.copy()
    env["PYTHON"] = str(selected)
    env.pop("VENV_PY", None)
    result = subprocess.run(
        [str(Path(__file__).resolve().parents[1] / "run-tests.sh")],
        env=env, capture_output=True, text=True, check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    return selected, values


def test_runner_selects_same_interpreter_for_installer_subprocesses(tmp_path):
    selected, values = _run_with_interpreter_probe(tmp_path)
    assert values["HELPER"] == str(selected)
    assert values["PATH_FIRST"] == str(selected.parent)


def test_runner_uses_disposable_home_and_empty_env_file(tmp_path):
    _, values = _run_with_interpreter_probe(tmp_path)
    test_home = Path(values["HOME"])
    assert test_home != Path.home()
    assert Path(values["APP_HOME"]) == test_home / ".machina"
    assert Path(values["ENV_FILE"]) == test_home / ".env"
    assert values["ENV_READY"] == "yes"
    assert not test_home.exists(), "runner cleans up the home it created"
