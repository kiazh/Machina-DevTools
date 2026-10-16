#!/usr/bin/env bash
# Run the Machina test suite from this checkout against a Machina checkout.
#
# Usage:
#   ./run-tests.sh [pytest args...]
#   PYTHON=/path/to/venv/bin/python ./run-tests.sh tests/test_skill_runtime.py -q
#   MACHINA_REPO=/path/to/Machina-Harness ./run-tests.sh -m db
#
# The Machina checkout is resolved in this order:
#   1. $MACHINA_REPO
#   2. a sibling checkout: ../Machina-Harness, ../machina, ../machina-harness
#
# It is made importable via PYTHONPATH (install-free local runs).  CI should
# prefer `pip install -e "$MACHINA_REPO"[dev,connectors,workflow-trial]` into
# a fresh virtualenv and can then drop PYTHONPATH.
set -euo pipefail

DEVTOOLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MACHINA_REPO:-}" ]]; then
    for candidate in \
        "$DEVTOOLS_ROOT/../Machina-Harness" \
        "$DEVTOOLS_ROOT/../machina" \
        "$DEVTOOLS_ROOT/../machina-harness"; do
        if [[ -f "$candidate/machina/__init__.py" ]]; then
            MACHINA_REPO="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi

if [[ -z "${MACHINA_REPO:-}" ]]; then
    echo "error: Machina checkout not found." >&2
    echo "Set MACHINA_REPO=/path/to/Machina-Harness (or place a sibling checkout)." >&2
    exit 2
fi
export MACHINA_REPO

# Make the checkout importable for this process and for every spawned
# subprocess (several tests run `python -m machina...` children).
export PYTHONPATH="$MACHINA_REPO${PYTHONPATH:+:$PYTHONPATH}"

# Never scatter bytecode into either checkout.
export PYTHONDONTWRITEBYTECODE=1

PYTHON="${PYTHON:-python3}"
TEST_PYTHON="$(command -v "$PYTHON")" || { echo "error: Python interpreter not found" >&2; exit 2; }
# Installer helpers spawn Python too; keep them on the selected environment.
export PATH="$(dirname "$TEST_PYTHON"):$PATH"
export VENV_PY="${VENV_PY:-$TEST_PYTHON}"

# A suite must never read/write the operator's config, auth, or MCP inventory.
# Individual fixtures can still supply their own home or dotenv explicitly.
TEST_RUN_HOME="$(mktemp -d "${TMPDIR:-/tmp}/machina-devtools-home.XXXXXX")"
trap 'rm -rf -- "$TEST_RUN_HOME"' EXIT
mkdir -p "$TEST_RUN_HOME/.machina"
: > "$TEST_RUN_HOME/.env"
cd "$DEVTOOLS_ROOT"
env HOME="$TEST_RUN_HOME" MACHINA_HOME="$TEST_RUN_HOME/.machina" \
    MACHINA_ENV_FILE="$TEST_RUN_HOME/.env" "$TEST_PYTHON" -m pytest "$@"
