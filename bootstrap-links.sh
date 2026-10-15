#!/usr/bin/env bash
# Create compatibility symlinks for repo-root artifacts the (unmodified) test
# suite reads through `Path(__file__).resolve().parents[1]` or through the
# process CWD.  Optional in the long term: the execution-day migration should
# replace those anchors with `tests.support.repo_paths.repo_root()`, after
# which this script can go away.
#
# Usage: ./bootstrap-links.sh [path/to/Machina-Harness]
set -euo pipefail

DEVTOOLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINA_REPO="${1:-${MACHINA_REPO:-}}"

if [[ -z "$MACHINA_REPO" ]]; then
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

if [[ -z "$MACHINA_REPO" || ! -f "$MACHINA_REPO/machina/__init__.py" ]]; then
    echo "error: pass the Machina checkout path or set MACHINA_REPO" >&2
    exit 2
fi
MACHINA_REPO="$(cd "$MACHINA_REPO" && pwd)"

# Directories and files tests read from the (old) repo root.  README.md and
# pyproject.toml are deliberately NOT linked: this repo owns both names.
LINKABLE=(
    machina
    skills
    research
    docs
    ui
    tools
    scripts
    docker-compose.yml
    install.sh
    .env.example
)

cd "$DEVTOOLS_ROOT"
for name in "${LINKABLE[@]}"; do
    if [[ -e "$name" || -L "$name" ]]; then
        echo "skip (exists): $name"
        continue
    fi
    ln -s "$MACHINA_REPO/$name" "$name"
    echo "linked: $name -> $MACHINA_REPO/$name"
done
echo "done. Known gap: tests reading README.md/pyproject.toml from the old repo root."
