# Machina-DevTools

Test and benchmark tooling for [Machina](../Machina-Harness) (the harness
lives in a separate repository and is checked out next to this one, or
wherever `MACHINA_REPO` points).

## Layout

```
.
├── tests/            # the Machina backend + integration test suite (moved from Machina-Harness, history preserved)
├── benchmarks/       # future home of standalone benchmark suites (see benchmarks/README.md)
├── conftest.py       # root shim: resolves the Machina checkout (MACHINA_REPO / sibling)
├── run-tests.sh      # canonical runner (resolves checkout, sets PYTHONPATH, runs pytest)
├── bootstrap-links.sh# optional compatibility symlinks for unmodified path-coupled tests
├── Makefile          # make test / test-backend / test-db
└── pyproject.toml    # pytest configuration (mirrors the harness markers)
```

The suite talks to the **Machina checkout**, never to a copy of the sources
inside this repo:

* Resolution order: `MACHINA_REPO` → `../Machina-Harness` → `../machina` →
  `../machina-harness` (same rules in `conftest.py` and `run-tests.sh`).
* `run-tests.sh` puts the checkout on `PYTHONPATH` and disables bytecode
  writes. CI installs it instead: `pip install -e "$MACHINA_REPO[dev,connectors]"`.

## Quickstart

```bash
git clone https://github.com/kiazh/Machina-DevTools.git
cd Machina-DevTools

# Point at your Machina checkout (a sibling ../Machina-Harness also works):
export MACHINA_REPO="$HOME/machina"

# Optional today, removable once tests stop anchoring paths at parents[1]:
./bootstrap-links.sh "$MACHINA_REPO"

# Unit smoke
PYTHON="$MACHINA_REPO/.venv/bin/python" ./run-tests.sh tests/test_skill_runtime.py -q

# Backend selection (what CI runs)
./run-tests.sh -m "not ui_integration and not workflow_trial"

# DB-backed suites (disposable database only!)
export MACHINA_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/machina_test
./run-tests.sh -m db
```

`make test` and `make test-backend` wrap the same runner; `make bootstrap`
creates the symlinks.

## CI

`.github/workflows/tests.yml` runs the backend selection against a checked-out
Machina-Harness (`main` by default; manual runs accept a `harness_ref` input).
It uses the same conventions as the harness CI: Python 3.11, SHA-pinned
actions, and the pgvector/pg16 service image for the disposable test database.

## Environment

| Variable | Purpose |
| --- | --- |
| `MACHINA_REPO` | Path to the Machina checkout (overrides sibling detection). |
| `MACHINA_TEST_DATABASE_URL` | Disposable PostgreSQL DSN; DB tests skip without it. Never point at a real database. |
| `MACHINA_TEST_PYTHON` | Interpreter path for `ui_integration` real-gateway tests (they skip/exclude it otherwise). |
| `PYTHON` | Interpreter used by `run-tests.sh` (default `python3`). |

## Markers

`ui_integration`, `workflow_trial`, `db`, `integration`, `load`. The backend
selection is `-m "not ui_integration and not workflow_trial"`; load/concurrent
tests are deselected with `-m "not load"` when needed.

## Known gaps (tracked follow-ups)

1. **Repo-root anchoring.** A few dozen test files resolve the harness root via
   `Path(__file__).resolve().parents[1]` / `parent.parent`, and some read
   harness files through the process CWD. Today the optional symlinks created
   by `bootstrap-links.sh`, plus the root `conftest.py` shim, bridge this; the
   durable fix is a shared `tests.support.repo_paths.repo_root()` helper using
   the same resolution order as `conftest.py`, plus a mechanical rewrite of
   those anchors.
2. **README.md / pyproject.toml.** These names belong to this repo, so the
   symlink bridge cannot cover the few tests that read them from the old root
   (e.g. `tests/test_final_pass_fixes.py` reads `pyproject.toml`).
3. **Benchmarks.** No standalone benchmark suite exists yet; see
   `benchmarks/README.md` for the inventory.
