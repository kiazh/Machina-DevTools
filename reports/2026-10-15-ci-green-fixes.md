# Lane D — CI-green fixes for the moved suite (2026-10-15)

CI run: `38786587642` (DevTools workflow "Tests", 9 failed / 5105 passed / 3 skipped
/ 9 deselected). Harness under test: `kiazh/Machina-Harness@c4b3182` (identical to
the local `/Users/shadow/machina` checkout at the time of this work).

All fixes are in **Machina-DevTools** (this repository); the harness checkout was
read-only throughout. Nothing here was pushed or committed — changes are left in
the working tree for review.

Note on this file's location: `docs/` in a bootstrapped clone is a **symlink to the
harness checkout** (`bootstrap-links.sh` creates it), so writing
`docs/2026-10-15-ci-green-fixes.md` would have modified Machina-Harness. Since the
lane is forbidden from touching the harness, the evidence lives in the real
directory `reports/` inside this repository instead. No test reads the repo-root
`docs/` path (verified), so the symlink was left untouched.

## How to run (mirrors CI)

```sh
cd <DevTools clone>
MACHINA_REPO=/Users/shadow/machina PYTHON=/Users/shadow/machina/.venv/bin/python \
  ./run-tests.sh <paths> -q -p no:cacheprovider
```

CI-only conditions were reproduced locally as follows:

```sh
# help-rendering drift (GITHUB_ACTIONS flips typer into styled output)
GITHUB_ACTIONS=true ... ./run-tests.sh tests/test_interop.py::test_new_flags_are_documented_in_help ...

# schema scan against a truly fresh database (local machina_test had leftovers)
MACHINA_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/machina_devtools_ci_repro ...
```

## Per-failure root causes and fixes

### 1. `test_final_pass_fixes.py::test_docker_context_and_docs_handle_telemetry_state`

- **Root cause:** the test anchored at `Path(__file__).resolve().parents[1]` (the
  DevTools root here) and read `.dockerignore` + `pyproject.toml`. In the harness
  repo both files sit at the repo root; in DevTools the root is a different repo
  that never had a `.dockerignore` (`pyproject.toml` exists but is DevTools' own,
  so the telemetry-limitation assertion would have failed next).
- **Fix:** read both through `tests.support.repo_paths.repo_path(...)`.
- **Files:** `tests/test_final_pass_fixes.py`.

### 2–4. `test_infra_fixes.py` (sandbox Dockerfile / CI / dependabot)

- **Root cause:** module-level `ROOT = Path(__file__).resolve().parents[1]` pointed
  at the DevTools root; `Dockerfile.sandbox`, `.github/workflows/ci.yml` and
  `.github/dependabot.yml` are harness files that were not symlinked by
  `bootstrap-links.sh`.
- **Fix:** `ROOT = repo_root()` (single anchor). Every artifact this module pins
  (sandbox Dockerfile, installer, pgvector helper, compose file, CI config,
  dependabot) lives in the harness checkout; resolution is now explicit instead of
  relying on the bootstrap symlinks.
- **Files:** `tests/test_infra_fixes.py`.

### 5. `test_integration_permissions.py::test_no_docker_needed_for_host`

- **Root cause:** `open("FEATURE_LEDGER.md")` resolved against the process CWD
  (DevTools root). The ledger lives in the harness root and is not one of the
  bootstrapped symlinks.
- **Fix:** `repo_path("FEATURE_LEDGER.md")`.
- **Files:** `tests/test_integration_permissions.py`.

### 6. `test_integration_permissions.py::test_launcher_propagates_mode_to_gateway`

- **Root cause (from the CI captured stdout):** `launch_ui()` exits with rc=1 when
  the UI tree is not usable; CI printed
  `✗ UI dependencies are not installed. Run: npm install ... <workspace>/Machina-Harness/ui`.
  CI checks out harness sources but never runs `npm install`, so
  `ui/node_modules/@earendil-works/pi-tui` is absent. The test monkeypatched node,
  the node version and `subprocess.call`, but not the UI install state — it passed
  locally only because the developer checkout has `node_modules`.
- **Fix:** pin a hermetic stub UI tree (empty `src/main.ts` + the expected
  `node_modules/@earendil-works/pi-tui` directory) through the supported
  `MACHINA_UI_DIR` override, so the test exercises LP-13 mode propagation instead
  of the machine's npm state. An extra assertion pins that the stub entry path
  reached `subprocess.call`, i.e. no early return happened.
- **Files:** `tests/test_integration_permissions.py`.

### 7–8. Help-output assertions (`test_interop.py::test_new_flags_are_documented_in_help`, `test_serve.py::TestServeCommand::test_serve_help`)

- **Root cause (reproduced locally):** typer's `rich_utils` sets
  `FORCE_TERMINAL = True if getenv("GITHUB_ACTIONS") or getenv("FORCE_COLOR") or
  getenv("PY_COLORS") else None`. GitHub Actions sets `GITHUB_ACTIONS`, so CI renders
  help with Rich styles; the option highlighter styles the first dash separately and
  the emitted reset splits the token in the raw string:

  ```text
  RAW     : '... \x1b[1;36m-\x1b[0m\x1b[1;36m-host\x1b[0m    \x1b[1;32m-h\x1b[0m ...'
  STRIPPED: '│ --host    -h      <str>  Bind host [default: 127.0.0.1] ...'
  ```

  `"--host" in result.output` is therefore False under `GITHUB_ACTIONS=true` with
  the exact CI versions (typer 0.27.3 / rich 15.0.0 / click 8.5.0 — the dev venv has
  the same versions, so this is rendering mode, not version drift).
- **Fix:** new `tests/support/ansi.py::strip_ansi()`; both tests assert the option
  tokens on the visible (ANSI-stripped) text. Assertions and tokens are unchanged —
  only the styling layer is normalized away.
- **Files:** `tests/support/ansi.py` (new), `tests/test_interop.py`, `tests/test_serve.py`.

### 9. `test_schema_conformance.py::test_static_python_sql_compiles_against_schema`

- **Root cause:** `static_queries()` compiles every literal DML statement found in
  the harness `machina/` tree against a fresh isolated schema. `machina/research/
  workflow_trial.py` + `workflow_benchmark.py` (the optional native-vs-LangGraph
  approval trial, `workflow_trial` marker, deselected in CI) contain DML against
  `workflow_trial_runs` / `workflow_trial_effects`. Those tables are created at
  runtime by the trial's own `setup()` DDL, and the LangGraph path additionally owns
  its checkpointer tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
  `checkpoint_migrations`) via `langgraph-checkpoint-postgres`'
  `AsyncPostgresSaver.setup()`. None of them are in `schema.sql`, so a fresh DB
  raises the 9 `UndefinedTableError`s seen in CI.
- **Why it passed locally:** `machina_test`'s `public` schema still contains those
  tables from earlier local workflow-trial runs; `isolated_schema` sets
  `search_path = <isolated>,public`, so the unqualified statements resolved against
  the leftovers. Reproduced locally with an empty scratch database — identical 9
  errors.
- **Fix:** explicit `_LANGGRAPH_OWNED_SOURCES` exclusion by repository-relative path
  in `static_queries()` (with the ownership comment above it), plus a control test
  `test_langgraph_owned_exclusion_control` that guards:
  1. the excluded paths still exist (no stale skip),
  2. the files do contribute DML when the skip is disabled (the exclusion is what
     makes the scan pass),
  3. a synthetic product-style module scanned by the same discoverer and compiled
     through the same `prepare()` gate still raises `UndefinedTableError` for a
     genuinely undefined table.
- **Files:** `tests/test_schema_conformance.py`.

## New shared helpers

- `tests/support/repo_paths.py` — `repo_root()` / `repo_path(rel)`; mirrors the root
  `conftest.py` resolution (`MACHINA_REPO` → sibling `Machina-Harness` / `machina` /
  `machina-harness`; keep in sync with conftest).
- `tests/support/ansi.py` — `strip_ansi(text)`; removes CSI/OSC escape sequences
  from captured CLI output.

## Verification

Commands and results (all from the clone root via `run-tests.sh`):

| Run | Condition | Result |
| --- | --- | --- |
| the 9 CI-failing tests + new control | `GITHUB_ACTIONS=true`, fresh scratch DB | **10 passed** |
| `test_infra_fixes, test_integration_permissions, test_final_pass_fixes, test_interop, test_serve, test_schema_conformance` | `GITHUB_ACTIONS=true`, fresh scratch DB, `-m "not ui_integration and not workflow_trial"` | **139 passed, 1 skipped** |
| same six files | default local env (no `GITHUB_ACTIONS`), `machina_test` DB with leftovers | **139 passed, 1 skipped** |
| whole suite collection | `--collect-only` | **5127 tests collected** (was 5126 + 1 new control) |

The single skip is the pre-existing Windows-only
`test_scope_windows_case_unc_and_symlink_lexical` (macOS).

## Other anchor sites spotted (not part of this lane's 9; left for the migration)

- 34 test modules still compute a repo root via `Path(__file__).resolve().parents[1]`
  or equivalents; those that read *harness* files only pass because
  `bootstrap-links.sh` symlinks `machina`, `skills`, `research`, `docs`, `ui`,
  `tools`, `scripts`, `docker-compose.yml`, `install.sh`, `.env.example` into the
  DevTools root. Examples: `test_gateway_images.py`, `test_installer_env.py`,
  `test_installer_quoting.py`, `test_code_quality.py`, `test_architecture_layers.py`,
  `tests/workflow_trial/*`.
- CWD-relative bare opens in `tests/test_integration_permissions.py`:
  `open("install.sh")` (×2), `open("tools/install-helpers.sh")`, `open("ui/src/app.ts")`
  — currently bridged by the same symlinks.
- `tests/test_schema_conformance.py` now resolves the scan root through
  `repo_path("machina")` (previously `parents[1] / "machina"`, which only worked via
  the `machina` symlink).
- README.md / pyproject.toml are deliberately *not* symlinked (DevTools owns both
  names); any test pinning the harness copies must use `repo_path`.
