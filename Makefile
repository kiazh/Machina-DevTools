# Machina-DevTools convenience targets.
PYTHON ?= python3
ARGS ?=

.PHONY: bootstrap test test-db test-backend lint

# Create the compatibility symlinks for repo-root artifacts (optional once
# tests stop counting parents[N] from __file__).
bootstrap:
	./bootstrap-links.sh

# Unfiltered local suite (same shape as the old in-repo habit).
test:
	PYTHON=$(PYTHON) ./run-tests.sh $(ARGS)

# Backend CI selection: excludes UI integration and workflow-trial groups.
test-backend:
	PYTHON=$(PYTHON) ./run-tests.sh -m "not ui_integration and not workflow_trial" $(ARGS)

# DB-backed suites only (needs MACHINA_TEST_DATABASE_URL pointed at a
# disposable database).
test-db:
	PYTHON=$(PYTHON) ./run-tests.sh -m db $(ARGS)
