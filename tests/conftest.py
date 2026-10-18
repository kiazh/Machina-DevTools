"""Shared test fixtures and global state isolation."""

import asyncio
import logging
import os

import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(items):
    """Expose actual DB fixtures through the documented release selection."""
    for item in items:
        if {"db_pool", "live_db"}.intersection(item.fixturenames):
            item.add_marker(pytest.mark.db)


@pytest.fixture(scope="session", autouse=True)
def gateway_scheduler_disabled():
    """Gateways never auto-start a background scheduler in the suite.

    The flag used to be set as a side effect of constructing ``Harness`` in
    ``tests/test_gateway.py`` and leaked into every later test (and any
    subprocess inheriting the environment). The suite owns it explicitly and
    restores the process environment afterwards.
    """
    name = "MACHINA_GATEWAY_NO_SCHEDULER"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    yield
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


@pytest.fixture(autouse=True)
def isolate_config_secrets(monkeypatch):
    """Isolate developer credentials across all supported secret sources.

    Individual tests can explicitly configure fake environment, file, or
    backend values using monkeypatch after this fixture runs.
    """
    from machina.core.config import LEGACY_ENV_VARS, SECRET_KEYS, config
    from machina.security import secrets

    known_values = set(secrets._redaction_values)
    secrets._redaction_values.clear()

    # New provider keys are isolated automatically as the config evolves.
    secret_keys = tuple(sorted(SECRET_KEYS))
    saved = {k: getattr(config, k) for k in secret_keys}
    monkeypatch.setenv("MACHINA_SECRET_BACKEND", "")
    for k in secret_keys:
        setattr(config, k, "")
        # Config.get() falls back to dotenv-loaded environment values. Blank
        # those too so a fake-provider test cannot spend a developer's keys.
        monkeypatch.setenv(f"MACHINA_{k.upper()}", "")
        monkeypatch.setenv(f"MACHINA_{k.upper()}_FILE", "")
        if k in LEGACY_ENV_VARS:
            monkeypatch.setenv(LEGACY_ENV_VARS[k], "")
            monkeypatch.setenv(f"{LEGACY_ENV_VARS[k]}_FILE", "")
    # Provenance signing is optional and not a provider credential; a
    # developer's .env must not turn it on for tests that stub the DB.
    monkeypatch.delenv("MACHINA_PROVENANCE_KEY", raising=False)
    monkeypatch.delenv("MACHINA_PROVENANCE_KEY_FILE", raising=False)
    yield
    secrets._redaction_values.clear()
    secrets._redaction_values.update(known_values)
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.fixture(scope="session", autouse=True)
def init_test_database():
    """Create the schema in the isolated test database once per session."""
    dsn = os.environ.get("MACHINA_TEST_DATABASE_URL")
    if not dsn:
        return
    from machina.db.connection import Database

    async def _init() -> None:
        test_db = Database(dsn=dsn)
        try:
            await test_db.connect()
            await test_db.initialize_schema()
        finally:
            await test_db.close()

    try:
        asyncio.run(_init())
    except Exception as error:
        # A DSN was explicitly provided, so a broken schema init is a test
        # environment failure, not a reason to silently skip every DB test
        # (which reads as a false green). Never echo the DSN: it may carry a
        # password.
        logger.exception("test database schema initialization failed")
        pytest.fail(
            "test database schema initialization failed; "
            f"fix MACHINA_TEST_DATABASE_URL before running DB tests ({type(error).__name__})",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    """Reset mutable global singleton state before each test.

    Modules import the singletons by name (``from machina.core.context import
    context_manager``), so rebinding the module global would not affect them.
    Instead, clear their state in place.
    """
    from machina.db.connection import db

    db._pool = None
    # Never let tests touch the developer database: use the isolated test DB,
    # or no DSN at all (DB-backed tests then skip).
    db.dsn = os.environ.get("MACHINA_TEST_DATABASE_URL", "")

    from machina.core.session import session_manager

    session_manager._cache.clear()

    from machina.core import provider as provider_module

    provider_module._cache_clear()

    # The ChatGPT catalog module keeps in-process live/CLI-cache state; one
    # test's seeded catalog must not leak into the next (nor the developer's
    # real ~/.codex/models_cache.json, which individual tests re-isolate via
    # a fake HOME/CODEX_HOME when they need a hermetic catalog).
    from machina.core.provider import catalog as provider_catalog

    provider_catalog.clear_catalog_cache()

    # In-memory config is process-global: one test's config.set would leak
    # into the next. Snapshot non-secret values and session overrides.
    from machina.core.config import DEFAULTS, SECRET_KEYS, config

    saved_config = {key: getattr(config, key, None) for key in DEFAULTS if key not in SECRET_KEYS}
    saved_overrides = dict(getattr(config, "_session_overrides", {}) or {})
    # Test defaults must not depend on a developer's saved full-host mode,
    # workspace, persona, or live environment overrides. Individual tests
    # explicitly configure the behavior they exercise after this fixture.
    from machina.core.config import LEGACY_ENV_VARS

    for key, value in DEFAULTS.items():
        if key not in SECRET_KEYS:
            setattr(config, key, value)
            monkeypatch.delenv(f"MACHINA_{key.upper()}", raising=False)
            if key in LEGACY_ENV_VARS:
                monkeypatch.delenv(LEGACY_ENV_VARS[key], raising=False)
    config._session_overrides.clear()
    from machina.core.session_mode import reset_for_tests as reset_session_modes

    reset_session_modes()

    # Optimization circuit breakers are process-global; one test's induced
    # timeout must not make a later test skip its subsystem.
    from machina.core.optimization.policy import reset_breakers

    reset_breakers()

    # The audit writer is a process-global, loop-bound singleton. A test that
    # initialized a gateway without closing it leaks an ownership reference and
    # a writer task bound to its now-dead event loop; a later test's stop()
    # then no-ops (observed as the r5_018 direct-submit race under the full
    # suite). Normalize to a clean generation before each test through the
    # explicit API; reset() logs when it discards live state so the leak stays
    # visible rather than silently vanishing.
    from machina.observability.audit import audit_persistence as audit_writer

    audit_writer.reset()

    yield

    for key, value in saved_config.items():
        setattr(config, key, value)
    config._session_overrides.clear()
    config._session_overrides.update(saved_overrides)
    reset_session_modes()


@pytest_asyncio.fixture(autouse=True)
async def drain_optimization_shadow_tasks():
    """Settle fire-and-forget optimization telemetry before the loop closes."""
    from machina.core.optimization.shadow import drain_shadow_tasks

    yield
    await drain_shadow_tasks(timeout=1.0)


@pytest_asyncio.fixture
async def db_pool():
    """A real test database, configured via MACHINA_TEST_DATABASE_URL.

    Skips the test when no test database is configured. Never hardcode
    credentials here.
    """
    dsn = os.environ.get("MACHINA_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("MACHINA_TEST_DATABASE_URL not set")
    from machina.db.connection import Database

    test_db = Database(dsn=dsn)
    await test_db.connect()
    await test_db.initialize_schema()
    yield test_db
    await test_db.close()
