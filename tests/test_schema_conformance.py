"""Fresh/upgrade/repeated migrations and actual PostgreSQL SQL contracts."""

import ast
import re
import uuid
from pathlib import Path

import asyncpg
import pytest

from tests.support.repo_paths import repo_path
from tests.support.schema import isolated_schema

pytestmark = [pytest.mark.integration, pytest.mark.db]


async def test_schema_loader_preserves_utf8_sql(db_pool, tmp_path):
    async with isolated_schema(db_pool, initialize=False) as (_, database):
        source = tmp_path / "utf8.sql"
        source.write_text(
            "CREATE TABLE utf8_probe(value text); INSERT INTO utf8_probe VALUES ('café ✓');",
            encoding="utf-8",
        )
        await database.initialize_schema(str(source))
        assert await database.fetchval("SELECT value FROM utf8_probe") == "café ✓"


async def test_migration_failure_rolls_back_all_statements(db_pool, tmp_path):
    async with isolated_schema(db_pool, initialize=False) as (schema, database):
        source = tmp_path / "invalid.sql"
        source.write_text(
            "CREATE TABLE migration_probe(value text); SELECT 1 / 0;", encoding="utf-8"
        )
        with pytest.raises(asyncpg.DivisionByZeroError):
            await database.initialize_schema(str(source))
        assert (
            await database.fetchval("SELECT to_regclass($1)", schema + ".migration_probe") is None
        )


async def test_legacy_session_upgrade_is_idempotent_and_preserves_data(db_pool):
    async with isolated_schema(db_pool, initialize=False) as (_, database):
        await database.execute("""CREATE TABLE sessions (
            id uuid PRIMARY KEY, title text, agent_id text DEFAULT 'harness', state_msgpack bytea,
            status text DEFAULT 'active', goal text, model text, provider text,
            context_budget int DEFAULT 8000, created_at timestamptz DEFAULT now(),
            last_activity timestamptz DEFAULT now())""")
        sid = uuid.uuid4()
        await database.execute(
            "INSERT INTO sessions(id, title, status, state_msgpack) VALUES ($1, $2, 'running', $3)",
            sid,
            "legacy session",
            b"legacy data",
        )
        await database.initialize_schema()
        await database.initialize_schema()
        row = await database.fetchrow(
            "SELECT title, state_msgpack, status, execution_mode FROM sessions WHERE id = $1", sid
        )
        assert dict(row) == {
            "title": "legacy session",
            "state_msgpack": b"legacy data",
            "status": "active",
            "execution_mode": "ask",
        }
        for mode in ("ask", "workspace", "sandbox", "full"):
            await database.execute(
                "UPDATE sessions SET execution_mode = $2 WHERE id = $1", sid, mode
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await database.execute(
                "UPDATE sessions SET execution_mode = 'invalid' WHERE id = $1", sid
            )


async def test_legacy_learning_reviews_upgrade_adds_auto_apply_and_reverted(db_pool):
    """Existing installations gain auto_applied and the 'reverted' status.

    The additive ALTERs in schema.sql must upgrade a pre-auto-apply
    ``learning_reviews`` table in place: existing rows stay, the boolean
    defaults off, 'reverted' becomes accepted, and unknown statuses stay
    rejected by the replaced CHECK constraint.
    """
    async with isolated_schema(db_pool, initialize=False) as (_, database):
        await database.execute("""CREATE TABLE sessions (
            id uuid PRIMARY KEY, title text, agent_id text DEFAULT 'harness', state_msgpack bytea,
            status text DEFAULT 'active', goal text, model text, provider text,
            context_budget int DEFAULT 8000, created_at timestamptz DEFAULT now(),
            last_activity timestamptz DEFAULT now())""")
        await database.execute("""CREATE TABLE learning_reviews (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            agent_id text NOT NULL, turn_hash text NOT NULL,
            status text NOT NULL CHECK (
                status IN ('reviewing', 'none', 'pending', 'approved', 'rejected', 'error')),
            name text, description text, triggers jsonb NOT NULL DEFAULT '[]',
            content text, reason text NOT NULL DEFAULT '',
            created_at timestamptz NOT NULL DEFAULT now(),
            reviewed_at timestamptz, UNIQUE (session_id, turn_hash))""")
        sid = uuid.uuid4()
        await database.execute("INSERT INTO sessions(id, title) VALUES ($1, 'legacy')", sid)
        legacy = await database.fetchval(
            "INSERT INTO learning_reviews(session_id, agent_id, turn_hash, status) "
            "VALUES ($1, 'harness', 'legacy-turn', 'approved') RETURNING id",
            sid,
        )
        await database.initialize_schema()
        await database.initialize_schema()
        row = await database.fetchrow(
            "SELECT status, auto_applied FROM learning_reviews WHERE id = $1", legacy
        )
        assert row["status"] == "approved"
        assert row["auto_applied"] is False
        await database.execute(
            "UPDATE learning_reviews SET status = 'reverted' WHERE id = $1", legacy
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await database.execute(
                "UPDATE learning_reviews SET status = 'nonsense' WHERE id = $1", legacy
            )


async def test_fresh_schema_fks_indexes_and_projections(db_pool):
    async with isolated_schema(db_pool) as (schema, database):
        await database.initialize_schema()
        from machina.core.scheduler import _COLUMNS as job_columns
        from machina.permissions.store import GRANT_COLUMNS

        for table, columns in (("jobs", job_columns), ("permission_grants", GRANT_COLUMNS)):
            assert await database.fetch(f"SELECT {columns} FROM {table} LIMIT 0") == []
        fks = await database.fetch(
            """SELECT c.relname AS child, p.relname AS parent, k.confdeltype::text AS deletion
            FROM pg_constraint k JOIN pg_class c ON c.oid = k.conrelid
            JOIN pg_class p ON p.oid = k.confrelid JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE k.contype = 'f' AND n.nspname = $1""",
            schema,
        )
        edges = {(row["child"], row["parent"], row["deletion"]) for row in fks}
        assert {
            ("context_chunks", "sessions", "c"),
            ("context_archive", "sessions", "c"),
            ("jobs", "sessions", "c"),
            ("memory_provenance", "memories", "c"),
            ("permission_approvals", "sessions", "c"),
            ("permission_grants", "sessions", "c"),
        } <= edges
        indexes = await database.fetch(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1", schema
        )
        names = {row["indexname"] for row in indexes}
        assert {
            "idx_sessions_claim",
            "idx_sessions_status_last_activity",
            "idx_context_chunks_session_type_created",
            "idx_context_chunks_embedding",
        } <= names
        sid = await database.fetchval(
            "INSERT INTO sessions(title) VALUES ('cascade probe') RETURNING id"
        )
        await database.execute(
            "INSERT INTO context_chunks(session_id, agent_id, chunk_type, payload_msgpack) VALUES ($1, 'harness', 'compression_summary', $2)",
            sid,
            b"payload",
        )
        await database.execute("DELETE FROM sessions WHERE id = $1", sid)
        assert (
            await database.fetchval(
                "SELECT count(*) FROM context_chunks WHERE session_id = $1", sid
            )
            == 0
        )


# Modules whose literal DML targets tables that are created at runtime, never
# by schema.sql, and therefore can never compile against a fresh isolated
# schema:
#   * research/workflow_trial.py + research/workflow_benchmark.py are the
#     optional native-vs-LangGraph approval trial (deselected in CI via the
#     `workflow_trial` marker). workflow_trial_runs / workflow_trial_effects
#     are created by the trial's own setup() DDL, and the LangGraph path
#     additionally owns its checkpointer tables (checkpoints,
#     checkpoint_blobs, checkpoint_writes, checkpoint_migrations) through
#     langgraph-checkpoint-postgres' AsyncPostgresSaver.setup().
# The workflow_trial suite (workflow-trial extra installed) is what covers
# these statements; the schema scan skips them by explicit path only.
_LANGGRAPH_OWNED_SOURCES = (
    "research/workflow_trial.py",
    "research/workflow_benchmark.py",
)


def static_queries(root: Path | None = None, *, include_langgraph_owned: bool = False):
    """Complete literal SQL passed to the database, excluding dynamic builders.

    SQLite-dialect statements are skipped per source file: a file that uses
    the sqlite3 module may contain ``?`` placeholders and SQLite-only syntax,
    which can never compile against PostgreSQL. The skip is file-scoped so a
    Postgres file using JSONB ``?`` operators is still scanned.

    ``_LANGGRAPH_OWNED_SOURCES`` are skipped by explicit path: their tables are
    created at runtime, not by ``schema.sql`` (see the constant's comment).
    ``include_langgraph_owned=True`` disables that skip so the exclusion can be
    controlled; the default is the production scan.
    """
    root = root if root is not None else repo_path("machina")
    for path in root.rglob("*.py"):
        if (
            not include_langgraph_owned
            and path.relative_to(root).as_posix() in _LANGGRAPH_OWNED_SOURCES
        ):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        sqlite_dialect = "sqlite3" in source

        def postgres_candidate(query: str) -> bool:
            if sqlite_dialect and "?" in query:
                return False
            return bool(re.match(r"\s*(SELECT|INSERT|UPDATE|DELETE)\b", query, re.I))

        constants = {}
        for assignment in tree.body:
            if isinstance(assignment, ast.Assign):
                try:
                    value = ast.literal_eval(assignment.value)
                except (ValueError, TypeError):
                    continue
                for target in assignment.targets:
                    if isinstance(target, ast.Name) and isinstance(value, str):
                        constants[target.id] = value
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not node.args
                or not isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr not in {"execute", "fetch", "fetchrow", "fetchval"}:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.JoinedStr):
                pieces = []
                for part in argument.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        pieces.append(part.value)
                    elif (
                        isinstance(part, ast.FormattedValue)
                        and isinstance(part.value, ast.Name)
                        and part.value.id in constants
                    ):
                        pieces.append(constants[part.value.id])
                    else:
                        break
                else:
                    query = "".join(pieces)
                    if postgres_candidate(query):
                        yield path.name + ":" + str(node.lineno), query
                continue
            try:
                query = ast.literal_eval(argument)
            except (ValueError, TypeError):
                continue
            if isinstance(query, str) and postgres_candidate(query):
                yield path.name + ":" + str(node.lineno), query


async def test_static_python_sql_compiles_against_schema(db_pool):
    async with isolated_schema(db_pool) as (_, database):
        errors, count = [], 0
        async with database.acquire() as connection:
            for location, query in static_queries():
                try:
                    await connection.prepare(query)
                    count += 1
                except asyncpg.PostgresError as error:
                    errors.append(f"{location}: {type(error).__name__}")
        assert count >= 100, "SQL discovery stopped covering production queries"
        assert errors == [], errors


async def test_langgraph_owned_exclusion_control(db_pool, tmp_path):
    """Control for the langgraph-owned exclusion above — not a weakening.

    Guards three properties: the exclusion names real files (no stale skip
    hiding a moved module), the excluded files do contribute literal DML when
    the skip is disabled (so the exclusion is what makes the scan pass), and
    the same ``prepare`` gate still rejects a genuinely undefined table in
    product-style SQL scanned by the very same discoverer.
    """
    root = repo_path("machina")
    for relative in _LANGGRAPH_OWNED_SOURCES:
        assert (root / relative).is_file(), f"stale exclusion (file moved?): {relative}"

    excluded_names = tuple(Path(relative).name + ":" for relative in _LANGGRAPH_OWNED_SOURCES)
    without_skip = [
        location
        for location, _ in static_queries(include_langgraph_owned=True)
        if location.startswith(excluded_names)
    ]
    assert without_skip, "excluded modules contribute no SQL; the exclusion is stale"
    assert [
        location for location, _ in static_queries() if location.startswith(excluded_names)
    ] == [], "langgraph-owned sources leaked back into the schema scan"

    (tmp_path / "synthetic_product_module.py").write_text(
        "async def run(conn):\n"
        "    return await conn.execute('SELECT 1 FROM table_that_is_not_in_the_schema')\n",
        encoding="utf-8",
    )
    scanned = list(static_queries(tmp_path))
    assert [query for _, query in scanned] == [
        "SELECT 1 FROM table_that_is_not_in_the_schema"
    ]

    async with (
        isolated_schema(db_pool) as (_, database),
        database.acquire() as connection,
    ):
        with pytest.raises(asyncpg.UndefinedTableError):
            await connection.prepare(scanned[0][1])
