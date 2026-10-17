"""B3 — compaction/status coexistence and release evidence (spec §17-B3, §14.7, §16.9).

Coverage map:

- **C13 coexistence & ownership**: the in-turn gate acquires no turn/mutation
  claim and cannot conflict with ``maybe_auto_compact``'s durable MUTATION
  claim. DB-backed proofs: (a) the gate completes while a compaction mutation
  is held and the claim is untouched; (b) auto-compaction defers while a turn
  holds the session and proceeds after release, with the no-progress guard
  preventing a second compaction/accounting pass; (c) a wired (flag-on) turn
  followed by the real ``TurnEngine.schedule_auto_compact`` produces the same
  ownership trace as flag-off — exactly one claim attempt per scheduling call.
- **C14 restart/reconnect durability** (DB-backed): recovered artifacts cite
  the stored revision/source_version through a fresh connection AND a fresh
  process; stale cursors (revision drift) are rejected end-to-end through the
  tool surface as structured stale errors, never wrong data; cross-session
  recovery stays denied.
- **C11 effective-budget change**: a mid-session model/budget change
  recomputes the fit against the effective route through the fake provider;
  the next gate uses the new budget and refuses when pins no longer fit.
- **C05 continuation**: signed ``provider_content`` blocks survive
  byte-for-byte through a wired gate into the next provider request.
- **§14.7 status**: one bounded ``request_projection_decision`` audit event
  per gate evaluation (shadow/wired/refusal/degraded) — labels and counts,
  never bodies.

No live provider calls: every provider is a scripted fake. DB-marked tests
run against ``MACHINA_TEST_DATABASE_URL`` and skip without it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from machina.core.agent import BaseReActAgent, ReActAgent
from machina.core.config import config
from machina.core.exceptions import ToolError
from machina.core.models import LLMResponse, Session
from machina.core.request_gate import evaluate_request
from machina.tools.base import ToolRegistry
from tests.support.fakes import (
    RejectingProvider,
    ScriptedProvider,
    VerifiedLookupOnly,
    VerifiedStore,
)

_RejectingProvider = RejectingProvider
_VerifiedLookupOnly = VerifiedLookupOnly
_VerifiedStore = VerifiedStore

REPO = Path(os.environ.get("MACHINA_REPO") or Path(__file__).resolve().parents[2] / "machina")
CORPUS = Path(__file__).resolve().parent / "data" / "request_projection" / "corpus.json"

GOAL = "Ship the B3 coexistence evidence safely."
CONSTRAINT = "Do not edit ui/."
FILLER = "deterministic B3 fixture line %03d for coexistence and release evidence"
ROUNDS = 4


# ─── fake providers (no live calls; scripted + payload capture) ──────────────
# Shared bodies live in tests/support/fakes.py (B3 was the superset of B2).


def _ScriptedProvider(responses, *, on_call=None, **kwargs):
    return ScriptedProvider(responses, on_call=on_call, model="fake-b3", **kwargs)


def _round_response(index: int) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake-b3",
        usage={"total_tokens": 1},
        tool_calls=[
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "session_recall",
                    "arguments": json.dumps({"query": f"round {index}"}),
                },
            }
        ],
    )


def _final_response() -> LLMResponse:
    return LLMResponse(content="all rounds complete", model="fake-b3", usage={"total_tokens": 1})


def _script() -> list[LLMResponse]:
    return [_round_response(i) for i in range(1, ROUNDS + 1)] + [_final_response()]


# ─── loop harness (mocked session/context, stub tool, no network) ────────────


def _stub_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(
        name="session_recall",
        description="stub recall for the B3 fixture",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    async def session_recall(query: str = "") -> str:
        lines = [FILLER % i for i in range(48)]
        return f"{query}: " + "\n".join(lines)

    return reg


def _wire_loop(monkeypatch, tmp_path, session) -> None:
    """Mock session/context plumbing for a full loop run (B2 harness shape)."""
    monkeypatch.setenv("MACHINA_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setattr(config, "workspace_root", str(tmp_path / "work"))
    monkeypatch.setattr(config, "execution_mode", "ask")
    monkeypatch.setattr(config, "max_tokens", 64)
    for field in ("memory_enabled", "rag_enabled", "personality_growth_enabled"):
        monkeypatch.setattr(config, field, False)
    monkeypatch.setattr("machina.core.agent.session_manager.get", AsyncMock(return_value=session))
    monkeypatch.setattr("machina.core.agent.session_manager.update_activity", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_recent_context", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "machina.core.agent.context_manager.search_archive_text", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("machina.core.agent.context_manager.add_chunks_batch", AsyncMock())
    monkeypatch.setattr(
        "machina.core.agent.context_manager.get_latest_images", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(BaseReActAgent, "_reserve_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(BaseReActAgent, "_finish_usage", AsyncMock())
    monkeypatch.setattr("machina.core.agent.registry", _stub_registry())


def _count(messages, tool_defs, provider) -> int:
    from machina.core.prompt import _request_context_tokens

    return _request_context_tokens(messages, tool_defs, provider)


def _freeze_gate_budget(monkeypatch, value: int) -> None:
    import machina.core.request_gate as gate_module

    monkeypatch.setattr(gate_module, "_effective_turn_budget", lambda session, provider: value)


def _corpus_messages(name: str = "multi_round") -> list[dict]:
    """Provider-shaped messages from the B1 corpus, with live-shaped handles."""
    case = json.loads(CORPUS.read_text())["cases"][name]
    messages: list[dict] = []
    for message in case["messages"]:
        converted = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            converted["tool_calls"] = copy.deepcopy(message["tool_calls"])
        if message.get("tool_call_id"):
            converted["tool_call_id"] = message["tool_call_id"]
        if message.get("artifact_handle"):
            converted["content"] += (
                f"\n[evidence retained as artifact {message['artifact_handle']} "
                f"({len(converted['content'])} chars); artifact_read pages the exact original]"
            )
        messages.append(converted)
    return messages


# ─── DB helpers (skips without MACHINA_TEST_DATABASE_URL) ────────────────────


@pytest.fixture
async def live_db():
    """The shared DB singleton, connected to the disposable test database."""
    if not os.environ.get("MACHINA_TEST_DATABASE_URL"):
        pytest.skip("MACHINA_TEST_DATABASE_URL not set")
    from machina.db.connection import db

    await db.connect()
    await db.initialize_schema()
    yield db
    await db.close()


async def _mk_session(db, *, agent_id: str = "harness", ancestors=None) -> uuid.UUID:
    session_id = uuid.uuid4()
    await db.execute(
        "INSERT INTO sessions (id, agent_id, ancestor_session_ids, ancestor_agent_ids) "
        "VALUES ($1, $2, $3::uuid[], $4::text[])",
        session_id,
        agent_id,
        ancestors or [],
        [agent_id] * len(ancestors or []),
    )
    return session_id


async def _drop_sessions(db, ids) -> None:
    if ids:
        await db.execute("DELETE FROM sessions WHERE id = ANY($1::uuid[])", list(ids))


@pytest.fixture
def claim_trace(monkeypatch):
    """Wrap the public claim functions so tests can assert what was attempted."""
    from machina.core import turns

    trace: list[str] = []

    def _wrap(name: str, label: str) -> None:
        original = getattr(turns, name)

        async def wrapped(session_id, *args, **kwargs):
            result = await original(session_id, *args, **kwargs)
            trace.append(f"{label}:{'ok' if result else 'none'}")
            return result

        monkeypatch.setattr(turns, name, wrapped)

    _wrap("try_begin_turn", "begin_turn")
    _wrap("end_turn", "end_turn")
    _wrap("begin_mutation", "begin_mutation")
    _wrap("end_mutation", "end_mutation")
    return trace


async def _drive_schedule_auto_compact(monkeypatch, session_id) -> int:
    """Execute the real ``TurnEngine.schedule_auto_compact`` and await its task."""
    import machina.core.turn_engine as engine_module

    created: list[asyncio.Task] = []
    real_create_task = asyncio.create_task

    def _capture(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        created.append(task)
        return task

    tracked: list[asyncio.Task] = []

    def _track(task):
        tracked.append(task)
        return task

    monkeypatch.setattr(engine_module, "runtime_services", SimpleNamespace(track=_track))
    fake_engine = SimpleNamespace(
        spec=SimpleNamespace(session_id=session_id),
        _record=lambda site, exc: None,
    )
    with mock.patch.object(asyncio, "create_task", _capture):
        engine_module.TurnEngine.schedule_auto_compact(fake_engine)
    assert len(created) == 1
    for task in created:
        await task
    assert tracked == created
    return len(created)


def _capture_audits(monkeypatch) -> list[tuple]:
    audits: list[tuple] = []
    monkeypatch.setattr(
        "machina.core.request_gate.audit_log",
        lambda *args, **kwargs: audits.append((args, kwargs)),
    )
    return audits


# ─── C13(a): the gate makes no claim and cannot disturb a held mutation ──────


@pytest.mark.asyncio
async def test_gate_completes_under_held_mutation_and_attempts_no_claim(
    live_db, monkeypatch, claim_trace
):
    from machina import services as services_module
    from machina.core import turns

    session_id = await _mk_session(live_db)
    compress = AsyncMock(return_value=None)
    monkeypatch.setattr(services_module, "compress_session", compress)
    try:
        token = await turns.begin_mutation(session_id)
        assert token
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] == token and state["claim_kind"] == "mutation"

        messages = _corpus_messages("multi_round")
        full = _count(messages, [], _RejectingProvider())
        _freeze_gate_budget(monkeypatch, full - 100)
        config.set("request_projection_enabled", True)
        monkeypatch.setattr("machina.core.request_gate.artifact_store", _VerifiedLookupOnly())

        claim_trace.clear()
        gate = await evaluate_request(
            messages,
            session=Session(id=session_id, goal=GOAL, state={"constraints": [CONSTRAINT]}),
            provider=_RejectingProvider(),
            tool_defs=[],
            session_id=session_id,
            agent_id="harness",
            user_message="go ahead",
        )
        # The gate completed its wired decision and attempted no claim at all.
        assert gate.failure is None
        assert gate.projected is True
        assert claim_trace == []

        # Ownership is byte-identical after the gate: still the compaction's.
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] == token and state["claim_kind"] == "mutation"

        # The other direction: a turn cannot start over a live mutation, and a
        # second compaction attempt is refused (no double claim, no report).
        assert await turns.try_begin_turn(session_id) is None
        assert await services_module.maybe_auto_compact(session_id) is None
        assert compress.call_count == 0
        assert await turns.end_mutation(session_id, token) is True
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] is None
    finally:
        await _drop_sessions(live_db, [session_id])


def test_gate_module_has_no_claim_coordination_imports():
    """Static companion to the runtime proof: the seam cannot claim by design."""
    import inspect

    import machina.core.request_gate as gate_module

    source = inspect.getsource(gate_module)
    for name in (
        "begin_mutation",
        "begin_turn",
        "end_mutation",
        "end_turn",
        "try_begin_turn",
        "machina.core.turns",
    ):
        assert name not in source


# ─── C13(b): auto-compaction defers on a live turn, then compacts once ───────


@pytest.mark.asyncio
async def test_auto_compaction_defers_on_live_turn_and_never_double_accounts(
    live_db, monkeypatch
):
    from machina import services as services_module
    from machina.core import turns

    compression = SimpleNamespace(compressed_tokens=42, original_tokens=99, method="truncate")
    compress = AsyncMock(return_value=compression)
    monkeypatch.setattr(services_module, "compress_session", compress)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))

    session_id = await _mk_session(live_db)
    try:
        token = await turns.try_begin_turn(session_id)
        assert token
        # (b) A live turn claim defers compaction at acquisition: no compression,
        # no mutation claim, the turn claim untouched.
        assert await services_module.maybe_auto_compact(session_id) is None
        assert compress.call_count == 0
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] == token and state["claim_kind"] == "turn"
        assert await turns.end_turn(session_id, token) is True

        # After release the same call runs exactly one compaction and records
        # progress once (the claim is released again).
        first = await services_module.maybe_auto_compact(session_id)
        assert first is not None and first["compressedTokens"] == 42
        assert compress.call_count == 1
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] is None
        from machina.core.session import session_manager

        row = await session_manager.get(session_id)
        assert (row.state or {}).get("last_auto_compact_tokens") == 42

        # No-progress guard: an immediate re-run skips without compressing or
        # re-accounting, and still leaves no claim behind.
        assert await services_module.maybe_auto_compact(session_id) is None
        assert compress.call_count == 1
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] is None
    finally:
        await _drop_sessions(live_db, [session_id])


# ─── C13(c): wired turn + schedule_auto_compact == flag-off ownership ────────


@pytest.mark.asyncio
async def test_wired_turn_then_scheduled_compaction_matches_flag_off_ownership(
    tmp_path, monkeypatch, live_db, claim_trace
):
    from machina import services as services_module
    from machina.core import turns

    compress = AsyncMock(return_value=None)
    monkeypatch.setattr(services_module, "compress_session", compress)
    monkeypatch.setattr(services_module, "session_context_window", AsyncMock(return_value=100))
    monkeypatch.setattr(services_module, "_estimate_next_request", AsyncMock(return_value=10**6))

    created: list[uuid.UUID] = []
    compress_calls: list[int] = []

    async def _ownership_run(flag: bool) -> list[str]:
        config.set("request_projection_enabled", flag)
        session_id = await _mk_session(live_db)
        created.append(session_id)
        session = Session(
            id=session_id,
            context_budget=8000,
            goal=GOAL,
            state={"constraints": [CONSTRAINT]},
        )
        _wire_loop(monkeypatch, tmp_path, session)
        claim_trace.clear()
        token = await turns.try_begin_turn(session_id)  # transport acquires
        assert token
        provider = _ScriptedProvider(_script())
        response = await ReActAgent(provider=provider).run(session_id, "go ahead", verbose=False)
        assert response.outcome == "completed"
        assert await turns.end_turn(session_id, token) is True  # release before scheduling
        before = compress.call_count
        assert await _drive_schedule_auto_compact(monkeypatch, session_id) == 1
        # Exactly one compaction attempt per scheduling call, in both modes.
        assert compress.call_count == before + 1
        compress_calls.append(compress.call_count)
        state = await turns._db_claim_state(session_id)
        assert state["claim_owner"] is None  # ownership fully released after cleanup
        from machina.core.session import session_manager

        row = await session_manager.get(session_id)
        assert "last_auto_compact_tokens" not in (row.state or {})
        return list(claim_trace)

    try:
        wired_trace = await _ownership_run(True)
        off_trace = await _ownership_run(False)
        assert wired_trace == off_trace
        assert wired_trace == [
            "begin_turn:ok",
            "end_turn:ok",
            "begin_mutation:ok",
            "end_mutation:ok",
        ]
        assert compress_calls == [1, 2]
    finally:
        await _drop_sessions(live_db, created)


# ─── C14: restart/reconnect durability (DB-backed) ───────────────────────────

_FRESH_PROCESS_READ = """
import asyncio, json, os, sys, uuid

async def main():
    from machina.db.connection import Database
    import machina.core.artifacts as artifacts

    db = Database(dsn=os.environ["MACHINA_TEST_DATABASE_URL"])
    await db.connect()
    artifacts.db = db
    try:
        page = await artifacts.artifact_store.read(
            sys.argv[1],
            session_id=uuid.UUID(sys.argv[2]),
            agent_id="harness",
            offset=0,
            limit=50,
        )
        print(
            "B3JSON "
            + json.dumps(
                {
                    "content_revision": page["content_revision"],
                    "source_version": page["source_version"],
                    "total_chars": page["total_chars"],
                    "content": page["content"],
                }
            )
        )
    finally:
        await db.close()

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_recovered_artifact_cites_revision_across_reconnect_and_fresh_process(
    live_db, monkeypatch
):
    from machina.core.artifacts import artifact_store
    from machina.db.connection import Database

    session_id = await _mk_session(live_db)
    url = os.environ["MACHINA_TEST_DATABASE_URL"]
    try:
        content = "B3-RECOVERY " + "".join(str(i % 10) for i in range(400))
        stored = await artifact_store.store(
            "tool:terminal",
            content,
            session_id=session_id,
            agent_id="harness",
            source_version="machina:b3-test",
            status="completed",
        )
        assert stored is not None
        assert stored["source_version"] == "machina:b3-test"

        # Fresh connection: a new pool/process-equivalent handle reads the same
        # stored binding (no in-process cache carries the record).
        fresh = Database(dsn=url)
        await fresh.connect()
        try:
            monkeypatch.setattr("machina.core.artifacts.db", fresh)
            page = await artifact_store.read(
                stored["artifact_id"], session_id=session_id, agent_id="harness", limit=50
            )
            meta = await artifact_store.describe(
                stored["artifact_id"], session_id=session_id, agent_id="harness"
            )
        finally:
            await fresh.close()
        assert page["content"] == content[:50]
        assert page["total_chars"] == len(content)
        assert page["content_revision"] == stored["content_revision"]
        assert page["source_version"] == "machina:b3-test"
        assert meta["content_revision"] == stored["content_revision"]
        assert meta["source_version"] == "machina:b3-test"

        # Fresh OS process: recovery cites the stored revision/source_version.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", _FRESH_PROCESS_READ, stored["artifact_id"], str(session_id)],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(REPO),
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        line = next(line for line in proc.stdout.splitlines() if line.startswith("B3JSON "))
        payload = json.loads(line.removeprefix("B3JSON "))
        assert payload["content_revision"] == stored["content_revision"]
        assert payload["source_version"] == "machina:b3-test"
        assert payload["total_chars"] == len(content)
        assert payload["content"] == content[:50]
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_stale_read_cursor_is_rejected_through_the_tool_surface(live_db):
    from machina.core.artifacts import ArtifactError, artifact_store
    from machina.tools import agents as agents_module
    from machina.tools.artifacts import artifact_read

    session_id = await _mk_session(live_db)
    try:
        content = "B3-STALE " + "x" * 200 + " RUN_MARKER " + "y" * 300
        stored = await artifact_store.store(
            "tool:terminal",
            content,
            session_id=session_id,
            agent_id="harness",
            source_version="machina:b3-test",
        )
        artifact_id = stored["artifact_id"]

        token = agents_module.current_session_id.set(session_id)
        token_agent = agents_module.current_agent_id.set("harness")
        try:
            first = await artifact_read(artifact_id, limit=60)
            header = first.splitlines()[0]
            assert f"revision={stored['content_revision']}" in header
            assert "more remains" in header
            cursor = re.search(r"next_cursor=(\S+)", first).group(1)

            # Resume works while the binding matches.
            resumed = await artifact_read(artifact_id, cursor=cursor)
            assert (
                f"of {len(content)} revision={stored['content_revision']}]"
                in resumed.splitlines()[0]
            )

            # Revision drift (a rebuilt/replaced evidence basis): the bound
            # cursor must be rejected, never silently paged.
            await live_db.execute(
                "UPDATE tool_artifacts SET content_revision = 'sha256:drifted' WHERE id = $1",
                uuid.UUID(artifact_id),
            )
            with pytest.raises(ToolError) as excinfo:
                await artifact_read(artifact_id, cursor=cursor)
            message = str(excinfo.value)
            assert "no longer matches" in message and "issue a new read" in message
            assert content[:40] not in message  # never wrong data

            # Structured at the store layer: a stable machine-readable code.
            with pytest.raises(ArtifactError) as store_error:
                await artifact_store.read(
                    artifact_id,
                    session_id=session_id,
                    agent_id="harness",
                    cursor=cursor,
                )
            assert store_error.value.code == "stale_cursor"

            # A plain new read (no cursor) is the explicit way to accept the
            # new basis — it states the drifted revision instead of guessing.
            fresh = await artifact_read(artifact_id, limit=60)
            assert "revision=sha256:drifted" in fresh.splitlines()[0]
        finally:
            agents_module.current_session_id.reset(token)
            agents_module.current_agent_id.reset(token_agent)
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_stale_search_cursor_is_rejected_through_the_tool_surface(live_db):
    from machina.core.artifacts import artifact_store
    from machina.tools import agents as agents_module
    from machina.tools.artifacts import artifact_search

    session_id = await _mk_session(live_db)
    try:
        older = await artifact_store.store(
            "tool:terminal",
            "older artifact without the marker " + "z" * 300,
            session_id=session_id,
            agent_id="harness",
        )
        newest = await artifact_store.store(
            "tool:terminal",
            "RUN_MARKER in the newest artifact " + "q" * 300,
            session_id=session_id,
            agent_id="harness",
        )
        assert older and newest

        token = agents_module.current_session_id.set(session_id)
        token_agent = agents_module.current_agent_id.set("harness")
        try:
            first = await artifact_search("RUN_MARKER", limit=1)
            assert f"artifact={newest['artifact_id']}" in first
            cursor = re.search(r"next_cursor=(\S+)", first).group(1)

            # The bound head's revision drifts (revision-invalidated evidence).
            await live_db.execute(
                "UPDATE tool_artifacts SET content_revision = 'sha256:drifted' WHERE id = $1",
                uuid.UUID(newest["artifact_id"]),
            )
            with pytest.raises(ToolError) as excinfo:
                await artifact_search("RUN_MARKER", limit=1, cursor=cursor)
            message = str(excinfo.value)
            assert "evidence basis changed" in message and "issue a new search" in message
            assert "RUN_MARKER in the newest" not in message  # never wrong data

            # A fresh search starts a new basis and still finds the content.
            fresh = await artifact_search("RUN_MARKER", limit=1)
            assert "RUN_MARKER" in fresh
        finally:
            agents_module.current_session_id.reset(token)
            agents_module.current_agent_id.reset(token_agent)
    finally:
        await _drop_sessions(live_db, [session_id])


@pytest.mark.asyncio
async def test_cross_session_recovery_stays_denied_through_tools(live_db):
    from machina.core.artifacts import artifact_store
    from machina.tools import agents as agents_module
    from machina.tools.artifacts import (
        artifact_describe,
        artifact_read,
        artifact_search,
    )

    owner = await _mk_session(live_db)
    stranger = await _mk_session(live_db)
    try:
        stored = await artifact_store.store(
            "tool:terminal",
            "secret evidence " * 20,
            session_id=owner,
            agent_id="harness",
        )
        artifact_id = stored["artifact_id"]

        token = agents_module.current_session_id.set(stranger)
        token_agent = agents_module.current_agent_id.set("harness")
        try:
            for call in (
                lambda: artifact_read(artifact_id),
                lambda: artifact_describe(artifact_id),
                lambda: artifact_search("secret", artifact_id=artifact_id),
            ):
                with pytest.raises(ToolError) as excinfo:
                    await call()
                assert "not accessible" in str(excinfo.value)
        finally:
            agents_module.current_session_id.reset(token)
            agents_module.current_agent_id.reset(token_agent)
    finally:
        await _drop_sessions(live_db, [owner, stranger])


# ─── C11: mid-session effective-budget change ────────────────────────────────


@pytest.mark.asyncio
async def test_mid_session_route_change_recomputes_the_budget_at_the_seam(monkeypatch):
    config.set("request_projection_enabled", True)
    session = Session(
        id=uuid.uuid4(),
        goal=GOAL,
        state={"constraints": [CONSTRAINT]},
        model="b3-large",
        context_window=6000,
        context_window_model="b3-large",
    )
    messages = [{"role": "user", "content": "go ahead"}]
    provider = _RejectingProvider()
    provider.model = "b3-large"
    kwargs = {
        "provider": provider,
        "tool_defs": [],
        "session_id": uuid.uuid4(),
        "agent_id": "harness",
        "user_message": "go ahead",
    }
    first = await evaluate_request(messages, session=session, **kwargs)
    assert first.effective_budget == 6000
    assert first.failure is None

    # Mid-session route change: the same session object now resolves to a much
    # smaller effective window; the next gate recomputes against it. F-01: the
    # provider carries the model actually serving the turn, so it follows the
    # route.
    provider.model = "b3-small"
    session.model = "b3-small"
    session.context_window = 64
    session.context_window_model = "b3-small"
    second = await evaluate_request(messages, session=session, **kwargs)
    assert second.effective_budget == 64
    assert second.failure is not None
    assert second.refusal is not None
    assert second.refusal.code == "pins_exceed_capacity"


@pytest.mark.asyncio
async def test_loop_refits_next_iteration_after_a_model_window_change(tmp_path, monkeypatch):
    session_id = uuid.uuid4()
    session = Session(
        id=session_id,
        context_budget=8000,
        goal=GOAL,
        state={"constraints": [CONSTRAINT]},
    )
    _wire_loop(monkeypatch, tmp_path, session)
    config.set("request_projection_enabled", True)
    audits = _capture_audits(monkeypatch)

    def _shrink_route(call_number: int) -> None:
        if call_number == 1:
            # The route changes between iterations (model pin + resolved
            # window). F-01: the provider carries the serving model, so the
            # route change updates it together with the session pin.
            provider.model = "b3-small"
            session.model = "b3-small"
            session.context_window = 64
            session.context_window_model = "b3-small"

    provider = _ScriptedProvider(_script(), on_call=_shrink_route)
    response = await ReActAgent(provider=provider).run(session_id, "go ahead", verbose=False)

    # Iteration 1 used the initial route; iteration 2 refit against the new
    # budget and refused before any second provider call.
    assert len(provider.calls) == 1
    assert response.outcome == "failed"
    assert "Pinned content alone exceeds the effective request budget." in response.content

    decisions = [
        kwargs
        for args, kwargs in audits
        if args == ("request_projection_decision",) and kwargs["mode"] == "wired"
    ]
    assert [d["would_fit"] for d in decisions] == [True, False]
    assert decisions[1]["refusal"] == "pins_exceed_capacity"
    assert decisions[0]["effective_budget_tokens"] > decisions[1]["effective_budget_tokens"]


# ─── C05: signed provider-native blocks through the wired gate ───────────────


@pytest.mark.asyncio
async def test_signed_provider_blocks_survive_byte_for_byte_into_next_request(
    tmp_path, monkeypatch
):
    session = Session(
        id=uuid.uuid4(),
        context_budget=8000,
        goal=GOAL,
        state={"constraints": [CONSTRAINT]},
    )
    _wire_loop(monkeypatch, tmp_path, session)
    store = _VerifiedStore()
    monkeypatch.setattr("machina.core.tool_views.artifact_store", store)
    monkeypatch.setattr("machina.core.request_gate.artifact_store", store)

    native = [
        {"type": "thinking", "thinking": "signed chain of thought", "signature": "sig-b3-1"},
        {"type": "redacted_thinking", "data": "opaque-encrypted-payload"},
    ]

    def _native_script() -> list[LLMResponse]:
        first = LLMResponse(
            content="",
            model="fake-b3",
            usage={"total_tokens": 1},
            raw={"provider_content": copy.deepcopy(native)},
            tool_calls=[
                {
                    "id": "call_native_1",
                    "type": "function",
                    "function": {"name": "session_recall", "arguments": '{"query": "native"}'},
                }
            ],
        )
        return [first, _final_response()]

    # Probe: huge budget -> the full second request including system pins.
    _freeze_gate_budget(monkeypatch, 10**9)
    config.set("request_projection_enabled", True)
    probe_provider = _ScriptedProvider(_native_script())
    probe = await ReActAgent(provider=probe_provider).run(session.id, "go ahead", verbose=False)
    assert probe.outcome == "completed" and len(probe_provider.calls) == 2
    full_messages = probe_provider.calls[1]["messages"]
    tool_defs = probe_provider.calls[1]["tools"]
    full_tokens = _count(full_messages, tool_defs, probe_provider)

    # Wired: the same request at its full size. Append-time reducers already
    # compact old observations; the signed-block contract is the focus here.
    _freeze_gate_budget(monkeypatch, full_tokens)
    config.set("request_projection_enabled", True)
    wired_provider = _ScriptedProvider(_native_script())
    wired = await ReActAgent(provider=wired_provider).run(session.id, "go ahead", verbose=False)
    assert wired.outcome == "completed"
    assert len(wired_provider.calls) == 2
    sent = wired_provider.calls[1]["messages"]
    assert _count(sent, wired_provider.calls[1]["tools"], wired_provider) <= full_tokens

    probe_assistant = next(
        m
        for m in full_messages
        if any(tc.get("id") == "call_native_1" for tc in m.get("tool_calls") or ())
    )
    wired_assistant = next(
        m
        for m in sent
        if any(tc.get("id") == "call_native_1" for tc in m.get("tool_calls") or ())
    )
    # C05: the signed blocks are the exact same JSON, deep-copied, in order.
    assert (
        json.dumps(wired_assistant["provider_content"], sort_keys=True, separators=(",", ":"))
        == json.dumps(native, sort_keys=True, separators=(",", ":"))
        == json.dumps(probe_assistant["provider_content"], sort_keys=True, separators=(",", ":"))
    )
    assert wired_assistant["provider_content"] is not native
    # C04: ids and arguments survive verbatim alongside the native blocks.
    assert wired_assistant["tool_calls"] == probe_assistant["tool_calls"]


# ─── §14.7: bounded honest status for the gate decision ─────────────────────


@pytest.mark.asyncio
async def test_request_projection_decision_is_emitted_bounded_in_both_modes(monkeypatch):
    messages = _corpus_messages("multi_round")
    session = Session(id=uuid.uuid4(), goal=GOAL, state={"constraints": [CONSTRAINT]})
    kwargs = {
        "provider": _RejectingProvider(),
        "tool_defs": [],
        "session_id": uuid.uuid4(),
        "agent_id": "harness",
        "user_message": "go ahead",
    }
    audits = _capture_audits(monkeypatch)
    monkeypatch.setattr("machina.core.request_gate.artifact_store", _VerifiedLookupOnly())

    config.set("request_projection_enabled", False)
    shadow = await evaluate_request(messages, session=session, **kwargs)
    assert shadow.projected is False and shadow.failure is None  # shadow never applies
    config.set("request_projection_enabled", True)
    full = _count(messages, [], _RejectingProvider())
    _freeze_gate_budget(monkeypatch, full - 100)
    wired = await evaluate_request(messages, session=session, **kwargs)

    events = [kwargs_ for args, kwargs_ in audits if args == ("request_projection_decision",)]
    assert len(events) == 2
    shadow_event, wired_event = events
    assert shadow_event["mode"] == "shadow" and shadow_event["projected"] is False
    assert shadow_event["would_fit"] is True and shadow_event["refusal"] is None
    assert shadow_event["estimated_tokens"] > 0 and shadow_event["passes"] >= 1
    assert shadow_event["degraded"] is False
    assert wired_event["mode"] == "wired" and wired_event["projected"] is True
    assert wired_event["would_fit"] is (wired.failure is None)
    assert wired_event["effective_budget_tokens"] == full - 100
    assert wired_event["request_tokens"] == wired.request_tokens

    # Bounded: no prompt bodies, pins, or message content anywhere in the events.
    blob = json.dumps(events, default=str)
    assert GOAL not in blob
    assert CONSTRAINT not in blob
    assert "deterministic round filler" not in blob
    for event in events:
        assert all(
            isinstance(value, (bool, int, str)) or value is None for value in event.values()
        )


@pytest.mark.asyncio
async def test_degraded_wiring_is_recorded_not_silent(monkeypatch):
    messages = [{"role": "user", "content": "go ahead"}]
    audits = _capture_audits(monkeypatch)
    config.set("request_projection_enabled", True)

    async def _boom(*args, **kwargs):
        raise RuntimeError("forced projection failure")

    monkeypatch.setattr("machina.core.request_gate._build_input", _boom)
    gate = await evaluate_request(
        messages,
        session=Session(id=uuid.uuid4(), goal=GOAL),
        provider=_RejectingProvider(),
        tool_defs=[],
        session_id=uuid.uuid4(),
        agent_id="harness",
        user_message="go ahead",
    )
    # Degraded wiring keeps the legacy request + hard guard (no lossy send).
    assert gate.failure is None
    assert gate.messages is messages
    assert gate.projected is False

    event = next(kwargs for args, kwargs in audits if args == ("request_projection_decision",))
    assert event["mode"] == "wired" and event["degraded"] is True
    assert event["would_fit"] is None and event["estimated_tokens"] is None
