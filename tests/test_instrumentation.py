"""Structured instrumentation: sink schema, wiring seams, and fail-open rules.

Covers the observable contract used by every instrumented seam:
JSON Lines schema, dedicated sink/rotation, redaction, broken-sink fallback,
status/duration on success/error/cancellation, correlation ids, and the
generated catalog's freshness.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import pytest

from machina.core.metrics import metrics
from machina.observability.instrumentation import instrument, log_event
from machina.tools.base import ToolRegistry

REQUIRED_KEYS = {"ts", "event", "level", "pid"}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "instrumentation.jsonl"
    monkeypatch.setenv("MACHINA_INSTRUMENTATION_LOG", str(path))
    return path


def _records(path: Path) -> list[dict]:
    assert path.exists(), "instrumentation sink was not created"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_event(path: Path, prefix: str) -> list[dict]:
    return [record for record in _records(path) if str(record.get("event", "")).startswith(prefix)]


def _last(path: Path, event: str) -> dict:
    matches = [record for record in _records(path) if record.get("event") == event]
    assert matches, f"no {event!r} record; got {[r.get('event') for r in _records(path)]}"
    return matches[-1]


# ─── sink schema ────────────────────────────────────────────────────────────


def test_jsonl_record_schema_and_duration(log_path):
    with instrument("unit", "op", answer=42) as span:
        span.note(extra="late")

    records = _by_event(log_path, "unit.op.")
    assert [record["status"] for record in records] == ["start", "ok"]
    for record in records:
        assert REQUIRED_KEYS <= set(record)
        assert record["pid"] == os.getpid()
        assert record["level"] in {"debug", "info", "warning", "error", "critical"}
        assert record["ts"].endswith("Z")
    assert records[0]["answer"] == 42
    assert records[1]["answer"] == 42
    assert records[1]["extra"] == "late"
    assert isinstance(records[1]["duration_ms"], (int, float))
    assert records[1]["duration_ms"] >= 0
    # One object per line (JSON Lines): every raw line parses.
    for line in log_path.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)


def test_log_event_accepts_arbitrary_fields(log_path):
    log_event("custom.event", level="debug", count=3, ok=True, detail="value")
    record = _last(log_path, "custom.event")
    assert record["count"] == 3
    assert record["ok"] is True
    assert record["level"] == "debug"


def test_finish_duration_is_monotonic_non_negative(log_path):
    import time

    with instrument("unit", "sleep"):
        time.sleep(0.01)
    record = _last(log_path, "unit.sleep.finish")
    assert record["duration_ms"] >= 5


# ─── redaction and payload hygiene ──────────────────────────────────────────


def test_secret_values_are_redacted(log_path):
    openai_key = "sk-" + "a" * 32
    bearer = "Bearer " + "b" * 40
    log_event("redaction.check", note=openai_key, auth=bearer)
    raw = log_path.read_text(encoding="utf-8")
    assert "a" * 32 not in raw
    assert "b" * 40 not in raw
    assert "REDACTED" in raw


def test_body_keys_are_hashed_not_logged(log_path):
    log_event("body.check", prompt="private prompt body", messages=[{"content": "private"}])
    raw = log_path.read_text(encoding="utf-8")
    record = _last(log_path, "body.check")
    assert "private prompt body" not in raw
    assert "private" not in raw
    assert record["prompt"].startswith("<redacted len=")
    assert record["messages"].startswith("<redacted len=")


def test_string_and_container_bounds(log_path):
    log_event("bounds.check", text="x" * 5000, items=list(range(100)))
    raw = log_path.read_text(encoding="utf-8")
    assert len(raw) < 5000
    # Body-like keys are digested, so use a non-body key for the truncation.
    log_event("bounds.check2", label="y" * 5000)
    record2 = _last(log_path, "bounds.check2")
    assert len(record2["label"]) <= 512 + len("...[truncated]")
    assert record2["label"].endswith("...[truncated]")


# ─── fail-open sink ─────────────────────────────────────────────────────────


def test_broken_sink_path_never_raises(tmp_path, monkeypatch):
    before = metrics.get_counter("instrumentation.events")
    monkeypatch.setenv(
        "MACHINA_INSTRUMENTATION_LOG", str(tmp_path / "missing" / "nested" / "events.jsonl")
    )
    # A parent path that is a file (not a directory) cannot be created either.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MACHINA_INSTRUMENTATION_LOG", str(blocker / "events.jsonl"))

    with instrument("unit", "broken", value=1):
        pass
    log_event("unit.broken.event")
    assert metrics.get_counter("instrumentation.events") >= before + 3
    assert not (blocker / "events.jsonl").exists()


def test_sink_is_rotated(log_path, monkeypatch):
    monkeypatch.setattr("machina.observability.instrumentation.MAX_LOG_BYTES", 500)
    for index in range(30):
        log_event("rotation.event", index=index)
    previous = log_path.with_name(log_path.name + ".1")
    assert previous.exists(), "rotation must keep one previous file"
    assert log_path.stat().st_size <= 500


# ─── status / exceptions / cancellation ────────────────────────────────────


def test_exception_logs_error_status_and_reraises(log_path):
    with pytest.raises(ValueError, match="boom"), instrument("unit", "fails"):
        raise ValueError("boom sk-" + "c" * 32)

    record = _last(log_path, "unit.fails.error")
    assert record["status"] == "error"
    assert record["error_class"] == "ValueError"
    assert "c" * 32 not in log_path.read_text(encoding="utf-8")
    assert isinstance(record["duration_ms"], (int, float))


async def test_cancellation_logs_cancelled_and_reraises(log_path):
    with pytest.raises(asyncio.CancelledError), instrument("unit", "cancelled"):
        raise asyncio.CancelledError()

    record = _last(log_path, "unit.cancelled.cancelled")
    assert record["status"] == "cancelled"
    assert isinstance(record["duration_ms"], (int, float))


def test_correlation_fields_from_contextvars(log_path):
    from machina.tools import agents

    session_id = uuid.uuid4()
    token = agents.current_session_id.set(session_id)
    try:
        log_event("correlation.check")
    finally:
        agents.current_session_id.reset(token)
    record = _last(log_path, "correlation.check")
    assert record["session_id"] == str(session_id)


# ─── wiring: tool ──────────────────────────────────────────────────────────


async def test_tool_execute_emits_start_and_finish(log_path):
    registry = ToolRegistry()

    @registry.register(description="echo")
    def echo(value: str) -> str:
        return value * 2

    result = await registry.execute("echo", value="hi")
    assert result == "hihi"
    records = _by_event(log_path, "tool.execute.")
    assert [record["status"] for record in records] == ["start", "ok"]
    assert records[0]["tool"] == "echo"
    assert records[0]["arg_keys"] == ["value"]
    assert records[0]["arg_count"] == 1
    assert records[1]["result_size"] == 4
    assert "hihi" not in log_path.read_text(encoding="utf-8")


async def test_tool_failure_emits_error(log_path):
    registry = ToolRegistry()

    @registry.register(description="boom")
    def boom() -> str:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        await registry.execute("boom")
    record = _last(log_path, "tool.execute.error")
    assert record["tool"] == "boom"
    assert record["error_class"] == "RuntimeError"


# ─── wiring: gateway RPC ───────────────────────────────────────────────────


async def test_gateway_rpc_dispatch_emits(log_path):
    from machina.gateway.server import Gateway

    gateway = Gateway(lambda _frame: None, owns_db=False)

    async def echo(params):
        return {"echo": params.get("value")}

    gateway.add_method("test.echo", echo)
    await gateway.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "test.echo",
                "params": {"value": "x", "session_id": "s-1"},
            }
        )
    )
    record = _last(log_path, "gateway.rpc.finish")
    assert record["method"] == "test.echo"
    assert record["transport"] == "stdio"
    assert record["session_id"] == "s-1"
    assert record["status"] == "ok"
    assert isinstance(record["duration_ms"], (int, float))


# ─── wiring: provider transport ────────────────────────────────────────────


class _FakeResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return {"usage": {"prompt_tokens": 3, "completion_tokens": 5}, "choices": []}

    def raise_for_status(self) -> None:  # pragma: no cover - success path only
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.base_url = httpx.URL("https://openrouter.ai/api/v1")

    async def post(self, url, json):
        return _FakeResponse()


async def test_provider_request_emits_metadata_only(log_path):
    from machina.core.provider.transport import _post_with_retry

    client = _FakeClient()
    response = await _post_with_retry(
        client,
        "/chat/completions",
        {"model": "test/model", "messages": [{"role": "user", "content": "secret prompt"}]},
    )
    assert response.status_code == 200
    record = _last(log_path, "provider.request.finish")
    assert record["provider"] == "openrouter"
    assert record["model"] == "test/model"
    assert record["stream"] is False
    assert record["message_count"] == 1
    assert record["prompt_tokens"] == 3
    assert record["completion_tokens"] == 5
    assert "secret prompt" not in log_path.read_text(encoding="utf-8")


# ─── wiring: skill search (owned by the skills lane) ───────────────────────


def _skills_search_instrumented() -> bool:
    import ast

    import machina.skills

    root = Path(machina.skills.__file__).resolve().parent
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - shipped code parses
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name not in {"log_event", "instrument", "instrumented"}:
                continue
            literal_args = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if any(
                value.startswith(("skills.search", "skills.candidate"))
                for value in literal_args
            ):
                return True
    return False


def test_skill_search_emits_skills_events(tmp_path, log_path):
    if not _skills_search_instrumented():
        pytest.skip("skills search instrumentation lane has not landed yet")
    from machina.skills.registry import SkillRegistry
    from machina.skills.runtime import search_skills

    registry = SkillRegistry(tmp_path / "skills")
    registry.create_skill("deploy", "Deploy services", "Steps", triggers=["deploy"])
    search_skills(registry, "deploy")

    records = [
        record
        for record in _records(log_path)
        if str(record.get("event", "")).startswith(("skills.search", "skills.candidate"))
    ]
    assert records, "skill search must emit a skills.search/skills.candidate event"


# ─── generated catalog ─────────────────────────────────────────────────────


def test_generated_catalog_is_current(tmp_path):
    import machina

    repo = Path(machina.__file__).resolve().parent.parent
    generator = repo / "tools" / "generate_instrumentation_seams.py"
    output = tmp_path / "instrumentation_seams.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--root",
            str(repo),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    committed = repo / "machina" / "observability" / "instrumentation_seams.py"
    assert output.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")


def test_catalog_covers_core_seams():
    from machina.observability.instrumentation_seams import SEAMS

    triples = {(component, operation) for component, operation, *_ in SEAMS}
    assert {
        ("tool", "execute"),
        ("gateway", "rpc"),
        ("provider", "request"),
        ("provider", "stream"),
        ("context", "add_chunk"),
        ("context", "add_chunks_batch"),
        ("rag", "index_document"),
        ("rag", "search"),
        ("scheduler", "execute"),
        ("turns", "acquire"),
        ("session", "create"),
        ("session", "fork"),
        ("session", "resume"),
        ("permissions", "guard"),
        ("memory", "retrieve"),
        ("mcp", "connect"),
        ("mcp", "call_tool"),
        ("mcp", "disconnect"),
    } <= triples
