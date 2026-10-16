"""Pre-wave review fixes (2026-10-16): pins for the HIGH/MID review items.

* harness_status truncation ladder drops recoverable sections (installed
  names, the duplicated FEATURES map) before learning.notes;
* firecrawl lifecycle: state-root resolution, PID-reuse identity, the
  start-chain fence and the fast-quit chain stop;
* benchmark: DevTools resolver validation/preference and the section loop
  surviving a SystemExit;
* RAG evaluation: default dataset resolves through the DevTools checkout;
* approval gate: a disabled gate returns the durable outcome, not the stale
  pending record.

The approval pin needs ``MACHINA_TEST_DATABASE_URL`` (``db_pool`` skips
otherwise). The firecrawl pins never touch a real Lima VM: everything runs
against stub ``limactl``/``up.sh`` scripts under a temp ``machina_home``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# 1. harness_status truncation ladder
# ---------------------------------------------------------------------------


async def test_harness_status_keeps_notes_and_drops_recoverable_sections_first(monkeypatch):
    """RED before the reorder: a payload just over the cap emptied
    ``learning.notes`` while the recoverable installed-names detail (and the
    duplicated FEATURES map) were still present.
    """
    from machina.core.models import Session
    from machina.tools import harness

    sid = uuid.uuid4()
    owner = "ladder-owner"
    monkeypatch.setattr(harness, "active_agent_scope", AsyncMock(return_value=(owner, sid)))
    monkeypatch.setattr(
        harness.session_manager, "get", AsyncMock(return_value=Session(id=sid, agent_id=owner))
    )
    monkeypatch.setattr(
        harness.growth_store,
        "get",
        AsyncMock(
            return_value={
                "notes": {"style:tone": "Speak plainly"},
                "interactions": 3,
                "feedback_count": 1,
            }
        ),
    )
    monkeypatch.setattr(
        harness.growth_store, "rsi_status", AsyncMock(return_value={"enabled": True, "revision": 1})
    )
    monkeypatch.setattr(harness, "configuration", lambda session: {"stub": True})
    monkeypatch.setattr(harness, "_persona_summary", AsyncMock(return_value={"available": False}))
    monkeypatch.setattr(harness, "_workspace_section", lambda: {"available": False})
    monkeypatch.setattr(harness.interop_mod, "state_summary", lambda: {"harnesses": []})
    # Sized so dropping the names alone takes the payload back under the cap.
    monkeypatch.setattr(
        harness,
        "_installed_inventory",
        lambda: {"skills_by_source": {"local": {"count": 1, "names": ["n" * 2200]}}},
    )

    result = await harness.harness_status()
    payload = json.loads(result)

    assert len(result) <= harness._MAX_STATUS_BYTES
    assert payload["learning"]["notes"] == {"style:tone": "Speak plainly"}
    assert "notes_omitted" not in payload["learning"]
    # The duplicated feature map survives too: it is only dropped when the
    # payload is still over the cap without the installed names.
    assert "mcp_call" in payload["features"]["mcp"]
    # The recoverable section did drop, with an honest marker.
    assert payload["installed"]["skills_by_source"]["local"]["count"] == 1
    assert "names" not in payload["installed"]["skills_by_source"]["local"]
    assert payload["installed"]["installed_omitted"] == ["local"]


# ---------------------------------------------------------------------------
# 2. firecrawl lifecycle
# ---------------------------------------------------------------------------


def _lifecycle_env(tmp_path, monkeypatch, *, boot_sleep: str = "0"):
    """Temp machina_home + stub limactl/up.sh; returns (fl, trace, limactl)."""
    from machina.core import firecrawl_lifecycle as fl
    from machina.core.config import config

    monkeypatch.setattr(config, "machina_home", str(tmp_path / "state"))
    monkeypatch.setattr(fl, "_warned", set())
    monkeypatch.setattr(fl, "_started_pid", None)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("MACHINA_HOME", raising=False)
    monkeypatch.delenv("MACHINA_MACHINA_HOME", raising=False)

    trace = tmp_path / "trace.log"
    firecrawl_dir = tmp_path / "firecrawl"
    firecrawl_dir.mkdir()
    up_script = firecrawl_dir / "up.sh"
    up_script.write_text(
        "#!/bin/sh\n"
        f'echo "up.sh $*" >> "{trace}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    up_script.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    limactl = bin_dir / "limactl"
    limactl.write_text(
        "#!/bin/sh\n"
        f'echo "limactl $*" >> "{trace}"\n'
        + (f'if [ "$1" = "start" ]; then sleep {boot_sleep}; fi\n' if boot_sleep else "")
        + "exit 0\n",
        encoding="utf-8",
    )
    limactl.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(config, "firecrawl_autostart", True)
    monkeypatch.setattr(config, "firecrawl_stop_on_exit", True)
    monkeypatch.setattr(config, "firecrawl_home", str(firecrawl_dir))
    return fl, trace, limactl


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_state_dir_honors_machina_home_and_falls_back_to_home(tmp_path, monkeypatch):
    from machina.core.config import config

    fl, _, _ = _lifecycle_env(tmp_path, monkeypatch)
    assert fl._state_dir() == tmp_path / "state"
    assert fl._run_dir() == tmp_path / "state" / "run"
    assert fl._log_path().parent == tmp_path / "state" / "logs"

    monkeypatch.setattr(config, "machina_home", "")
    monkeypatch.delenv("MACHINA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert fl._state_dir() == tmp_path / "home" / ".machina"


def test_marker_records_process_and_chain_identity(tmp_path, monkeypatch):
    fl, _, _ = _lifecycle_env(tmp_path, monkeypatch)
    marker = fl._marker_path(os.getpid())
    fl._write_marker(marker, os.getpid(), child_pid=1234567, child_started="token")
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["process_started"] == fl._process_started(os.getpid())
    assert fl._process_started(os.getpid()) != ""
    assert data["child_pid"] == 1234567
    assert data["child_started"] == "token"


def test_prune_rejects_recycled_pids_and_keeps_verified_owners(tmp_path, monkeypatch):
    fl, _, _ = _lifecycle_env(tmp_path, monkeypatch)
    run_dir = fl._run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Recycled PID: the process is alive but its start token does not match.
    recycled = fl._marker_path(os.getpid())
    recycled.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "process_started": "Thu Jan  1 00:00:00 1970",
            }
        ),
        encoding="utf-8",
    )

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _wait_for(lambda: fl._process_started(other.pid) != "")
        live = fl._marker_path(other.pid)
        live.write_text(
            json.dumps(
                {
                    "pid": other.pid,
                    "started_at": time.time(),
                    "process_started": fl._process_started(other.pid),
                }
            ),
            encoding="utf-8",
        )

        fl._prune_dead_markers()

        assert not recycled.exists(), "a recycled-PID marker must not pin the stack"
        assert live.exists(), "a verified live session marker must be kept"
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_prune_ages_out_unverifiable_legacy_markers(tmp_path, monkeypatch):
    fl, _, _ = _lifecycle_env(tmp_path, monkeypatch)
    run_dir = fl._run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    fresh = fl._marker_path(os.getpid())
    fresh.write_text(
        json.dumps({"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8"
    )
    fl._prune_dead_markers()
    assert fresh.exists(), "a fresh legacy marker stays conservative"

    aged = fl._marker_path(os.getpid())
    aged.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.time() - fl._MARKER_MAX_AGE_SECONDS - 60,
            }
        ),
        encoding="utf-8",
    )
    fl._prune_dead_markers()
    assert not aged.exists(), "an aged unverifiable marker must not block teardown forever"


def test_terminate_chain_stops_only_its_verified_process_group(tmp_path, monkeypatch):
    fl, _, _ = _lifecycle_env(tmp_path, monkeypatch)
    child = subprocess.Popen(["/bin/sh", "-c", "sleep 30"], start_new_session=True)
    try:
        started = fl._process_started(child.pid)
        assert started

        # Identity mismatch (recycled PID): nothing is signalled.
        fl._terminate_chain(child.pid, "Thu Jan  1 00:00:00 1970")
        assert child.poll() is None

        fl._terminate_chain(child.pid, started)
        child.wait(timeout=10)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_fence_denies_up_without_a_live_marker(tmp_path, monkeypatch):
    fl, trace, limactl = _lifecycle_env(tmp_path, monkeypatch)
    up_script = Path(fl._firecrawl_home()) / "up.sh"
    command = fl._start_command(
        str(limactl),
        fl._launcher_argv(up_script, "up"),
    )

    completed = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=15
    )

    assert completed.returncode == 1
    assert "skipping up.sh up" in completed.stdout
    assert trace.read_text(encoding="utf-8").splitlines() == ["limactl start docker"]


def test_fence_allows_up_with_a_live_marker(tmp_path, monkeypatch):
    fl, trace, limactl = _lifecycle_env(tmp_path, monkeypatch)
    run_dir = fl._run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = fl._marker_path(os.getpid())
    fl._write_marker(marker, os.getpid())

    up_script = Path(fl._firecrawl_home()) / "up.sh"
    command = fl._start_command(
        str(limactl),
        fl._launcher_argv(up_script, "up"),
    )
    completed = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=15
    )

    assert completed.returncode == 0
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "limactl start docker",
        "up.sh up",
    ]


def test_fast_quit_during_boot_is_fenced_even_when_the_chain_survives(tmp_path, monkeypatch):
    """RED before the fence: a quit during the boot left the detached chain to
    run ``up.sh up`` afterwards with no marker (containers + VM orphaned)."""
    fl, trace, _ = _lifecycle_env(tmp_path, monkeypatch, boot_sleep="1")

    # The detached chain survives this stop (kill misses): only the fence can
    # prevent `up.sh up` after the marker is gone.
    monkeypatch.setattr(fl, "_terminate_chain", lambda *args, **kwargs: None)

    assert fl.start_if_enabled() is True
    assert fl.stop_if_enabled() is True

    def _fence_ran() -> bool:
        log = fl._log_path()
        return log.exists() and "skipping up.sh up" in log.read_text(encoding="utf-8")

    assert _wait_for(_fence_ran), "the chain never hit the marker fence"
    assert "up.sh up" not in trace.read_text(encoding="utf-8")
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert "up.sh down" in lines
    assert "limactl stop docker" in lines


def test_fast_quit_stops_the_still_booting_chain_before_teardown(tmp_path, monkeypatch):
    """The teardown must signal the recorded chain (kill path, not fence)."""
    fl, trace, _ = _lifecycle_env(tmp_path, monkeypatch, boot_sleep="30")

    assert fl.start_if_enabled() is True
    marker = fl._marker_path(os.getpid())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert isinstance(data["child_pid"], int) and data["child_started"]

    assert fl.stop_if_enabled() is True
    assert _wait_for(lambda: "up.sh down" in trace.read_text(encoding="utf-8"))
    time.sleep(1.5)  # the stub limactl sleeps 30 s; give the kill time to hold
    assert "up.sh up" not in trace.read_text(encoding="utf-8")
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(data["child_pid"])],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert state == "" or state.startswith("Z"), f"the boot chain still runs ({state})"


# ---------------------------------------------------------------------------
# 3. benchmark DevTools resolver + section loop
# ---------------------------------------------------------------------------


def _devtools_like(root: Path) -> Path:
    from machina.research import benchmark

    (root / "tests").mkdir(parents=True)
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    for required in benchmark._DEVTOOLS_REQUIRED_TESTS:
        (root / required).write_text("", encoding="utf-8")
    return root


def test_devtools_resolver_requires_the_harness_modules(tmp_path):
    from machina.research import benchmark

    candidate = tmp_path / "checkout"
    (candidate / "tests").mkdir(parents=True)
    (candidate / "tests" / "__init__.py").write_text("", encoding="utf-8")
    assert benchmark._is_devtools_checkout(candidate) is False

    (candidate / "tests" / "test_replay_harness.py").write_text("", encoding="utf-8")
    assert benchmark._is_devtools_checkout(candidate) is False

    (candidate / "tests" / "test_retrieval_eval.py").write_text("", encoding="utf-8")
    assert benchmark._is_devtools_checkout(candidate) is True


def test_devtools_resolver_prefers_siblings_over_cwd(tmp_path, monkeypatch):
    from machina.research import benchmark

    repo = tmp_path / "Machina"
    repo.mkdir()
    sibling = _devtools_like(tmp_path / "Machina-DevTools")
    cwd = tmp_path / "cwd-with-tests-only"
    (cwd / "tests").mkdir(parents=True)
    (cwd / "tests" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(benchmark, "_REPO_ROOT", repo)
    monkeypatch.delenv("MACHINA_DEVTOOLS", raising=False)
    monkeypatch.chdir(cwd)

    assert benchmark._devtools_root_or_none() == sibling.resolve()


def test_devtools_resolver_env_override_wins(tmp_path, monkeypatch):
    from machina.research import benchmark

    env_checkout = _devtools_like(tmp_path / "env-devtools")
    monkeypatch.setenv("MACHINA_DEVTOOLS", str(env_checkout))
    assert benchmark._devtools_root_or_none() == env_checkout.resolve()


def test_ensure_tests_importable_raises_runtime_error_not_system_exit(monkeypatch):
    from machina.research import benchmark

    monkeypatch.setattr(benchmark, "_devtools_root_or_none", lambda: None)
    with pytest.raises(RuntimeError, match="Machina-DevTools"):
        benchmark._ensure_tests_importable()


def test_system_exit_section_records_an_error_and_the_run_continues(tmp_path, monkeypatch):
    """RED before the hardening: a SystemExit from one section aborted every
    later section instead of recording the error payload and continuing."""
    from machina.research import benchmark

    def explode():
        raise SystemExit("fixture exit")

    monkeypatch.setattr(benchmark, "_environment", explode)
    monkeypatch.setattr(benchmark, "_incident_replay", lambda: {"ran": True})
    out = tmp_path / "bench.json"

    rc = benchmark.main(["--section", "environment,replay", "--json", str(out)])

    assert rc == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "SystemExit" in data["environment"]["error"]
    assert data["replay"]["ran"] is True


# ---------------------------------------------------------------------------
# 4. RAG evaluation default dataset
# ---------------------------------------------------------------------------


def test_rag_evaluation_default_dataset_resolves_through_devtools(tmp_path, monkeypatch):
    from machina.rag import evaluation
    from machina.research import benchmark

    monkeypatch.setattr(benchmark, "_devtools_root_or_none", lambda: tmp_path)
    assert evaluation._devtools_dataset_path() == (
        tmp_path / "tests" / "data" / "rag" / "machina_docs_v1.json"
    )

    monkeypatch.setattr(benchmark, "_devtools_root_or_none", lambda: None)
    assert evaluation._devtools_dataset_path() is None


def test_rag_evaluation_default_dataset_exists_in_this_checkout():
    from machina.rag import evaluation

    path = evaluation._devtools_dataset_path()
    assert path is not None, "run the suite from (or next to) the DevTools checkout"
    assert path.is_file()


# ---------------------------------------------------------------------------
# 5. Approval gate: disabled gate returns the durable outcome
# ---------------------------------------------------------------------------

_APPROVAL_TARGETS = (
    "machina.memory.store.db",
    "machina.memory.identity.db",
    "machina.memory.approval.db",
)


async def test_disabled_gate_submit_returns_the_stored_outcome(db_pool, monkeypatch):
    """RED before the fix: submit() auto-approved but returned status=pending
    with memory_id=None for a memory that really was stored."""
    from machina.core.config import config
    from machina.memory.approval import ApprovalStatus, MemoryApprovalGate
    from machina.memory.models import MemoryEntry

    for target in _APPROVAL_TARGETS:
        monkeypatch.setattr(target, db_pool)
    monkeypatch.setattr(config, "memory_approval_enabled", False, raising=False)
    gate = MemoryApprovalGate(enabled=False)
    agent_id = f"prewave-approval-{uuid.uuid4()}"
    try:
        result = await gate.submit("Gate-disabled facts are durable", "fact", agent_id=agent_id)

        assert isinstance(result, MemoryEntry), "the outcome must name a stored memory"
        row = await db_pool.fetchrow(
            "SELECT status, memory_id FROM pending_memories WHERE agent_id = $1", agent_id
        )
        assert row is not None
        assert row["status"] == ApprovalStatus.APPROVED.value
        assert row["memory_id"] == result.id
    finally:
        await db_pool.execute("DELETE FROM memories WHERE agent_id = $1", agent_id)
        await db_pool.execute("DELETE FROM pending_memories WHERE agent_id = $1", agent_id)
