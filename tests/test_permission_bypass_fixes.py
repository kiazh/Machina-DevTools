"""Combined #7/#9: permission classification must fail closed and normalize wrappers.

#7: ``all(...)`` over an empty target list is vacuously true, so a file
operation whose path argument was not recognized could be auto-allowed.
#9: ``uv run`` was not unwrapped before the sensitive-write check, and
``git --no-pager`` was not normalized consistently between the routine-command
and recursive-traversal classifiers.
"""

from __future__ import annotations

import pytest

from machina.permissions.commands import routine_project_command
from machina.permissions.policy import (
    ActionRequest,
    _recursive_traversal,
    _sensitive_capabilities,
    decide,
    normalize_request,
)


def _request(tmp_path, **kwargs):
    return normalize_request(
        ActionRequest(
            mode="ask",
            cwd=str(tmp_path),
            workspace_root=str(tmp_path),
            **kwargs,
        )
    )


@pytest.mark.parametrize("operation", ["file.read", "file.write"])
def test_empty_file_targets_are_not_vacuously_inside_the_workspace(tmp_path, operation):
    decision = decide(_request(tmp_path, operation=operation, targets=[]))
    assert decision.verdict == "pending", (
        f"{operation} with no resolved targets was auto-allowed by vacuous all()"
    )


def test_uv_run_ruff_cannot_write_dot_machina_without_review(tmp_path):
    request = _request(
        tmp_path,
        operation="process.exec",
        argv=["uv", "run", "ruff", "check", "--fix", ".machina/x.py"],
        capabilities=["opaque_execution"],
    )
    capabilities = _sensitive_capabilities(request)
    assert "security_settings" in capabilities
    assert decide(request).verdict == "pending"


def test_git_no_pager_grep_is_normalized_but_still_gated(tmp_path):
    argv = ["git", "--no-pager", "grep", "SECRET"]
    # Both classifiers must see the same subcommand: routine knows it is a
    # project command, traversal knows it walks the repository.
    assert routine_project_command(argv, str(tmp_path), tmp_path) is True
    assert _recursive_traversal(argv, str(tmp_path)) is True
    request = _request(
        tmp_path,
        operation="process.exec",
        argv=argv,
        capabilities=["opaque_execution"],
    )
    assert "opaque_execution_review" in _sensitive_capabilities(request)
    assert decide(request).verdict == "pending"
