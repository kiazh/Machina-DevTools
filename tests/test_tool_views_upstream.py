"""Combined #31: unaudited tools are not treated as unknown-completeness."""

from __future__ import annotations

from machina.core.tool_views import _upstream_truncated


def test_unaudited_tools_are_not_upstream_truncated():
    assert _upstream_truncated("web_extract", "payload") is False
    assert _upstream_truncated("mcp_call", "payload") is False
    assert _upstream_truncated("terminal", "\n[output truncated]") is True
