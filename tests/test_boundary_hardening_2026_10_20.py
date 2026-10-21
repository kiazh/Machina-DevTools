"""Regression checks for boundary hardening found in the full review."""

from __future__ import annotations

from collections import deque

from machina.api.rate_limit import RateLimitMiddleware
from machina.connectors.mcp_client import _bounded_json
from machina.gateway.__main__ import _request_method


def test_gateway_method_probe_rejects_huge_and_deep_json():
    assert _request_method('{"jsonrpc":"2.0","id":' + "9" * 5000 + ',"method":"initialize"}') is None
    assert _request_method("[" * 2000 + "]" * 2000) is None


def test_bounded_json_does_not_materialize_oversized_values():
    assert _bounded_json({"ok": "value"}, 1024) == '{"ok": "value"}'
    assert _bounded_json({"blob": "x" * 2048}, 1024) is None


def test_rate_limit_prunes_ready_bucket_map():
    middleware = RateLimitMiddleware(app=None, requests_per_minute=10)
    middleware._last_prune = 0.0
    now = 61.0
    middleware._ready_calls.update(
        {f"client-{i}": deque([now - 1]) for i in range(10_001)}
    )
    middleware._prune(middleware._ready_calls, now)
    assert len(middleware._ready_calls) <= 10_000
