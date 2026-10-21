"""HTTP readiness probes stay rate-limited while /health remains cheap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from machina.api.rate_limit import RateLimitMiddleware


def _request(path: str, host: str = "203.0.113.7"):
    return SimpleNamespace(url=SimpleNamespace(path=path), client=SimpleNamespace(host=host))


async def _next(_request):
    return "ok"


@pytest.mark.asyncio
async def test_ready_probe_is_throttled_but_health_is_not():
    middleware = RateLimitMiddleware(app=None, requests_per_minute=1, ready_requests_per_minute=1)
    ready = _request("/ready")
    assert await middleware.dispatch(ready, _next) == "ok"
    limited = await middleware.dispatch(ready, _next)
    assert limited.status_code == 429

    health = _request("/health")
    assert await middleware.dispatch(health, _next) == "ok"
    assert await middleware.dispatch(health, _next) == "ok"
