"""Tests for API bug fixes: thread safety, session ownership, auth bypass, and TTL."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

# Set a test API key before importing the app
os.environ["MACHINA_API_KEY"] = "test-key-123"

from machina.api.app import create_app
from machina.core.models import Session

TEST_API_KEY = "test-key-123"


def make_session(**kwargs) -> Session:
    """Create a Session with sensible defaults for testing."""
    defaults = dict(
        id=uuid.uuid4(),
        title="Test Session",
        model="test-model",
        provider="test-provider",
        status="active",
        last_activity=datetime.now(),
    )
    defaults.update(kwargs)
    return Session(**defaults)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
async def client():
    with patch("machina.api.app.db.connect", new_callable=AsyncMock):
        with patch("machina.api.app.db.close", new_callable=AsyncMock):
            app = create_app()
            mock_gw = AsyncMock()
            mock_gw.call.return_value = {"result": "ok"}
            mock_gw.close = AsyncMock()
            app.state.gateway = mock_gw
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c


class TestDocumentOwnership:
    """Document endpoints must verify session ownership."""

    async def test_index_document_rejects_wrong_agent(self, client, auth_headers):
        """POST /api/v1/documents must reject when agent doesn't own the session."""
        session = make_session(agent_id="owner-agent")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            resp = await client.post(
                "/api/v1/documents",
                headers=auth_headers,
                json={
                    "sessionId": str(session.id),
                    "source": "test",
                    "content": "hello",
                    "agent": "intruder-agent",
                },
            )
            assert resp.status_code == 403

    async def test_index_document_accepts_correct_agent(self, client, auth_headers):
        """POST /api/v1/documents must accept when agent owns the session."""
        session = make_session(agent_id="owner-agent")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            with patch(
                "machina.tools.rag.acquire_rag_pipeline", new_callable=AsyncMock
            ) as mock_pipeline:
                mock_pipeline.return_value = AsyncMock()
                mock_pipeline.return_value.index_document = AsyncMock(
                    return_value=[object()]
                )
                resp = await client.post(
                    "/api/v1/documents",
                    headers=auth_headers,
                    json={
                        "sessionId": str(session.id),
                        "source": "test",
                        "content": "hello",
                        "agent": "owner-agent",
                    },
                )
                assert resp.status_code == 200

    async def test_index_document_accepts_default_agent(self, client, auth_headers):
        """POST /api/v1/documents must accept when no agent is specified (uses session's agent_id)."""
        session = make_session(agent_id="harness")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            with patch(
                "machina.tools.rag.acquire_rag_pipeline", new_callable=AsyncMock
            ) as mock_pipeline:
                mock_pipeline.return_value = AsyncMock()
                mock_pipeline.return_value.index_document = AsyncMock(
                    return_value=[object()]
                )
                resp = await client.post(
                    "/api/v1/documents",
                    headers=auth_headers,
                    json={
                        "sessionId": str(session.id),
                        "source": "test",
                        "content": "hello",
                    },
                )
                assert resp.status_code == 200

    async def test_search_documents_rejects_wrong_agent(self, client, auth_headers):
        """GET /api/v1/documents/search must reject when agent doesn't own the session."""
        session = make_session(agent_id="owner-agent")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            resp = await client.get(
                f"/api/v1/documents/search?sessionId={session.id}&query=test&agent=intruder-agent",
                headers=auth_headers,
            )
            assert resp.status_code == 403

    async def test_search_documents_accepts_correct_agent(self, client, auth_headers):
        """GET /api/v1/documents/search must accept when agent owns the session."""
        session = make_session(agent_id="owner-agent")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            with patch(
                "machina.tools.rag.acquire_rag_pipeline", new_callable=AsyncMock
            ) as mock_pipeline:
                mock_pipeline.return_value = AsyncMock()
                mock_pipeline.return_value.search = AsyncMock(return_value=[])
                resp = await client.get(
                    f"/api/v1/documents/search?sessionId={session.id}&query=test&agent=owner-agent",
                    headers=auth_headers,
                )
                assert resp.status_code == 200

    async def test_search_documents_accepts_default_agent(self, client, auth_headers):
        """GET /api/v1/documents/search must accept when no agent is specified."""
        session = make_session(agent_id="harness")

        with patch(
            "machina.api.app.session_manager.get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session
            with patch(
                "machina.tools.rag.acquire_rag_pipeline", new_callable=AsyncMock
            ) as mock_pipeline:
                mock_pipeline.return_value = AsyncMock()
                mock_pipeline.return_value.search = AsyncMock(return_value=[])
                resp = await client.get(
                    f"/api/v1/documents/search?sessionId={session.id}&query=test",
                    headers=auth_headers,
                )
                assert resp.status_code == 200


# ─── Bug 3: MACHINA_API_KEY=disabled bypass ────────────────────────────


class TestAuthDisabledBypass:
    """MACHINA_API_KEY=disabled must not bypass auth."""

    async def test_disabled_key_still_requires_auth(self):
        """When API key is 'disabled', requests without a key must still be rejected."""
        from machina.api.auth import require_api_key

        # Temporarily set the key to 'disabled'
        with patch("machina.api.auth.configured_key", return_value="disabled"):
            # A request without any auth headers should be rejected
            with pytest.raises(HTTPException) as exc_info:
                await require_api_key(authorization=None, x_api_key=None)
            assert exc_info.value.status_code == 401

    async def test_disabled_key_with_correct_key_succeeds(self):
        """When API key is 'disabled', presenting 'disabled' as the key should succeed."""
        from machina.api.auth import require_api_key

        with patch("machina.api.auth.configured_key", return_value="disabled"):
            # Presenting 'disabled' as the key should succeed
            result = await require_api_key(
                authorization="Bearer disabled",
                x_api_key=None,
            )
            assert result is None  # No exception raised

    async def test_disabled_key_with_wrong_key_rejected(self):
        """When API key is 'disabled', presenting a wrong key must be rejected."""
        from machina.api.auth import require_api_key

        with patch("machina.api.auth.configured_key", return_value="disabled"):
            with pytest.raises(HTTPException) as exc_info:
                await require_api_key(
                    authorization="Bearer wrong-key",
                    x_api_key=None,
                )
            assert exc_info.value.status_code == 401

    async def test_empty_configured_key_never_authenticates_empty_input(self):
        """An empty configured value must be unavailable, not an empty-key match."""
        from machina.api.auth import require_api_key

        with patch("machina.api.auth.configured_key", return_value=""):
            with pytest.raises(HTTPException) as exc_info:
                await require_api_key(authorization="Bearer ", x_api_key="")
            assert exc_info.value.status_code == 503
