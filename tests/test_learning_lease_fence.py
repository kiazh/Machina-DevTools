"""Combined #10: a reclaimed learning-review worker cannot publish late.

Worker A claims the row and stalls in the provider call. Its lease expires and
worker B reclaims the same row. When A resumes, every completion update must
require A's claim token; otherwise A's late result overwrites B's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid

import pytest

from machina.core.config import config
from machina.core.models import LLMResponse
from machina.core.session import SessionManager
from machina.skills.learning import LearningReviewer
from machina.skills.registry import SkillRegistry

USER = "Document the deployment workflow and steps"
RESPONSE = "First do health checks then inspect logs and verify readiness."


def _candidate(name: str) -> str:
    return json.dumps(
        {
            "name": name,
            "description": "Lease fence test skill",
            "triggers": ["lease-fence"],
            "content": "Check health and readiness before declaring the task complete.",
        }
    )


class _Provider:
    model = "lease-fence"

    def __init__(self, content: str, entered: asyncio.Event | None = None,
                 release: asyncio.Event | None = None) -> None:
        self.content = content
        self.entered = entered
        self.release = release

    async def complete(self, **kwargs):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return LLMResponse(content=self.content, model=self.model, usage={"total_tokens": 30})


async def test_expired_worker_cannot_overwrite_the_reclaiming_worker(db_pool, monkeypatch, tmp_path):
    monkeypatch.setattr("machina.skills.learning.db", db_pool)
    monkeypatch.setattr("machina.core.session.db", db_pool)
    monkeypatch.setattr(config, "learning_review_auto_apply", False)
    monkeypatch.setattr(config, "learning_review_max_per_session", 5)
    agent_id = f"lease-{uuid.uuid4()}"
    session = await SessionManager().create(title="lease fence", agent_id=agent_id)
    reviewer = LearningReviewer(registry=SkillRegistry(tmp_path / "skills"))
    digest = hashlib.sha256((USER + "\0" + RESPONSE).encode()).hexdigest()
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        worker_a = asyncio.create_task(
            reviewer.review_turn(
                session.id,
                agent_id,
                USER,
                RESPONSE,
                _Provider(_candidate("lease-a"), entered, release),
            )
        )
        await asyncio.wait_for(entered.wait(), 5)

        # A's lease expires while it is still in the provider call.
        await db_pool.execute(
            "UPDATE learning_reviews SET lease_expires_at = now() - interval '1 second' "
            "WHERE session_id = $1 AND turn_hash = $2",
            session.id,
            digest,
        )

        result_b = await reviewer.review_turn(
            session.id, agent_id, USER, RESPONSE, _Provider(_candidate("lease-b"))
        )
        assert result_b is not None and result_b["status"] == "pending"
        assert result_b["name"] == "lease-b"

        release.set()
        result_a = await asyncio.wait_for(worker_a, 5)
        assert result_a is None, "the expired worker published after losing its claim"

        row = await db_pool.fetchrow(
            "SELECT status, name, lease_token FROM learning_reviews "
            "WHERE session_id = $1 AND turn_hash = $2",
            session.id,
            digest,
        )
        assert row["status"] == "pending"
        assert row["name"] == "lease-b"
        assert row["lease_token"] is None
    finally:
        release.set()
        await db_pool.execute("DELETE FROM sessions WHERE id = $1", session.id)
