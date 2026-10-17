"""Shared fakes for handler seams and the loop-wiring suites.

These classes were copy-pasted across several test modules (``FakeGateway`` in
four workspace/skills suites; the scripted/rejecting providers and verified
artifact stores in the B2/B3 loop-wiring suites). They are hoisted here
verbatim (plus the union of their optional attributes) so drift is fixed once.

Nothing here touches a real provider, database, or filesystem.
"""

from __future__ import annotations

import copy
import hashlib

from machina.core.models import LLMResponse, StreamEvent


class FakeGateway:
    """Minimal gateway surface for handler tests.

    ``require_db`` counts calls (skills tests assert the guard ran); workspace
    tests ignore the counter. ``workspace_service`` starts unset so the
    workspace feature can attach one exactly as production does.
    """

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.methods: dict = {}
        self.workspace_service = None
        self.guards = 0

    def require_db(self) -> None:
        self.guards += 1

    def _write(self, frame: dict) -> None:
        self.frames.append(frame)

    def add_method(self, name, handler) -> None:
        self.methods[name] = handler


class ScriptedProvider:
    """Deterministic provider; every call snapshots its kwargs."""

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        on_call=None,
        model: str = "fake",
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.on_call = on_call
        self.model = model

    async def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.on_call is not None:
            self.on_call(len(self.calls))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    async def stream_complete(self, **kwargs):
        response = await self.complete(**kwargs)
        if response.content:
            yield StreamEvent(type="text", content=response.content)
        yield StreamEvent(type="done", response=response)


class RejectingProvider:
    """Fails the test if any seam tries to call a provider itself."""

    model = "reject"

    async def complete(self, **kwargs):
        raise AssertionError("provider must not be called during projection")

    async def stream_complete(self, **kwargs):
        raise AssertionError("provider must not be called during projection")
        yield  # pragma: no cover - generator protocol


class VerifiedStore:
    """Retains stored content, answers the projector's lookup contract."""

    def __init__(self) -> None:
        self.stored: dict[str, str] = {}
        self.lookups: list[str] = []
        self.store_calls = 0

    async def store(self, kind, content, **kwargs):
        self.store_calls += 1
        # Deterministic handle: identical content -> identical handle, so two
        # scripted runs produce byte-identical payloads.
        handle = "hand-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        self.stored[handle] = content
        return {"artifact_id": handle, "kind": kind, "content_bytes": len(content)}

    async def lookup(self, handle, *, session_id, agent_id):
        self.lookups.append(str(handle))
        if str(handle) not in self.stored:
            return None
        return {
            "available": True,
            "complete": True,
            "handle": str(handle),
            "revision": f"rev-{handle}",
            "summary_view": f"verified summary for {handle}",
        }


class VerifiedLookupOnly:
    """Verified lookup for handles it was never asked to retain (corpus ids)."""

    def __init__(self) -> None:
        self.lookups: list[str] = []

    async def lookup(self, handle, *, session_id, agent_id):
        self.lookups.append(str(handle))
        return {
            "available": True,
            "complete": True,
            "handle": str(handle),
            "revision": f"rev-{handle}",
            "summary_view": f"verified summary for {handle}",
        }
