"""Combined #41: large MCP structuredContent is truncated, not discarded."""

from __future__ import annotations

import json
from types import SimpleNamespace

from machina.connectors.mcp_client import MCPConnection


def test_oversized_structured_content_becomes_bounded_text():
    connection = MCPConnection.__new__(MCPConnection)
    connection.config = SimpleNamespace(result_limit=200)
    connection._redact = lambda text: text
    connection._secret_values = set()

    structured = {"payload": "a" * 1000}
    result = SimpleNamespace(content=[], structuredContent=structured, is_error=False)
    mcp_types = SimpleNamespace(
        TextContent=type("TextContent", (), {}),
        ImageContent=type("ImageContent", (), {}),
        AudioContent=type("AudioContent", (), {}),
        EmbeddedResource=type("EmbeddedResource", (), {}),
        ResourceLink=type("ResourceLink", (), {}),
    )

    outcome = connection._normalize("tool", result, mcp_types)

    assert outcome.truncated is True
    assert outcome.structured is None
    assert outcome.text.startswith('{"payload":')
    assert '"aaaa' in outcome.text
    assert len(outcome.text.encode("utf-8")) <= 200
