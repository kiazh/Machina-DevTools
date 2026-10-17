"""HTTP v2 approval normalization preserves the originating card metadata."""

import pytest

from machina.api.routes.shared import _serialize_event
from machina.core.models import StreamEvent


@pytest.mark.parametrize("durable", [False, True, None])
@pytest.mark.parametrize("nested", [False, True])
def test_http_v2_approval_preserves_card_metadata(durable, nested):
    proposal = {"tool": "write_file", "argv": [], "content_digest": "reviewed"}
    card = {
        "turn_id": "origin",
        "file_diff": {"before": "old", "after": "new"},
        "durable": durable,
        "status": "pending",
    }
    card.update({"proposal": proposal} if nested else proposal)
    event = StreamEvent(
        type="needs_approval", content="request", tool_name="file.write", tool_args=card
    )
    data = _serialize_event(event, protocol_version=2, session_id="session", turn_id="current")
    assert data["originTurnId"] == "origin"
    assert data["fileDiff"] == card["file_diff"]
    assert data["durable"] is durable
    assert data["tool"] == "write_file"


def test_http_v2_approval_defaults_to_the_emitting_turn():
    event = StreamEvent(
        type="needs_approval", content="request", tool_name="file.write", tool_args={}
    )
    data = _serialize_event(event, protocol_version=2, session_id="session", turn_id="current")
    assert data["originTurnId"] == "current"
