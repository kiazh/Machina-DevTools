"""Unit tests for the pure Content-Length framing layer (``LspFraming``).

No subprocess here: these pin the byte-level contract — arbitrary chunk
boundaries, several frames per read, UTF-8 byte counts, and every bounded
rejection path.
"""

from __future__ import annotations

import json

import pytest

from machina.workspace.lsp import LspFraming, LspProtocolError

REPRESENTATIVE = {
    "jsonrpc": "2.0",
    "id": 7,
    "method": "textDocument/hover",
    "params": {
        "textDocument": {"uri": "file:///tmp/naïve-🦀.py"},
        "position": {"line": 3, "character": 12},
    },
}


def _frame(message: dict) -> bytes:
    return LspFraming.encode(message)


def test_encode_decode_roundtrip():
    framing = LspFraming()
    frames = framing.feed(_frame(REPRESENTATIVE))
    assert frames == [REPRESENTATIVE]
    framing.eof()


def test_headers_are_ascii_crlf_with_byte_length():
    encoded = _frame({"jsonrpc": "2.0", "method": "x", "params": {"t": "héllo 🦀"}})
    header, _, body = encoded.partition(b"\r\n\r\n")
    assert header.startswith(b"Content-Length: ")
    assert header.isascii()
    assert encoded[len(header) + 4 :] == body
    assert int(header.split(b":")[1]) == len(body)
    body_text = body.decode("utf-8")
    assert len(body_text) < len(body)  # multi-byte characters
    assert json.loads(body_text)["params"]["t"] == "héllo 🦀"


def test_split_at_every_byte_boundary():
    encoded = _frame(REPRESENTATIVE)
    for split in range(len(encoded) + 1):
        framing = LspFraming()
        frames = framing.feed(encoded[:split])
        frames.extend(framing.feed(encoded[split:]))
        assert frames == [REPRESENTATIVE], f"split at {split}"
        framing.eof()


def test_byte_at_a_time_matches_whole_frame():
    encoded = _frame(REPRESENTATIVE)
    framing = LspFraming()
    frames = []
    for index in range(len(encoded)):
        frames.extend(framing.feed(encoded[index : index + 1]))
    assert frames == [REPRESENTATIVE]


def test_three_frames_in_one_read():
    messages = [{"jsonrpc": "2.0", "id": n, "result": {"n": n}} for n in (1, 2, 3)]
    framing = LspFraming()
    frames = framing.feed(b"".join(_frame(message) for message in messages))
    assert frames == messages


def test_frames_split_across_reads_with_trailing_partial_frame():
    first = _frame({"jsonrpc": "2.0", "id": 1, "result": "one"})
    second = _frame({"jsonrpc": "2.0", "id": 2, "result": "two"})
    framing = LspFraming()
    assert framing.feed(first + second[:11]) == [{"jsonrpc": "2.0", "id": 1, "result": "one"}]
    assert framing.feed(second[11:]) == [{"jsonrpc": "2.0", "id": 2, "result": "two"}]


def test_empty_feed_returns_no_frames():
    framing = LspFraming()
    assert framing.feed(b"") == []
    framing.eof()


def test_negative_content_length_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: -5\r\n\r\n{}")


def test_duplicate_content_length_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 2\r\nContent-Length: 3\r\n\r\n{}")
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}")


def test_missing_content_length_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Type: application/json\r\n\r\n{}")


def test_non_numeric_content_length_rejected():
    for value in (b"", b"abc", b"1.5", b"0x10"):
        framing = LspFraming()
        with pytest.raises(LspProtocolError):
            framing.feed(b"Content-Length: " + value + b"\r\n\r\n")


def test_oversized_body_rejected():
    framing = LspFraming(max_body=1024)
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 1025\r\n\r\n{}")
    # Default body limit is 8 MiB: a declared 1 GiB frame is refused.
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 1073741824\r\n\r\n")


def test_body_at_the_limit_is_accepted():
    payload = json.dumps({"pad": "x" * 100})
    body = payload.encode("utf-8")
    framing = LspFraming(max_body=len(body))
    frames = framing.feed(b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
    assert frames == [json.loads(payload)]


def test_header_too_large_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"X-Padding: " + b"a" * (16 * 1024) + b"\r\n\r\n{}")


def test_header_at_exact_limit_is_accepted():
    body = b"{}"
    prefix = b"Content-Length: %d\r\nX-Padding: " % len(body)
    header_length = 16 * 1024
    padding = b"a" * (header_length - len(prefix))
    encoded = prefix + padding + b"\r\n\r\n" + body
    framing = LspFraming()
    assert framing.feed(encoded) == [{}]


def test_eof_mid_body_rejected():
    encoded = _frame(REPRESENTATIVE)
    header, _, body = encoded.partition(b"\r\n\r\n")
    framing = LspFraming()
    assert framing.feed(header + b"\r\n\r\n" + body[:3]) == []
    with pytest.raises(LspProtocolError):
        framing.eof()


def test_eof_mid_header_rejected():
    framing = LspFraming()
    assert framing.feed(b"Content-Leng") == []
    with pytest.raises(LspProtocolError):
        framing.eof()


def test_garbage_before_header_is_protocol_error_and_sticky():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b'{"jsonrpc":"2.0","id":1,"method":"x"}\n')
    # The torn stream stays failed instead of silently resynchronising.
    with pytest.raises(LspProtocolError):
        framing.feed(_frame({"jsonrpc": "2.0", "id": 2, "result": None}))
    with pytest.raises(LspProtocolError):
        framing.eof()


def test_invalid_header_line_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Not-A-Header\r\n\r\n{}")


def test_blank_frames_are_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"\r\n\r\n{}")


def test_invalid_or_non_object_body_rejected():
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 7\r\n\r\nnotjson")
    framing = LspFraming()
    with pytest.raises(LspProtocolError):
        framing.feed(b"Content-Length: 2\r\n\r\n[]")


def test_unknown_headers_are_ignored():
    framing = LspFraming()
    body = b'{"jsonrpc":"2.0","id":1,"result":true}'
    encoded = (
        b"Content-Length: %d\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n%s"
        % (len(body), body)
    )
    assert framing.feed(encoded) == [{"jsonrpc": "2.0", "id": 1, "result": True}]
