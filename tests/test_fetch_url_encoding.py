"""Outbound request targets must be valid ASCII without changing URL semantics."""

import pytest
from tests.test_r5_014_host_authority import _install_fake_transport

from machina.security.fetch import FetchError, pinned_fetch


@pytest.mark.parametrize(
    ("url", "host", "target"),
    [
        (
            "http://example.com/café?q=naïve value",
            "example.com",
            b"/caf%C3%A9?q=na%C3%AFve%20value",
        ),
        ("http://bücher.example/", "xn--bcher-kva.example", b"/"),
        ("http://example.com/a%2Fb?q=x%20y&v=%26", "example.com", b"/a%2Fb?q=x%20y&v=%26"),
    ],
)
def test_unicode_urls_are_encoded_for_the_wire(monkeypatch, url, host, target):
    _, sent, _ = _install_fake_transport(
        monkeypatch, {host: (["93.184.216.34"], lambda: (200, {"Content-Length": "2"}, [b"ok"]))}
    )
    assert pinned_fetch(url) == "ok"
    assert sent[0].split(b"\r\n")[0] == b"GET " + target + b" HTTP/1.1"
    assert b"Host: " + host.encode("ascii") + b"\r\n" in sent[0]


@pytest.mark.parametrize("suffix", ["/x\r\nX-Injected: yes", "/x\tvalue", "/x\x00value", "/\ud800"])
def test_invalid_url_text_fails_before_connect(monkeypatch, suffix):
    connected, sent, _ = _install_fake_transport(
        monkeypatch, {"example.com": (["93.184.216.34"], lambda: (200, {}, [b"ok"]))}
    )
    with pytest.raises(FetchError):
        pinned_fetch("http://example.com" + suffix)
    assert connected == sent == []


def test_redirect_target_is_encoded_again(monkeypatch):
    responses = iter(
        [(302, {"Location": "/a b?q=x y"}, []), (200, {"Content-Length": "2"}, [b"ok"])]
    )
    _, sent, _ = _install_fake_transport(
        monkeypatch, {"example.com": (["93.184.216.34"], lambda: next(responses))}
    )
    assert pinned_fetch("http://example.com/") == "ok"
    assert sent[1].startswith(b"GET /a%20b?q=x%20y HTTP/1.1\r\n")
