"""Log redaction must cover the final formatted representation and early imports."""

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from machina.security.secrets import remember_secret_value


@dataclass
class Payload:
    token: str


@pytest.mark.parametrize("kind", ["url", "path", "bytes", "set", "dataclass"])
def test_non_string_log_arguments_are_redacted(caplog, kind):
    secret = "opaque-secret-log-canary"
    remember_secret_value(secret)
    item = {
        "url": httpx.URL(f"https://example.com/?credential={secret}"),
        "path": Path("/tmp") / secret,
        "bytes": secret.encode(),
        "set": {secret},
        "dataclass": Payload(secret),
    }[kind]
    with caplog.at_level(logging.WARNING, logger="machina.object-log-test"):
        logging.getLogger("machina.object-log-test").warning(
            "object=%s count=%d ratio=%.2f", item, 3, 1.25
        )
    assert secret not in caplog.text
    assert "count=3 ratio=1.25" in caplog.records[-1].getMessage()


def test_first_log_before_redaction_import_is_safe():
    script = """
import io, logging, sys
import machina
assert "machina.memory.redaction" not in sys.modules
output = io.StringIO()
logger = logging.getLogger("machina.first-log")
logger.addHandler(logging.StreamHandler(output))
logger.warning("key=%s", "sk-logCanary1234567890abcdefghij")
print(output.getvalue())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "sk-logCanary1234567890abcdefghij" not in result.stdout
    assert "[REDACTED" in result.stdout
