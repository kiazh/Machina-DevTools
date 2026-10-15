"""Normalize captured CLI output before asserting on its visible text.

Typer renders Rich-styled help whenever ``GITHUB_ACTIONS``, ``FORCE_COLOR``
or ``PY_COLORS`` is set (``typer.rich_utils.FORCE_TERMINAL``) — GitHub
Actions sets ``GITHUB_ACTIONS`` for every step.  Rich applies its styles
*inside* option tokens (e.g. ``-\\x1b[0m-host`` for ``--host``), so a raw
substring check for ``--host`` fails in CI even though the help clearly
shows it.  Strip the escape sequences first and assert on the visible text.
"""

from __future__ import annotations

import re

# CSI sequences (colors, cursor movement) and OSC sequences (hyperlinks).
_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x1b\x07]*(?:\x07|\x1b\\)"
)


def strip_ansi(text: str) -> str:
    """Return ``text`` with ANSI escape sequences removed."""
    return _ANSI_ESCAPE.sub("", text)
