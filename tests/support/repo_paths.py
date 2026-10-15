"""Resolve files that live in the Machina checkout under test.

The suite was written inside Machina-Harness, where ``Path(__file__)``-relative
anchors pointed at the repository root.  In this repository those tests run
from a separate checkout, so anchors must resolve the *harness* checkout
instead.  This helper mirrors the resolution in the root ``conftest.py``
(keep the two in sync):

* ``MACHINA_REPO`` wins when set (the root conftest exports it, so it is
  always set under pytest);
* otherwise a sibling checkout is used (``../Machina-Harness``, ``../machina``,
  ``../machina-harness``).

Use ``repo_root()`` / ``repo_path(rel)`` for repository files instead of
``Path(__file__).resolve().parents[1]`` or the process CWD: in this checkout
the DevTools root is not the harness root, and only a subset of harness
artifacts are symlinked in by ``bootstrap-links.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_path", "repo_root"]

_DEVTOOLS_ROOT = Path(__file__).resolve().parents[2]

_SIBLING_CANDIDATES = (
    "Machina-Harness",
    "machina",
    "machina-harness",
)


def repo_root() -> Path:
    """Absolute path of the Machina (harness) checkout under test."""
    candidates: list[Path] = []
    env = os.environ.get("MACHINA_REPO")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(_DEVTOOLS_ROOT.parent / name for name in _SIBLING_CANDIDATES)
    for candidate in candidates:
        if (candidate / "machina" / "__init__.py").is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise RuntimeError(
        "Machina checkout not found. Set MACHINA_REPO or place a sibling "
        f"checkout next to {_DEVTOOLS_ROOT}.\nTried:\n  {tried}"
    )


def repo_path(rel: str | os.PathLike[str]) -> Path:
    """Path of ``rel`` inside the Machina checkout (``repo_root() / rel``)."""
    return repo_root() / rel
