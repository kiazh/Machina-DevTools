"""Machina-DevTools root conftest — locate the Machina checkout.

The test suite originally lived inside the Machina-Harness repository and
anchored file discovery at ``Path(__file__).resolve().parents[1]`` (that repo
root).  In this repository the same tests must resolve the *harness* checkout
instead.  This shim, loaded before ``tests/conftest.py``, resolves that
checkout once per session and makes it importable:

* ``MACHINA_REPO`` env var wins when set;
* otherwise a sibling checkout is used (``../Machina-Harness``, ``../machina``,
  ``../machina-harness``);
* the resolved checkout is exported as ``MACHINA_REPO`` so tests can opt in,
  and prepended to ``sys.path`` so ``import machina`` works even without an
  editable install (CI additionally does ``pip install -e "$MACHINA_REPO"``).

Execution-day note: individual tests that count ``parents[N]`` from
``__file__``, or read repository files through the process CWD, should be
migrated to a shared helper (``tests.support.repo_paths.repo_root()``) that
uses the same resolution rules.  This shim is the *minimum* that keeps an
unmodified snapshot of the suite runnable from a separate checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEVTOOLS_ROOT = Path(__file__).resolve().parent

_SIBLING_CANDIDATES = (
    "Machina-Harness",
    "machina",
    "machina-harness",
)


def _resolve_repo() -> Path:
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


_REPO = _resolve_repo()
os.environ.setdefault("MACHINA_REPO", str(_REPO))

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
