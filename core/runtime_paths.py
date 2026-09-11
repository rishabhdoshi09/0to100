"""Single source of truth for where QuantTerm's mutable runtime state lives.

Every durable artifact the product writes — logs, SQLite stores, scans,
recommendations, evidence — belongs under one root. That root is the repository
checkout in production and a temporary directory under test.

Why this exists: modules resolved their own paths from ``__file__``, so the
test suite wrote real artifacts into the developer's checkout. A long-term
shortlist containing the fixture symbols ``AAA`` and ``BBB`` reached
``logs/product/latest_long_term_scan.json``, and the running desk then read
them as genuine evidence and told the operator a scan was available. Research
evidence for a fabricated ticker (``QTTRUTHA``) landed in the real evidence
store the same way.

The root is resolved on every call rather than cached at import, because tests
set ``QT_RUNTIME_ROOT`` after modules are already imported.
"""
from __future__ import annotations

import os
from pathlib import Path

# The checkout itself. Production writes here; this is also the tree the test
# guard protects.
REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_VAR = "QT_RUNTIME_ROOT"


def runtime_root() -> Path:
    """Base directory for all mutable runtime state."""
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return REPO_ROOT


def runtime_path(*parts: str | os.PathLike[str]) -> Path:
    """A path under the runtime root. Nothing is created here."""
    return runtime_root().joinpath(*[str(part) for part in parts])


def logs_dir() -> Path:
    """The runtime ``logs/`` directory."""
    return runtime_path("logs")


def logs_path(*parts: str | os.PathLike[str]) -> Path:
    """A path under the runtime ``logs/`` directory."""
    return logs_dir().joinpath(*[str(part) for part in parts])


def ensure_logs_path(*parts: str | os.PathLike[str]) -> Path:
    """Like :func:`logs_path`, but the parent directory is created."""
    target = logs_path(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def is_redirected() -> bool:
    """True when runtime state is pointed somewhere other than the checkout."""
    return runtime_root().resolve() != REPO_ROOT.resolve()
