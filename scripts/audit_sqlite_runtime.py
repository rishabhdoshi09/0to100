#!/usr/bin/env python3
"""Print a non-destructive audit of every SQLite store under QuantTerm logs."""
from __future__ import annotations

import json
from pathlib import Path
import sys


# When a script is executed as ``python scripts/foo.py``, Python puts the
# ``scripts`` directory (not the repository root) on sys.path.  QuantTerm's
# application packages live at the repository root, so make that import root
# explicit before importing product.*.  This keeps the CLI usable exactly the
# way operators invoke it on macOS while remaining harmless under ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from product.sqlite_audit import audit_runtime, summary  # noqa: E402


def main() -> int:
    rows = audit_runtime()
    payload = {
        "summary": summary(rows),
        "databases": [row.as_dict() for row in rows],
    }
    print(json.dumps(payload, indent=2))
    # The command is diagnostic.  A nonzero exit means at least one base file
    # could not be validated even in immutable mode; WAL-sidecar constraints do
    # not by themselves fail the command.
    return 2 if int(payload["summary"]["needs_investigation"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
