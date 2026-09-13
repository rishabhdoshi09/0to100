#!/usr/bin/env python3
"""Print a non-destructive audit of every SQLite store under QuantTerm logs."""
from __future__ import annotations

import json

from product.sqlite_audit import audit_runtime, summary


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
