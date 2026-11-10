"""Re-parse stored official XBRL filings with the current parser (no network).

    python scripts/reparse_pit_xbrl.py --dry-run
    python scripts/reparse_pit_xbrl.py --warehouse /path/to/pit_warehouse.db

Writes superseding revisions; originals are kept for audit. Run while the desk
is stopped (it writes the production warehouse).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    from product.pit_reparse import reparse_legacy_xbrl

    report = reparse_legacy_xbrl(
        warehouse_path=args.warehouse,
        evidence_root=args.evidence_root,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
