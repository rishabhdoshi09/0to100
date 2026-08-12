"""Command-line entrypoint for one explicit read-only Zerodha snapshot capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.reconciliation.snapshot_store import (
    BrokerSnapshotStore,
    capture_and_store_zerodha_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Zerodha orders, trades, positions, margins and GTT state read-only; "
            "persist the result for reconciliation. No order is placed or modified."
        )
    )
    parser.add_argument(
        "--db",
        default="logs/reconciliation/broker_snapshots.db",
        help="SQLite snapshot ledger path",
    )
    parser.add_argument(
        "--observed-at",
        default=None,
        help="Optional ISO timestamp for deterministic testing; normally omitted",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless both account and protection lanes are complete",
    )
    return parser


def main(argv=None, *, client=None) -> int:
    args = build_parser().parse_args(argv)
    store = BrokerSnapshotStore(Path(args.db))
    bundle = capture_and_store_zerodha_snapshot(
        store,
        client,
        observed_at=args.observed_at,
    )
    payload = {
        "snapshot_id": bundle.account.snapshot_id,
        "observed_at": bundle.account.observed_at,
        "account_complete": bundle.account.complete,
        "protections_complete": bundle.protections_complete,
        "complete": bundle.complete,
        "orders": len(bundle.account.orders),
        "trades": len(bundle.account.trades),
        "positions": len(bundle.account.positions),
        "protections": len(bundle.protections),
        "errors": list(bundle.errors),
        "store": store.summary(),
        "mutations_enabled": False,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 2 if args.require_complete and not bundle.complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
