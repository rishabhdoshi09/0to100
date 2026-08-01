"""CLI for one explicit read-only Zerodha observation and reconciliation cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.oms.store import OmsStore
from execution.protection.store import ProtectionStore
from execution.reconciliation.snapshot_store import BrokerSnapshotStore
from execution.reconciliation.store import ReconciliationReportStore
from execution.reconciliation.zerodha_cycle import run_zerodha_observation_cycle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Zerodha state read-only, reconcile durable OMS state, and verify "
            "protection. This command cannot place, modify or cancel broker orders."
        )
    )
    parser.add_argument("--oms-db", default="logs/oms/orders.db")
    parser.add_argument("--protection-db", default="logs/protection/plans.db")
    parser.add_argument(
        "--snapshot-db", default="logs/reconciliation/broker_snapshots.db"
    )
    parser.add_argument(
        "--report-db", default="logs/reconciliation/reports.db"
    )
    parser.add_argument("--observed-at", default=None)
    parser.add_argument("--internal-cash", type=float, default=None)
    parser.add_argument("--internal-margin", type=float, default=None)
    parser.add_argument(
        "--no-repairs",
        action="store_true",
        help="Compare only; do not apply deterministic internal OMS catch-up repairs",
    )
    parser.add_argument(
        "--require-entry-ready",
        action="store_true",
        help="Exit non-zero unless the observation has no entry blockers",
    )
    return parser


def main(argv=None, *, client=None) -> int:
    args = build_parser().parse_args(argv)
    oms = OmsStore(Path(args.oms_db))
    protection = ProtectionStore(Path(args.protection_db))
    snapshots = BrokerSnapshotStore(Path(args.snapshot_db))
    reports = ReconciliationReportStore(Path(args.report_db))
    result = run_zerodha_observation_cycle(
        client=client,
        oms_store=oms,
        protection_store=protection,
        snapshot_store=snapshots,
        report_store=reports,
        observed_at=args.observed_at,
        internal_cash=args.internal_cash,
        internal_available_margin=args.internal_margin,
        apply_repairs=not args.no_repairs,
    )
    payload = {
        "snapshot_id": result.snapshot_id,
        "snapshot_complete": result.snapshot_complete,
        "entries_allowed": result.entries_allowed,
        "blockers": list(result.blockers),
        "internal_positions": [
            position.as_dict() for position in result.internal_state.positions
        ],
        "unresolved_order_ids": list(result.internal_state.unresolved_order_ids),
        "reconciliation": {
            "report_id": result.reconciliation.report.report_id,
            "status": result.reconciliation.report.status,
            "entry_freeze_required": (
                result.reconciliation.report.entry_freeze_required
            ),
            "applied_repairs": list(result.reconciliation.applied_repairs),
            "quarantined_orders": list(result.reconciliation.quarantined_orders),
            "errors": list(result.reconciliation.errors),
        },
        "protection": {
            "entry_freeze_required": result.protection.entry_freeze_required,
            "verified_plans": list(result.protection.verified_plans),
            "recovery_plans": list(result.protection.recovery_plans),
            "orphan_plans": list(result.protection.orphan_plans),
            "errors": list(result.protection.errors),
        },
        "broker_mutations_enabled": False,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 2 if args.require_entry_ready and not result.entries_allowed else 0


if __name__ == "__main__":
    raise SystemExit(main())
