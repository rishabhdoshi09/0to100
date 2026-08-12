"""One explicit read-only Zerodha observation cycle.

The cycle captures and stores broker evidence, derives internal positions from the durable OMS,
reconciles only complete authoritative lanes, and synchronises protection plans. It performs no
broker mutation and exposes no submission capability.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from execution.protection.service import ProtectionSyncResult, sync_protection
from execution.reconciliation.internal_state import (
    InternalStateProjection,
    project_internal_positions,
)
from execution.reconciliation.service import (
    ReconciliationRunResult,
    run_reconciliation,
)
from execution.reconciliation.snapshot_store import (
    BrokerSnapshotStore,
    capture_and_store_zerodha_snapshot,
)


@dataclass(frozen=True)
class ZerodhaObservationResult:
    snapshot_id: str
    snapshot_complete: bool
    internal_state: InternalStateProjection
    reconciliation: ReconciliationRunResult
    protection: ProtectionSyncResult
    entries_allowed: bool
    blockers: tuple[str, ...]


def run_zerodha_observation_cycle(
    *,
    client,
    oms_store,
    protection_store,
    snapshot_store: BrokerSnapshotStore,
    report_store,
    observed_at: datetime | str | None = None,
    internal_cash: float | None = None,
    internal_available_margin: float | None = None,
    apply_repairs: bool = True,
) -> ZerodhaObservationResult:
    """Capture, reconcile and verify protection from read-only broker state."""
    bundle = capture_and_store_zerodha_snapshot(
        snapshot_store,
        client,
        observed_at=observed_at,
    )
    internal_state = project_internal_positions(oms_store, protection_store)
    account = bundle.account

    if bundle.protections_complete:
        protected_by_symbol: dict[str, int] = {}
        for protection in bundle.protections:
            if not protection.active:
                continue
            symbol = protection.symbol.upper()
            protected_by_symbol[symbol] = (
                protected_by_symbol.get(symbol, 0) + max(0, int(protection.quantity))
            )
        positions = tuple(
            replace(
                position,
                protected_quantity=min(
                    max(0, int(position.quantity)),
                    protected_by_symbol.get(position.symbol.upper(), 0),
                ),
            )
            for position in account.positions
        )
        account = replace(account, positions=positions)
    else:
        # Position quantity may be present, but protection completeness is part of the
        # authority needed to declare the position lane reconciled for new risk.
        account = replace(
            account,
            positions_complete=False,
            errors=tuple(account.errors) + ("protection snapshot incomplete",),
        )

    reconciliation = run_reconciliation(
        oms_store=oms_store,
        broker=account,
        internal_positions=internal_state.positions,
        internal_cash=internal_cash,
        internal_available_margin=internal_available_margin,
        report_store=report_store,
        apply_repairs=apply_repairs,
    )
    protection = sync_protection(
        oms_store=oms_store,
        protection_store=protection_store,
        broker_protections=bundle.protections,
        broker_snapshot_complete=bundle.protections_complete,
    )

    blockers: list[str] = []
    if not bundle.account.complete:
        blockers.append("BROKER_ACCOUNT_SNAPSHOT_INCOMPLETE")
    if not bundle.protections_complete:
        blockers.append("BROKER_PROTECTION_SNAPSHOT_INCOMPLETE")
    if internal_state.unresolved_order_ids:
        blockers.append("INTERNAL_ORDER_STATE_UNRESOLVED")
    if reconciliation.report.entry_freeze_required:
        blockers.append("RECONCILIATION_ENTRY_FREEZE")
    if protection.entry_freeze_required:
        blockers.append("PROTECTION_ENTRY_FREEZE")
    if reconciliation.errors:
        blockers.append("RECONCILIATION_APPLY_ERROR")
    if protection.errors:
        blockers.append("PROTECTION_SYNC_ERROR")

    unique_blockers = tuple(dict.fromkeys(blockers))
    return ZerodhaObservationResult(
        snapshot_id=bundle.account.snapshot_id,
        snapshot_complete=bundle.complete,
        internal_state=internal_state,
        reconciliation=reconciliation,
        protection=protection,
        entries_allowed=not unique_blockers,
        blockers=unique_blockers,
    )
