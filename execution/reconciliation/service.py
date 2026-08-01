"""Controlled reconciliation service over the pure engine and durable OMS.

Only deterministic broker facts are applied automatically. Incomplete snapshots never mutate
OMS state. Conflicts quarantine mapped orders and leave entries frozen for operator review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.reconciliation.engine import reconcile
from execution.reconciliation.models import (
    AUTO_REPAIR,
    MANUAL_REVIEW,
    QUARANTINE,
    BrokerAccountSnapshot,
    InternalPositionSnapshot,
    ReconciliationReport,
)
from execution.reconciliation.store import ReconciliationReportStore


@dataclass(frozen=True)
class ReconciliationRunResult:
    report: ReconciliationReport
    applied_repairs: tuple[str, ...] = ()
    quarantined_orders: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def run_reconciliation(
    *,
    oms_store: OmsStore,
    broker: BrokerAccountSnapshot,
    internal_positions: Iterable[InternalPositionSnapshot],
    internal_cash: float | None = None,
    internal_available_margin: float | None = None,
    report_store: ReconciliationReportStore | None = None,
    apply_repairs: bool = True,
) -> ReconciliationRunResult:
    """Compare, apply deterministic repairs, quarantine conflicts, and report final state."""
    positions = tuple(internal_positions)
    initial = _build_report(
        oms_store=oms_store,
        broker=broker,
        internal_positions=positions,
        internal_cash=internal_cash,
        internal_available_margin=internal_available_margin,
    )
    applied: list[str] = []
    quarantined: list[str] = []
    errors: list[str] = []

    if apply_repairs and broker.complete:
        for repair in initial.repairs:
            try:
                _apply_repair(oms_store, repair)
                applied.append(repair.action_id)
            except Exception as exc:
                errors.append(f"{repair.action_id}:{type(exc).__name__}:{exc}")
                if repair.order_id:
                    try:
                        oms_store.quarantine(
                            repair.order_id,
                            code="RECONCILIATION_REPAIR_FAILED",
                            message=str(exc),
                            actor="reconciliation",
                        )
                        quarantined.append(repair.order_id)
                    except Exception as quarantine_exc:
                        errors.append(
                            f"quarantine:{repair.order_id}:{type(quarantine_exc).__name__}:{quarantine_exc}"
                        )

        for issue in initial.issues:
            if not issue.order_id:
                continue
            try:
                if issue.action == QUARANTINE:
                    oms_store.quarantine(
                        issue.order_id,
                        code=issue.code,
                        message=issue.message,
                        actor="reconciliation",
                    )
                    quarantined.append(issue.order_id)
                elif issue.action == MANUAL_REVIEW:
                    order = oms_store.get(issue.order_id)
                    if order.status == OM.SUBMISSION_PENDING:
                        oms_store.mark_submission_uncertain(
                            issue.order_id,
                            reason=issue.message,
                            error_code=issue.code,
                            actor="reconciliation",
                        )
                    else:
                        oms_store.quarantine(
                            issue.order_id,
                            code=issue.code,
                            message=issue.message,
                            actor="reconciliation",
                        )
                    quarantined.append(issue.order_id)
            except Exception as exc:
                errors.append(f"issue:{issue.issue_id}:{type(exc).__name__}:{exc}")

    final = _build_report(
        oms_store=oms_store,
        broker=broker,
        internal_positions=positions,
        internal_cash=internal_cash,
        internal_available_margin=internal_available_margin,
    )
    if report_store is not None:
        report_store.record(final)
    return ReconciliationRunResult(
        report=final,
        applied_repairs=tuple(dict.fromkeys(applied)),
        quarantined_orders=tuple(dict.fromkeys(quarantined)),
        errors=tuple(errors),
    )


def _build_report(
    *,
    oms_store: OmsStore,
    broker: BrokerAccountSnapshot,
    internal_positions: tuple[InternalPositionSnapshot, ...],
    internal_cash: float | None,
    internal_available_margin: float | None,
) -> ReconciliationReport:
    orders = oms_store.list_orders()
    fills = {order.order_id: oms_store.fills(order.order_id) for order in orders}
    return reconcile(
        internal_orders=orders,
        internal_fills=fills,
        internal_positions=internal_positions,
        broker=broker,
        internal_cash=internal_cash,
        internal_available_margin=internal_available_margin,
    )


def _apply_repair(oms_store: OmsStore, repair) -> None:
    payload = dict(repair.payload or {})
    if repair.action_type == "ACKNOWLEDGE":
        oms_store.acknowledge(
            repair.order_id,
            broker_order_id=repair.broker_order_id,
            external_event_id=repair.external_event_id,
            actor="reconciliation",
        )
        return
    if repair.action_type == "RECORD_FILL":
        oms_store.record_fill(
            repair.order_id,
            external_fill_id=str(payload.get("trade_id") or repair.external_event_id),
            quantity=int(payload.get("quantity") or 0),
            price=float(payload.get("price") or 0),
            filled_at=str(payload.get("executed_at") or "") or None,
            broker_order_id=repair.broker_order_id,
            metadata={"reconciled": True, "broker_snapshot_trade": payload},
            actor="reconciliation",
        )
        return
    if repair.action_type == "REJECT":
        oms_store.reject(
            repair.order_id,
            reason=str(payload.get("reason") or "broker rejected order"),
            external_event_id=repair.external_event_id,
        )
        return
    if repair.action_type == "CANCEL":
        oms_store.cancel(
            repair.order_id,
            reason=str(payload.get("reason") or "broker cancelled order"),
            external_event_id=repair.external_event_id,
        )
        return
    raise ValueError(f"unsupported reconciliation repair {repair.action_type}")
