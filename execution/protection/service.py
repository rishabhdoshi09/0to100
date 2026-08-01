"""Protection Manager orchestration over OMS fills and supplied broker protection snapshots.

The service creates or resizes durable plans, updates OMS protection state only after fills,
and reconciles external protection facts. It contains no broker submission or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.protection import models as P
from execution.protection.store import ProtectionStore

_EXPECTED_AT_BROKER = frozenset({
    P.SUBMISSION_PENDING,
    P.ACTIVE,
    P.VERIFIED,
    P.ADJUSTMENT_REQUIRED,
    P.CANCEL_PENDING,
    P.RECOVERY_REQUIRED,
})


@dataclass(frozen=True)
class ProtectionSyncResult:
    plans_created_or_updated: tuple[str, ...]
    verified_plans: tuple[str, ...]
    recovery_plans: tuple[str, ...]
    orphan_plans: tuple[str, ...]
    oms_protected_orders: tuple[str, ...]
    errors: tuple[str, ...]
    entry_freeze_required: bool


def sync_protection(
    *,
    oms_store: OmsStore,
    protection_store: ProtectionStore,
    broker_protections: Iterable[P.BrokerProtectionSnapshot] = (),
    broker_snapshot_complete: bool = False,
) -> ProtectionSyncResult:
    """Synchronise protection plans from durable OMS state and optional broker facts."""
    plans_touched: list[str] = []
    verified: list[str] = []
    recovery: list[str] = []
    orphans: list[str] = []
    oms_protected: list[str] = []
    errors: list[str] = []

    orders = oms_store.list_orders()
    order_by_id = {order.order_id: order for order in orders}
    for order in orders:
        try:
            plan = protection_store.get_by_order(order.order_id)
            if order.status == OM.CLOSED:
                if plan is None:
                    continue
                if plan.status in {P.REQUIRED, P.FAILED}:
                    plan = protection_store.mark_cancelled(plan.plan_id, actor="protection_manager")
                elif plan.status not in {P.CANCELLED, P.CANCEL_PENDING}:
                    plan = protection_store.request_cancel(
                        plan.plan_id,
                        reason="OMS position closed",
                    )
                plans_touched.append(plan.plan_id)
                continue
            if order.status in OM.TERMINAL_STATUSES:
                continue
            if order.filled_quantity <= 0:
                continue

            plan = protection_store.ensure_for_order(order)
            plans_touched.append(plan.plan_id)

            # OMS exposes the full-fill protection lifecycle. Partial fills remain
            # PARTIALLY_FILLED while their independent plan protects cumulative fills.
            if order.status == OM.FILLED:
                order = oms_store.mark_protection_pending(order.order_id)
            if plan.fully_protected and order.status == OM.PROTECTION_PENDING:
                oms_store.mark_protected(
                    order.order_id,
                    protection_reference=plan.broker_protection_id or plan.stop_reference,
                    external_event_id=f"protect:{plan.plan_id}:{plan.version}",
                )
                oms_protected.append(order.order_id)
        except Exception as exc:
            errors.append(f"oms:{order.order_id}:{type(exc).__name__}:{exc}")
            plan = protection_store.get_by_order(order.order_id)
            if plan is not None:
                try:
                    plan = protection_store.require_recovery(
                        plan.plan_id,
                        code="PROTECTION_SYNC_FAILED",
                        message=str(exc),
                    )
                    recovery.append(plan.plan_id)
                except Exception as recovery_exc:
                    errors.append(
                        f"recovery:{order.order_id}:{type(recovery_exc).__name__}:{recovery_exc}"
                    )

    broker_items = tuple(broker_protections)
    if broker_snapshot_complete:
        by_id = {
            item.broker_protection_id: item
            for item in broker_items
            if item.broker_protection_id
        }
        by_order = {item.order_id: item for item in broker_items if item.order_id}
        matched_ids: set[str] = set()
        for plan in protection_store.list_plans():
            if plan.status == P.CANCELLED or plan.status == P.ORPHANED:
                continue
            broker = None
            if plan.broker_protection_id:
                broker = by_id.get(plan.broker_protection_id)
            if broker is None:
                broker = by_order.get(plan.order_id)
            if broker is None:
                if plan.status in _EXPECTED_AT_BROKER:
                    try:
                        updated = protection_store.require_recovery(
                            plan.plan_id,
                            code="BROKER_PROTECTION_MISSING",
                            message="complete broker snapshot has no matching protection",
                        )
                        recovery.append(updated.plan_id)
                    except Exception as exc:
                        errors.append(f"missing:{plan.plan_id}:{type(exc).__name__}:{exc}")
                continue
            matched_ids.add(broker.broker_protection_id)
            try:
                if plan.status == P.CANCEL_PENDING:
                    if not broker.active:
                        protection_store.mark_cancelled(
                            plan.plan_id,
                            external_event_id=f"cancel:{broker.broker_protection_id}:{broker.updated_at}",
                            actor="protection_reconciliation",
                        )
                    continue
                updated = protection_store.verify(
                    plan.plan_id,
                    broker,
                    external_event_id=f"verify:{broker.broker_protection_id}:{broker.updated_at}",
                )
                plans_touched.append(updated.plan_id)
                if updated.status == P.VERIFIED:
                    verified.append(updated.plan_id)
                    order = order_by_id.get(updated.order_id)
                    if order is not None and order.status == OM.FILLED:
                        order = oms_store.mark_protection_pending(order.order_id)
                    if order is not None and order.status == OM.PROTECTION_PENDING:
                        oms_store.mark_protected(
                            order.order_id,
                            protection_reference=(
                                updated.broker_protection_id or updated.stop_reference
                            ),
                            external_event_id=f"protect:{updated.plan_id}:{updated.version}",
                        )
                        oms_protected.append(order.order_id)
                elif updated.status == P.RECOVERY_REQUIRED:
                    recovery.append(updated.plan_id)
            except Exception as exc:
                errors.append(f"verify:{plan.plan_id}:{type(exc).__name__}:{exc}")
                try:
                    updated = protection_store.require_recovery(
                        plan.plan_id,
                        code="PROTECTION_VERIFY_FAILED",
                        message=str(exc),
                    )
                    recovery.append(updated.plan_id)
                except Exception:
                    pass

        known_broker_ids = {
            plan.broker_protection_id
            for plan in protection_store.list_plans()
            if plan.broker_protection_id
        }
        for broker in broker_items:
            if not broker.active:
                continue
            if broker.broker_protection_id in matched_ids | known_broker_ids:
                continue
            try:
                orphan = protection_store.record_orphan(broker)
                orphans.append(orphan.plan_id)
            except Exception as exc:
                errors.append(
                    f"orphan:{broker.broker_protection_id}:{type(exc).__name__}:{exc}"
                )

    summary = protection_store.summary()
    entry_freeze = (
        not broker_snapshot_complete
        or bool(summary["entry_freeze_required"])
        or bool(recovery)
        or bool(orphans)
        or bool(errors)
    )
    return ProtectionSyncResult(
        plans_created_or_updated=tuple(dict.fromkeys(plans_touched)),
        verified_plans=tuple(dict.fromkeys(verified)),
        recovery_plans=tuple(dict.fromkeys(recovery)),
        orphan_plans=tuple(dict.fromkeys(orphans)),
        oms_protected_orders=tuple(dict.fromkeys(oms_protected)),
        errors=tuple(errors),
        entry_freeze_required=entry_freeze,
    )
