"""Synchronise simulated PaperBook exits into durable OMS and protection state."""
from __future__ import annotations

from dataclasses import dataclass

from execution.oms import models as OM
from execution.protection import models as PM


@dataclass(frozen=True)
class PaperExitSyncResult:
    order_id: str
    order_status: str
    protection_status: str
    already_closed: bool = False


def sync_paper_close(pipeline, trade) -> PaperExitSyncResult | None:
    """Close the one active OMS owner of a simulated strategy-symbol position.

    The function is idempotent. A missing active owner is a no-op because an earlier invocation
    may already have closed it. Multiple active owners are quarantined instead of guessed.
    """
    candidates = [
        order for order in pipeline.oms.list_orders()
        if order.strategy_id == trade.strategy_id
        and order.symbol == trade.symbol
        and order.filled_quantity > 0
        and order.status not in OM.TERMINAL_STATUSES
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        for order in candidates:
            pipeline.oms.quarantine(
                order.order_id,
                code="MULTIPLE_ACTIVE_PAPER_POSITION_OWNERS",
                message=f"{trade.strategy_id}:{trade.symbol} has {len(candidates)} active OMS owners",
                actor="paper_exit",
            )
        raise RuntimeError("multiple active OMS owners for one PaperBook position")

    order = candidates[0]
    if order.status in {OM.UNKNOWN, OM.QUARANTINED, OM.RECOVERY_REQUIRED}:
        return PaperExitSyncResult(
            order_id=order.order_id,
            order_status=order.status,
            protection_status="RECOVERY_REQUIRED",
        )

    plan = pipeline.protection.get_by_order(order.order_id)
    protection_status = plan.status if plan is not None else "MISSING"
    if plan is not None and plan.status not in {PM.CANCELLED, PM.CANCEL_PENDING}:
        plan = pipeline.protection.request_cancel(
            plan.plan_id,
            reason=f"paper position closed: {trade.exit_reason}",
            actor="paper_exit",
        )
    if plan is not None and plan.status == PM.CANCEL_PENDING:
        plan = pipeline.protection.mark_cancelled(
            plan.plan_id,
            external_event_id=f"paper-protection-cancelled-{order.order_id}",
            actor="paper_protection",
        )
    if plan is not None:
        protection_status = plan.status

    if order.status in {OM.PROTECTED, OM.PROTECTION_PENDING, OM.FILLED}:
        order = pipeline.oms.mark_exit_pending(
            order.order_id,
            reason=trade.exit_reason,
            actor="paper_exit",
        )
    if order.status == OM.EXIT_PENDING:
        order = pipeline.oms.mark_closed(
            order.order_id,
            reason=trade.exit_reason,
            actor="paper_exit",
        )

    return PaperExitSyncResult(
        order_id=order.order_id,
        order_status=order.status,
        protection_status=protection_status,
        already_closed=False,
    )
