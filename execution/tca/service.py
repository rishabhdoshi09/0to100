"""Idempotent TCA generation from canonical intelligence and durable OMS evidence."""
from __future__ import annotations

from typing import Any, Callable

from execution.oms.store import OmsStore
from execution.tca.analyzer import assess_entry_execution
from execution.tca.store import TcaStore


class TradeIntentNotFound(LookupError):
    pass


def assess_oms_order(
    *,
    event_store,
    oms_store: OmsStore,
    tca_store: TcaStore,
    order_id: str,
    submission_reference_price: float | None = None,
    explicit_fees: float = 0.0,
    estimated_spread_bps: float = 0.0,
    estimated_market_impact: float = 0.0,
    opportunity_cost: float = 0.0,
    metadata: dict[str, Any] | None = None,
):
    """Build and persist one assessment from immutable records."""
    order = oms_store.get(order_id)
    intent = next(
        (
            record for record in event_store.of_type("TradeIntent")
            if record.record_id == order.trade_intent_id
        ),
        None,
    )
    if intent is None:
        raise TradeIntentNotFound(order.trade_intent_id)
    assessment = assess_entry_execution(
        intent=intent,
        order=order,
        transitions=oms_store.history(order_id),
        fills=oms_store.fills(order_id),
        submission_reference_price=submission_reference_price,
        explicit_fees=explicit_fees,
        estimated_spread_bps=estimated_spread_bps,
        estimated_market_impact=estimated_market_impact,
        opportunity_cost=opportunity_cost,
        metadata=metadata,
    )
    return tca_store.record(assessment)


def assess_filled_orders(
    *,
    event_store,
    oms_store: OmsStore,
    tca_store: TcaStore,
    submission_reference_price_fn: Callable[[Any], float | None] | None = None,
    explicit_fees_fn: Callable[[Any], float] | None = None,
) -> dict[str, Any]:
    """Assess every OMS order with at least one durable fill; repeated runs are safe."""
    recorded: list[str] = []
    skipped: list[dict[str, str]] = []
    for order in oms_store.list_orders():
        fills = oms_store.fills(order.order_id)
        if not fills:
            continue
        try:
            assessment = assess_oms_order(
                event_store=event_store,
                oms_store=oms_store,
                tca_store=tca_store,
                order_id=order.order_id,
                submission_reference_price=(
                    submission_reference_price_fn(order)
                    if submission_reference_price_fn else None
                ),
                explicit_fees=float(explicit_fees_fn(order)) if explicit_fees_fn else 0.0,
            )
            recorded.append(assessment.assessment_id)
        except Exception as exc:
            skipped.append({
                "order_id": order.order_id,
                "reason": f"{type(exc).__name__}:{exc}",
            })
    return {
        "recorded": list(dict.fromkeys(recorded)),
        "skipped": skipped,
        "recorded_count": len(set(recorded)),
        "skipped_count": len(skipped),
    }
