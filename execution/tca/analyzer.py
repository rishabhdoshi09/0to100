"""Pure entry-execution transaction-cost analysis from immutable intent and OMS evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Iterable

from execution.tca.models import EntryExecutionAssessment


class TcaInputError(ValueError):
    pass


def assess_entry_execution(
    *,
    intent,
    order,
    transitions: Iterable,
    fills: Iterable,
    submission_reference_price: float | None = None,
    explicit_fees: float = 0.0,
    estimated_spread_bps: float = 0.0,
    estimated_market_impact: float = 0.0,
    opportunity_cost: float = 0.0,
    metadata: dict | None = None,
) -> EntryExecutionAssessment:
    """Attribute one entry from signal time through final observed fill.

    The actual fill already embeds spread and market impact. Their estimates are reported as
    diagnostics and are not added again to implementation shortfall. The total is:

      decision-to-submission price cost
      + submission-to-fill price cost
      + explicit fees
      + externally supplied opportunity cost.
    """
    if str(getattr(intent, "record_id", "")) != str(getattr(order, "trade_intent_id", "")):
        raise TcaInputError("intent does not own the supplied OMS order")
    fill_rows = tuple(fills)
    if not fill_rows:
        raise TcaInputError("at least one durable fill is required")
    quantity = sum(int(fill.quantity) for fill in fill_rows)
    if quantity <= 0:
        raise TcaInputError("cumulative fill quantity must be positive")
    fill_value = sum(int(fill.quantity) * float(fill.price) for fill in fill_rows)
    average_fill = fill_value / quantity
    decision_price = float(getattr(intent, "intended_entry", 0.0) or 0.0)
    if decision_price <= 0 or average_fill <= 0:
        raise TcaInputError("decision and fill prices must be positive")
    for name, value in {
        "explicit_fees": explicit_fees,
        "estimated_spread_bps": estimated_spread_bps,
        "estimated_market_impact": estimated_market_impact,
        "opportunity_cost": opportunity_cost,
    }.items():
        if float(value) < 0:
            raise TcaInputError(f"{name} cannot be negative")

    warnings: list[str] = []
    if submission_reference_price is None:
        submission_price = decision_price
        warnings.append("SUBMISSION_REFERENCE_PRICE_UNAVAILABLE")
    else:
        submission_price = float(submission_reference_price)
        if submission_price <= 0:
            raise TcaInputError("submission_reference_price must be positive")

    side = str(getattr(order, "side", "BUY") or "BUY").upper()
    sign = 1.0 if side == "BUY" else -1.0
    decision_to_submission = sign * (submission_price - decision_price) * quantity
    submission_to_fill = sign * (average_fill - submission_price) * quantity
    total_price_shortfall = decision_to_submission + submission_to_fill
    notional = decision_price * quantity
    spread_cost = notional * float(estimated_spread_bps) / 10_000.0
    implementation_shortfall = (
        total_price_shortfall + float(explicit_fees) + float(opportunity_cost)
    )
    implementation_bps = implementation_shortfall / notional * 10_000.0 if notional else 0.0

    transitions = tuple(transitions)
    timestamps = _transition_timestamps(transitions)
    fills_sorted = sorted(fill_rows, key=lambda fill: (_parse(fill.filled_at) or _epoch(), fill.fill_id))
    signal_at = str(getattr(intent, "event_ts", "") or "")
    first_fill_at = str(fills_sorted[0].filled_at or "")
    final_fill_at = str(fills_sorted[-1].filled_at or "")

    approved_quantity = int(getattr(order, "approved_quantity", 0) or 0)
    if approved_quantity > 0 and quantity < approved_quantity:
        warnings.append("PARTIAL_FILL_ONLY")
    if approved_quantity > 0 and quantity > approved_quantity:
        warnings.append("FILL_EXCEEDS_APPROVED_QUANTITY")
    order_filled = int(getattr(order, "filled_quantity", 0) or 0)
    if order_filled != quantity:
        warnings.append("OMS_FILL_TOTAL_MISMATCH")
    order_average = float(getattr(order, "average_fill_price", 0.0) or 0.0)
    if order_average > 0 and abs(order_average - average_fill) > 1e-9:
        warnings.append("OMS_AVERAGE_FILL_MISMATCH")

    required_timestamps = {
        "signal_at": signal_at,
        "risk_approved_at": timestamps.get("RISK_APPROVED", ""),
        "submission_prepared_at": timestamps.get("SUBMISSION_PREPARED", ""),
        "final_fill_at": final_fill_at,
    }
    for name, value in required_timestamps.items():
        if not value:
            warnings.append(f"{name.upper()}_MISSING")
    complete = (
        not warnings
        and approved_quantity > 0
        and quantity == approved_quantity
        and submission_reference_price is not None
    )

    payload = {
        "order_id": order.order_id,
        "trade_intent_id": intent.record_id,
        "quantity": quantity,
        "average_fill_price": average_fill,
        "explicit_fees": float(explicit_fees),
        "submission_reference_price": submission_price,
        "estimated_spread_bps": float(estimated_spread_bps),
        "estimated_market_impact": float(estimated_market_impact),
        "opportunity_cost": float(opportunity_cost),
        "fill_ids": [fill.fill_id for fill in fills_sorted],
        "transition_ids": [transition.transition_id for transition in transitions],
    }
    assessment_id = f"tca-{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]}"
    return EntryExecutionAssessment(
        assessment_id=assessment_id,
        order_id=order.order_id,
        trade_intent_id=intent.record_id,
        target_portfolio_id=str(getattr(intent, "target_portfolio_id", "") or ""),
        strategy_id=str(getattr(intent, "strategy_id", "") or ""),
        symbol=str(getattr(intent, "symbol", "") or "").upper(),
        side=side,
        quantity=quantity,
        decision_price=decision_price,
        submission_reference_price=submission_price,
        average_fill_price=average_fill,
        notional=notional,
        decision_to_submission_cost=decision_to_submission,
        submission_to_fill_cost=submission_to_fill,
        total_price_shortfall=total_price_shortfall,
        explicit_fees=float(explicit_fees),
        estimated_spread_cost=spread_cost,
        estimated_market_impact=float(estimated_market_impact),
        opportunity_cost=float(opportunity_cost),
        implementation_shortfall=implementation_shortfall,
        implementation_shortfall_bps=implementation_bps,
        signal_at=signal_at,
        intent_persisted_at=timestamps.get("INTENT_ACCEPTED", ""),
        risk_approved_at=timestamps.get("RISK_APPROVED", ""),
        submission_prepared_at=timestamps.get("SUBMISSION_PREPARED", ""),
        broker_acknowledged_at=timestamps.get("BROKER_ACKNOWLEDGED", ""),
        first_fill_at=first_fill_at,
        final_fill_at=final_fill_at,
        signal_to_risk_seconds=_seconds(signal_at, timestamps.get("RISK_APPROVED", "")),
        risk_to_submission_seconds=_seconds(
            timestamps.get("RISK_APPROVED", ""),
            timestamps.get("SUBMISSION_PREPARED", ""),
        ),
        submission_to_ack_seconds=_seconds(
            timestamps.get("SUBMISSION_PREPARED", ""),
            timestamps.get("BROKER_ACKNOWLEDGED", ""),
        ),
        ack_to_first_fill_seconds=_seconds(
            timestamps.get("BROKER_ACKNOWLEDGED", ""),
            first_fill_at,
        ),
        submission_to_final_fill_seconds=_seconds(
            timestamps.get("SUBMISSION_PREPARED", ""),
            final_fill_at,
        ),
        complete=complete,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=dict(metadata or {}),
    )


def _transition_timestamps(transitions: Iterable) -> dict[str, str]:
    out: dict[str, str] = {}
    for transition in sorted(
        transitions,
        key=lambda item: (int(getattr(item, "sequence", 0)), str(getattr(item, "transition_id", ""))),
    ):
        event_type = str(getattr(transition, "event_type", "") or "")
        event_at = str(getattr(transition, "event_at", "") or "")
        if event_type and event_at and event_type not in out:
            out[event_type] = event_at
    return out


def _seconds(start: str, end: str) -> float | None:
    left = _parse(start)
    right = _parse(end)
    if left is None or right is None:
        return None
    return max(0.0, (right - left).total_seconds())


def _parse(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _epoch() -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
