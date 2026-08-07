"""Typed transaction-cost and execution-timeline assessments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntryExecutionAssessment:
    assessment_id: str
    order_id: str
    trade_intent_id: str
    target_portfolio_id: str
    strategy_id: str
    symbol: str
    side: str
    quantity: int
    decision_price: float
    submission_reference_price: float
    average_fill_price: float
    notional: float
    decision_to_submission_cost: float
    submission_to_fill_cost: float
    total_price_shortfall: float
    explicit_fees: float
    estimated_spread_cost: float
    estimated_market_impact: float
    opportunity_cost: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
    signal_at: str
    intent_persisted_at: str
    risk_approved_at: str
    submission_prepared_at: str
    broker_acknowledged_at: str
    first_fill_at: str
    final_fill_at: str
    signal_to_risk_seconds: float | None
    risk_to_submission_seconds: float | None
    submission_to_ack_seconds: float | None
    ack_to_first_fill_seconds: float | None
    submission_to_final_fill_seconds: float | None
    complete: bool
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
