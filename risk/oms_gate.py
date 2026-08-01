"""Bridge independent Risk Governor decisions into durable OMS transitions.

This module never submits to a broker. It reads a PROPOSED OMS order, evaluates reconciled
state, persists the decision, and advances the order only to RISK_APPROVED or REJECTED.
"""
from __future__ import annotations

from dataclasses import dataclass

from execution.oms import models as OM
from execution.oms.store import OmsStore
from risk.governor import (
    APPROVE,
    REDUCE,
    GovernorDecision,
    PortfolioRiskState,
    RiskLimits,
    evaluate,
    request_from_oms,
)
from risk.governor_store import RiskDecisionStore


@dataclass(frozen=True)
class OmsRiskResult:
    decision: GovernorDecision
    order: OM.OrderSnapshot


def evaluate_oms_order(
    oms_store: OmsStore,
    order_id: str,
    *,
    state: PortfolioRiskState,
    limits: RiskLimits | None = None,
    sector: str = "",
    correlation_cluster: str = "",
    decision_store: RiskDecisionStore | None = None,
) -> OmsRiskResult:
    """Evaluate and durably apply one risk decision to a PROPOSED OMS order."""
    order = oms_store.get(order_id)
    if order.status != OM.PROPOSED:
        raise OM.IllegalTransition(
            f"risk evaluation requires PROPOSED order; found {order.status}"
        )
    request = request_from_oms(
        order,
        sector=sector,
        correlation_cluster=correlation_cluster,
    )
    decision = evaluate(request, state, limits)
    if decision_store is not None:
        decision_store.record(decision)

    if decision.action in {APPROVE, REDUCE} and decision.approved_quantity > 0:
        updated = oms_store.approve_risk(
            order_id,
            risk_decision_id=decision.decision_id,
            approved_quantity=decision.approved_quantity,
            reason="; ".join(decision.reasons),
            external_event_id=decision.decision_id,
        )
    else:
        updated = oms_store.transition(
            order_id,
            OM.REJECTED,
            event_type="RISK_REJECTED",
            actor="risk_governor",
            reason="; ".join(decision.reasons),
            external_event_id=decision.decision_id,
            updates={
                "last_error_code": decision.action,
                "last_error_message": "; ".join(decision.reasons),
            },
            metadata={
                "decision_id": decision.decision_id,
                "decision_action": decision.action,
                "requested_quantity": decision.requested_quantity,
                "approved_quantity": decision.approved_quantity,
            },
        )
    return OmsRiskResult(decision=decision, order=updated)
