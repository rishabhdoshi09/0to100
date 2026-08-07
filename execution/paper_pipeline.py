"""Production-parity PAPER execution through OMS, Risk, Protection and TCA.

This adapter is deliberately simulated and has no broker dependency. It exercises the same
institutional state contracts intended for production while keeping PaperBook as the canonical
simulated position ledger. Every external-looking identifier is explicitly prefixed ``paper-``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.protection.models import BrokerProtectionSnapshot
from execution.protection.store import ProtectionStore
from execution.tca.service import assess_oms_order
from execution.tca.store import TcaStore
from risk.governor import (
    PendingOrderState,
    PortfolioRiskState,
    PositionState,
    RiskLimits,
)
from risk.governor_store import RiskDecisionStore
from risk.oms_gate import evaluate_oms_order

PAPER_BROKER_CONNECTED = True
BROKER_MUTATIONS_ENABLED = False


@dataclass(frozen=True)
class PaperExecutionResult:
    order_id: str
    status: str
    position: Any | None
    risk_decision_id: str = ""
    protection_plan_id: str = ""
    tca_assessment_id: str = ""
    reason: str = ""
    resumed: bool = False

    @property
    def opened(self) -> bool:
        return self.position is not None and self.status == OM.PROTECTED


class PaperExecutionPipeline:
    """Crash-resumable simulated EMS over durable institutional state."""

    def __init__(
        self,
        *,
        oms_store: OmsStore,
        risk_store: RiskDecisionStore,
        protection_store: ProtectionStore,
        tca_store: TcaStore,
        event_store,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        self.oms = oms_store
        self.risk = risk_store
        self.protection = protection_store
        self.tca = tca_store
        self.events = event_store
        self.risk_limits = risk_limits or RiskLimits()

    def execute(self, intent, *, book, date: str, runtime_state=None) -> PaperExecutionResult:
        """Execute or resume one PAPER intent idempotently.

        The order is durable before risk approval. The PaperBook position is opened only after
        risk approval and simulated acknowledgement. A crash after opening but before recording
        the fill is repaired from the existing PaperBook position on the next call.
        """
        order = self.oms.ingest_intent(intent)
        initial_status = order.status
        resumed = initial_status != OM.PROPOSED

        if order.status in {OM.REJECTED, OM.CANCELLED, OM.EXPIRED}:
            return PaperExecutionResult(
                order_id=order.order_id,
                status=order.status,
                position=None,
                risk_decision_id=order.risk_decision_id,
                reason=order.last_error_message or order.status,
                resumed=resumed,
            )
        if order.status in {OM.UNKNOWN, OM.QUARANTINED, OM.RECOVERY_REQUIRED}:
            return PaperExecutionResult(
                order_id=order.order_id,
                status=order.status,
                position=self._book_position(book, order),
                risk_decision_id=order.risk_decision_id,
                reason="ambiguous PAPER order state requires recovery",
                resumed=True,
            )

        risk_decision_id = order.risk_decision_id
        if order.status == OM.PROPOSED:
            risk_result = evaluate_oms_order(
                self.oms,
                order.order_id,
                state=self._risk_state(book, date, runtime_state, exclude_order_id=order.order_id),
                limits=self.risk_limits,
                decision_store=self.risk,
            )
            order = risk_result.order
            risk_decision_id = risk_result.decision.decision_id
            if not risk_result.decision.approved:
                return PaperExecutionResult(
                    order_id=order.order_id,
                    status=order.status,
                    position=None,
                    risk_decision_id=risk_decision_id,
                    reason="; ".join(risk_result.decision.reasons),
                    resumed=resumed,
                )

        submission_token = f"paper-submit-{order.order_id}"
        paper_broker_order_id = f"paper-order-{order.order_id}"
        if order.status == OM.RISK_APPROVED:
            order = self.oms.prepare_submission(
                order.order_id,
                submission_token=submission_token,
                actor="paper_ems",
            )
        if order.status == OM.SUBMISSION_PENDING:
            order = self.oms.acknowledge(
                order.order_id,
                broker_order_id=paper_broker_order_id,
                external_event_id=f"paper-ack-{order.order_id}",
                actor="paper_broker",
            )

        position = self._book_position(book, order)
        if order.status == OM.BROKER_ACKNOWLEDGED and position is None:
            position = book.open_position(
                order.strategy_id,
                order.symbol,
                order.intended_entry,
                order.stop_price,
                order.target_price,
                date,
                int(getattr(intent, "holding_horizon_days", 0) or 0),
                risk_pct_of_capital=order.intended_risk_pct,
                quantity=order.approved_quantity,
            )
            if position is None:
                order = self.oms.cancel(
                    order.order_id,
                    reason="PaperBook refused the independently approved exact quantity",
                    external_event_id=f"paper-book-refusal-{order.order_id}",
                )
                return PaperExecutionResult(
                    order_id=order.order_id,
                    status=order.status,
                    position=None,
                    risk_decision_id=risk_decision_id,
                    reason="BOOK_REFUSED_AFTER_RISK_APPROVAL",
                    resumed=resumed,
                )

        if order.status == OM.BROKER_ACKNOWLEDGED and position is not None:
            if int(position.qty) != int(order.approved_quantity):
                order = self.oms.quarantine(
                    order.order_id,
                    code="PAPER_BOOK_QUANTITY_MISMATCH",
                    message=(
                        f"book={position.qty}; approved={order.approved_quantity}"
                    ),
                    actor="paper_recovery",
                )
                return PaperExecutionResult(
                    order_id=order.order_id,
                    status=order.status,
                    position=position,
                    risk_decision_id=risk_decision_id,
                    reason=order.last_error_message,
                    resumed=True,
                )
            order = self.oms.record_fill(
                order.order_id,
                external_fill_id=f"paper-fill-{order.order_id}",
                quantity=int(position.qty),
                price=float(position.entry_price),
                filled_at=date,
                broker_order_id=paper_broker_order_id,
                metadata={"mode": "PAPER", "broker_mutations_enabled": False},
                actor="paper_broker",
            )

        if order.status == OM.PARTIALLY_FILLED:
            order = self.oms.quarantine(
                order.order_id,
                code="UNEXPECTED_PAPER_PARTIAL_FILL",
                message="the deterministic PAPER adapter must fill approved quantity atomically",
                actor="paper_recovery",
            )
            return PaperExecutionResult(
                order_id=order.order_id,
                status=order.status,
                position=position,
                risk_decision_id=risk_decision_id,
                reason=order.last_error_message,
                resumed=True,
            )

        plan_id = ""
        if order.status in {OM.FILLED, OM.PROTECTION_PENDING}:
            plan = self.protection.ensure_for_order(order, actor="paper_protection")
            plan_id = plan.plan_id
            if order.status == OM.FILLED:
                order = self.oms.mark_protection_pending(
                    order.order_id,
                    actor="paper_protection",
                )
            plan = self._ensure_paper_protected(plan, order)
            if not plan.fully_protected:
                order = self.oms.transition(
                    order.order_id,
                    OM.RECOVERY_REQUIRED,
                    event_type="PAPER_PROTECTION_UNVERIFIED",
                    actor="paper_protection",
                    reason=plan.last_error_message or plan.status,
                    updates={
                        "last_error_code": "PAPER_PROTECTION_UNVERIFIED",
                        "last_error_message": plan.last_error_message or plan.status,
                    },
                )
                return PaperExecutionResult(
                    order_id=order.order_id,
                    status=order.status,
                    position=position,
                    risk_decision_id=risk_decision_id,
                    protection_plan_id=plan.plan_id,
                    reason=order.last_error_message,
                    resumed=resumed,
                )
            if order.status == OM.PROTECTION_PENDING:
                order = self.oms.mark_protected(
                    order.order_id,
                    protection_reference=plan.broker_protection_id,
                    external_event_id=f"paper-protected-{order.order_id}",
                    actor="paper_protection",
                )

        if order.status == OM.PROTECTED:
            plan = self.protection.get_by_order(order.order_id)
            plan_id = plan.plan_id if plan is not None else plan_id
            assessment = assess_oms_order(
                event_store=self.events,
                oms_store=self.oms,
                tca_store=self.tca,
                order_id=order.order_id,
                submission_reference_price=order.intended_entry,
                explicit_fees=0.0,
                metadata={
                    "mode": "PAPER",
                    "broker_mutations_enabled": False,
                    "simulated_broker_order_id": paper_broker_order_id,
                },
            )
            return PaperExecutionResult(
                order_id=order.order_id,
                status=order.status,
                position=position or self._book_position(book, order),
                risk_decision_id=risk_decision_id,
                protection_plan_id=plan_id,
                tca_assessment_id=assessment.assessment_id,
                resumed=resumed,
            )

        return PaperExecutionResult(
            order_id=order.order_id,
            status=order.status,
            position=position,
            risk_decision_id=risk_decision_id,
            protection_plan_id=plan_id,
            reason="PAPER execution did not reach protected state",
            resumed=resumed,
        )

    def _ensure_paper_protected(self, plan, order):
        broker_id = f"paper-protection-{order.order_id}"
        request_token = f"paper-protection-request-{order.order_id}"
        if plan.status not in {"ACTIVE", "VERIFIED"}:
            if plan.status != "SUBMISSION_PENDING":
                plan = self.protection.prepare_submission(
                    plan.plan_id,
                    request_token=request_token,
                    actor="paper_protection",
                )
            if plan.status == "SUBMISSION_PENDING":
                plan = self.protection.acknowledge(
                    plan.plan_id,
                    broker_protection_id=broker_id,
                    protected_quantity=order.filled_quantity,
                    stop_reference=f"{broker_id}:stop",
                    target_reference=f"{broker_id}:target",
                    external_event_id=f"paper-protection-ack-{order.order_id}",
                    actor="paper_protection",
                )
        snapshot = BrokerProtectionSnapshot(
            broker_protection_id=broker_id,
            order_id=order.order_id,
            symbol=order.symbol,
            active=True,
            quantity=order.filled_quantity,
            stop_price=order.stop_price,
            target_price=order.target_price,
            stop_reference=f"{broker_id}:stop",
            target_reference=f"{broker_id}:target",
            updated_at=order.updated_at,
            raw={"mode": "PAPER", "broker_mutations_enabled": False},
        )
        return self.protection.verify(
            plan.plan_id,
            snapshot,
            external_event_id=f"paper-protection-verify-{order.order_id}",
            actor="paper_protection",
        )

    def _risk_state(self, book, date: str, runtime_state, *, exclude_order_id: str):
        positions = tuple(
            PositionState(
                symbol=position.symbol,
                quantity=int(position.qty),
                market_price=float(position.entry_price),
                stop_price=float(position.stop_price),
                protected_quantity=int(position.qty),
            )
            for position in book.open.values()
        )
        pending: list[PendingOrderState] = []
        for order in self.oms.list_orders(statuses=OM.PENDING_EXPOSURE_STATUSES):
            if order.order_id == exclude_order_id or order.remaining_quantity <= 0:
                continue
            pending.append(
                PendingOrderState(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    remaining_quantity=order.remaining_quantity,
                    reference_price=order.intended_entry,
                    stop_price=order.stop_price,
                    side=order.side,
                    uncertain=order.status in OM.UNCERTAIN_STATUSES,
                )
            )
        equity = float(book.equity())
        position_cost = sum(
            float(position.entry_price) * int(position.qty)
            for position in book.open.values()
        )
        cash = max(0.0, float(book.capital) + float(book.realized_pnl) - position_cost)
        peak = max([float(book.capital), *[float(value) for value in book.equity_curve]])
        reconciled = bool(getattr(runtime_state, "reconciled", True))
        semantic = {
            "date": date,
            "equity": equity,
            "cash": cash,
            "positions": [position.__dict__ for position in positions],
            "pending": [order.__dict__ for order in pending],
            "reconciled": reconciled,
        }
        fingerprint = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        return PortfolioRiskState(
            snapshot_id=f"paper-state-{fingerprint}",
            as_of=date,
            reconciled=reconciled,
            data_fresh=True,
            broker_connected=PAPER_BROKER_CONNECTED,
            cash=cash,
            available_margin=cash,
            equity=equity,
            start_day_equity=float(book.capital),
            peak_equity=peak,
            realized_pnl=float(book.realized_pnl),
            unrealized_pnl=equity - float(book.capital) - float(book.realized_pnl),
            positions=positions,
            pending_orders=tuple(pending),
            margin_state_known=True,
            data_age_seconds=0.0,
            active_incidents=(),
        )

    @staticmethod
    def _book_position(book, order):
        return book.open.get((order.strategy_id, order.symbol))
