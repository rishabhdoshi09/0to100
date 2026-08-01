"""Independent, deterministic Risk Governor for QuantTerm.

The governor consumes only reconciled portfolio/order state and a broker-neutral order request.
It has no strategy, UI, LLM, broker or database dependency. It may APPROVE, REDUCE, REJECT or
FREEZE; uncertainty always blocks new entries.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

APPROVE = "APPROVE"
REDUCE = "REDUCE"
REJECT = "REJECT"
FREEZE = "FREEZE"

ENTRY = "ENTRY"
EXIT = "EXIT"


@dataclass(frozen=True)
class PositionState:
    symbol: str
    quantity: int
    market_price: float
    stop_price: float = 0.0
    sector: str = ""
    correlation_cluster: str = ""
    protected_quantity: int = 0

    @property
    def market_value(self) -> float:
        return max(0, self.quantity) * max(0.0, self.market_price)

    @property
    def open_risk(self) -> float:
        if self.quantity <= 0:
            return 0.0
        if self.stop_price <= 0 or self.stop_price >= self.market_price:
            return self.market_value
        return self.quantity * (self.market_price - self.stop_price)

    @property
    def unprotected_quantity(self) -> int:
        return max(0, self.quantity - max(0, self.protected_quantity))


@dataclass(frozen=True)
class PendingOrderState:
    order_id: str
    symbol: str
    remaining_quantity: int
    reference_price: float
    stop_price: float = 0.0
    side: str = "BUY"
    sector: str = ""
    correlation_cluster: str = ""
    uncertain: bool = False

    @property
    def capital_exposure(self) -> float:
        if self.side.upper() != "BUY":
            return 0.0
        return max(0, self.remaining_quantity) * max(0.0, self.reference_price)

    @property
    def risk_exposure(self) -> float:
        if self.side.upper() != "BUY" or self.remaining_quantity <= 0:
            return 0.0
        if self.stop_price <= 0 or self.stop_price >= self.reference_price:
            return self.capital_exposure
        return self.remaining_quantity * (self.reference_price - self.stop_price)


@dataclass(frozen=True)
class PortfolioRiskState:
    snapshot_id: str
    as_of: str
    reconciled: bool
    data_fresh: bool
    broker_connected: bool
    cash: float
    available_margin: float
    equity: float
    start_day_equity: float
    peak_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: tuple[PositionState, ...] = ()
    pending_orders: tuple[PendingOrderState, ...] = ()
    margin_state_known: bool = True
    data_age_seconds: float = 0.0
    active_incidents: tuple[str, ...] = ()

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.start_day_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 1.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_pct: float = 0.01
    max_name_exposure_pct: float = 0.10
    max_total_open_risk_pct: float = 0.05
    max_pending_risk_pct: float = 0.03
    max_gross_exposure_pct: float = 0.95
    max_sector_exposure_pct: float = 0.25
    max_cluster_exposure_pct: float = 0.20
    max_order_value_pct: float = 0.10
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    min_cash_buffer_pct: float = 0.05
    max_positions: int = 5
    max_data_age_seconds: float = 120.0
    require_protection: bool = True
    freeze_on_any_incident: bool = True


@dataclass(frozen=True)
class RiskRequest:
    order_id: str
    symbol: str
    side: str
    requested_quantity: int
    reference_price: float
    stop_price: float
    purpose: str = ENTRY
    sector: str = ""
    correlation_cluster: str = ""


@dataclass(frozen=True)
class GovernorDecision:
    decision_id: str
    action: str
    approved_quantity: int
    requested_quantity: int
    reasons: tuple[str, ...]
    order_id: str
    symbol: str
    state_snapshot_id: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.action in {APPROVE, REDUCE} and self.approved_quantity > 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    request: RiskRequest,
    state: PortfolioRiskState,
    limits: RiskLimits | None = None,
) -> GovernorDecision:
    """Evaluate one order request against reconciled worst-case portfolio state."""
    limits = limits or RiskLimits()
    request = _normalize_request(request)
    hard_reasons = _common_hard_blocks(request, state, limits)

    if request.purpose == EXIT:
        return _evaluate_exit(request, state, limits, hard_reasons)

    metrics = _portfolio_metrics(state)
    if hard_reasons:
        action = FREEZE if any(reason.startswith(("DAILY_LOSS", "DRAWDOWN", "INCIDENT"))
                               for reason in hard_reasons) else REJECT
        return _decision(request, state, limits, action, 0, hard_reasons, metrics)

    risk_per_share = request.reference_price - request.stop_price
    if risk_per_share <= 0:
        return _decision(
            request,
            state,
            limits,
            REJECT,
            0,
            ("INVALID_ENTRY_STOP",),
            metrics,
        )

    equity = state.equity
    current_symbols = {position.symbol.upper() for position in state.positions if position.quantity > 0}
    pending_symbols = {
        order.symbol.upper()
        for order in state.pending_orders
        if order.side.upper() == "BUY" and order.remaining_quantity > 0
    }
    symbol = request.symbol.upper()

    current_name_value = sum(
        position.market_value for position in state.positions if position.symbol.upper() == symbol
    )
    current_name_value += sum(
        order.capital_exposure for order in state.pending_orders if order.symbol.upper() == symbol
    )
    current_sector_value = sum(
        position.market_value for position in state.positions if position.sector == request.sector
    ) + sum(
        order.capital_exposure for order in state.pending_orders if order.sector == request.sector
    )
    current_cluster_value = sum(
        position.market_value
        for position in state.positions
        if position.correlation_cluster == request.correlation_cluster
    ) + sum(
        order.capital_exposure
        for order in state.pending_orders
        if order.correlation_cluster == request.correlation_cluster
    )

    capacities: dict[str, int] = {
        "REQUESTED": request.requested_quantity,
        "TRADE_RISK": _qty((equity * limits.max_risk_per_trade_pct), risk_per_share),
        "NAME": _qty(equity * limits.max_name_exposure_pct - current_name_value,
                     request.reference_price),
        "GROSS": _qty(equity * limits.max_gross_exposure_pct - metrics["gross_exposure"],
                      request.reference_price),
        "OPEN_RISK": _qty(equity * limits.max_total_open_risk_pct - metrics["open_risk"],
                          risk_per_share),
        "PENDING_RISK": _qty(equity * limits.max_pending_risk_pct - metrics["pending_risk"],
                             risk_per_share),
        "ORDER_VALUE": _qty(equity * limits.max_order_value_pct, request.reference_price),
        "CASH": _qty(
            state.cash - equity * limits.min_cash_buffer_pct,
            request.reference_price,
        ),
        "MARGIN": _qty(state.available_margin, request.reference_price),
    }
    if request.sector:
        capacities["SECTOR"] = _qty(
            equity * limits.max_sector_exposure_pct - current_sector_value,
            request.reference_price,
        )
    if request.correlation_cluster:
        capacities["CLUSTER"] = _qty(
            equity * limits.max_cluster_exposure_pct - current_cluster_value,
            request.reference_price,
        )
    if symbol not in current_symbols | pending_symbols and len(current_symbols | pending_symbols) >= limits.max_positions:
        capacities["MAX_POSITIONS"] = 0

    approved_quantity = min(capacities.values()) if capacities else 0
    metrics.update({
        "risk_per_share": risk_per_share,
        "capacities": capacities,
        "current_name_value": current_name_value,
        "current_sector_value": current_sector_value,
        "current_cluster_value": current_cluster_value,
    })
    binding = tuple(sorted(key for key, value in capacities.items() if value == approved_quantity))
    if approved_quantity <= 0:
        reasons = tuple(f"NO_CAPACITY_{key}" for key in binding) or ("NO_APPROVABLE_QUANTITY",)
        return _decision(request, state, limits, REJECT, 0, reasons, metrics)
    if approved_quantity < request.requested_quantity:
        return _decision(
            request,
            state,
            limits,
            REDUCE,
            approved_quantity,
            tuple(f"REDUCED_BY_{key}" for key in binding),
            metrics,
        )
    return _decision(request, state, limits, APPROVE, approved_quantity, ("ALL_LIMITS_PASS",), metrics)


def request_from_oms(
    order,
    *,
    sector: str = "",
    correlation_cluster: str = "",
    purpose: str = ENTRY,
) -> RiskRequest:
    """Convert an immutable OMS order snapshot into a risk request."""
    return RiskRequest(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        requested_quantity=order.requested_quantity,
        reference_price=order.intended_entry,
        stop_price=order.stop_price,
        purpose=purpose,
        sector=sector,
        correlation_cluster=correlation_cluster,
    )


def _common_hard_blocks(
    request: RiskRequest,
    state: PortfolioRiskState,
    limits: RiskLimits,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not state.snapshot_id:
        reasons.append("STATE_SNAPSHOT_MISSING")
    if not state.reconciled:
        reasons.append("STATE_UNRECONCILED")
    if not state.broker_connected:
        reasons.append("BROKER_DISCONNECTED")
    if not state.margin_state_known:
        reasons.append("MARGIN_STATE_UNKNOWN")
    if state.equity <= 0 or state.cash < 0 or state.available_margin < 0:
        reasons.append("INVALID_CAPITAL_STATE")
    uncertain = [order.order_id for order in state.pending_orders if order.uncertain]
    if uncertain:
        reasons.append("UNCERTAIN_ORDER_STATE")
    if request.purpose == ENTRY:
        if not state.data_fresh or state.data_age_seconds > limits.max_data_age_seconds:
            reasons.append("STALE_MARKET_DATA")
        if limits.require_protection and any(
            position.unprotected_quantity > 0 for position in state.positions
        ):
            reasons.append("UNPROTECTED_POSITION")
        if state.daily_pnl <= -abs(state.start_day_equity * limits.max_daily_loss_pct):
            reasons.append("DAILY_LOSS_LIMIT_BREACHED")
        if state.drawdown_pct >= limits.max_drawdown_pct:
            reasons.append("DRAWDOWN_LIMIT_BREACHED")
        if limits.freeze_on_any_incident and state.active_incidents:
            reasons.append("INCIDENT_FREEZE")
    return tuple(dict.fromkeys(reasons))


def _evaluate_exit(
    request: RiskRequest,
    state: PortfolioRiskState,
    limits: RiskLimits,
    hard_reasons: tuple[str, ...],
) -> GovernorDecision:
    metrics = _portfolio_metrics(state)
    # Exits remain possible during stale data, loss and drawdown states, but never when
    # broker/reconciliation/capital state is unknown or the same symbol has uncertain orders.
    exit_blockers = tuple(
        reason for reason in hard_reasons
        if reason in {
            "STATE_SNAPSHOT_MISSING",
            "STATE_UNRECONCILED",
            "BROKER_DISCONNECTED",
            "MARGIN_STATE_UNKNOWN",
            "INVALID_CAPITAL_STATE",
        }
    )
    if any(order.uncertain and order.symbol.upper() == request.symbol.upper()
           for order in state.pending_orders):
        exit_blockers += ("UNCERTAIN_SYMBOL_ORDER_STATE",)
    if exit_blockers:
        return _decision(request, state, limits, REJECT, 0, exit_blockers, metrics)
    held = sum(
        max(0, position.quantity)
        for position in state.positions
        if position.symbol.upper() == request.symbol.upper()
    )
    approved = min(request.requested_quantity, held)
    if approved <= 0:
        return _decision(request, state, limits, REJECT, 0, ("NO_POSITION_TO_EXIT",), metrics)
    action = APPROVE if approved == request.requested_quantity else REDUCE
    reasons = ("EXIT_APPROVED",) if action == APPROVE else ("REDUCED_TO_HELD_QUANTITY",)
    return _decision(request, state, limits, action, approved, reasons, metrics)


def _portfolio_metrics(state: PortfolioRiskState) -> dict[str, Any]:
    gross = sum(position.market_value for position in state.positions)
    pending_capital = sum(order.capital_exposure for order in state.pending_orders)
    open_risk = sum(position.open_risk for position in state.positions)
    pending_risk = sum(order.risk_exposure for order in state.pending_orders)
    return {
        "equity": state.equity,
        "cash": state.cash,
        "available_margin": state.available_margin,
        "gross_exposure": gross + pending_capital,
        "position_exposure": gross,
        "pending_capital": pending_capital,
        "open_risk": open_risk + pending_risk,
        "position_risk": open_risk,
        "pending_risk": pending_risk,
        "daily_pnl": state.daily_pnl,
        "drawdown_pct": state.drawdown_pct,
        "unprotected_quantity": sum(position.unprotected_quantity for position in state.positions),
        "uncertain_orders": [order.order_id for order in state.pending_orders if order.uncertain],
    }


def _normalize_request(request: RiskRequest) -> RiskRequest:
    side = request.side.upper()
    purpose = request.purpose.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if purpose not in {ENTRY, EXIT}:
        raise ValueError("purpose must be ENTRY or EXIT")
    values = (request.reference_price, request.stop_price)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("request contains non-finite prices")
    if request.requested_quantity <= 0 or request.reference_price <= 0:
        raise ValueError("request quantity and reference price must be positive")
    return RiskRequest(
        order_id=str(request.order_id),
        symbol=str(request.symbol).upper(),
        side=side,
        requested_quantity=int(request.requested_quantity),
        reference_price=float(request.reference_price),
        stop_price=float(request.stop_price),
        purpose=purpose,
        sector=str(request.sector),
        correlation_cluster=str(request.correlation_cluster),
    )


def _decision(
    request: RiskRequest,
    state: PortfolioRiskState,
    limits: RiskLimits,
    action: str,
    approved_quantity: int,
    reasons: tuple[str, ...],
    metrics: dict[str, Any],
) -> GovernorDecision:
    payload = {
        "request": asdict(request),
        "state": asdict(state),
        "limits": asdict(limits),
        "action": action,
        "approved_quantity": approved_quantity,
        "reasons": reasons,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    decision_id = f"risk-{hashlib.sha256(blob).hexdigest()[:20]}"
    return GovernorDecision(
        decision_id=decision_id,
        action=action,
        approved_quantity=max(0, int(approved_quantity)),
        requested_quantity=request.requested_quantity,
        reasons=tuple(reasons),
        order_id=request.order_id,
        symbol=request.symbol,
        state_snapshot_id=state.snapshot_id,
        metrics=dict(metrics),
    )


def _qty(available_amount: float, unit_amount: float) -> int:
    if not math.isfinite(float(available_amount)) or not math.isfinite(float(unit_amount)):
        return 0
    if available_amount <= 0 or unit_amount <= 0:
        return 0
    return max(0, int(math.floor(available_amount / unit_amount)))
