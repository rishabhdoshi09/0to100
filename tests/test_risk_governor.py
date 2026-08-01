from __future__ import annotations

from dataclasses import replace

from execution.oms import REJECTED, RISK_APPROVED
from execution.oms.store import OmsStore
from research.intelligence import schemas as SC
from risk.governor import (
    APPROVE,
    ENTRY,
    EXIT,
    FREEZE,
    REDUCE,
    REJECT,
    PendingOrderState,
    PortfolioRiskState,
    PositionState,
    RiskLimits,
    RiskRequest,
    evaluate,
)
from risk.governor_store import RiskDecisionStore
from risk.oms_gate import evaluate_oms_order


def _state(**overrides):
    values = {
        "snapshot_id": "risk-state-1",
        "as_of": "2026-08-01T10:00:00+05:30",
        "reconciled": True,
        "data_fresh": True,
        "broker_connected": True,
        "cash": 100_000.0,
        "available_margin": 100_000.0,
        "equity": 100_000.0,
        "start_day_equity": 100_000.0,
        "peak_equity": 100_000.0,
        "positions": (),
        "pending_orders": (),
        "data_age_seconds": 5.0,
    }
    values.update(overrides)
    return PortfolioRiskState(**values)


def _request(**overrides):
    values = {
        "order_id": "oms-1",
        "symbol": "AAA",
        "side": "BUY",
        "requested_quantity": 50,
        "reference_price": 100.0,
        "stop_price": 90.0,
        "purpose": ENTRY,
        "sector": "INDUSTRIALS",
        "correlation_cluster": "cluster-1",
    }
    values.update(overrides)
    return RiskRequest(**values)


def _intent(quantity=50):
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle",
        symbol="AAA",
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=quantity,
        required_quantity=quantity,
    )


def test_healthy_reconciled_state_approves_requested_quantity():
    decision = evaluate(_request(), _state())

    assert decision.action == APPROVE
    assert decision.approved is True
    assert decision.approved_quantity == 50
    assert decision.reasons == ("ALL_LIMITS_PASS",)


def test_governor_reduces_to_binding_house_limit():
    decision = evaluate(_request(requested_quantity=200), _state())

    assert decision.action == REDUCE
    assert decision.approved_quantity == 100
    assert "REDUCED_BY_TRADE_RISK" in decision.reasons
    assert decision.metrics["capacities"]["TRADE_RISK"] == 100


def test_uncertain_or_unreconciled_state_rejects_new_entries():
    uncertain = PendingOrderState(
        order_id="unknown-order",
        symbol="BBB",
        remaining_quantity=10,
        reference_price=100,
        stop_price=90,
        uncertain=True,
    )
    state = _state(reconciled=False, pending_orders=(uncertain,))

    decision = evaluate(_request(), state)

    assert decision.action == REJECT
    assert decision.approved_quantity == 0
    assert "STATE_UNRECONCILED" in decision.reasons
    assert "UNCERTAIN_ORDER_STATE" in decision.reasons


def test_stale_data_or_unprotected_positions_block_new_risk():
    position = PositionState(
        symbol="BBB",
        quantity=10,
        market_price=100,
        stop_price=90,
        protected_quantity=0,
    )
    state = _state(data_fresh=False, data_age_seconds=500, positions=(position,))

    decision = evaluate(_request(), state)

    assert decision.action == REJECT
    assert "STALE_MARKET_DATA" in decision.reasons
    assert "UNPROTECTED_POSITION" in decision.reasons


def test_daily_loss_and_drawdown_trigger_freeze():
    state = _state(
        equity=89_000,
        start_day_equity=100_000,
        peak_equity=110_000,
        cash=89_000,
        available_margin=89_000,
    )

    decision = evaluate(_request(), state)

    assert decision.action == FREEZE
    assert "DAILY_LOSS_LIMIT_BREACHED" in decision.reasons
    assert "DRAWDOWN_LIMIT_BREACHED" in decision.reasons


def test_pending_exposure_reduces_new_name_capacity():
    pending = PendingOrderState(
        order_id="pending-1",
        symbol="AAA",
        remaining_quantity=90,
        reference_price=100,
        stop_price=90,
        sector="INDUSTRIALS",
        correlation_cluster="cluster-1",
    )
    state = _state(pending_orders=(pending,))

    decision = evaluate(_request(requested_quantity=50), state)

    assert decision.action == REDUCE
    assert decision.approved_quantity == 10
    assert "REDUCED_BY_NAME" in decision.reasons


def test_exit_remains_available_during_stale_data_and_loss_state():
    held = PositionState(
        symbol="AAA",
        quantity=20,
        market_price=100,
        stop_price=90,
        protected_quantity=20,
    )
    state = _state(
        data_fresh=False,
        data_age_seconds=999,
        equity=95_000,
        start_day_equity=100_000,
        peak_equity=100_000,
        cash=93_000,
        available_margin=93_000,
        positions=(held,),
        active_incidents=("daily-loss",),
    )
    request = _request(
        side="SELL",
        purpose=EXIT,
        requested_quantity=25,
        stop_price=0,
    )

    decision = evaluate(request, state)

    assert decision.action == REDUCE
    assert decision.approved_quantity == 20
    assert decision.reasons == ("REDUCED_TO_HELD_QUANTITY",)


def test_exit_blocks_when_same_symbol_order_state_is_uncertain():
    held = PositionState(
        symbol="AAA",
        quantity=20,
        market_price=100,
        protected_quantity=20,
    )
    uncertain = PendingOrderState(
        order_id="unknown-exit",
        symbol="AAA",
        remaining_quantity=20,
        reference_price=100,
        side="SELL",
        uncertain=True,
    )
    request = _request(side="SELL", purpose=EXIT, requested_quantity=20, stop_price=0)

    decision = evaluate(request, _state(positions=(held,), pending_orders=(uncertain,)))

    assert decision.action == REJECT
    assert "UNCERTAIN_SYMBOL_ORDER_STATE" in decision.reasons


def test_identical_inputs_produce_identical_decision_id():
    first = evaluate(_request(), _state())
    second = evaluate(_request(), _state())

    assert first.decision_id == second.decision_id
    changed = evaluate(_request(requested_quantity=49), _state())
    assert changed.decision_id != first.decision_id


def test_oms_bridge_persists_and_applies_approved_decision(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = oms.ingest_intent(_intent(quantity=50))
    decisions = RiskDecisionStore(tmp_path / "risk.db")

    result = evaluate_oms_order(
        oms,
        order.order_id,
        state=_state(),
        decision_store=decisions,
        sector="INDUSTRIALS",
        correlation_cluster="cluster-1",
    )

    assert result.decision.action == APPROVE
    assert result.order.status == RISK_APPROVED
    assert result.order.risk_decision_id == result.decision.decision_id
    assert decisions.latest(order.order_id)["decision_id"] == result.decision.decision_id


def test_oms_bridge_terminally_rejects_failed_risk_decision(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = oms.ingest_intent(_intent(quantity=50))
    decisions = RiskDecisionStore(tmp_path / "risk.db")

    result = evaluate_oms_order(
        oms,
        order.order_id,
        state=_state(broker_connected=False),
        decision_store=decisions,
    )

    assert result.decision.action == REJECT
    assert result.order.status == REJECTED
    assert result.order.last_error_code == REJECT
    assert "BROKER_DISCONNECTED" in result.order.last_error_message


def test_custom_limits_reduce_without_mutating_request():
    request = _request(requested_quantity=50)
    limits = replace(RiskLimits(), max_risk_per_trade_pct=0.002)

    decision = evaluate(request, _state(), limits)

    assert decision.action == REDUCE
    assert decision.approved_quantity == 20
    assert request.requested_quantity == 50
