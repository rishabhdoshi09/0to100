from types import SimpleNamespace

from execution.oms import models as OM
from execution.oms.store import OmsStore
from execution.protection.store import ProtectionStore
from operations.zerodha_observer import SingleObserverLock
from research.intelligence.schemas import TradeIntent
from risk.governor import PortfolioRiskState, RiskRequest, evaluate


def _intent():
    return TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        cycle_id="cycle",
        symbol="AAA",
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        holding_horizon_days=20,
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=10,
        required_quantity=10,
    )


def test_timeout_after_possible_acceptance_never_auto_resubmits(tmp_path):
    path = tmp_path / "oms.db"
    store = OmsStore(path)
    order = store.ingest_intent(_intent())
    order = store.approve_risk(
        order.order_id,
        risk_decision_id="risk-1",
        approved_quantity=10,
    )
    order = store.prepare_submission(order.order_id, submission_token="submit-once")
    order = store.mark_submission_uncertain(
        order.order_id,
        reason="network timeout after request bytes were sent",
    )

    restarted = OmsStore(path).get(order.order_id)
    assert restarted.status == OM.RECOVERY_REQUIRED
    assert restarted.submission_token == "submit-once"
    try:
        OmsStore(path).prepare_submission(order.order_id, submission_token="submit-twice")
    except OM.IllegalTransition:
        pass
    else:
        raise AssertionError("uncertain submission was allowed to resubmit automatically")


def test_restart_after_fill_before_protection_freezes_entries(tmp_path):
    oms_path = tmp_path / "oms.db"
    protection_path = tmp_path / "protection.db"
    oms = OmsStore(oms_path)
    order = oms.ingest_intent(_intent())
    order = oms.approve_risk(order.order_id, risk_decision_id="risk-1", approved_quantity=10)
    order = oms.prepare_submission(order.order_id, submission_token="paper-submit")
    order = oms.acknowledge(order.order_id, broker_order_id="paper-order")
    order = oms.record_fill(
        order.order_id,
        external_fill_id="paper-fill",
        quantity=10,
        price=100,
    )
    assert order.status == OM.FILLED

    restarted_oms = OmsStore(oms_path)
    restarted_protection = ProtectionStore(protection_path)
    plan = restarted_protection.ensure_for_order(restarted_oms.get(order.order_id))

    assert plan.status == "REQUIRED"
    assert restarted_protection.summary()["entry_freeze_required"] is True
    assert restarted_protection.summary()["unsafe_plan_ids"] == [plan.plan_id]


def test_stale_market_data_freezes_new_risk():
    state = PortfolioRiskState(
        snapshot_id="state",
        as_of="2026-08-01T10:00:00+05:30",
        reconciled=True,
        data_fresh=False,
        broker_connected=True,
        cash=100_000,
        available_margin=100_000,
        equity=100_000,
        start_day_equity=100_000,
        peak_equity=100_000,
        data_age_seconds=600,
    )
    decision = evaluate(
        RiskRequest(
            order_id="order",
            symbol="AAA",
            side="BUY",
            requested_quantity=10,
            reference_price=100,
            stop_price=90,
        ),
        state,
    )

    assert decision.approved is False
    assert "STALE_MARKET_DATA" in decision.reasons


def test_duplicate_observer_process_cannot_own_same_lock(tmp_path):
    first = SingleObserverLock(tmp_path / "observer.lock")
    second = SingleObserverLock(tmp_path / "observer.lock")
    try:
        assert first.acquire() is True
        assert second.acquire() is False
    finally:
        second.release()
        first.release()
