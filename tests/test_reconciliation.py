from __future__ import annotations

from execution.oms import BROKER_ACKNOWLEDGED, FILLED, QUARANTINED, SUBMISSION_PENDING
from execution.oms.store import OmsStore
from execution.reconciliation import (
    HEALTHY,
    INCOMPLETE,
    QUARANTINED as REPORT_QUARANTINED,
    REPAIRABLE,
    BrokerAccountSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    InternalPositionSnapshot,
    ReconciliationReportStore,
    reconcile,
    run_reconciliation,
)
from research.intelligence import schemas as SC


def _intent(quantity=10, symbol="AAA"):
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01",
        cycle_id="cycle",
        symbol=symbol,
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio",
        target_position_id=f"position-{symbol}",
        desired_quantity=quantity,
        required_quantity=quantity,
    )


def _submitted(store: OmsStore, quantity=10, symbol="AAA"):
    order = store.ingest_intent(_intent(quantity, symbol))
    order = store.approve_risk(
        order.order_id,
        risk_decision_id=f"risk-{symbol}",
        approved_quantity=quantity,
    )
    return store.prepare_submission(order.order_id, submission_token=f"submit-{symbol}")


def _broker(
    *,
    orders=(),
    trades=(),
    positions=(),
    complete=True,
    cash=100_000,
    margin=100_000,
    errors=(),
):
    return BrokerAccountSnapshot(
        snapshot_id="broker-snapshot-1",
        observed_at="2026-08-01T10:00:00+05:30",
        source="test-broker",
        orders=tuple(orders),
        trades=tuple(trades),
        positions=tuple(positions),
        cash=cash,
        available_margin=margin,
        orders_complete=complete,
        trades_complete=complete,
        positions_complete=complete,
        account_complete=complete,
        errors=tuple(errors),
    )


def _broker_order(order, *, status="OPEN", filled=0, quantity=None):
    return BrokerOrderSnapshot(
        broker_order_id=f"broker-{order.symbol}",
        status=status,
        symbol=order.symbol,
        side=order.side,
        quantity=order.approved_quantity if quantity is None else quantity,
        filled_quantity=filled,
        average_price=100 if filled else 0,
        client_order_ref=order.trade_intent_id,
        submission_token=order.submission_token,
    )


def test_incomplete_snapshot_freezes_and_never_mutates_oms(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _submitted(oms)
    broker = _broker(complete=False, errors=("positions endpoint unavailable",))

    result = run_reconciliation(
        oms_store=oms,
        broker=broker,
        internal_positions=(),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.report.status == INCOMPLETE
    assert result.report.entry_freeze_required is True
    assert result.applied_repairs == ()
    assert oms.get(order.order_id).status == SUBMISSION_PENDING
    assert "BROKER_SNAPSHOT_INCOMPLETE" in result.report.summary["issue_counts"]


def test_open_broker_order_is_deterministically_acknowledged(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _submitted(oms)
    broker_order = _broker_order(order)

    pre = reconcile(
        internal_orders=oms.list_orders(),
        internal_fills={order.order_id: oms.fills(order.order_id)},
        internal_positions=(),
        broker=_broker(orders=(broker_order,)),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )
    assert pre.status == REPAIRABLE
    assert [repair.action_type for repair in pre.repairs] == ["ACKNOWLEDGE"]

    result = run_reconciliation(
        oms_store=oms,
        broker=_broker(orders=(broker_order,)),
        internal_positions=(),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.report.status == HEALTHY
    assert oms.get(order.order_id).status == BROKER_ACKNOWLEDGED
    assert oms.get(order.order_id).broker_order_id == broker_order.broker_order_id
    assert len(result.applied_repairs) == 1


def test_missing_broker_fill_is_caught_up_idempotently(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _submitted(oms, quantity=10)
    broker_order = _broker_order(order, status="COMPLETE", filled=10)
    trade = BrokerTradeSnapshot(
        trade_id="trade-1",
        broker_order_id=broker_order.broker_order_id,
        symbol="AAA",
        side="BUY",
        quantity=10,
        price=101,
        executed_at="2026-08-01T10:01:00+05:30",
    )
    broker_position = BrokerPositionSnapshot(
        symbol="AAA",
        quantity=10,
        average_price=101,
        protected_quantity=10,
    )
    internal_position = InternalPositionSnapshot(
        symbol="AAA",
        quantity=10,
        average_price=101,
        protected_quantity=10,
    )
    snapshot = _broker(
        orders=(broker_order,),
        trades=(trade,),
        positions=(broker_position,),
    )

    first = run_reconciliation(
        oms_store=oms,
        broker=snapshot,
        internal_positions=(internal_position,),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )
    second = run_reconciliation(
        oms_store=oms,
        broker=snapshot,
        internal_positions=(internal_position,),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    filled = oms.get(order.order_id)
    assert filled.status == FILLED
    assert filled.filled_quantity == 10
    assert filled.average_fill_price == 101
    assert len(oms.fills(order.order_id)) == 1
    assert first.report.status == HEALTHY
    assert second.report.status == HEALTHY
    assert second.applied_repairs == ()


def test_unknown_broker_order_freezes_entries(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    unknown = BrokerOrderSnapshot(
        broker_order_id="broker-unknown",
        status="OPEN",
        symbol="XYZ",
        side="BUY",
        quantity=5,
    )

    report = reconcile(
        internal_orders=oms.list_orders(),
        internal_fills={},
        internal_positions=(),
        broker=_broker(orders=(unknown,)),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert report.status == REPORT_QUARANTINED
    assert report.entry_freeze_required is True
    assert "UNKNOWN_BROKER_ORDER" in report.summary["issue_counts"]


def test_broker_overfill_is_preserved_and_quarantined(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _submitted(oms, quantity=5)
    broker_order = _broker_order(order, status="COMPLETE", filled=6)
    trade = BrokerTradeSnapshot(
        trade_id="trade-over",
        broker_order_id=broker_order.broker_order_id,
        symbol="AAA",
        side="BUY",
        quantity=6,
        price=100,
        executed_at="2026-08-01T10:01:00+05:30",
    )
    position = BrokerPositionSnapshot(
        symbol="AAA",
        quantity=6,
        protected_quantity=6,
    )
    internal_position = InternalPositionSnapshot(
        symbol="AAA",
        quantity=6,
        protected_quantity=6,
    )

    result = run_reconciliation(
        oms_store=oms,
        broker=_broker(orders=(broker_order,), trades=(trade,), positions=(position,)),
        internal_positions=(internal_position,),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    reconciled_order = oms.get(order.order_id)
    assert reconciled_order.status == QUARANTINED
    assert reconciled_order.filled_quantity == 6
    assert reconciled_order.last_error_code in {"BROKER_OVERFILL", "RECONCILIATION_REPAIR_FAILED"}
    assert result.report.entry_freeze_required is True


def test_internal_fill_ahead_of_complete_broker_book_is_quarantined(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _submitted(oms, quantity=5)
    oms.record_fill(
        order.order_id,
        external_fill_id="internal-only-fill",
        quantity=5,
        price=100,
    )
    current = oms.get(order.order_id)
    broker_order = _broker_order(current, status="OPEN", filled=0)

    report = reconcile(
        internal_orders=oms.list_orders(),
        internal_fills={order.order_id: oms.fills(order.order_id)},
        internal_positions=(),
        broker=_broker(orders=(broker_order,)),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert report.status == REPORT_QUARANTINED
    assert "INTERNAL_FILL_AHEAD_OF_BROKER" in report.summary["issue_counts"]


def test_position_and_protection_mismatches_freeze_entries(tmp_path):
    report = reconcile(
        internal_orders=(),
        internal_fills={},
        internal_positions=(
            InternalPositionSnapshot(symbol="AAA", quantity=10, protected_quantity=10),
        ),
        broker=_broker(positions=(
            BrokerPositionSnapshot(symbol="AAA", quantity=8, protected_quantity=0),
        )),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert report.status == REPORT_QUARANTINED
    assert report.entry_freeze_required is True
    assert "POSITION_QUANTITY_MISMATCH" in report.summary["issue_counts"]
    assert "BROKER_POSITION_UNPROTECTED" in report.summary["issue_counts"]


def test_cash_and_margin_mismatches_are_not_silently_overwritten():
    report = reconcile(
        internal_orders=(),
        internal_fills={},
        internal_positions=(),
        broker=_broker(cash=90_000, margin=80_000),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert report.status == REPORT_QUARANTINED
    assert "CASH_MISMATCH" in report.summary["issue_counts"]
    assert "MARGIN_MISMATCH" in report.summary["issue_counts"]


def test_reconciliation_reports_persist_idempotently(tmp_path):
    report = reconcile(
        internal_orders=(),
        internal_fills={},
        internal_positions=(),
        broker=_broker(),
        internal_cash=100_000,
        internal_available_margin=100_000,
    )
    store = ReconciliationReportStore(tmp_path / "reports.db")

    store.record(report)
    store.record(report)

    assert store.summary()["reports"] == 1
    assert store.latest()["report_id"] == report.report_id
    assert store.latest()["status"] == HEALTHY
