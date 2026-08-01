from execution.oms.store import OmsStore
from execution.tca import TcaStore, TradeIntentNotFound, assess_filled_orders, assess_oms_order
from research.intelligence import schemas as SC
from research.intelligence.event_store import EventStore


def _intent(symbol="AAA"):
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        cycle_id="cycle",
        symbol=symbol,
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio",
        target_position_id=f"position-{symbol}",
        desired_quantity=10,
        required_quantity=10,
    )


def _filled_oms_order(oms, intent):
    order = oms.ingest_intent(intent)
    order = oms.approve_risk(order.order_id, risk_decision_id="risk-1")
    order = oms.prepare_submission(order.order_id, submission_token="submit-1")
    order = oms.acknowledge(
        order.order_id,
        broker_order_id="broker-1",
        external_event_id="ack-1",
    )
    return oms.record_fill(
        order.order_id,
        external_fill_id="fill-1",
        quantity=10,
        price=102,
        filled_at="2026-08-01T09:32:00+05:30",
        broker_order_id="broker-1",
    )


def test_tca_service_links_intent_oms_transitions_and_fills(tmp_path):
    events = EventStore()
    intent = _intent()
    events.append(intent)
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_oms_order(oms, intent)
    tca = TcaStore(tmp_path / "tca.db")

    assessment = assess_oms_order(
        event_store=events,
        oms_store=oms,
        tca_store=tca,
        order_id=order.order_id,
        submission_reference_price=101,
        explicit_fees=5,
    )

    assert assessment.trade_intent_id == intent.record_id
    assert assessment.order_id == order.order_id
    assert assessment.quantity == 10
    assert assessment.total_price_shortfall == 20
    assert assessment.implementation_shortfall == 25
    assert tca.latest_for_order(order.order_id)["assessment_id"] == assessment.assessment_id


def test_batch_tca_is_idempotent(tmp_path):
    events = EventStore()
    intent = _intent()
    events.append(intent)
    oms = OmsStore(tmp_path / "oms.db")
    _filled_oms_order(oms, intent)
    tca = TcaStore(tmp_path / "tca.db")

    first = assess_filled_orders(
        event_store=events,
        oms_store=oms,
        tca_store=tca,
        submission_reference_price_fn=lambda order: 101,
        explicit_fees_fn=lambda order: 5,
    )
    second = assess_filled_orders(
        event_store=events,
        oms_store=oms,
        tca_store=tca,
        submission_reference_price_fn=lambda order: 101,
        explicit_fees_fn=lambda order: 5,
    )

    assert first["recorded_count"] == 1
    assert second["recorded_count"] == 1
    assert first["recorded"] == second["recorded"]
    assert tca.summary()["assessments"] == 1


def test_tca_service_refuses_missing_canonical_intent(tmp_path):
    intent = _intent()
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_oms_order(oms, intent)
    tca = TcaStore(tmp_path / "tca.db")

    try:
        assess_oms_order(
            event_store=EventStore(),
            oms_store=oms,
            tca_store=tca,
            order_id=order.order_id,
            submission_reference_price=101,
        )
    except TradeIntentNotFound as exc:
        assert intent.record_id in str(exc)
    else:
        raise AssertionError("TCA was created without canonical TradeIntent evidence")
