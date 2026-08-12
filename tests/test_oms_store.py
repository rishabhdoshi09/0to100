from __future__ import annotations

import pytest

from execution.oms import (
    BROKER_ACKNOWLEDGED,
    FILLED,
    IllegalTransition,
    IdempotencyConflict,
    InvalidIntent,
    OmsStore,
    PARTIALLY_FILLED,
    PROPOSED,
    PROTECTED,
    PROTECTION_PENDING,
    QUARANTINED,
    RECOVERY_REQUIRED,
    RISK_APPROVED,
    SUBMISSION_PENDING,
)
from research.intelligence import schemas as SC


def _intent(*, symbol="AAA", quantity=10, record_id=""):
    return SC.TradeIntent(
        strategy_id="strategy-1",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        record_id=record_id,
        cycle_id="cycle-1",
        symbol=symbol,
        direction="LONG",
        entry_rule="target_portfolio_delta",
        intended_entry=100,
        intended_risk_pct=0.5,
        max_capital=quantity * 100,
        stop_rule="absolute_price",
        stop_price=90,
        exit_rule="absolute_target",
        target_price=120,
        holding_horizon_days=20,
        target_portfolio_id="portfolio-1",
        target_position_id=f"position-{symbol}",
        current_quantity=0,
        pending_quantity=0,
        desired_quantity=quantity,
        required_quantity=quantity,
    )


def _approved(store: OmsStore, *, quantity=10):
    order = store.ingest_intent(_intent(quantity=quantity))
    return store.approve_risk(
        order.order_id,
        risk_decision_id="risk-1",
        approved_quantity=quantity,
    )


def test_intent_ingestion_is_idempotent_and_persisted_before_submission(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    intent = _intent()

    first = store.ingest_intent(intent)
    second = store.ingest_intent(intent)

    assert first.order_id == second.order_id
    assert first.status == PROPOSED
    assert first.trade_intent_id == intent.record_id
    assert len(store.list_orders()) == 1
    history = store.history(first.order_id)
    assert len(history) == 1
    assert history[0].event_type == "INTENT_ACCEPTED"
    assert history[0].to_status == PROPOSED


def test_same_idempotency_key_cannot_own_different_intent_content(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    first = _intent(symbol="AAA", record_id="intent-fixed")
    second = _intent(symbol="BBB", record_id="intent-fixed")
    store.ingest_intent(first)

    with pytest.raises(IdempotencyConflict):
        store.ingest_intent(second)


def test_oms_rejects_unlinked_or_zero_quantity_intents(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    unlinked = SC.TradeIntent(
        strategy_id="strategy",
        symbol="AAA",
        intended_entry=100,
        stop_price=90,
        target_price=120,
        required_quantity=10,
    )
    zero = _intent(quantity=0)

    with pytest.raises(InvalidIntent):
        store.ingest_intent(unlinked)
    with pytest.raises(InvalidIntent):
        store.ingest_intent(zero)


def test_risk_and_submission_state_survive_restart(tmp_path):
    path = tmp_path / "oms.db"
    store = OmsStore(path)
    order = _approved(store)
    assert order.status == RISK_APPROVED

    pending = store.prepare_submission(order.order_id, submission_token="submit-1")
    assert pending.status == SUBMISSION_PENDING
    assert pending.submission_token == "submit-1"

    restarted = OmsStore(path)
    restored = restarted.get(order.order_id)
    assert restored.status == SUBMISSION_PENDING
    assert restored.approved_quantity == 10
    assert restored.submission_token == "submit-1"
    assert [item.to_status for item in restarted.history(order.order_id)] == [
        PROPOSED,
        RISK_APPROVED,
        SUBMISSION_PENDING,
    ]


def test_illegal_direct_submission_fails_closed(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = store.ingest_intent(_intent())

    with pytest.raises(IllegalTransition):
        store.prepare_submission(order.order_id, submission_token="submit-1")

    assert store.get(order.order_id).status == PROPOSED


def test_uncertain_submission_cannot_be_automatically_resubmitted(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store)
    order = store.prepare_submission(order.order_id, submission_token="submit-1")
    order = store.mark_submission_uncertain(order.order_id, reason="broker response lost")

    assert order.status == RECOVERY_REQUIRED
    with pytest.raises(IllegalTransition):
        store.prepare_submission(order.order_id, submission_token="submit-2")

    recovered = store.acknowledge(
        order.order_id,
        broker_order_id="broker-1",
        external_event_id="ack-1",
    )
    assert recovered.status == BROKER_ACKNOWLEDGED
    assert recovered.broker_order_id == "broker-1"


def test_partial_and_full_fills_are_idempotent(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store, quantity=10)
    store.prepare_submission(order.order_id, submission_token="submit-1")
    store.acknowledge(order.order_id, broker_order_id="broker-1", external_event_id="ack-1")

    partial = store.record_fill(
        order.order_id,
        external_fill_id="fill-1",
        quantity=4,
        price=101,
        broker_order_id="broker-1",
    )
    duplicate = store.record_fill(
        order.order_id,
        external_fill_id="fill-1",
        quantity=4,
        price=101,
        broker_order_id="broker-1",
    )
    complete = store.record_fill(
        order.order_id,
        external_fill_id="fill-2",
        quantity=6,
        price=102,
        broker_order_id="broker-1",
    )

    assert partial.status == PARTIALLY_FILLED
    assert duplicate.version == partial.version
    assert len(store.fills(order.order_id)) == 2
    assert complete.status == FILLED
    assert complete.filled_quantity == 10
    assert complete.remaining_quantity == 0
    assert complete.average_fill_price == pytest.approx(101.6)


def test_fill_can_arrive_before_acknowledgement(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store, quantity=5)
    store.prepare_submission(order.order_id, submission_token="submit-1")

    filled = store.record_fill(
        order.order_id,
        external_fill_id="fill-fast",
        quantity=5,
        price=100.5,
        broker_order_id="broker-fast",
    )

    assert filled.status == FILLED
    assert filled.broker_order_id == "broker-fast"
    assert filled.filled_quantity == 5


def test_broker_overfill_is_preserved_and_quarantined(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store, quantity=5)
    store.prepare_submission(order.order_id, submission_token="submit-1")

    quarantined = store.record_fill(
        order.order_id,
        external_fill_id="fill-over",
        quantity=6,
        price=100,
    )

    assert quarantined.status == QUARANTINED
    assert quarantined.filled_quantity == 6
    assert quarantined.last_error_code == "BROKER_OVERFILL"
    assert len(store.fills(order.order_id)) == 1


def test_external_ack_event_is_idempotent(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store)
    store.prepare_submission(order.order_id, submission_token="submit-1")

    first = store.acknowledge(
        order.order_id,
        broker_order_id="broker-1",
        external_event_id="ack-1",
    )
    second = store.acknowledge(
        order.order_id,
        broker_order_id="broker-1",
        external_event_id="ack-1",
    )

    assert first.version == second.version
    assert len(store.history(order.order_id)) == 4


def test_pending_exposure_counts_uncertain_orders_at_worst_case(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store, quantity=10)
    store.prepare_submission(order.order_id, submission_token="submit-1")
    store.record_fill(
        order.order_id,
        external_fill_id="fill-1",
        quantity=4,
        price=101,
    )
    store.mark_submission_uncertain(order.order_id, reason="position endpoint unavailable")

    exposure = store.pending_exposure()

    assert exposure["pending_quantities"] == {"AAA": 6}
    assert exposure["pending_risk_amounts"] == {"AAA": 60.0}
    assert exposure["pending_capital_amounts"] == {"AAA": 600.0}
    assert order.order_id in exposure["uncertain_order_ids"]


def test_protection_lifecycle_is_explicit(tmp_path):
    store = OmsStore(tmp_path / "oms.db")
    order = _approved(store, quantity=2)
    store.prepare_submission(order.order_id, submission_token="submit-1")
    store.record_fill(
        order.order_id,
        external_fill_id="fill-1",
        quantity=2,
        price=100,
    )

    pending = store.mark_protection_pending(order.order_id)
    protected = store.mark_protected(
        order.order_id,
        protection_reference="gtt-1",
        external_event_id="protection-1",
    )

    assert pending.status == PROTECTION_PENDING
    assert protected.status == PROTECTED
    assert store.summary()["recovery_required"] == []
