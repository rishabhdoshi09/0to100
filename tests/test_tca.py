from types import SimpleNamespace

import pytest

from execution.oms.models import FillSnapshot, TransitionSnapshot
from execution.tca import TcaInputError, TcaStore, assess_entry_execution
from research.intelligence import schemas as SC


def _intent(*, side="LONG"):
    return SC.TradeIntent(
        strategy_id="strategy",
        strategy_version=1,
        rules_hash="rules",
        data_snapshot_id="snapshot",
        source="target_portfolio",
        event_ts="2026-08-01T09:30:00+05:30",
        cycle_id="cycle",
        symbol="AAA",
        direction=side,
        intended_entry=100,
        intended_risk_pct=0.5,
        stop_price=90,
        target_price=120,
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=10,
        required_quantity=10,
    )


def _order(intent, *, side="BUY", approved=10, filled=10, average=102):
    return SimpleNamespace(
        order_id="oms-1",
        trade_intent_id=intent.record_id,
        target_portfolio_id="portfolio",
        strategy_id="strategy",
        symbol="AAA",
        side=side,
        approved_quantity=approved,
        filled_quantity=filled,
        average_fill_price=average,
    )


def _transition(sequence, event_type, event_at):
    return TransitionSnapshot(
        transition_id=f"transition-{sequence}",
        order_id="oms-1",
        sequence=sequence,
        from_status="",
        to_status="",
        event_type=event_type,
        event_at=event_at,
        actor="test",
        reason="",
        external_event_id="",
        metadata={},
    )


def _timeline():
    return (
        _transition(1, "INTENT_ACCEPTED", "2026-08-01T09:30:05+05:30"),
        _transition(2, "RISK_APPROVED", "2026-08-01T09:31:00+05:30"),
        _transition(3, "SUBMISSION_PREPARED", "2026-08-01T09:32:00+05:30"),
        _transition(4, "BROKER_ACKNOWLEDGED", "2026-08-01T09:32:10+05:30"),
    )


def _fills():
    return (
        FillSnapshot(
            fill_id="fill-1",
            order_id="oms-1",
            external_fill_id="broker-fill-1",
            quantity=4,
            price=102,
            filled_at="2026-08-01T09:32:30+05:30",
            metadata={},
        ),
        FillSnapshot(
            fill_id="fill-2",
            order_id="oms-1",
            external_fill_id="broker-fill-2",
            quantity=6,
            price=102,
            filled_at="2026-08-01T09:32:40+05:30",
            metadata={},
        ),
    )


def test_tca_attributes_delay_fill_and_explicit_cost_without_double_counting():
    intent = _intent()
    assessment = assess_entry_execution(
        intent=intent,
        order=_order(intent),
        transitions=_timeline(),
        fills=_fills(),
        submission_reference_price=101,
        explicit_fees=5,
        estimated_spread_bps=10,
        estimated_market_impact=3,
        opportunity_cost=2,
    )

    assert assessment.complete is True
    assert assessment.quantity == 10
    assert assessment.decision_to_submission_cost == 10
    assert assessment.submission_to_fill_cost == 10
    assert assessment.total_price_shortfall == 20
    assert assessment.explicit_fees == 5
    assert assessment.estimated_spread_cost == 1
    assert assessment.estimated_market_impact == 3
    assert assessment.implementation_shortfall == 27
    assert assessment.implementation_shortfall_bps == 270
    assert assessment.signal_to_risk_seconds == 60
    assert assessment.risk_to_submission_seconds == 60
    assert assessment.submission_to_ack_seconds == 10
    assert assessment.ack_to_first_fill_seconds == 20
    assert assessment.submission_to_final_fill_seconds == 40
    assert assessment.warnings == ()


def test_partial_fill_is_assessed_but_not_labelled_complete():
    intent = _intent()
    fills = (_fills()[0],)
    assessment = assess_entry_execution(
        intent=intent,
        order=_order(intent, filled=4, average=102),
        transitions=_timeline(),
        fills=fills,
        submission_reference_price=101,
    )

    assert assessment.complete is False
    assert assessment.quantity == 4
    assert "PARTIAL_FILL_ONLY" in assessment.warnings


def test_missing_submission_benchmark_is_explicit():
    intent = _intent()
    assessment = assess_entry_execution(
        intent=intent,
        order=_order(intent),
        transitions=_timeline(),
        fills=_fills(),
    )

    assert assessment.complete is False
    assert assessment.submission_reference_price == 100
    assert assessment.total_price_shortfall == 20
    assert "SUBMISSION_REFERENCE_PRICE_UNAVAILABLE" in assessment.warnings


def test_sell_price_improvement_uses_the_correct_sign():
    intent = _intent(side="SHORT")
    fills = (
        FillSnapshot(
            fill_id="sell-fill",
            order_id="oms-1",
            external_fill_id="sell-fill",
            quantity=10,
            price=101,
            filled_at="2026-08-01T09:32:30+05:30",
            metadata={},
        ),
    )
    assessment = assess_entry_execution(
        intent=intent,
        order=_order(intent, side="SELL", average=101),
        transitions=_timeline(),
        fills=fills,
        submission_reference_price=100,
    )

    assert assessment.total_price_shortfall == -10
    assert assessment.implementation_shortfall == -10


def test_intent_order_mismatch_fails_closed():
    intent = _intent()
    wrong_order = _order(intent)
    wrong_order.trade_intent_id = "different-intent"

    with pytest.raises(TcaInputError):
        assess_entry_execution(
            intent=intent,
            order=wrong_order,
            transitions=_timeline(),
            fills=_fills(),
            submission_reference_price=101,
        )


def test_assessment_id_is_deterministic_and_store_is_idempotent(tmp_path):
    intent = _intent()
    kwargs = dict(
        intent=intent,
        order=_order(intent),
        transitions=_timeline(),
        fills=_fills(),
        submission_reference_price=101,
        explicit_fees=5,
    )
    first = assess_entry_execution(**kwargs)
    second = assess_entry_execution(**kwargs)
    assert first.assessment_id == second.assessment_id

    store = TcaStore(tmp_path / "tca.db")
    store.record(first)
    store.record(second)

    assert store.summary()["assessments"] == 1
    assert store.summary()["complete_assessments"] == 1
    assert store.latest_for_order("oms-1")["assessment_id"] == first.assessment_id


def test_negative_cost_inputs_are_rejected():
    intent = _intent()
    with pytest.raises(TcaInputError):
        assess_entry_execution(
            intent=intent,
            order=_order(intent),
            transitions=_timeline(),
            fills=_fills(),
            submission_reference_price=101,
            explicit_fees=-1,
        )
