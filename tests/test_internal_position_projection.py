from execution.oms.store import OmsStore
from execution.protection import BrokerProtectionSnapshot, ProtectionStore
from execution.protection.models import VERIFIED
from execution.reconciliation.internal_state import project_internal_positions
from research.intelligence import schemas as SC


def _intent(symbol="AAA", quantity=10):
    return SC.TradeIntent(
        strategy_id=f"strategy-{symbol}",
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


def _filled(oms, *, symbol="AAA", approved=10, fill=10):
    order = oms.ingest_intent(_intent(symbol, approved))
    order = oms.approve_risk(
        order.order_id,
        risk_decision_id=f"risk-{symbol}",
        approved_quantity=approved,
    )
    order = oms.prepare_submission(order.order_id, submission_token=f"submit-{symbol}")
    return oms.record_fill(
        order.order_id,
        external_fill_id=f"fill-{symbol}",
        quantity=fill,
        price=101,
        broker_order_id=f"broker-{symbol}",
    )


def test_filled_orders_reconstruct_expected_position(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled(oms)

    projection = project_internal_positions(oms)

    assert projection.reconciled_candidate is True
    assert projection.filled_order_ids == (order.order_id,)
    assert projection.positions[0].symbol == "AAA"
    assert projection.positions[0].quantity == 10
    assert projection.positions[0].average_price == 101
    assert projection.positions[0].protected_quantity == 0


def test_verified_plan_contributes_exact_protected_quantity(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled(oms)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit")
    protections.acknowledge(
        plan.plan_id,
        broker_protection_id="gtt-1",
        protected_quantity=10,
        stop_reference="stop-1",
        target_reference="target-1",
    )
    verified = protections.verify(
        plan.plan_id,
        BrokerProtectionSnapshot(
            broker_protection_id="gtt-1",
            order_id=order.order_id,
            symbol="AAA",
            active=True,
            quantity=10,
            stop_price=90,
            target_price=120,
            stop_reference="stop-1",
            target_reference="target-1",
        ),
    )
    assert verified.status == VERIFIED

    projection = project_internal_positions(oms, protections)

    assert projection.positions[0].protected_quantity == 10


def test_cancelled_partial_fill_remains_an_expected_position(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    partial = _filled(oms, approved=10, fill=4)
    oms.cancel(partial.order_id, reason="remaining quantity cancelled")

    projection = project_internal_positions(oms)

    assert projection.positions[0].quantity == 4
    assert partial.order_id in projection.filled_order_ids


def test_closed_order_is_removed_from_expected_position(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled(oms)
    oms.mark_protection_pending(order.order_id)
    oms.mark_protected(order.order_id, protection_reference="gtt-1")
    oms.mark_exit_pending(order.order_id, reason="exit submitted")
    oms.mark_closed(order.order_id, reason="broker position flat")

    projection = project_internal_positions(oms)

    assert projection.positions == ()


def test_uncertain_filled_order_remains_visible_and_unresolved(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    partial = _filled(oms, approved=10, fill=4)
    oms.mark_submission_uncertain(partial.order_id, reason="broker endpoint unavailable")

    projection = project_internal_positions(oms)

    assert projection.positions[0].quantity == 4
    assert projection.unresolved_order_ids == (partial.order_id,)
    assert projection.reconciled_candidate is False
