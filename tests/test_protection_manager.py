from __future__ import annotations

from execution.oms import FILLED, PARTIALLY_FILLED, PROTECTED, PROTECTION_PENDING
from execution.oms.store import OmsStore
from execution.protection import (
    ACTIVE,
    ADJUSTMENT_REQUIRED,
    CANCEL_PENDING,
    ORPHANED,
    RECOVERY_REQUIRED,
    REQUIRED,
    SUBMISSION_PENDING,
    VERIFIED,
    BrokerProtectionSnapshot,
    ProtectionStore,
    sync_protection,
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


def _filled_order(oms: OmsStore, *, approved=10, first_fill=10, symbol="AAA"):
    order = oms.ingest_intent(_intent(approved, symbol))
    order = oms.approve_risk(
        order.order_id,
        risk_decision_id=f"risk-{symbol}",
        approved_quantity=approved,
    )
    order = oms.prepare_submission(order.order_id, submission_token=f"submit-{symbol}")
    return oms.record_fill(
        order.order_id,
        external_fill_id=f"fill-{symbol}-1",
        quantity=first_fill,
        price=100,
        broker_order_id=f"broker-{symbol}",
    )


def _broker(plan, *, quantity=None, stop=90, target=120, active=True, updated="v1"):
    return BrokerProtectionSnapshot(
        broker_protection_id=f"protect-broker-{plan.symbol}",
        order_id=plan.order_id,
        symbol=plan.symbol,
        active=active,
        quantity=plan.required_quantity if quantity is None else quantity,
        stop_price=stop,
        target_price=target,
        stop_reference=f"stop-{plan.symbol}",
        target_reference=f"target-{plan.symbol}",
        updated_at=updated,
    )


def test_partial_fill_creates_exact_protection_requirement(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms, approved=10, first_fill=4)
    protections = ProtectionStore(tmp_path / "protection.db")

    plan = protections.ensure_for_order(order)

    assert order.status == PARTIALLY_FILLED
    assert plan.status == REQUIRED
    assert plan.required_quantity == 4
    assert plan.protected_quantity == 0
    assert protections.summary()["entry_freeze_required"] is True


def test_prepare_acknowledge_and_verify_survive_restart(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms)
    path = tmp_path / "protection.db"
    protections = ProtectionStore(path)
    plan = protections.ensure_for_order(order)
    pending = protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")
    active = protections.acknowledge(
        plan.plan_id,
        broker_protection_id="protect-broker-AAA",
        protected_quantity=10,
        stop_reference="stop-AAA",
        target_reference="target-AAA",
        external_event_id="ack-1",
    )
    verified = protections.verify(
        plan.plan_id,
        _broker(active),
        external_event_id="verify-1",
    )

    assert pending.status == SUBMISSION_PENDING
    assert active.status == ACTIVE
    assert verified.status == VERIFIED
    assert verified.fully_protected is True
    assert protections.summary()["entry_freeze_required"] is False

    restarted = ProtectionStore(path)
    restored = restarted.get(plan.plan_id)
    assert restored.status == VERIFIED
    assert restored.protected_quantity == 10
    assert [item.to_status for item in restarted.history(plan.plan_id)] == [
        REQUIRED,
        SUBMISSION_PENDING,
        ACTIVE,
        VERIFIED,
    ]


def test_new_fill_after_verified_plan_requires_quantity_adjustment(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    partial = _filled_order(oms, approved=10, first_fill=4)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(partial)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")
    protections.acknowledge(
        plan.plan_id,
        broker_protection_id="protect-broker-AAA",
        protected_quantity=4,
        stop_reference="stop-AAA",
    )
    verified = protections.verify(plan.plan_id, _broker(plan, quantity=4))
    assert verified.status == VERIFIED

    full = oms.record_fill(
        partial.order_id,
        external_fill_id="fill-AAA-2",
        quantity=6,
        price=100,
        broker_order_id="broker-AAA",
    )
    resized = protections.ensure_for_order(full)

    assert full.status == FILLED
    assert resized.status == ADJUSTMENT_REQUIRED
    assert resized.required_quantity == 10
    assert resized.protected_quantity == 4
    assert resized.missing_quantity == 6


def test_new_fill_while_submission_pending_requires_recovery(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    partial = _filled_order(oms, approved=10, first_fill=4)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(partial)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")

    full = oms.record_fill(
        partial.order_id,
        external_fill_id="fill-AAA-2",
        quantity=6,
        price=100,
        broker_order_id="broker-AAA",
    )
    resized = protections.ensure_for_order(full)

    assert resized.status == RECOVERY_REQUIRED
    assert resized.required_quantity == 10
    assert resized.last_error_code == "PENDING_REQUEST_QUANTITY_STALE"


def test_under_cover_and_wrong_prices_never_verify(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")
    under = protections.acknowledge(
        plan.plan_id,
        broker_protection_id="protect-broker-AAA",
        protected_quantity=4,
        stop_reference="stop-AAA",
    )
    assert under.status == ADJUSTMENT_REQUIRED

    protections.prepare_submission(plan.plan_id, request_token="protect-submit-2")
    protections.acknowledge(
        plan.plan_id,
        broker_protection_id="protect-broker-AAA",
        protected_quantity=10,
        stop_reference="stop-AAA",
    )
    wrong = protections.verify(
        plan.plan_id,
        _broker(plan, stop=89),
        external_event_id="verify-wrong",
    )

    assert wrong.status == RECOVERY_REQUIRED
    assert wrong.last_error_code == "STOP_PRICE_MISMATCH"


def test_sync_marks_full_order_protected_only_after_broker_verification(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms)
    protections = ProtectionStore(tmp_path / "protection.db")

    first = sync_protection(
        oms_store=oms,
        protection_store=protections,
        broker_snapshot_complete=False,
    )
    plan = protections.get_by_order(order.order_id)
    assert plan is not None
    assert plan.status == REQUIRED
    assert oms.get(order.order_id).status == PROTECTION_PENDING
    assert first.entry_freeze_required is True

    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")
    second = sync_protection(
        oms_store=oms,
        protection_store=protections,
        broker_protections=(_broker(plan),),
        broker_snapshot_complete=True,
    )

    assert protections.get(plan.plan_id).status == VERIFIED
    assert oms.get(order.order_id).status == PROTECTED
    assert order.order_id in second.oms_protected_orders
    assert second.entry_freeze_required is False


def test_partial_fill_can_be_verified_without_overwriting_fill_state(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms, approved=10, first_fill=4)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")

    result = sync_protection(
        oms_store=oms,
        protection_store=protections,
        broker_protections=(_broker(plan, quantity=4),),
        broker_snapshot_complete=True,
    )

    assert protections.get(plan.plan_id).status == VERIFIED
    assert oms.get(order.order_id).status == PARTIALLY_FILLED
    assert result.entry_freeze_required is False


def test_complete_snapshot_detects_orphan_protection(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    protections = ProtectionStore(tmp_path / "protection.db")
    orphan = BrokerProtectionSnapshot(
        broker_protection_id="orphan-1",
        order_id="unknown-order",
        symbol="XYZ",
        active=True,
        quantity=5,
        stop_price=90,
        target_price=120,
        stop_reference="stop-orphan",
    )

    result = sync_protection(
        oms_store=oms,
        protection_store=protections,
        broker_protections=(orphan,),
        broker_snapshot_complete=True,
    )

    assert len(result.orphan_plans) == 1
    plan = protections.get(result.orphan_plans[0])
    assert plan.status == ORPHANED
    assert result.entry_freeze_required is True


def test_closed_oms_position_requests_protection_cancellation(tmp_path):
    oms = OmsStore(tmp_path / "oms.db")
    order = _filled_order(oms)
    protections = ProtectionStore(tmp_path / "protection.db")
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")
    protections.acknowledge(
        plan.plan_id,
        broker_protection_id="protect-broker-AAA",
        protected_quantity=10,
        stop_reference="stop-AAA",
    )
    protections.verify(plan.plan_id, _broker(plan))
    oms.mark_protection_pending(order.order_id)
    oms.mark_protected(order.order_id, protection_reference="protect-broker-AAA")
    oms.mark_exit_pending(order.order_id, reason="target reached")
    oms.mark_closed(order.order_id, reason="broker position flat")

    sync_protection(
        oms_store=oms,
        protection_store=protections,
        broker_snapshot_complete=False,
    )

    assert protections.get(plan.plan_id).status == CANCEL_PENDING
