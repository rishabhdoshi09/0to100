from execution.oms import PROTECTED
from execution.oms.store import OmsStore
from execution.protection import ProtectionStore
from execution.reconciliation.snapshot_store import BrokerSnapshotStore
from execution.reconciliation.store import ReconciliationReportStore
from execution.reconciliation.zerodha_cycle import run_zerodha_observation_cycle
from research.intelligence import schemas as SC


def _intent(quantity=10):
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


def _filled_order(oms):
    order = oms.ingest_intent(_intent())
    order = oms.approve_risk(
        order.order_id,
        risk_decision_id="risk-1",
        approved_quantity=10,
    )
    order = oms.prepare_submission(order.order_id, submission_token="submit-1")
    return oms.record_fill(
        order.order_id,
        external_fill_id="trade-1",
        quantity=10,
        price=101,
        broker_order_id="broker-order-1",
    )


class HealthyClient:
    def __init__(self, order_id):
        self.order_id = order_id

    def orders(self):
        return [
            {
                "order_id": "broker-order-1",
                "status": "COMPLETE",
                "tradingsymbol": "AAA",
                "transaction_type": "BUY",
                "quantity": 10,
                "filled_quantity": 10,
                "average_price": 101,
                "tag": "submit-1",
            }
        ]

    def trades(self):
        return [
            {
                "trade_id": "trade-1",
                "order_id": "broker-order-1",
                "tradingsymbol": "AAA",
                "transaction_type": "BUY",
                "quantity": 10,
                "average_price": 101,
            }
        ]

    def positions(self):
        return {
            "net": [
                {
                    "tradingsymbol": "AAA",
                    "quantity": 10,
                    "average_price": 101,
                    "product": "CNC",
                }
            ]
        }

    def margins(self):
        return {"equity": {"available": {"cash": 100_000}, "net": 100_000}}

    def get_gtts(self):
        return [
            {
                "id": "gtt-1",
                "status": "active",
                "metadata": {"order_id": self.order_id},
                "condition": {
                    "tradingsymbol": "AAA",
                    "trigger_values": [90, 120],
                },
                "orders": [
                    {
                        "transaction_type": "SELL",
                        "quantity": 10,
                        "price": 89.5,
                        "order_id": "stop-1",
                    },
                    {
                        "transaction_type": "SELL",
                        "quantity": 10,
                        "price": 120,
                        "order_id": "target-1",
                    },
                ],
                "updated_at": "2026-08-01T10:00:00+05:30",
            }
        ]

    def place_order(self, **kwargs):
        raise AssertionError("observation cycle attempted broker mutation")


class IncompletePositionsClient(HealthyClient):
    def positions(self):
        raise TimeoutError("positions unavailable")


class IncompleteProtectionClient(HealthyClient):
    def get_gtts(self):
        raise TimeoutError("gtt unavailable")


def _stores(tmp_path):
    return (
        OmsStore(tmp_path / "oms.db"),
        ProtectionStore(tmp_path / "protection.db"),
        BrokerSnapshotStore(tmp_path / "snapshots.db"),
        ReconciliationReportStore(tmp_path / "reconciliation.db"),
    )


def test_complete_cycle_reconciles_and_verifies_protection_without_broker_mutation(tmp_path):
    oms, protections, snapshots, reports = _stores(tmp_path)
    order = _filled_order(oms)
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")

    result = run_zerodha_observation_cycle(
        client=HealthyClient(order.order_id),
        oms_store=oms,
        protection_store=protections,
        snapshot_store=snapshots,
        report_store=reports,
        observed_at="2026-08-01T04:30:00+00:00",
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.snapshot_complete is True
    assert result.reconciliation.report.healthy is True
    assert result.protection.entry_freeze_required is False
    assert result.entries_allowed is True
    assert result.blockers == ()
    assert oms.get(order.order_id).status == PROTECTED
    assert protections.get(plan.plan_id).fully_protected is True
    assert snapshots.summary()["complete_snapshots"] == 1
    assert reports.summary()["latest_status"] == "HEALTHY"


def test_incomplete_positions_lane_freezes_without_deterministic_repairs(tmp_path):
    oms, protections, snapshots, reports = _stores(tmp_path)
    order = _filled_order(oms)
    plan = protections.ensure_for_order(order)
    protections.prepare_submission(plan.plan_id, request_token="protect-submit-1")

    result = run_zerodha_observation_cycle(
        client=IncompletePositionsClient(order.order_id),
        oms_store=oms,
        protection_store=protections,
        snapshot_store=snapshots,
        report_store=reports,
        observed_at="2026-08-01T04:30:00+00:00",
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.snapshot_complete is False
    assert result.entries_allowed is False
    assert "BROKER_ACCOUNT_SNAPSHOT_INCOMPLETE" in result.blockers
    assert "RECONCILIATION_ENTRY_FREEZE" in result.blockers
    assert result.reconciliation.applied_repairs == ()
    assert reports.summary()["latest_status"] == "INCOMPLETE"


def test_incomplete_protection_lane_makes_position_authority_incomplete(tmp_path):
    oms, protections, snapshots, reports = _stores(tmp_path)
    order = _filled_order(oms)

    result = run_zerodha_observation_cycle(
        client=IncompleteProtectionClient(order.order_id),
        oms_store=oms,
        protection_store=protections,
        snapshot_store=snapshots,
        report_store=reports,
        observed_at="2026-08-01T04:30:00+00:00",
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.snapshot_complete is False
    assert result.entries_allowed is False
    assert "BROKER_PROTECTION_SNAPSHOT_INCOMPLETE" in result.blockers
    assert "RECONCILIATION_ENTRY_FREEZE" in result.blockers
    assert "PROTECTION_ENTRY_FREEZE" in result.blockers
    assert result.reconciliation.report.status == "INCOMPLETE"


def test_unknown_broker_order_freezes_even_when_all_lanes_are_complete(tmp_path):
    oms, protections, snapshots, reports = _stores(tmp_path)

    class UnknownOrderClient(HealthyClient):
        def __init__(self):
            super().__init__("unknown-internal")

        def positions(self):
            return {"net": []}

        def trades(self):
            return []

        def get_gtts(self):
            return []

    result = run_zerodha_observation_cycle(
        client=UnknownOrderClient(),
        oms_store=oms,
        protection_store=protections,
        snapshot_store=snapshots,
        report_store=reports,
        observed_at="2026-08-01T04:30:00+00:00",
        internal_cash=100_000,
        internal_available_margin=100_000,
    )

    assert result.snapshot_complete is True
    assert result.entries_allowed is False
    assert "RECONCILIATION_ENTRY_FREEZE" in result.blockers
    assert "UNKNOWN_BROKER_ORDER" in result.reconciliation.report.summary["issue_counts"]
