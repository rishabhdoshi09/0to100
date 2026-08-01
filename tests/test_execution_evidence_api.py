from types import SimpleNamespace

from execution.oms.models import FillSnapshot, TransitionSnapshot
from execution.protection.store import ProtectionStore
from execution.reconciliation.models import HEALTHY, ReconciliationReport
from execution.reconciliation.store import ReconciliationReportStore
from execution.tca.analyzer import assess_entry_execution
from execution.tca.store import TcaStore
from research.intelligence import schemas as SC

import terminal_product_api


def test_reconciliation_endpoint_projects_latest_persisted_report(tmp_path, monkeypatch):
    path = tmp_path / "reconciliation.db"
    store = ReconciliationReportStore(path)
    report = ReconciliationReport(
        report_id="recon-1",
        broker_snapshot_id="broker-snapshot-1",
        observed_at="2026-08-01T10:00:00+05:30",
        status=HEALTHY,
        entry_freeze_required=False,
        summary={"snapshot_complete": True},
    )
    store.record(report)
    monkeypatch.setattr(terminal_product_api, "RECONCILIATION_DB", path)

    payload = terminal_product_api.reconciliation_status()

    assert payload["available"] is True
    assert payload["certified_for_live"] is False
    assert payload["broker_snapshot_connected"] is False
    assert payload["summary"]["latest_status"] == HEALTHY
    assert payload["latest"]["report_id"] == report.report_id


def test_protection_endpoint_projects_quantity_aware_plans(tmp_path, monkeypatch):
    path = tmp_path / "protection.db"
    store = ProtectionStore(path)
    order = SimpleNamespace(
        order_id="oms-1",
        symbol="AAA",
        filled_quantity=4,
        stop_price=90,
        target_price=120,
    )
    plan = store.ensure_for_order(order)
    monkeypatch.setattr(terminal_product_api, "PROTECTION_DB", path)

    payload = terminal_product_api.protection_status()

    assert payload["available"] is True
    assert payload["exchange_adapter_connected"] is False
    assert payload["certified_for_live"] is False
    assert payload["summary"]["entry_freeze_required"] is True
    assert payload["plans"][0]["plan_id"] == plan.plan_id
    assert payload["plans"][0]["required_quantity"] == 4


def test_tca_endpoint_projects_only_persisted_assessments(tmp_path, monkeypatch):
    path = tmp_path / "tca.db"
    intent = SC.TradeIntent(
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
        target_portfolio_id="portfolio",
        target_position_id="position",
        desired_quantity=10,
        required_quantity=10,
    )
    order = SimpleNamespace(
        order_id="oms-1",
        trade_intent_id=intent.record_id,
        side="BUY",
        approved_quantity=10,
        filled_quantity=10,
        average_fill_price=102,
    )
    transitions = (
        TransitionSnapshot(
            transition_id="trn-1", order_id="oms-1", sequence=1,
            from_status="", to_status="PROPOSED", event_type="INTENT_ACCEPTED",
            event_at="2026-08-01T09:30:05+05:30", actor="oms", reason="",
            external_event_id="", metadata={},
        ),
        TransitionSnapshot(
            transition_id="trn-2", order_id="oms-1", sequence=2,
            from_status="PROPOSED", to_status="RISK_APPROVED", event_type="RISK_APPROVED",
            event_at="2026-08-01T09:31:00+05:30", actor="risk", reason="",
            external_event_id="", metadata={},
        ),
        TransitionSnapshot(
            transition_id="trn-3", order_id="oms-1", sequence=3,
            from_status="RISK_APPROVED", to_status="SUBMISSION_PENDING",
            event_type="SUBMISSION_PREPARED",
            event_at="2026-08-01T09:32:00+05:30", actor="ems", reason="",
            external_event_id="", metadata={},
        ),
    )
    fills = (
        FillSnapshot(
            fill_id="fill-1", order_id="oms-1", external_fill_id="broker-fill-1",
            quantity=10, price=102, filled_at="2026-08-01T09:32:30+05:30", metadata={},
        ),
    )
    assessment = assess_entry_execution(
        intent=intent,
        order=order,
        transitions=transitions,
        fills=fills,
        submission_reference_price=101,
        explicit_fees=5,
    )
    store = TcaStore(path)
    store.record(assessment)
    monkeypatch.setattr(terminal_product_api, "TCA_DB", path)

    payload = terminal_product_api.tca_status()

    assert payload["available"] is True
    assert payload["live_fill_feed_connected"] is False
    assert payload["summary"]["assessments"] == 1
    assert payload["assessments"][0]["assessment_id"] == assessment.assessment_id


def test_execution_evidence_endpoints_do_not_create_missing_state(tmp_path, monkeypatch):
    recon = tmp_path / "missing-reconciliation.db"
    protection = tmp_path / "missing-protection.db"
    tca = tmp_path / "missing-tca.db"
    monkeypatch.setattr(terminal_product_api, "RECONCILIATION_DB", recon)
    monkeypatch.setattr(terminal_product_api, "PROTECTION_DB", protection)
    monkeypatch.setattr(terminal_product_api, "TCA_DB", tca)

    assert terminal_product_api.reconciliation_status()["available"] is False
    assert terminal_product_api.protection_status()["available"] is False
    assert terminal_product_api.tca_status()["available"] is False
    assert recon.exists() is False
    assert protection.exists() is False
    assert tca.exists() is False
