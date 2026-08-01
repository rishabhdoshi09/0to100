from datetime import datetime, timezone

from product.institutional_readiness import BLOCKED, READY, build_institutional_readiness


def _base_payload(**overrides):
    payload = {
        "data": {
            "bhavcopy": {"ready": True, "sessions": 500, "minimum_sessions": 60},
            "snapshot": {"ready": True, "snapshot_id": "snap-1"},
        },
        "market": {"available": True},
        "scan": {"available": True},
        "paper": {"available": True},
        "autonomy": {"available": True},
        "operations": {"running": True},
        "capabilities": {},
        "now": datetime(2026, 8, 1, 6, 30, tzinfo=timezone.utc),
    }
    payload.update(overrides)
    return payload


def test_research_ready_does_not_unlock_live_execution():
    report = build_institutional_readiness(**_base_payload())

    assert report["system_state"] == "PAPER_ONLY"
    assert report["deployment"]["paper"]["allowed"] is True
    assert report["deployment"]["limited_live"]["allowed"] is False
    assert report["deployment"]["live"]["allowed"] is False
    assert next(item for item in report["domains"] if item["key"] == "research")["status"] == READY
    assert next(item for item in report["domains"] if item["key"] == "execution")["status"] == BLOCKED


def test_history_and_snapshot_do_not_certify_institutional_data():
    report = build_institutional_readiness(**_base_payload())
    data = next(item for item in report["domains"] if item["key"] == "data")

    assert data["status"] != READY
    assert "corporate_actions_point_in_time" in data["blockers"]
    assert "universe_history_point_in_time" in data["blockers"]
    assert "symbol_lineage_complete" in data["blockers"]


def test_missing_capability_flags_fail_closed():
    report = build_institutional_readiness(**_base_payload(capabilities={"durable_oms": True}))
    execution = next(item for item in report["domains"] if item["key"] == "execution")

    assert execution["status"] == BLOCKED
    assert "broker_event_ingestion" in execution["blockers"]
    assert "ambiguous_order_quarantine" in execution["blockers"]


def test_limited_live_requires_all_domains_and_explicit_owner_approval():
    capabilities = {
        "registered_candidate": True,
        "historical_net_edge_passed": True,
        "forward_evidence_passed": True,
        "capacity_assessed": True,
        "corporate_actions_point_in_time": True,
        "universe_history_point_in_time": True,
        "symbol_lineage_complete": True,
        "trading_calendar_validated": True,
        "fundamental_availability_dates_or_not_required": True,
        "strategy_version_frozen": True,
        "feature_parity_certified": True,
        "universe_rule_parity_certified": True,
        "cost_model_parity_certified": True,
        "exit_rule_parity_certified": True,
        "canonical_target_portfolio": True,
        "portfolio_constraints_enforced": True,
        "open_order_exposure_included": True,
        "durable_oms": True,
        "broker_event_ingestion": True,
        "idempotent_submission": True,
        "partial_fill_recovery": True,
        "ambiguous_order_quarantine": True,
        "independent_risk_governor": True,
        "pending_exposure_risk": True,
        "independent_kill_switch": True,
        "fail_closed_live_gate": True,
        "startup_reconciliation": True,
        "continuous_broker_reconciliation": True,
        "position_and_cash_reconciliation": True,
        "mismatch_quarantine": True,
        "protection_manager": True,
        "partial_fill_protection": True,
        "protection_verification": True,
        "orphan_protection_detection": True,
        "duplicate_worker_detection": True,
        "restart_recovery_tested": True,
        "failure_injection_tested": True,
        "operator_runbook_available": True,
    }
    report = build_institutional_readiness(**_base_payload(capabilities=capabilities))
    assert report["deployment"]["limited_live"]["status"] == BLOCKED
    assert "limited_live_owner_approval_missing" in report["deployment"]["limited_live"]["blockers"]

    capabilities["limited_live_owner_approved"] = True
    report = build_institutional_readiness(**_base_payload(capabilities=capabilities))
    assert report["system_state"] == "LIMITED_LIVE_ELIGIBLE"
    assert report["deployment"]["limited_live"]["allowed"] is True
    assert report["deployment"]["live"]["allowed"] is False
