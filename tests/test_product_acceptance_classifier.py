"""Acceptance classification must inspect durable operation results, not only status."""
from __future__ import annotations

from scripts.run_product_acceptance import (
    classify_operation,
    collect_nested_blockers,
    grade_due_diligence,
    grade_forward_soak,
    grade_learning_dashboard,
    grade_market_reports,
    grade_paper_cycle_execution,
    grade_paper_status,
    grade_recommendations,
    grade_simulator,
    grade_stock_intelligence,
    is_expected_external_blocker,
    product_acceptance_verdict,
)


def test_data_prepare_succeeded_with_nested_fno_blocked_is_degraded():
    graded = classify_operation({
        "status": "SUCCEEDED",
        "result": {
            "history": {"current": True, "available_session": "2026-09-04"},
            "fno": {
                "blocked": True,
                "code": "FNO_UNIVERSE_UNAVAILABLE",
                "error": "Zerodha login or instrument cache is required",
            },
            "degraded": True,
            "missing_lanes": ["fno"],
        },
    })
    assert graded["status"] == "DEGRADED"
    assert "fno" in graded["blocker_reason"].lower()
    assert "FNO_UNIVERSE_UNAVAILABLE" in graded["blocker_reason"]


def test_clean_data_prepare_succeeded_is_pass():
    graded = classify_operation({
        "status": "SUCCEEDED",
        "result": {
            "history": {"current": True},
            "fno": {"mapped_underlyings": 180, "blocked": False},
            "degraded": False,
            "missing_lanes": [],
        },
    })
    assert graded == {"status": "PASS", "blocker_reason": ""}


def test_failed_operation_is_fail():
    graded = classify_operation({
        "status": "FAILED",
        "error_code": "RuntimeError",
        "error_message": "worker exploded",
        "result": {"degraded": True},
    })
    assert graded["status"] == "FAIL"
    assert "RuntimeError" in graded["blocker_reason"]


def test_cancelled_operation_is_fail():
    graded = classify_operation({"status": "CANCELLED", "error_message": "owner cancelled"})
    assert graded["status"] == "FAIL"


def test_explicit_external_blocker_is_blocked():
    graded = classify_operation({
        "status": "BLOCKED",
        "error_code": "FNO_UNIVERSE_UNAVAILABLE",
        "error_message": "LOGIN REQUIRED",
        "result": {"mapped_underlyings": 0},
    })
    assert graded["status"] == "BLOCKED"
    assert "FNO_UNIVERSE_UNAVAILABLE" in graded["blocker_reason"]


def test_blocked_without_explicit_blocker_is_fail():
    graded = classify_operation({
        "status": "BLOCKED",
        "error_code": "",
        "error_message": "",
        "result": {},
    })
    assert graded["status"] == "FAIL"
    assert "without explicit blocker" in graded["blocker_reason"]


def test_internal_exception_is_not_downgraded_to_degraded():
    graded = classify_operation({
        "status": "FAILED",
        "error_code": "TypeError",
        "error_message": "NoneType",
    })
    assert graded["status"] == "FAIL"
    nested = collect_nested_blockers({"degraded": True, "missing_lanes": ["fno"]})
    assert any("fno" in item for item in nested)


def test_empty_json_is_not_stock_intelligence_pass():
    assert grade_stock_intelligence({}, "TCS")["status"] == "FAIL"
    assert grade_stock_intelligence({"available": True}, "TCS")["status"] == "FAIL"
    ok = grade_stock_intelligence({
        "schema_version": 1,
        "symbol": "TCS",
        "sources": [{"name": "Official price history", "available": True}],
        "technical": {"available": True},
        "fundamentals": {"available": True},
    }, "TCS")
    assert ok["status"] == "PASS"


def test_empty_json_is_not_due_diligence_pass():
    assert grade_due_diligence({"ok": True}, "TCS")["status"] == "FAIL"
    ok = grade_due_diligence({
        "schema_version": 7,
        "symbol": "TCS",
        "engine": "StockResearchEngine",
        "kpis": [{"id": "sales"}],
        "research_coverage": {"coverage_pct": 40},
        "vs_technical_setup": "NEUTRAL",
    }, "TCS")
    assert ok["status"] == "PASS"


def test_recommendations_require_structured_categories():
    assert grade_recommendations({"hello": True})["status"] == "FAIL"
    assert grade_recommendations({"categories": [], "generated_at": "now"})["status"] == "FAIL"
    ok = grade_recommendations({
        "categories": [{"id": "momentum_breakouts", "cards": [{"symbol": "TCS"}]}],
        "generated_at": "2026-09-06T00:00:00Z",
        "scan_scanned_at": "2026-09-06T00:00:00Z",
        "records_status": "CURRENT",
    })
    assert ok["status"] == "PASS"
    assert ok["cards"] == 1


def test_market_reports_require_durable_reports():
    assert grade_market_reports({"missing_lanes": ["news"]})["status"] == "FAIL"
    ok = grade_market_reports({
        "reports": [{"id": "market_pulse_2026-09-06", "title": "Market Pulse", "date": "2026-09-06"}],
        "missing_lanes": ["fno"],
    })
    assert ok["status"] == "PASS"


def test_simulator_running_is_not_pass():
    assert grade_simulator({"status": "RUNNING", "decision": {"action": "WAIT"}})["status"] == "FAIL"
    assert grade_simulator({"status": "HISTORICAL_DECISION_UNAVAILABLE"})["status"] == "DEGRADED"
    ok = grade_simulator({
        "status": "SUCCEEDED",
        "schema_version": 1,
        "kind": "PAST_DECISION_SIMULATION",
        "original": {"action": "WAIT", "reason_code": "NO_SETUP"},
    })
    assert ok["status"] == "PASS"


def test_learning_and_soak_require_contracts():
    assert grade_learning_dashboard({"policies": []})["status"] == "FAIL"
    assert grade_learning_dashboard({
        "schema_version": 1,
        "live_locked": True,
        "policies": [],
        "counterfactuals": {"frozen": 0},
    })["status"] == "PASS"
    assert grade_forward_soak({"ok": True})["status"] == "FAIL"
    assert grade_forward_soak({
        "verification": {"lanes": {"SCAN": "PASS"}, "live_locked": True},
        "live_locked": True,
    })["status"] == "PASS"


def test_paper_cycle_request_is_not_execution():
    assert grade_paper_status({"available": True}, live_locked=True)["status"] == "FAIL"
    assert grade_paper_status({
        "available": True,
        "last_cycle": {},
        "open_positions": [],
    }, live_locked=True)["status"] == "PASS"
    unobserved = grade_paper_cycle_execution(observed=False)
    assert unobserved["status"] == "DEGRADED"
    assert "not observed" in unobserved["blocker_reason"]
    done = grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "paper cycle: NO_ELIGIBLE_TRADE"},
        last_cycle={"eligibility": "NO_ELIGIBLE_TRADE", "cycle_id": "c1"},
        observed=True,
    )
    assert done["status"] == "PASS"
    assert "NO_ELIGIBLE_TRADE" in done["blocker_reason"]


def test_verdict_allows_only_expected_fno_blocker():
    rows = [
        {"feature": "Canonical stack / readiness", "status": "PASS"},
        {"feature": "Market Scan", "status": "PASS"},
        {
            "feature": "Data refresh",
            "status": "DEGRADED",
            "blocker_reason": "degraded=true; fno blocked: FNO_UNIVERSE_UNAVAILABLE",
        },
        {
            "feature": "F&O refresh",
            "status": "BLOCKED",
            "blocker_reason": "FNO_UNIVERSE_UNAVAILABLE LOGIN REQUIRED",
        },
        {"feature": "Recommendations", "status": "PASS"},
    ]
    assert product_acceptance_verdict(rows, live_locked=True)["verdict"] == "PRODUCT ACCEPTANCE PASS"
    assert is_expected_external_blocker(rows[2]) is True

    hold_degraded = product_acceptance_verdict(
        rows + [{"feature": "News refresh", "status": "DEGRADED", "blocker_reason": "timeout"}],
        live_locked=True,
    )
    assert hold_degraded["verdict"] == "PRODUCT ACCEPTANCE HOLD"

    hold_fail = product_acceptance_verdict(
        [{"feature": "Market Scan", "status": "FAIL", "blocker_reason": "boom"}],
        live_locked=True,
    )
    assert hold_fail["verdict"] == "PRODUCT ACCEPTANCE HOLD"
    assert product_acceptance_verdict(rows, live_locked=False)["verdict"] == "PRODUCT ACCEPTANCE HOLD"
