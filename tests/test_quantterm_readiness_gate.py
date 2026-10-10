"""A public/live readiness badge must fail closed on incomplete evidence."""
from __future__ import annotations

from scripts.quantterm_readiness_gate import evaluate


def valid_responses():
    return {
        "health": {
            "ok": True, "live_locked": True, "live_lock_verified": True,
            "live_execution_authorized": False,
        },
        "access": {
            "public_read_only": True,
            "operator_token_configured": True,
            "mutation_policy": "OPERATOR_TOKEN_REQUIRED",
        },
        "dashboard": {
            "generated_at": "2026-10-11T00:30:00+00:00",
            "dashboard_cache": {"status": "FRESH"},
            "data": {"history_current": True},
            "scan": {
                "available": True,
                "scanned_at": "2026-10-11T00:20:00+00:00",
                "requested_universe": 2000,
                "universe_size": 1980,
                "coverage": {"checked": 1980, "analysis_errors": 0},
                "desk_overlays": {
                    "recommendations": "saved",
                    "decision_discovery": "saved",
                },
                "provenance": {
                    "data_current": True, "price_data_as_of": "2026-10-09",
                },
            },
            "paper": {"available": True, "last_cycle": {"status": "OBSERVING"}},
            "long_term": {"available": False},
        },
        "decision-simulation-gate": {
            "status_cache": {"status": "FRESH"},
            "scan_fresh": True, "approved": False,
        },
    }


def timings():
    return {key: 0.1 for key in valid_responses()}


def test_local_readiness_requires_current_sources_but_not_paper_approval():
    result = evaluate(valid_responses(), timings_s=timings(), public=True)
    assert result["verdict"] == "LOCAL_READ_GATE_PASS"
    assert result["blockers"] == []


def test_stale_dashboard_or_gate_never_produces_pass():
    responses = valid_responses()
    responses["dashboard"]["dashboard_cache"]["status"] = "STALE"
    responses["decision-simulation-gate"]["status_cache"]["status"] = "BOOTSTRAPPING"
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert any("FRESH" in entry for entry in result["blockers"])


def test_unlocked_live_money_is_always_hold():
    responses = valid_responses()
    responses["health"]["live_locked"] = False
    responses["health"]["live_execution_authorized"] = True
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert any("Live" in x for x in result["blockers"])


def test_unverified_coverage_or_missing_provenance_is_hold():
    responses = valid_responses()
    responses["dashboard"]["scan"]["requested_universe"] = 2384
    responses["dashboard"]["scan"]["coverage"]["checked"] = 2179
    responses["dashboard"]["scan"]["provenance"]["data_current"] = False
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert result["metrics"]["scan_coverage_pct"] < 95
    assert any("coverage" in x.lower() for x in result["blockers"])


def test_public_mode_never_passes_without_mutation_authorization():
    responses = valid_responses()
    responses["access"]["operator_token_configured"] = False
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "HOLD"
    assert evaluate(responses, timings_s=timings(), public=False)["verdict"] == "LOCAL_READ_GATE_PASS"


def test_slow_or_missing_api_endpoint_fails():
    assert evaluate(valid_responses(), timings_s={
        **timings(), "dashboard": 50.0,
    })["verdict"] == "HOLD"
    partial = valid_responses()
    del partial["dashboard"]
    assert evaluate(partial, timings_s=timings())["verdict"] == "HOLD"


def test_completed_scan_without_decision_publication_is_hold():
    responses = valid_responses()
    responses["dashboard"]["scan"]["desk_overlays"]["decision_discovery"] = "error"
    report = evaluate(responses, timings_s=timings())
    assert report["verdict"] == "HOLD"
    assert any("Decision discovery" in x for x in report["blockers"])
