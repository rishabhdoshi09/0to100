from __future__ import annotations

from datetime import datetime, timezone

from product.operator_health import enrich_autonomy_payload


def _epoch(day: str) -> float:
    return datetime.fromisoformat(f"{day}T12:00:00+00:00").timestamp()


def test_historical_failures_do_not_poison_current_health(monkeypatch):
    monkeypatch.setattr("product.operator_health._today", lambda: "2026-08-28")
    payload = {
        "running": True,
        "plain_state": "old",
        "explanation": "old",
        "active_failures": [],
        "active_job": {},
        "jobs_recent": [
            {
                "job_type": "data_refresh",
                "status": "PERMANENT_FAILED",
                "scheduled_for": _epoch("2026-08-27"),
                "critical": True,
                "error_code": "OLD_FAILURE",
            },
            {
                "job_type": "learning_cycle",
                "status": "SUCCEEDED",
                "scheduled_for": _epoch("2026-08-28"),
                "critical": False,
            },
        ],
    }

    out = enrich_autonomy_payload(payload)
    assert out["operator_state"] == "HEALTHY"
    assert out["active_failures"] == []
    assert out["historical_job_counts"]["PERMANENT_FAILED"] == 1
    assert out["learning_status"] == "CURRENT"
    assert "Historical ledger" in out["explanation"]


def test_background_refresh_is_working_not_frozen(monkeypatch):
    monkeypatch.setattr("product.operator_health._today", lambda: "2026-08-28")
    payload = {
        "running": True,
        "active_failures": [],
        "active_job": {},
        "jobs_recent": [
            {
                "job_type": "data_refresh",
                "status": "PENDING",
                "scheduled_for": _epoch("2026-08-28"),
                "critical": True,
                "error_code": "DATA_REFRESH_IN_PROGRESS",
            }
        ],
    }

    out = enrich_autonomy_payload(payload)
    assert out["operator_state"] == "WORKING"
    assert out["data_refresh_background"] is True
    assert out["learning_status"] == "WAITING_FOR_FRESH_EOD_DATA"
    assert "background" in out["plain_state"].lower()


def test_current_permanent_failure_is_visible(monkeypatch):
    monkeypatch.setattr("product.operator_health._today", lambda: "2026-08-28")
    payload = {
        "running": True,
        "active_failures": [],
        "active_job": {},
        "jobs_recent": [
            {
                "job_type": "outcome_resolution",
                "status": "PERMANENT_FAILED",
                "scheduled_for": _epoch("2026-08-28"),
                "critical": True,
                "error_code": "OUTCOME_ERROR",
            }
        ],
    }

    out = enrich_autonomy_payload(payload)
    assert out["operator_state"] == "DEGRADED"
    assert any(item.startswith("JOB_FAILED:outcome_resolution") for item in out["active_failures"])
