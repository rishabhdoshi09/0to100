"""Zerodha is an execution/live-data lane, not a global QuantTerm health switch."""
from __future__ import annotations

from product import operator_health
from product import readiness


def test_valid_auth_and_snapshot_is_broker_ready(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "broker_live",
        lambda: {"ready": True, "status": "SESSION_VALID", "reason": "", "error_code": ""},
    )
    monkeypatch.setattr(readiness, "kite_snapshot_id", lambda: "snap-123")

    state = readiness.broker_status()

    assert state["state"] == "READY"
    assert state["auth_ready"] is True
    assert state["live_data_ready"] is True
    assert state["execution_ready"] is True
    assert state["login_required"] is False


def test_missing_token_requires_login_without_implying_system_failure(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "broker_live",
        lambda: {
            "ready": False,
            "status": "TOKEN_MISSING",
            "reason": "daily Zerodha login is required",
            "error_code": "KITE_TOKEN_MISSING",
        },
    )
    monkeypatch.setattr(readiness, "kite_snapshot_id", lambda: "")

    state = readiness.broker_status()

    assert state["state"] == "LOGIN_REQUIRED"
    assert state["login_required"] is True
    assert state["live_data_ready"] is False
    assert state["execution_ready"] is False
    assert state["reason_code"] == "KITE_TOKEN_MISSING"


def test_valid_session_without_snapshot_is_not_mislabeled_login_required(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "broker_live",
        lambda: {"ready": True, "status": "SESSION_VALID", "reason": "", "error_code": ""},
    )
    monkeypatch.setattr(readiness, "kite_snapshot_id", lambda: "")

    state = readiness.broker_status()

    assert state["state"] == "SNAPSHOT_REQUIRED"
    assert state["auth_ready"] is True
    assert state["login_required"] is False
    assert state["live_data_ready"] is False


def test_provider_outage_is_not_presented_as_login_required(monkeypatch):
    monkeypatch.setattr(
        readiness,
        "broker_live",
        lambda: {
            "ready": False,
            "status": "PROVIDER_UNAVAILABLE",
            "reason": "provider timed out",
            "error_code": "TIMEOUTERROR",
        },
    )
    monkeypatch.setattr(readiness, "kite_snapshot_id", lambda: "snap-old")

    state = readiness.broker_status()

    assert state["state"] == "UNAVAILABLE"
    assert state["login_required"] is False
    assert state["live_data_ready"] is False


def test_operator_health_keeps_broker_login_as_separate_lane(monkeypatch):
    monkeypatch.setattr(operator_health, "_today", lambda: "2026-09-03")
    monkeypatch.setattr(
        operator_health,
        "_broker_lane",
        lambda: {
            "state": "LOGIN_REQUIRED",
            "ready": False,
            "live_data_ready": False,
            "execution_ready": False,
            "auth_ready": False,
            "login_required": True,
            "auth_status": "TOKEN_MISSING",
            "reason_code": "KITE_TOKEN_MISSING",
            "detail": "daily Zerodha login is required",
            "snapshot_id": "",
        },
    )

    out = operator_health.enrich_autonomy_payload({
        "running": True,
        "active_failures": [],
        "jobs_recent": [],
        "active_job": {},
    })

    assert out["operator_state"] == "HEALTHY"
    assert out["broker"]["state"] == "LOGIN_REQUIRED"
    assert out["broker"]["live_data_ready"] is False
    assert out["active_failures"] == []
