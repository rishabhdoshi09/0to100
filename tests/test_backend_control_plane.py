"""Backend Control Plane — projection over existing Home authorities."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from product.backend_control_plane import FORBIDDEN_CONTROLS, SAFE_CONTROLS, _scrub_technical
from product.home_os import FAILED_RECOVERABLE, LOGIN_REQUIRED, NORMAL, NO_TRADE, PAUSED, PROBLEM, build_home_os
from product.runtime_capabilities import by_id, home_actions

IST = ZoneInfo("Asia/Kolkata")


def _open() -> datetime:
    return datetime(2026, 9, 1, 10, 45, tzinfo=IST)


def _ready_dash(**auto):
    autonomy = {"state": "RUNNING", "running": True, "operator_state": "HEALTHY", "active_failures": []}
    autonomy.update(auto)
    return {
        "autonomy": autonomy,
        "data": {
            "ready": True,
            "bhavcopy": {
                "ready": True,
                "latest_date": "2026-09-01",
                "current": True,
                "expected_latest_completed_session": "2026-09-01",
                "available_session": "2026-09-01",
                "reason_code": "HISTORY_CURRENT",
                "symbols": 3367,
                "sessions": 1798,
                "source": "official_nse_bhavcopy",
            },
        },
    }


def test_ready_data_lane_explains_readiness_without_primary():
    os = build_home_os(
        dashboard=_ready_dash(),
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "headline": "No paper trade taken", "decision": "NO_TRADE", "reasons": ["ENTRY_TOO_EXTENDED"], "taken": []},
        soak={"real_forward_observations": 0, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    data = os["system"]["data"]
    assert data["status"] == "Ready"
    assert data["status_code"] == "READY"
    assert data["what"]
    assert data["current"]
    assert data["next"]
    assert data.get("primary_action") in (None, {})
    assert not data.get("primary_action")
    refresh = [a for a in data.get("secondary_actions") or [] if a["control"] == "REFRESH_DATA_NOW"]
    assert refresh and refresh[0]["label"] == "Refresh"
    assert os["live_locked"] is True


def test_working_data_lane_shows_current_next_without_duplicate_refresh():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "DATA_REFRESHING",
                "running": True,
                "operator_state": "WORKING",
                "data_refresh_background": True,
                "active_job": {"job_type": "data_refresh", "job_id": "job-1", "started_at": 1},
            },
            "data": {"ready": False, "bhavcopy": {"ready": False, "latest_date": "2026-08-28", "current": False, "expected_latest_completed_session": "2026-09-01", "available_session": "2026-08-28", "reason_code": "HISTORY_STALE"}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={},
        operations={"active": [{"kind": "DATA_PREPARE", "status": "RUNNING", "operation_id": "op-9", "message": "snapshot refresh is still running"}], "recent": []},
        now=_open(),
    )
    data = os["system"]["data"]
    auto = os["system"]["automation"]
    assert data["status"] == "Working"
    assert data["current"]
    assert data["next"]
    assert not data.get("primary_action")
    assert not any((a.get("control") == "REFRESH_DATA_NOW") for a in data.get("secondary_actions") or [])
    assert auto["status"] in {"Working", "Waiting"}
    assert auto.get("waiting_for") == "Market data" or auto["status"] == "Working"
    assert auto.get("current")
    assert auto.get("next")
    assert not auto.get("primary_action")


def test_waiting_dependency_explains_official_session():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "DATA_READY", "running": True, "operator_state": "HEALTHY"},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "latest_date": "2026-08-28",
                    "current": False,
                    "expected_latest_completed_session": "2026-09-01",
                    "available_session": "2026-08-28",
                    "reason_code": "HISTORY_STALE",
                    "symbols": 3000,
                    "sessions": 1700,
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-08-28T05:00:00+00:00", "records": [1]},
        operations={"active": [], "recent": []},
        now=_open(),
    )
    data = os["system"]["data"]
    assert data["status"] == "Waiting"
    assert "Official session" in (data.get("waiting_for") or "") or "later official" in (data.get("meaning") or "").lower() or "waiting for market data" in (data.get("summary") or "").lower()
    assert data.get("dependencies")


def test_needs_you_zerodha_primary_is_login_without_secrets():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "AUTH_REQUIRED",
                "running": True,
                "live_feed": {"connected": False, "access_token": "SHOULD_NOT_LEAK", "last_error": "token=abc"},
            },
            "data": {"ready": True},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["state"] in {LOGIN_REQUIRED, NORMAL, NO_TRADE}
    assert os["need_me"] is True
    assert os["broker"]["login_required"] is True
    zed = os["system"]["zerodha"]
    assert zed["status"] == "Needs you"
    assert zed["primary_action"]["label"] == "Login to Zerodha"
    assert zed["primary_action"]["kind"] == "instruction"
    assert "WAITING FOR ZERODHA LOGIN" in zed["detail"]
    dumped = str(zed)
    assert "SHOULD_NOT_LEAK" not in dumped
    assert "token=abc" not in dumped
    assert "access_token" not in (zed.get("technical") or {})


def test_problem_retry_maps_to_refresh_data_now():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": False}},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={},
        operations={"active": [], "recent": [{"kind": "DATA_PREPARE", "status": "FAILED", "error": "HISTORY_STALE"}]},
        now=_open(),
    )
    assert os["state"] == FAILED_RECOVERABLE
    data = os["system"]["data"]
    assert data["status"] == "Problem"
    assert data["primary_action"]["control"] == "REFRESH_DATA_NOW"
    assert data["primary_action"]["label"] == "Retry"
    assert data["primary_action"]["control"] in SAFE_CONTROLS
    assert data["primary_action"]["control"] not in FORBIDDEN_CONTROLS


def test_paper_pause_resume_map_to_existing_controls():
    paused = build_home_os(
        dashboard=_ready_dash(),
        paper={"enabled": False, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["PAPER_TRADING_DISABLED"]},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [1]},
        now=_open(),
    )
    assert paused["state"] == PAUSED
    bot = paused["system"]["paper_bot"]
    assert bot["status"] == "Needs you"
    assert bot["primary_action"]["control"] == "RESUME_NEW_PAPER_ENTRIES"
    assert bot["primary_action"]["label"] == "Resume"

    running = build_home_os(
        dashboard=_ready_dash(),
        paper={"enabled": True, "open_positions": [{"symbol": "TCS", "entry_price": 100, "stop_price": 94, "target_price": 115, "risk_amount": 1000}], "closed_trades": [], "open_risk": 1000},
        why={"available": True, "taken": [{"symbol": "TCS"}], "headline": "Took 1 paper trade(s): TCS"},
        soak={"real_forward_observations": 1, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    on = running["system"]["paper_bot"]
    assert on["status"] == "Ready"
    assert not on.get("primary_action")
    pause = [a for a in on.get("secondary_actions") or [] if a["control"] == "PAUSE_NEW_PAPER_ENTRIES"]
    assert pause
    assert on["positions"][0]["symbol"] == "TCS"
    assert on["positions"][0]["stop"] == 94
    assert on["positions"][0]["target"] == 115


def test_learning_verify_uses_canonical_control():
    os = build_home_os(
        dashboard=_ready_dash(),
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["NO_TRADE"], "decision": "NO_TRADE"},
        soak={
            "real_forward_observations": 47,
            "settled_trades": 21,
            "execution_adjusted_coverage_pct": 83,
            "insufficient_evidence": True,
            "FORWARD_SOAK_STATUS": "COLLECTING",
            "gross_expectancy": 0.4,
            "execution_adjusted_expectancy": 0.1,
        },
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [1]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    learn = os["system"]["learning"]
    verify = [a for a in learn.get("secondary_actions") or [] if a["control"] == "VERIFY_FORWARD_SOAK"]
    assert verify and verify[0]["label"] == "Verify now"
    assert by_id("forward_soak_verify")["control"] == "VERIFY_FORWARD_SOAK"
    assert "Real observations: 47" in (learn.get("current") or learn.get("summary") or "")
    assert learn["technical"]["note"]
    assert "proven alpha" in learn["technical"]["note"].lower()


def test_historical_job_failures_are_not_current_automation_problems():
    os = build_home_os(
        dashboard=_ready_dash(
            jobs={"PENDING": 1, "SUCCEEDED": 80, "FAILED": 20, "BLOCKED": 46},
            historical_job_counts={"FAILED": 20, "BLOCKED": 46},
            current_job_counts={"SUCCEEDED": 3},
            current_failed_jobs=[],
            current_blocked_critical_jobs=[],
            active_failures=[],
            operator_state="HEALTHY",
        ),
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["NO_TRADE"], "decision": "NO_TRADE"},
        soak={"real_forward_observations": 0, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [1]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    auto = os["system"]["automation"]
    assert auto["status"] != "Problem"
    assert auto.get("technical", {}).get("historical_note")


def test_live_money_lane_stays_locked_and_exposes_no_execution_control():
    os = build_home_os(
        dashboard=_ready_dash(),
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["NO_TRADE"], "decision": "NO_TRADE"},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [1]},
        now=_open(),
    )
    assert os["live_locked"] is True
    assert os["state"] != PROBLEM
    dumped = str(os["system"]) + str(os.get("check_system"))
    for banned in FORBIDDEN_CONTROLS:
        assert banned not in dumped
    for lane in os["system"].values():
        for action in [lane.get("primary_action"), *(lane.get("secondary_actions") or [])]:
            if not action:
                continue
            assert action.get("control") in SAFE_CONTROLS or action.get("kind") == "instruction"
    check = os["check_system"]
    assert check["read_only"] is True
    live = next(row for row in check["lanes"] if row["id"] == "live_money")
    assert live["status"] == "Locked"
    assert check["action"]["control"] == "CHECK_SYSTEM"
    assert check["action"]["kind"] == "refresh"


def test_check_system_capability_is_read_only_home_action():
    cap = by_id("check_system")
    assert cap["mode"] == "HOME_ACTION"
    assert cap["read_only"] is True
    assert cap["affects_live_money"] is False
    assert cap["control"] == "CHECK_SYSTEM"
    assert any(row["control"] == "CHECK_SYSTEM" for row in home_actions())


def test_scrub_never_keeps_tokens():
    clean = _scrub_technical({"session_state": "ok", "access_token": "abc", "nested": {"api_secret": "x", "symbols_ticking": 4}})
    assert "access_token" not in clean
    assert "api_secret" not in clean.get("nested", {})
    assert clean["nested"]["symbols_ticking"] == 4
