"""Phase 13 Home operating-system journeys. Same truth as professional mode."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from product.home_os import (
    FAILED_RECOVERABLE,
    LOGIN_REQUIRED,
    MARKET_CLOSED_COMPLETE,
    NO_TRADE,
    NORMAL,
    PAUSED,
    PREPARING,
    build_home_os,
)
from product.operator_language import simple_reason
from product.runtime_capabilities import audit_rows, automatic, by_id, home_actions


IST = ZoneInfo("Asia/Kolkata")


def _open() -> datetime:
    return datetime(2026, 9, 1, 10, 45, tzinfo=IST)


def _eod() -> datetime:
    return datetime(2026, 9, 1, 19, 10, tzinfo=IST)


def test_journey_b_zerodha_login_required():
    os = build_home_os(
        dashboard={"autonomy": {"state": "AUTH_REQUIRED", "running": True}, "data": {"ready": True}},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["need_me"] is True
    assert os["broker"]["login_required"] is True
    assert os["primary_action"]["label"] == "Login to Zerodha"
    assert os["primary_action"]["kind"] == "instruction"
    assert os["system"]["zerodha"]["status"] == "Needs you"
    assert os["live_locked"] is True
    assert os["state"] in {LOGIN_REQUIRED, NORMAL, NO_TRADE, MARKET_CLOSED_COMPLETE}


def test_observing_with_auth_health_still_needs_zerodha_login():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "OBSERVING",
                "running": True,
                "reason_code": "auth_health",
                "explanation": "daily Zerodha login is required",
            },
            "data": {"ready": True},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["need_me"] is True
    assert os["broker"]["login_required"] is True
    assert os["system"]["zerodha"]["status"] == "Needs you"
    assert os["live_locked"] is True


def test_observing_auth_health_without_explanation_is_not_broker_ready():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "OBSERVING",
                "running": True,
                "reason_code": "auth_health",
                "explanation": "",
            },
            "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["broker"]["login_required"] is True
    assert os["broker"]["status"] != "READY"
    assert os["system"]["zerodha"]["status"] == "Needs you"


def test_empty_autonomy_snapshot_is_not_broker_ready():
    os = build_home_os(
        dashboard={"autonomy": {}, "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}}},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["broker"]["login_required"] is True
    assert os["broker"]["status"] != "READY"
    assert os["system"]["zerodha"]["status"] == "Needs you"


def test_observing_auth_missing_failure_without_reason_code_is_not_broker_ready():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "OBSERVING",
                "running": True,
                "reason_code": "paper_cycle",
                "explanation": "paper cycle: no-op · CAPABILITY_BLOCKED",
                "active_failures": ["auth_missing", "live_feed_stale", "snapshot_stale"],
            },
            "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        now=_open(),
    )
    assert os["broker"]["login_required"] is True
    assert os["broker"]["status"] != "READY"
    assert os["system"]["zerodha"]["status"] == "Needs you"


def test_journey_c_no_trade_is_healthy():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": True, "bhavcopy": {"ready": True}}},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={
            "available": True,
            "headline": "No paper trade taken: ENTRY_TOO_EXTENDED",
            "decision": "NO_TRADE",
            "reasons": ["ENTRY_TOO_EXTENDED"],
            "rejections": [{"symbol": "TCS", "reason_code": "ENTRY_TOO_EXTENDED"}],
            "taken": [],
            "as_of": "2026-09-01",
        },
        journal={"latest": {"as_of": "2026-09-01", "taken": [], "rejections": [{"symbol": "TCS", "reason_code": "ENTRY_TOO_EXTENDED"}], "cycle_reasons": ["ENTRY_TOO_EXTENDED"]}},
        soak={"real_forward_observations": 0, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    assert os["state"] == NO_TRADE
    assert os["need_me"] is False
    assert "nothing was good enough" in os["headline"].lower() or "did not find a setup" in os["headline"].lower()
    assert os["paper_bot"]["todays_entries"] == 0
    assert "Waiting" in os["paper_bot"]["why"] or "good enough" in os["paper_bot"]["why"].lower()
    assert os["live_locked"] is True


def test_journey_d_data_failure_offers_one_retry():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": False}},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={},
        operations={"active": [], "recent": [{"kind": "DATA_PREPARE", "status": "FAILED"}]},
        now=_open(),
    )
    assert os["state"] == FAILED_RECOVERABLE
    assert os["primary_action"]["control"] == "REFRESH_DATA_NOW"
    assert os["primary_action"]["label"] == "Retry"
    assert os["need_me"] is True


def test_journey_e_restart_explains_recovery_without_new_trade():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": True, "bhavcopy": {"ready": True}}},
        paper={"enabled": True, "open_positions": [{"symbol": "TCS", "stop_price": 94, "target_price": 115}], "closed_trades": [], "open_risk": 1000},
        why={"available": True, "taken": [{"symbol": "TCS"}], "rejections": [], "headline": "Took 1 paper trade(s): TCS"},
        journal={"latest": {"taken": [{"symbol": "TCS"}], "as_of": "2026-09-01"}},
        soak={"real_forward_observations": 1, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
        recovered=["open paper position", "forward evidence"],
    )
    assert os["paper_bot"]["positions_open"] == 1
    assert os["paper_bot"]["todays_entries"] == 1
    assert any("Recovered" in row["text"] for row in os["recent_activity"])
    assert os["state"] in {NORMAL, NO_TRADE}


def test_journey_f_market_closed_is_understandable_on_home():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": True, "bhavcopy": {"ready": True}}},
        paper={"enabled": True, "open_positions": [], "closed_trades": [{"symbol": "TCS", "exit_reason": "TARGET"}]},
        why={"available": True, "taken": [{"symbol": "TCS"}], "headline": "Took 1"},
        journal={"latest": {"taken": [{"symbol": "TCS"}], "as_of": "2026-09-01"}},
        soak={"real_forward_observations": 1, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        soak_verify={"lanes": {"SCAN": "PASS", "FORWARD SETTLEMENT": "PASS", "LEARNING INGESTION": "PASS"}, "generated_at": "2026-09-01T13:46:00+00:00"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_eod(),
    )
    assert os["state"] in {MARKET_CLOSED_COMPLETE, NO_TRADE, NORMAL}
    assert os["today"]["market_open"] is False
    assert os["yesterday"]["settlement"] is True
    assert os["need_me"] is False


def test_journey_a_automatic_verify_persists_without_cli(tmp_path, monkeypatch):
    from product.forward_soak import persist_soak_verification, load_latest_verification

    monkeypatch.setenv("QT_FORWARD_SOAK_VERIFY", str(tmp_path / "verify.json"))
    first = persist_soak_verification(force=True)
    second = persist_soak_verification(min_interval_s=3600)
    assert first["source"] == "verify_persisted_soak"
    assert first["live_locked"] is True
    assert second["generated_at"] == first["generated_at"]
    assert load_latest_verification()["lanes"]["LIVE MONEY"] == "LOCKED"


def test_observe_only_does_not_pause_paper():
    os = build_home_os(
        dashboard={
            "autonomy": {
                "state": "RUNNING",
                "running": True,
                "owner_state": {"observe_only_date": "2026-09-01", "new_entries_paused": False},
            },
            "data": {"ready": True, "bhavcopy": {"ready": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["ENTRY_TOO_EXTENDED"], "rejections": [{"symbol": "TCS"}]},
        journal={"latest": {"taken": [], "rejections": [{"symbol": "TCS"}], "cycle_reasons": ["ENTRY_TOO_EXTENDED"]}},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )
    assert os["observe_only"] is True
    assert os["paper_bot"]["on"] is True
    assert os["paper_bot"]["paused"] is False
    assert os["live_locked"] is True
    assert "observe only" in os["subtext"].lower() or "paper still" in os["subtext"].lower()


def test_paused_paper_is_a_home_action():
    os = build_home_os(
        dashboard={"autonomy": {"state": "RUNNING", "running": True}, "data": {"ready": True, "bhavcopy": {"ready": True}}},
        paper={"enabled": False, "open_positions": [], "closed_trades": []},
        why={"available": True, "taken": [], "reasons": ["PAPER_TRADING_DISABLED"]},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [1]},
        now=_open(),
    )
    assert os["state"] == PAUSED
    assert os["primary_action"]["control"] == "RESUME_NEW_PAPER_ENTRIES"


def test_simple_language_does_not_change_codes():
    assert "Waiting" in simple_reason("ENTRY_TOO_EXTENDED")
    assert "too much similar risk" in simple_reason("PORTFOLIO_BLOCK")
    assert simple_reason("NO_TRADE") == "Nothing was good enough today."


def test_capability_inventory_keeps_engineering_out_of_home():
    ids = {row["capability_id"] for row in automatic()}
    assert "market_scan" in ids
    assert "forward_soak_verify" in ids
    assert "pytest_suite" not in ids
    for row in home_actions():
        assert row["affects_live_money"] is False
        assert row["mode"] == "HOME_ACTION"
    assert by_id("pytest_suite")["mode"] == "DEVELOPER_ONLY"
    assert by_id("issue92_dod")["mode"] == "DEVELOPER_ONLY"
    table = audit_rows()
    normal = [r for r in table if r["New mode"] != "DEVELOPER_ONLY" and r["Capability"] != "Start QuantTerm"]
    assert all(r["Still requires terminal?"] == "no" for r in normal)
    assert by_id("zerodha_observation")["read_only"] is True
    assert by_id("zerodha_observation")["affects_live_money"] is False


def test_news_refresh_is_not_preparing_official_data():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        operations={"active": [{"kind": "NEWS_REFRESH", "status": "RUNNING"}], "recent": []},
        now=_open(),
    )
    assert os["state"] != PREPARING
    assert "Preparing official data" not in (os["now"] or "")


def test_home_does_not_claim_settlement_from_stale_soak_off_session():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "PENDING"},
        soak_verify={"lanes": {"FORWARD SETTLEMENT": "PENDING"}, "generated_at": "2026-08-01T10:00:00+00:00"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        operations={"active": [], "recent": []},
        now=datetime(2026, 9, 1, 23, 50, tzinfo=IST),
    )
    assert os["now"] != "End-of-day settlement"


def test_radar_home_payload_includes_home_os(monkeypatch):
    import product.observer_api as observer

    class _Core:
        @staticmethod
        def _scan_payload():
            return {"scanned_at": "2026-09-01T05:00:00+00:00", "records": [], "universe_size": 0}

        @staticmethod
        def _market_payload():
            return {"health": "Quiet", "breadth": "—", "nifty_change_1d": 0, "vix": None, "leaders": [], "laggards": [], "trade_stance": "Open"}

        @staticmethod
        def _long_term_payload():
            return {"records": [], "scanned_at": ""}

    monkeypatch.setattr(observer, "core", _Core)
    payload = observer.radar_home_workspace()
    assert "home_os" in payload
    assert payload["home_os"]["live_locked"] is True
    assert payload["home_os"]["headline"]


def test_radar_home_uses_autonomy_payload_for_broker_status(monkeypatch):
    import product.observer_api as observer

    class _Core:
        @staticmethod
        def _scan_payload():
            return {"scanned_at": "2026-09-01T05:00:00+00:00", "records": [], "universe_size": 0}

        @staticmethod
        def _market_payload():
            return {"health": "Quiet", "breadth": "—", "nifty_change_1d": 0, "vix": None, "leaders": [], "laggards": [], "trade_stance": "Open"}

        @staticmethod
        def _long_term_payload():
            return {"records": [], "scanned_at": ""}

        @staticmethod
        def _autonomy_payload():
            return {
                "state": "OBSERVING",
                "running": True,
                "reason_code": "paper_cycle",
                "explanation": "paper cycle: no-op",
                "active_failures": ["auth_missing"],
                "capability_notes": [],
            }

    monkeypatch.setattr(observer, "core", _Core)
    payload = observer.radar_home_workspace()
    assert payload["home_os"]["broker"]["login_required"] is True
    assert payload["home_os"]["broker"]["status"] != "READY"


def test_data_ready_reconciles_from_real_history_when_no_data_payload_is_wired(monkeypatch):
    """Reproduces the reported readiness contradiction.

    ``radar_home_workspace`` (the real production caller) invokes
    ``build_home_os`` without a ``data=`` payload -- only ``scan``, ``radar``
    and ``autonomy`` are passed. That left ``data_d``/``bhav`` structurally
    empty, so the old ``data_ready = bool(data_d.get("ready") or
    bhav.get("ready"))`` computation was permanently False no matter what the
    real, already-persisted official history freshness said -- producing
    reason_code=HISTORY_CURRENT / history_current=True alongside a stuck
    data_ready=False and "Waiting" / "Getting the latest market data" UI.
    """
    import data.bhavcopy_runtime as bhav_runtime

    current_freshness = {
        "current": True,
        "usable_for_scan": True,
        "publication_pending": False,
        "expected_latest_completed_session": "2026-09-17",
        "available_session": "2026-09-17",
        "stale_sessions": 0,
        "reason_code": "HISTORY_CURRENT",
        "ready": True,
    }
    monkeypatch.setattr(
        bhav_runtime, "official_history_freshness", lambda *a, **k: dict(current_freshness)
    )

    # Mirrors the real call shape: no dashboard/data/paper/why/soak payload.
    os = build_home_os(now=_open())

    data_lane = os["system"]["data"]
    assert data_lane["technical"]["reason_code"] == "HISTORY_CURRENT"
    assert data_lane["technical"]["history_current"] is True
    assert data_lane["technical"]["data_ready"] is True
    assert data_lane["status_code"] != "WAITING"
    assert data_lane["current"] != "Getting the latest market data"


def test_home_best_trades_excludes_rejections_waits_and_non_buy_gate_rows(monkeypatch):
    """Home's primary list is canonical production selection, never a mixed journal."""
    import product.decision_simulation_gate as gate

    monkeypatch.setattr(
        gate,
        "status",
        lambda: {
            "scan_fresh": True,
            "discovery_ready": True,
            "best_trades": [
                {
                    "symbol": "TCS",
                    "setup_label": "Ready to trade",
                    "decision": "BUY",
                    "discovery_decision": "ENTER_NOW",
                    "reason_code": "ELIGIBLE",
                },
                {
                    "symbol": "BADWAIT",
                    "setup_label": "Ready to trade",
                    "decision": "WAIT",
                    "discovery_decision": "ENTER_NOW",
                    "reason_code": "ENTRY_TOO_EXTENDED",
                },
            ],
        },
    )

    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True, "broker": {"ready": True}},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "latest_date": "2026-09-01",
                    "current": True,
                    "reason_code": "HISTORY_CURRENT",
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={
            "available": True,
            "taken": [],
            "rejections": [
                {
                    "symbol": "AUROPHARMA",
                    "status": "Ready to trade",
                    "reason_code": "ENTRY_TOO_EXTENDED",
                }
            ],
            "waits": [
                {
                    "symbol": "EMIL",
                    "status": "Ready to trade",
                    "reason_code": "EVIDENCE_WEAK",
                }
            ],
            "reasons": ["ENTRY_TOO_EXTENDED"],
        },
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={
            "scanned_at": "2026-09-01T05:00:00+00:00",
            "records": [{"symbol": "TCS"}],
        },
        reco={"schema_version": 4, "categories": []},
        now=_open(),
    )

    assert [row["found"].split()[0] for row in os["opportunities"]] == ["TCS"]
    assert os["opportunities"][0]["label"] == "BUY"
    rendered = " ".join(str(row) for row in os["opportunities"])
    assert "AUROPHARMA" not in rendered
    assert "EMIL" not in rendered
    assert "BADWAIT" not in rendered


def test_market_closed_never_claims_complete_when_required_verifier_lanes_fail():
    lanes = {
        "SCAN": "FAIL",
        "RECOMMENDATIONS": "PASS",
        "SELECTION": "FAIL",
        "AUTOPILOT": "FAIL",
        "PAPER EXECUTION": "FAIL",
        "EXIT SUPERVISION": "UNKNOWN",
        "FORWARD SETTLEMENT": "PENDING",
        "LEARNING INGESTION": "PENDING",
        "EXECUTION REALITY": "PENDING",
        "LIVE MONEY": "LOCKED",
    }
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "latest_date": "2026-09-01",
                    "current": True,
                    "reason_code": "HISTORY_CURRENT",
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False, "taken": [], "rejections": [], "waits": []},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        soak_verify={"lanes": lanes},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_eod(),
    )

    assert os["state"] != MARKET_CLOSED_COMPLETE
    assert "not complete" in os["headline"].lower()
    assert "scan fail" in os["subtext"].lower()
    assert os["need_me"] is False
    assert os["verify"]["lanes"] == lanes


def test_market_closed_complete_requires_all_core_verifier_lanes():
    lanes = {
        "SCAN": "PASS",
        "RECOMMENDATIONS": "PASS",
        "SELECTION": "PASS",
        "AUTOPILOT": "PASS",
        "PAPER EXECUTION": "PASS",
        "EXIT SUPERVISION": "PASS",
        "FORWARD SETTLEMENT": "PASS",
        "LEARNING INGESTION": "PASS",
        "EXECUTION REALITY": "PASS",
        "LIVE MONEY": "LOCKED",
    }
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "latest_date": "2026-09-01",
                    "current": True,
                    "reason_code": "HISTORY_CURRENT",
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": [{"symbol": "TCS"}]},
        why={"available": True, "taken": [{"symbol": "TCS"}], "rejections": [], "waits": []},
        journal={"latest": {"taken": [{"symbol": "TCS"}], "as_of": "2026-09-01"}},
        soak={"real_forward_observations": 1, "insufficient_evidence": True},
        soak_verify={"lanes": lanes},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=_eod(),
    )

    assert os["state"] == MARKET_CLOSED_COMPLETE
    assert "complete" in os["headline"].lower()


def test_home_activity_reports_scanned_universe_not_qualified_row_count():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "latest_date": "2026-09-01",
                    "current": True,
                    "reason_code": "HISTORY_CURRENT",
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False, "taken": [], "rejections": [], "waits": []},
        scan={
            "scanned_at": "2026-09-01T05:00:00+00:00",
            "scanned": 598,
            "universe_size": 598,
            "records": [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}],
        },
        reco={"schema_version": 4, "categories": []},
        soak_verify={"lanes": {}},
        now=_eod(),
    )

    activity = " ".join(row.get("text", "") for row in os.get("recent_activity", []))
    assert "598 names checked" in activity
    assert "3 names checked" not in activity
