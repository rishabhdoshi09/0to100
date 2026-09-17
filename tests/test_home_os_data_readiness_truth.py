from __future__ import annotations

from datetime import datetime, timezone

import product.home_os as home_os


def _safe_live_lock() -> dict[str, object]:
    return {
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "live_lock_status": "LOCKED",
        "live_lock_reason": "test interlock",
        "live_lock_source": "test",
    }


def _build(monkeypatch, *, current: bool, available: str, expected: str, stale_sessions: int, raw_ready: bool):
    monkeypatch.setattr(home_os, "_live_safety_projection", _safe_live_lock)
    return home_os.build_home_os(
        dashboard={},
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        autonomy={
            "available": True,
            "running": True,
            "state": "DATA_READY",
            "broker": {"ready": True, "live_data_ready": True},
        },
        data={
            "ready": raw_ready,
            "bhavcopy": {
                "ready": raw_ready,
                "latest_date": available,
                "symbols": 3388,
                "sessions": 1810,
                "source": "official_bhavcopy",
            },
            "history_freshness": {
                "current": current,
                "available_session": available,
                "expected_latest_completed_session": expected,
                "stale_sessions": stale_sessions,
                "reason_code": "HISTORY_CURRENT" if current else "HISTORY_STALE",
            },
        },
        why={},
        soak={},
        soak_verify={},
        scan={},
        reco={},
        journal={},
        operations={"active": [], "recent": []},
        radar={},
        now=datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc),
    )


def test_current_official_session_is_ready_even_if_legacy_ready_flag_is_false(monkeypatch):
    payload = _build(
        monkeypatch,
        current=True,
        available="2026-09-17",
        expected="2026-09-17",
        stale_sessions=0,
        raw_ready=False,
    )
    lane = payload["system"]["data"]
    assert lane["status"] == "Ready"
    assert lane["technical"]["history_current"] is True
    assert lane["technical"]["data_ready"] is True
    assert payload["today"]["data_fresh"] is True


def test_stale_official_session_cannot_become_ready_from_legacy_flag(monkeypatch):
    payload = _build(
        monkeypatch,
        current=False,
        available="2026-09-16",
        expected="2026-09-17",
        stale_sessions=1,
        raw_ready=True,
    )
    lane = payload["system"]["data"]
    assert lane["status"] != "Ready"
    assert lane["technical"]["history_current"] is False
    assert lane["technical"]["data_ready"] is False
    assert payload["today"]["data_fresh"] is False


def test_raw_ready_remains_valid_when_no_completed_session_contract_exists(monkeypatch):
    payload = _build(
        monkeypatch,
        current=True,
        available="",
        expected="",
        stale_sessions=0,
        raw_ready=True,
    )
    assert payload["system"]["data"]["status"] == "Ready"
    assert payload["today"]["data_fresh"] is True
