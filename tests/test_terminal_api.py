from __future__ import annotations

import inspect

import terminal_api


def test_capability_strings_are_not_coerced_by_python_truthiness():
    assert terminal_api._capability("allowed") == "allowed"
    assert terminal_api._capability("limited") == "limited"
    assert terminal_api._capability("blocked") == "blocked"
    assert terminal_api._capability("") == "blocked"
    assert terminal_api._capability(None) == "blocked"


def test_terminal_controls_have_no_live_broker_or_order_action():
    assert terminal_api._ALLOWED_CONTROLS == {
        "RUN_SCAN_NOW",
        "RUN_LONG_TERM_SCAN_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_FNO_NOW",
        "RUN_CYCLE_NOW",
        "REFRESH_DATA_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
    }
    source = inspect.getsource(terminal_api.control).lower()
    assert "broker" not in source
    assert "order" not in source


def test_paper_payload_exposes_daily_learning_and_keeps_live_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_MEMORY", str(tmp_path / "paper_memory.json"))
    from product.paper_learning import remember_paper_book
    remember_paper_book(
        [
            {"symbol": "TCS", "pnl": -110, "realized_R": -1.1, "exit_date": "2026-08-20", "exit_reason": "STOP"},
            {"symbol": "TCS", "pnl": -40, "realized_R": -0.4, "exit_date": "2026-08-22", "exit_reason": "STOP"},
        ],
        as_of="2026-08-24",
    )
    payload = terminal_api._paper_payload()
    learning = payload["learning"]
    assert learning["live_locked"] is True
    assert learning["closed_trades"] == 2
    assert learning["cooldown"][0]["symbol"] == "TCS"
    assert "owner approval" in learning["disclaimer"].lower()


def test_market_controls_are_dispatched_outside_paper_autonomy():
    assert terminal_api._OPERATION_CONTROLS == {
        "RUN_SCAN_NOW": "MARKET_SCAN",
        "RUN_LONG_TERM_SCAN_NOW": "LONG_TERM_SCAN",
        "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
        "REFRESH_NEWS_NOW": "NEWS_REFRESH",
        "REFRESH_FNO_NOW": "FNO_REFRESH",
        "REFRESH_DATA_NOW": "DATA_PREPARE",
    }
    assert terminal_api._AUTONOMY_CONTROLS == {
        "RUN_CYCLE_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
    }


def test_json_safe_strips_nan_and_inf():
    payload = terminal_api._json_safe({"ok": 1.0, "bad": float("nan"), "rows": [float("inf"), 2.0]})
    assert payload == {"ok": 1.0, "bad": None, "rows": [None, 2.0]}


def test_dashboard_keeps_last_scan_when_a_lane_explodes(monkeypatch):
    scan = {
        "available": True,
        "scanned_at": "2026-08-24T04:41:16+00:00",
        "universe_size": 2,
        "summary": {},
        "records": [{"symbol": "TCS", "score": 80, "signals": ["MOMENTUM"]}],
    }
    monkeypatch.setattr(terminal_api, "_scan_payload", lambda: scan)
    monkeypatch.setattr(terminal_api, "_market_payload", lambda: {"available": True, "health": "Neutral"})
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})

    def boom(*_args, **_kwargs):
        raise RuntimeError("bhavcopy cache exploded")

    monkeypatch.setattr(terminal_api, "_data_payload", boom)
    payload = terminal_api.dashboard()
    assert payload["scan"]["records"][0]["symbol"] == "TCS"
    assert payload["data"]["scan_saved"] is True
    assert "degraded" in str(payload.get("error", "")).lower()


def test_radar_home_keeps_watchlist_when_sepa_ranking_fails(monkeypatch):
    from product import observer_api

    scan = {
        "available": True,
        "scanned_at": "2026-08-24T04:41:16+00:00",
        "universe_size": 2,
        "summary": {},
        "records": [
            {
                "symbol": "TCS",
                "score": 80,
                "signals": ["MOMENTUM"],
                "price": 100,
                "entry": 101,
                "stop": 95,
                "target": 120,
                "verdict": "WATCH",
                "chase_risk": False,
                "reasons": ["trend"],
            }
        ],
    }
    monkeypatch.setattr(observer_api.core, "_scan_payload", lambda: scan)
    monkeypatch.setattr(
        observer_api.core,
        "_market_payload",
        lambda: {
            "available": True,
            "health": "Neutral",
            "summary": "ok",
            "trade_stance": "Wait",
            "breadth": "mixed",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": 0.2,
            "nifty_change_5d": 0.1,
            "vix": 12.0,
            "technical_details": {},
        },
    )
    monkeypatch.setattr(observer_api.core, "_long_term_payload", lambda: {"available": False, "records": []})

    def boom(*_args, **_kwargs):
        raise RuntimeError("sepa unavailable")

    monkeypatch.setattr("product.sepa_setup.public_best_setups", boom)
    home = observer_api.radar_home_workspace()
    assert home["lanes"]["momentum"][0]["symbol"] == "TCS"
    assert home["best_setups"] == []
    assert "unavailable" in home["best_setups_note"].lower()
    assert "telegram" in home
    assert "headline" in home["telegram"]
    assert "desk_pipeline" in home
