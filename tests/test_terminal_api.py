from __future__ import annotations

import inspect
import time

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
        "REFRESH_MARKET_REPORT_NOW",
        "REFRESH_FNO_NOW",
        "RUN_CYCLE_NOW",
        "REFRESH_DATA_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
        "OBSERVE_ONLY_TODAY",
        "CLEAR_OBSERVE_ONLY",
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
        "RUN_LONG_TERM_SCAN_NOW": "MARKET_SCAN",
        "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
        "REFRESH_NEWS_NOW": "NEWS_REFRESH",
        "REFRESH_MARKET_REPORT_NOW": "MARKET_REPORT",
        "REFRESH_FNO_NOW": "FNO_REFRESH",
        "REFRESH_DATA_NOW": "DATA_PREPARE",
    }
    assert terminal_api._AUTONOMY_CONTROLS == {
        "RUN_CYCLE_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
        "OBSERVE_ONLY_TODAY",
        "CLEAR_OBSERVE_ONLY",
    }


def test_ops_runtime_treats_live_lock_owner_as_running(tmp_path, monkeypatch):
    import json
    import os

    ops = tmp_path / "market_ops"
    ops.mkdir()
    (ops / "worker.lock").write_text(str(os.getpid()), encoding="utf-8")
    (ops / "runtime.json").write_text(
        json.dumps({"process_running": False, "worker_pid": 1, "heartbeat_epoch": 0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(terminal_api, "OPS_ROOT", ops)
    monkeypatch.setattr(terminal_api, "OPS_RUNTIME", ops / "runtime.json")
    payload = terminal_api._ops_runtime_payload()
    assert payload["running"] is True
    assert payload["worker_pid"] == os.getpid()
    assert payload.get("recovering") is True


def test_health_is_a_cheap_liveness_probe():
    import inspect

    src = inspect.getsource(terminal_api.health)
    assert "_autonomy_payload" not in src
    assert "_operations_payload" not in src
    payload = terminal_api.health()
    assert payload["ok"] is True
    assert payload["service"] == "quantterm-terminal-api"


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


def test_dashboard_slims_scan_keeps_universe_and_conviction_input(monkeypatch):
    records = [{"symbol": f"S{i:03d}", "score": i, "composite": i} for i in range(120)]
    scan = {
        "available": True,
        "scanned_at": "2026-09-02T05:00:00+00:00",
        "universe_size": 1842,
        "summary": {"with_any_setup": 12},
        "records": records,
    }
    seen: dict[str, int] = {}
    monkeypatch.setattr(terminal_api, "_scan_payload", lambda: scan)
    monkeypatch.setattr(
        terminal_api,
        "_market_payload",
        lambda: {
            "available": True,
            "health": "Mixed",
            "summary": "ok",
            "trade_stance": "Wait",
            "breadth": "mixed",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": 0.1,
            "nifty_change_5d": 0.2,
            "vix": 12.0,
            "nifty_price": 25000,
            "technical_details": {},
        },
    )
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False, "articles": []})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})
    monkeypatch.setattr(terminal_api, "_scan_progress_payload", lambda: {})

    def fake_data(scan_arg, *_args):
        return {
            "ready": False,
            "snapshot": {},
            "bhavcopy": {},
            "scan_saved": True,
            "scan_records": len(scan_arg.get("records") or []),
            "long_term_saved": False,
            "long_term_records": 0,
            "blockers": [],
        }

    def fake_conviction(scan_arg, _market):
        seen["n"] = len(scan_arg.get("records") or [])
        return [{"symbol": "S119"}]

    monkeypatch.setattr(terminal_api, "_data_payload", fake_data)
    monkeypatch.setattr(terminal_api, "_conviction", fake_conviction)

    payload = terminal_api.dashboard()
    assert payload["scan"]["universe_size"] == 1842
    assert payload["scan"]["dashboard_records_shown"] == 80
    assert len(payload["scan"]["records"]) == 80
    assert payload["scan"]["records"][0]["symbol"] == "S119"
    assert payload["data"]["scan_records"] == 120
    assert seen["n"] == 120


def test_dashboard_returns_without_waiting_for_regime_fetch(monkeypatch):
    def hang():
        time.sleep(8)
        raise AssertionError("regime fetch must stay off the dashboard request")

    monkeypatch.setattr("core.regime_engine.compute_regime", hang)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: None)
    monkeypatch.setattr(
        terminal_api,
        "_scan_payload",
        lambda: {"available": True, "scanned_at": "2026-09-02T05:00:00+00:00", "universe_size": 10, "summary": {}, "records": []},
    )
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False, "articles": []})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})
    monkeypatch.setattr(terminal_api, "_scan_progress_payload", lambda: {})
    monkeypatch.setattr(
        terminal_api,
        "_data_payload",
        lambda *_args: {
            "ready": False,
            "snapshot": {},
            "bhavcopy": {},
            "scan_saved": True,
            "scan_records": 0,
            "long_term_saved": False,
            "long_term_records": 0,
            "blockers": [],
        },
    )
    started = time.monotonic()
    payload = terminal_api.dashboard()
    assert time.monotonic() - started < 1.5
    assert payload["market"]["available"] is False
    assert "assembl" in payload["market"]["summary"].lower()


def test_market_payload_does_not_block_on_regime_fetch(monkeypatch):
    monkeypatch.setattr(terminal_api, "_warm_regime", lambda: None)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: None)
    payload = terminal_api._market_payload()
    assert payload["available"] is False
    assert "assembl" in payload["summary"].lower()
    assert "do not infer" in payload["trade_stance"].lower()


def test_data_payload_does_not_unpickle_bhavcopy_inline(monkeypatch):
    seen: list[bool] = []

    def fake_status(*, load_cache: bool = False):
        seen.append(load_cache)
        return {
            "ready": False,
            "cache_exists": True,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_files": 0,
            "minimum_sessions": 60,
        }

    monkeypatch.setattr(terminal_api, "_warm_bhavcopy_cache", lambda: None)
    monkeypatch.setattr("data.bhavcopy_runtime.status", fake_status)
    monkeypatch.setattr(terminal_api, "_snapshot_payload", lambda: {"ready": False})
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda history, load_cache=False, **_kwargs: {"current": True},
    )
    payload = terminal_api._data_payload(
        {"available": True, "records": [1, 2]},
        {"available": False, "records": []},
        {"running": True},
        {"available": True},
        {"available": True},
    )
    assert seen == [False]
    assert payload["scan_records"] == 2
    assert any("still loading" in item.lower() for item in payload["blockers"])


def test_peek_cached_regime_is_missing_until_computed():
    from core import regime_engine

    regime_engine._CACHE.clear()
    assert regime_engine.peek_cached_regime() is None
    assert terminal_api._slim_ranked_records(
        {"universe_size": 3, "records": [{"symbol": "A", "score": 1}, {"symbol": "B", "score": 9}]}
    )["records"][0]["symbol"] == "B"
