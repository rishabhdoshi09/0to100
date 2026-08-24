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
