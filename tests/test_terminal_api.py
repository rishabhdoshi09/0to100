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
        "RUN_FULL_UNIVERSE_BACKTEST_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
    }
    source = inspect.getsource(terminal_api.control).lower()
    assert "broker" not in source
    assert "order" not in source


def test_market_controls_are_dispatched_outside_paper_autonomy():
    assert terminal_api._OPERATION_CONTROLS == {
        "RUN_SCAN_NOW": "MARKET_SCAN",
        "RUN_LONG_TERM_SCAN_NOW": "LONG_TERM_SCAN",
        "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
        "REFRESH_NEWS_NOW": "NEWS_REFRESH",
        "REFRESH_FNO_NOW": "FNO_REFRESH",
        "REFRESH_DATA_NOW": "DATA_PREPARE",
        "RUN_FULL_UNIVERSE_BACKTEST_NOW": "FULL_UNIVERSE_BACKTEST",
    }
    assert terminal_api._AUTONOMY_CONTROLS == {
        "RUN_CYCLE_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
    }


def test_queue_message_is_honest_about_worker_offline():
    msg = terminal_api._queue_message_for_control("MARKET_SCAN", {"running": False, "active": {}})
    assert "OFFLINE" in msg
    online = terminal_api._queue_message_for_control(
        "MARKET_SCAN",
        {"running": True, "worker_pid": 42, "active": {}},
    )
    assert "ONLINE" in online
    assert "42" in online
