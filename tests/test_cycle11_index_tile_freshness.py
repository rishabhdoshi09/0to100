"""Cycle 11: the market report's index tiles say WHICH session they describe, and when it is stale.

23 Nov (owner's Mac): during the session the report's "Latest session" tile showed NIFTY
+1.122% (the 20 Nov close against 19 Nov, from the index store) beside the F&O context's
live -1.406% (23 Nov intraday; the session closed -1.683%). Both were right, but the tile
carried no date, so a previous session's move read as today's. With Kite logged out the
index store stopped at 23 Nov: on 26 Nov the same tile showed -1.683% as the "latest session",
two sessions behind. Each tile now carries its session (as_of), source and basis, and is
stale when that session is older than the expected latest completed session.
"""
from __future__ import annotations

import pytest

from reports import street_pulse as SP


@pytest.fixture
def prints(monkeypatch):
    def install(as_of, expected):
        monkeypatch.setattr("data.index_store.latest_index_print",
                            lambda ticker: {"price": 25366.4, "chg_pct": -1.68, "as_of": as_of,
                                            "source": "kite_index_history"} if ticker == "^NSEI" else {})
        if isinstance(expected, Exception):
            monkeypatch.setattr(SP, "completed_session", lambda now=None: None)
        else:
            monkeypatch.setattr(SP, "completed_session", lambda now=None: expected)
        return SP._market_snapshot()["indices"][0]
    return install


def test_a_tile_two_sessions_behind_is_stale_and_dated(prints):
    tile = prints("2026-11-23", "2026-11-26")
    assert tile["as_of"] == "2026-11-23" and tile["stale"] is True
    assert tile["source"] == "kite_index_history" and tile["basis"] == "completed_session_close"


def test_a_current_tile_is_not_stale(prints):
    tile = prints("2026-11-26", "2026-11-26")
    assert tile["as_of"] == "2026-11-26" and tile["stale"] is False


def test_a_tile_without_a_session_is_stale(prints):
    assert prints("", "2026-11-26")["stale"] is True


def test_an_unknown_expected_session_is_unknown_not_fresh(prints):
    tile = prints("2026-11-26", RuntimeError("no calendar"))
    assert tile["stale"] is None and tile["as_of"] == "2026-11-26"


# ── GPT review of cycle 11 (C11-01, C11-02) ──

def test_C11_02_after_the_close_the_previous_session_is_stale(monkeypatch):
    """16:00 IST on a session day: that session has closed although its archive is not out
    until 18:00. A tile still showing the day before is stale."""
    from datetime import datetime

    from research.intelligence.data import nse_calendar as C

    monkeypatch.setattr(C, "load_holidays", lambda: set())
    assert SP.completed_session(datetime(2026, 11, 26, 16, 0)) == "2026-11-26"
    assert SP.completed_session(datetime(2026, 11, 26, 15, 0)) == "2026-11-25"
    assert SP.completed_session(datetime(2026, 11, 28, 11, 0)) == "2026-11-27"     # Saturday
    monkeypatch.setattr(C, "_now_ist", lambda: datetime(2026, 11, 26, 16, 0))
    monkeypatch.setattr("data.index_store.latest_index_print",
                        lambda ticker: {"price": 25366.4, "chg_pct": -1.68, "as_of": "2026-11-25"})
    assert SP._market_snapshot()["indices"][0]["stale"] is True


def test_C11_01_a_served_market_report_keeps_the_tile_session_and_rejudges_it(monkeypatch):
    from product.recommendations_workspace import _report_item

    monkeypatch.setattr(SP, "completed_session", lambda now=None: "2026-11-26")
    tile = {"name": "NIFTY 50", "price": 25366.4, "chg_pct": -1.68, "as_of": "2026-11-23", "stale": False,
            "source": "kite_index_history", "basis": "completed_session_close"}
    item = _report_item({"date": "2026-11-26", "pulse": {"snapshot": {"indices": [tile]}}}, today="2026-11-26")
    (got,) = item["snapshot"]["indices"]
    assert got["as_of"] == "2026-11-23" and got["stale"] is True            # re-judged when served
    assert got["source"] == "kite_index_history" and got["basis"] == "completed_session_close"


def test_C11_01_a_saved_report_without_tile_dates_is_unknown_not_current(monkeypatch):
    from product.recommendations_workspace import _report_item

    item = _report_item({"date": "2026-11-20", "pulse": {"snapshot": {"indices": [
        {"name": "NIFTY 50", "price": 25800.6, "chg_pct": 1.12}]}}}, today="2026-11-26")
    (got,) = item["snapshot"]["indices"]
    assert got["as_of"] is None and got["stale"] is None


def test_C11_01_the_takeaway_text_names_the_session(monkeypatch):
    monkeypatch.setattr(SP, "completed_session", lambda now=None: "2026-11-26")
    monkeypatch.setattr("data.index_store.latest_index_print",
                        lambda ticker: {"price": 25366.4, "chg_pct": -1.68, "as_of": "2026-11-23"}
                        if ticker == "^NSEI" else {})
    monkeypatch.setattr(SP, "_scan_rows_latest", lambda: ([], 0))
    monkeypatch.setattr(SP, "_movers_from_bhav", lambda: ([], []))
    pulse = SP.build_pulse(force=True)
    assert any("session 2026-11-23, stale" in t for t in pulse["takeaways"]), pulse["takeaways"]
