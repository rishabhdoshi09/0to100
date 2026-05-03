"""Tests for SQLite portfolio tracker."""
import os
import tempfile

import pytest

from sq_ai.portfolio.tracker import PortfolioTracker


@pytest.fixture()
def tracker(tmp_path):
    db = tmp_path / "test.db"
    return PortfolioTracker(db_path=str(db))


def test_open_close_trade_pnl(tracker):
    tid = tracker.open_trade("RELIANCE.NS", entry_price=2000, qty=10,
                             stop=1900, target=2300)
    assert tid > 0
    pos = tracker.open_positions()
    assert len(pos) == 1
    assert pos[0]["symbol"] == "RELIANCE.NS"

    pnl = tracker.close_trade(tid, exit_price=2100)
    assert pnl == (2100 - 2000) * 10
    assert tracker.open_positions() == []
    closed = tracker.closed_trades()
    assert len(closed) == 1
    assert closed[0]["pnl"] == pnl


def test_signals_log_and_fetch(tracker):
    tracker.log_signal("TCS.NS", "BUY", 0.72, 2, "ml+regime ok",
                       extra={"size_pct": 5})
    tracker.log_signal("INFY.NS", "HOLD", 0.30, 1, "low confidence")
    sigs = tracker.latest_signals(limit=10)
    assert len(sigs) == 2
    assert sigs[0]["symbol"] == "INFY.NS"     # latest first


def test_equity_curve(tracker):
    tracker.record_equity(1_000_000, 1_000_000, date="2024-01-01")
    tracker.record_equity(1_010_000, 990_000, date="2024-01-02")
    curve = tracker.equity_curve()
    assert len(curve) == 2
    assert curve[-1]["equity"] == 1_010_000


def test_screener_save_and_fetch(tracker):
    tracker.save_screener("2024-02-01", [
        {"symbol": "A.NS", "score": 1.2, "reasoning": "x"},
        {"symbol": "B.NS", "score": 1.0, "reasoning": "y"},
    ])
    out = tracker.latest_screener()
    assert len(out) == 2
    assert out[0]["rank"] == 1
    assert out[0]["symbol"] == "A.NS"


def test_prices_upsert(tracker):
    tracker.upsert_price("X.NS", "2024-01-01", 100, 110, 99, 105, 12345)
    tracker.upsert_price("X.NS", "2024-01-01", 100, 111, 99, 106, 12345)  # update
    assert tracker.latest_close("X.NS") == 106
