"""Tests for watchlist + presets + reports persistence."""
import json

from sq_ai.backend.watchlist import WatchlistService
from sq_ai.portfolio.tracker import PortfolioTracker


def test_watchlist_add_normalises_symbol(tmp_path):
    svc = WatchlistService(PortfolioTracker(db_path=str(tmp_path / "t.db")))
    out = svc.add("reliance")
    assert out["symbol"] == "RELIANCE.NS"


def test_watchlist_add_keeps_dotted_form(tmp_path):
    svc = WatchlistService(PortfolioTracker(db_path=str(tmp_path / "t.db")))
    out = svc.add("INFY.NS", note="long")
    assert out["symbol"] == "INFY.NS"
    items = svc.list()
    assert items[0]["symbol"] == "INFY.NS"
    assert items[0]["note"] == "long"


def test_watchlist_remove(tmp_path):
    svc = WatchlistService(PortfolioTracker(db_path=str(tmp_path / "t.db")))
    svc.add("INFY.NS")
    out = svc.remove("INFY.NS")
    assert out["status"] == "ok"
    assert svc.list() == []


def test_watchlist_empty_symbol_raises(tmp_path):
    svc = WatchlistService(PortfolioTracker(db_path=str(tmp_path / "t.db")))
    try:
        svc.add("  ")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_preset_save_list_delete(tmp_path):
    t = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    t.preset_save("aggressive", {"rsi": {"min": 70}, "macd": "bullish"})
    rows = t.preset_list()
    assert len(rows) == 1
    assert rows[0]["filters"]["rsi"]["min"] == 70
    n = t.preset_delete("aggressive")
    assert n == 1
    assert t.preset_list() == []


def test_report_record_and_list(tmp_path):
    t = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    t.report_record("daily_2024_01_01.pdf", summary="markets up")
    rows = t.report_list()
    assert len(rows) == 1
    assert rows[0]["filename"] == "daily_2024_01_01.pdf"
    assert "markets up" in rows[0]["summary"]


def test_earnings_save_and_list(tmp_path):
    t = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    t.earnings_save("RELIANCE.NS", "Q3-2024",
                    call_date="2024-01-15",
                    transcript_url="http://example.com/r.pdf",
                    highlights={"highlights": ["x"], "tone": "positive"},
                    guidance={"revenue": "10% growth"})
    rows = t.earnings_list("RELIANCE.NS")
    assert len(rows) == 1
    assert rows[0]["highlights"]["tone"] == "positive"
    assert rows[0]["guidance"]["revenue"] == "10% growth"
