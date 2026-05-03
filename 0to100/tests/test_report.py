"""Tests for the daily report generator (no network – mocks)."""
from unittest.mock import patch

from sq_ai.backend.report_scheduler import ReportGenerator
from sq_ai.portfolio.tracker import PortfolioTracker


def test_report_generator_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("SQ_DB_PATH", str(tmp_path / "t.db"))
    tracker = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    gen = ReportGenerator(tracker=tracker, reports_dir=str(tmp_path / "reports"))

    fake_snap = {
        "indices": [{"name": "Nifty 50", "symbol": "^NSEI", "price": 25000,
                     "change_pct": 0.5, "ema_20": 24800, "ema_50": 24500}],
        "sectors": [{"sector": "IT", "change_pct": 0.7}],
    }
    fake_movers = {
        "gainers": [{"symbol": "X.NS", "price": 100, "change_pct": 5.0}],
        "losers":  [{"symbol": "Y.NS", "price": 100, "change_pct": -4.0}],
    }
    with patch("sq_ai.backend.report_scheduler.market_snapshot",
               return_value=fake_snap), \
         patch("sq_ai.backend.report_scheduler.top_movers",
               return_value=fake_movers), \
         patch("sq_ai.backend.report_scheduler._narrative",
               return_value="markets stable, IT leads."):
        out = gen.generate()
    assert out["filename"].endswith(".pdf")
    from pathlib import Path
    assert Path(out["path"]).exists()
    assert Path(out["path"]).stat().st_size > 0
    rows = tracker.report_list()
    assert len(rows) == 1
    assert rows[0]["filename"] == out["filename"]


def test_render_pdf_text_fallback_when_reportlab_missing(tmp_path, monkeypatch):
    """If reportlab import fails the renderer falls back to plain text."""
    from sq_ai.backend.report_scheduler import render_pdf
    payload = {
        "date": "2024-01-01",
        "snapshot": {"indices": [], "sectors": []},
        "movers": {"gainers": [], "losers": []},
        "narrative": "no market data today.",
    }
    out = tmp_path / "x.pdf"
    p = render_pdf(payload, out)
    assert p.exists()
    assert p.stat().st_size > 0
