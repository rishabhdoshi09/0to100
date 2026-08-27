"""Reco desk UI: professional nav, visible backtest, paper-loss loop. No money-path changes."""
from __future__ import annotations

from pathlib import Path

from product.paper_lessons import (
    BACKTEST_DOES_NOT_CHANGE,
    BACKTEST_PURPOSE,
    paper_loss_lessons,
)
from product.scan_store import build_scan_payload
from ui.desk_board import reco_card_html, setup_badge

ROOT = Path(__file__).resolve().parents[1]


def test_app_points_at_the_vite_desk():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src
    assert "st.Page" not in src
    assert "run_desk.sh" in src
    assert "127.0.0.1:5173" in src
    nav = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "Home" in nav and "Market Scanner" in nav
    assert "Recommendations" in nav and "Market Reports" in nav
    assert "Long-Term Picks" in nav
    desk = (ROOT / "scripts" / "run_desk.sh").read_text(encoding="utf-8")
    assert "run_quantterm_complete.sh" in desk
    assert "streamlit" not in desk


def test_desk_board_does_not_start_scanners():
    src = (ROOT / "ui" / "desk_board.py").read_text(encoding="utf-8")
    assert "start_background_scan" not in src
    assert "force_rescan" not in src
    assert ".start()" not in src
    assert "place_order" not in src


def test_reco_card_uses_saved_scan_levels():
    payload = build_scan_payload(
        {"AAA": "Alpha Ltd"},
        [{"symbol": "AAA", "signals": ["MOMENTUM"], "score": 80.0, "verdict": "BUY",
          "reasons": ["Up 8% in 5 days"], "price": 203.4, "entry": 211, "stop": 186, "target": 261}],
    )
    row = payload["records"][0]
    assert setup_badge(row) == ("Buy Setup", "buy")
    html = reco_card_html(row)
    assert "AAA" in html and "Buy Setup" in html
    assert "Entry" in html and "Stop" in html and "Target" in html
    assert "Up 8% in 5 days" in html


def test_paper_losses_become_backtest_next_steps():
    closed = [
        {"symbol": "TCS", "pnl": -1200.0, "realized_R": -1.1, "exit_reason": "STOP",
         "entry_date": "2026-08-10", "exit_date": "2026-08-12"},
        {"symbol": "INFY", "pnl": 800.0, "realized_R": 0.7, "exit_reason": "TARGET",
         "entry_date": "2026-08-11", "exit_date": "2026-08-13"},
        {"symbol": "TCS", "pnl": -400.0, "realized_R": -0.4, "exit_reason": "MAX_HOLD",
         "entry_date": "2026-08-01", "exit_date": "2026-08-08"},
        {"symbol": "WIPRO", "pnl": -90.0, "realized_R": -0.2, "exit_reason": "STOP",
         "entry_date": "2026-08-14", "exit_date": "2026-08-15"},
    ]
    lessons = paper_loss_lessons(closed, limit=5)
    symbols = [row["symbol"] for row in lessons]
    assert "INFY" not in symbols
    assert symbols[0] == "WIPRO"          # newest loss first
    assert "TCS" in symbols
    assert symbols.count("TCS") == 1      # one next-step per name
    tcs = next(row for row in lessons if row["symbol"] == "TCS")
    assert tcs["exit_reason"] == "STOP"   # the later TCS loss, not the older one
    assert "Backtest" in tcs["next_step"]
    assert tcs["does_not_change"] == BACKTEST_DOES_NOT_CHANGE


def test_backtest_page_states_use_case_and_frozen_rules():
    src = (ROOT / "ui" / "retail_backtest_data.py").read_text(encoding="utf-8")
    assert "BACKTEST_PURPOSE" in src
    assert "BACKTEST_DOES_NOT_CHANGE" in src
    assert "qt_backtest_symbol" in src
    paper = (ROOT / "ui" / "retail_trade_market.py").read_text(encoding="utf-8")
    assert "render_paper_loss_followup" in paper
    assert "Paper Desk" in paper
    assert BACKTEST_PURPOSE
    assert "does not change today's BUY list" in BACKTEST_DOES_NOT_CHANGE.lower() or \
           "does not change" in BACKTEST_DOES_NOT_CHANGE.lower()


def test_help_explains_backtest_after_paper_loss():
    src = (ROOT / "ui" / "retail_pages.py").read_text(encoding="utf-8")
    assert "After a paper loss" in src
    assert "does not change today's BUY list" in src
    home = (ROOT / "ui" / "retail_home_momentum.py").read_text(encoding="utf-8")
    assert "render_today_board" in home
    assert "render_sepa_best_setups" in home
    assert "render_bot_learning" in home
    paper = (ROOT / "ui" / "retail_trade_market.py").read_text(encoding="utf-8")
    assert "render_bot_learning" in paper
    pages = (ROOT / "ui" / "desk_pages.py").read_text(encoding="utf-8")
    assert "Best Setups" in pages
    assert "render_sepa_setups" in pages
    assert "render_bot_learning" in pages
    help_src = (ROOT / "ui" / "retail_pages.py").read_text(encoding="utf-8")
    assert "Path to real money" in help_src
    assert "the bot cannot open that door" in help_src.lower()
