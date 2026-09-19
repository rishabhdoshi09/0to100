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
    assert "run_quantterm_complete.sh" in src
    assert "run_desk.sh" in src
    assert "127.0.0.1:5173" in src

    # Navigation has one authority: MarketSidebar consumes the canonical
    # contract instead of duplicating route names in the rendering layer.
    sidebar = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    navigation = (ROOT / "frontend" / "src" / "navigation.ts").read_text(encoding="utf-8")
    assert "WORKSPACE_NAV" in sidebar and "TOOL_GROUPS" in sidebar
    assert "WORKSPACE_NAV" in navigation and "TOOL_GROUPS" in navigation
    for route in ("Home", "Market Scanner", "Recommendations", "Market Reports", "Strategies", "Learning", "Coverage"):
        assert route in navigation

    desk = (ROOT / "scripts" / "run_desk.sh").read_text(encoding="utf-8")
    assert "run_quantterm_complete.sh" in desk
    assert "streamlit" not in desk


def test_desk_board_does_not_start_scanners():
    src = (ROOT / "ui" / "desk_board.py").read_text(encoding="utf-8")
    assert "start_background_scan" not in src
    assert "force_rescan" not in src
    assert ".start()" not in src
    assert "place_order" not in src
