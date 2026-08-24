"""The 5173 terminal must look like the RecoWealth desk, not the radar nav."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_reco_desk():
    src = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "NSE DESK" in src
    assert "Today" in src and "Setups" in src and "Paper Desk" in src
    assert "Backtest" in src and "Portfolio" in src and "Desk" in src
    assert "MARKET RADAR" not in src
    assert "DISCOVERY" not in src
    assert "place_order" not in src


def test_terminal_theme_uses_reco_palette():
    css = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")
    assert "#0a0e17" in css
    assert "#38bdf8" in css
    assert "34px 34px" not in css
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "NSE Desk" in html
    assert "JetBrains+Mono" in html


def test_today_has_reco_setup_cards():
    src = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "reco-card" in src
    assert "TODAY · NSE DESK" in src
    assert "How to use this desk" in src
