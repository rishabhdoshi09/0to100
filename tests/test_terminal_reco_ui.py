"""The 5173 terminal must look like Reco Wealth, not the radar nav."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_reco_desk():
    src = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "Reco Wealth" in src
    assert "Today" in src and "Setups" in src and "Paper Desk" in src
    assert "Backtest" in src and "Portfolio" in src and "Desk" in src
    assert "MARKET RADAR" not in src
    assert "DISCOVERY" not in src
    assert "place_order" not in src


def test_terminal_theme_uses_reco_wealth_palette():
    css = (ROOT / "frontend" / "src" / "recoWealth.css").read_text(encoding="utf-8")
    assert "#004d40" in css
    assert "rw-stock-card" in css
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "Reco Wealth" in html
    assert "Source+Serif" in html


def test_today_has_reco_setup_cards():
    src = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "reco-card" in src
    assert "rw-stock-card" in src
    assert "TODAY · RECO WEALTH" in src
    assert "Best Setups · SEPA qualified" in src
    assert "Target" in src and "Upside from entry" in src
    assert "SEPA qualified" in src


def test_today_hides_quantterm_radar_chrome():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "rw-app" in app
    assert "Reco Wealth" in app
    assert "RecoWealth desk is waiting for the market API" in app
    sidebar = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "QUANTTERM" not in sidebar
