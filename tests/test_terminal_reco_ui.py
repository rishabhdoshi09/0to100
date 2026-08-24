"""The 5173 terminal keeps HUD chrome + Reco-light research islands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_quantterm_discovery_desk():
    src = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "QUANTTERM" in src
    assert "JARVIS DESK" in src
    assert "DISCOVERY" in src
    assert "TOOLS & EVIDENCE" in src
    for route in (
        "Home",
        "Market Scanner",
        "Recommendations",
        "Market Reports",
        "Stock Intelligence",
        "Long-Term Picks",
        "Compare",
        "Watchlist",
        "Education",
        "F&O Desk",
        "My Holdings",
        "System Health",
        "Backtest",
    ):
        assert route in src
    assert "Reco Wealth" not in src
    assert "place_order" not in src


def test_terminal_uses_hud_chrome_and_reco_light_islands():
    main = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
    assert "ironman-hud.css" in main
    assert "recommendations.css" in main
    assert "recoWealth.css" not in main
    hud = (ROOT / "frontend" / "src" / "ironman-hud.css").read_text(encoding="utf-8")
    assert "#03070f" in hud
    assert "#3de7ff" in hud
    assert "#f0c14b" in hud
    reco = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".reco-light" in reco
    assert "#f4f7f5" in reco
    assert "#1b6b45" in reco
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "QuantTerm" in html
    assert "Orbitron" in html
    assert "Fraunces" in html


def test_home_is_three_lane_radar_with_two_best_of_panels():
    src = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "BEST TECHNICAL BREAKOUT" in src
    assert "BEST AMONG BREAKOUTS" in src
    assert "Breakouts" in src and "Momentum" in src and "Long-Term Picks" in src
    assert "TODAY · RECO WEALTH" not in src
    assert "rw-stock-card" not in src
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "hud-shell" in app
    assert "Search any NSE share" in app
    assert "ZERODHA OK" in app
    assert "run_quantterm_complete.sh" in app


def test_recommendations_are_exclusive_decision_cards():
    src = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "Wealth Builders" in src or "wealth_builders" in src
    assert "See evidence" in src
    assert "Key points" in src
    assert "reco-pick-together" in src
    assert "key_points" in src
    assert "Full research" in src
    css = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".reco-pick-together" in css
    py = (ROOT / "product" / "recommendations_workspace.py").read_text(encoding="utf-8")
    assert "wealth_builders" in py
    assert "recovery_setups" in py
    assert "one primary category" in py.lower() or "ONE primary category" in py


def test_market_reports_renders_sourced_desk_note_not_a_blog():
    src = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "DeskNoteMagazine" in src
    assert "desk_note" in src
    assert "not a buy list" in src.lower()
    assert "Stock Intelligence" in src
    css = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".desk-note" in css
    assert ".desk-tile" in css
    py = (ROOT / "product" / "desk_note.py").read_text(encoding="utf-8")
    assert "WRAP_SLOTS" in py
    assert "MIX_SHIFT_DESKS" in py
    assert "places_orders" in py
    for banned in ("₹600 crore", "20,000 MTPA", "65% pre-booked", "Q1 FY27"):
        assert banned not in py
