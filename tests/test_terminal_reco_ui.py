"""The 5173 terminal is a Reco-light research desk, not an Iron Man HUD."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_quantterm_discovery_desk():
    src = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "QUANTTERM" in src
    assert "RESEARCH DESK" in src
    assert "JARVIS DESK" not in src
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


def test_terminal_uses_reco_desk_not_hud_chrome():
    main = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
    assert "reco-desk.css" in main
    assert "recommendations.css" in main
    assert "ironman-hud.css" not in main
    assert "recoWealth.css" not in main
    reco = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".reco-light" in reco
    assert "#f4f7f5" in reco
    assert "#1b6b45" in reco
    desk = (ROOT / "frontend" / "src" / "reco-desk.css").read_text(encoding="utf-8")
    assert ".terminal-root.reco-desk" in desk
    assert "#f4f7f5" in desk
    assert "#1b6b45" in desk
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "QuantTerm" in html
    assert "Fraunces" in html
    assert "Orbitron" not in html
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "reco-desk" in app
    assert "hud-shell" not in app
    assert "DATA INCOMPLETE" not in app
    assert "PREPARING DATA" in app
    assert "KeepPage" in app
    assert "quantterm-nav" in app or "readDeskNav" in app
    assert "sessionMemory" in app or "readSessionJson" in app
    radar = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "recall" in radar and "remember" in radar


def test_home_is_three_lane_radar_with_two_best_of_panels():
    src = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "BEST TECHNICAL BREAKOUT" in src
    assert "BEST AMONG THE BEST" in src
    assert "Breakouts" in src and "Momentum" in src and "Long-Term Picks" in src
    assert "TODAY · RECO WEALTH" not in src
    assert "rw-stock-card" not in src
    assert "Make ready" not in src
    assert "Preparing official history and scan" in src
    assert "Official prices" in src
    assert "radar-pipeline" in src
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "Search any NSE share" in app
    assert "ZERODHA OK" in app


def test_recommendations_are_exclusive_decision_cards():
    src = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "Wealth Builders" in src or "wealth_builders" in src
    assert "See evidence" in src
    assert "Key points" in src
    assert "reco-pick-together" in src
    assert "key_points" in src
    assert "Case memory" in src
    assert "reco-case" in src
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
