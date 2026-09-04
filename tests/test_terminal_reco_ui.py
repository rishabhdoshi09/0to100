"""The 5173 terminal is a Reco-light research desk, not an Iron Man HUD."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_quantterm_discovery_desk():
    src = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    assert "QUANTTERM" in src
    assert "JARVIS DESK" not in src
    assert "PRIMARY_NAV" in src
    assert "ADVANCED_NAV" in src
    assert "OPERATE" in src
    assert "Advanced" in src
    for route in (
        "Home",
        "Market Scanner",
        "Recommendations",
        "Market Reports",
        "Stock Intelligence",
        "Compare",
        "Watchlist",
        "Strategies",
        "Learning",
        "Coverage",
        "Paper Portfolio",
        "System Health",
        "Backtest",
    ):
        assert route in src
    assert "Reco Wealth" not in src
    assert "place_order" not in src
    assert "reco-filter-btn" not in (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")


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
    assert "deskStartupLabel" in app
    startup = (ROOT / "frontend" / "src" / "deskStartupState.ts").read_text(encoding="utf-8")
    assert "PREPARING DATA" in startup
    assert "RESOURCE EXHAUSTED" in startup
    assert "RESOURCE UNKNOWN" in startup
    assert "KeepPage" in app
    assert "quantterm-nav" in app or "readDeskNav" in app
    assert "sessionMemory" in app or "readSessionJson" in app
    radar = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "recall" in radar and "remember" in radar
    assert "Investigate" in radar
    views = (ROOT / "frontend" / "src" / "productViews.tsx").read_text(encoding="utf-8")
    assert "Investigate" in views
    assert "fetchDueDiligence" in views
    assert "acquireDueDiligence" in views
    assert "Acquire from the internet" in views
    assert "vs_technical_setup" in views
    assert "evidence_pack" in views
    assert "Complete missing research data" in views
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "if (selected) setQuery(selected)" in app


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
    assert "Investigate" in src
    css = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".reco-pick-together" in css
    py = (ROOT / "product" / "recommendations_workspace.py").read_text(encoding="utf-8")
    assert "wealth_builders" in py
    assert "recovery_setups" in py
    assert "one primary category" in py.lower() or "ONE primary category" in py
    assert "two independent evidence families" in py.lower() or "buy_requires_two_independent_evidence_families" in py
    assert "NO HIGH-CONVICTION" in (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "family_confirms" in src or "families" in src


def test_market_reports_renders_sourced_desk_note_not_a_blog():
    src = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "DeskNoteMagazine" in src
    assert "desk_note" in src
    assert "not a buy list" in src.lower()
    assert "Stock Intelligence" in src
    assert "From the last market scan" in src
    assert "Scan market" in src
    assert "Build report" in src
    assert "bootstrapProduct" in src
    assert "needs_refresh" in src
    assert "DailyWrapList" in src
    assert "Here's the wrap of the day" in (ROOT / "frontend" / "src" / "dailyWrap.tsx").read_text(encoding="utf-8") or "Here&apos;s the wrap of the day" in (ROOT / "frontend" / "src" / "dailyWrap.tsx").read_text(encoding="utf-8")
    radar = (ROOT / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    assert "DailyWrapList" in radar
    assert "Run long-term scan" not in radar
    assert "one scan fills every tab" in radar
    reco_src = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "Refresh funds" in reco_src
    assert "Refresh long-term" not in reco_src
    css = (ROOT / "frontend" / "src" / "recommendations.css").read_text(encoding="utf-8")
    assert ".desk-note" in css
    assert ".desk-tile" in css
    assert "counter(wrap)" in css
    py = (ROOT / "product" / "desk_note.py").read_text(encoding="utf-8")
    assert "WRAP_SLOTS" in py
    assert "MIX_SHIFT_DESKS" in py
    assert "places_orders" in py
    for banned in ("₹600 crore", "20,000 MTPA", "65% pre-booked", "Q1 FY27"):
        assert banned not in py


def test_research_desk_exposes_parity_journal_and_health_lanes():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "StrategiesView" in app
    assert "LearningJournalView" in app
    assert "CoverageView" in app
    assert "ProductionBacktestView" in app
    assert "SystemHealthView" in app
    research = (ROOT / "frontend" / "src" / "researchDeskViews.tsx").read_text(encoding="utf-8")
    assert "BACKTEST PARITY: UNVERIFIED" in research
    assert "AI is learning" not in research
    reco = (ROOT / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    assert "BACKTEST PARITY" in reco
    assert "fundamental_disagreement" in reco
    views = (ROOT / "frontend" / "src" / "productViews.tsx").read_text(encoding="utf-8")
    assert "WHY SHOULD I NOT BUY THIS" in views
    assert "thesis_breakers" in views
