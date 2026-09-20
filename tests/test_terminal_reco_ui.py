"""The 5173 terminal is a Reco-light research desk, not an Iron Man HUD."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_sidebar_is_quantterm_discovery_desk():
    sidebar = (ROOT / "frontend" / "src" / "MarketSidebar.tsx").read_text(encoding="utf-8")
    navigation = (ROOT / "frontend" / "src" / "navigation.ts").read_text(encoding="utf-8")

    assert "QUANTTERM" in sidebar
    assert "JARVIS DESK" not in sidebar
    assert "WORKSPACE_NAV" in sidebar
    assert "TOOL_GROUPS" in sidebar
    assert "Tools" in sidebar
    for workspace in ("Today", "Opportunities", "Research", "Portfolio", "System"):
        assert workspace in navigation

    # Route inventory belongs to navigation.ts. Keeping these assertions out of
    # MarketSidebar prevents tests from forcing a second navigation authority.
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
        assert route in navigation
