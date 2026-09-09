from __future__ import annotations

import re

import terminal_api as core
from terminal_product_api_parallel import app


def _normalized_routes() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for route in app.routes:
        path = str(getattr(route, "path", "") or "")
        if not path.startswith("/api/"):
            continue
        path = re.sub(r"\{[^}]+\}", "{}", path)
        for method in set(getattr(route, "methods", set()) or set()):
            out.add((method.upper(), path))
    return out


def test_every_primary_visible_desk_surface_has_a_real_backend_route():
    """UI navigation must terminate in an actual API contract, not a decorative page."""
    routes = _normalized_routes()
    expected = {
        ("GET", "/api/health"),
        ("GET", "/api/dashboard"),
        ("GET", "/api/operations"),
        ("GET", "/api/operations/{}"),
        ("POST", "/api/controls/{}"),
        ("GET", "/api/chart/{}"),
        ("GET", "/api/data-readiness"),
        ("GET", "/api/news"),
        ("GET", "/api/education"),
        ("GET", "/api/fno"),
        ("GET", "/api/product-readiness"),
        ("POST", "/api/product-bootstrap"),
        ("GET", "/api/desk-pipeline"),
        ("GET", "/api/radar-home"),
        ("GET", "/api/scanner-workspace/{}"),
        ("GET", "/api/recommendations-workspace"),
        ("GET", "/api/market-reports-workspace"),
        ("GET", "/api/stock-intelligence/{}"),
        ("POST", "/api/stock-intelligence/{}/refresh-fundamentals"),
        ("GET", "/api/due-diligence/{}"),
        ("POST", "/api/due-diligence/{}/acquire"),
        ("GET", "/api/stock-investigator/suggest"),
        ("GET", "/api/trade-plan/{}"),
        ("GET", "/api/compare"),
        ("GET", "/api/watchlist"),
        ("POST", "/api/watchlist"),
        ("DELETE", "/api/watchlist/{}"),
        ("GET", "/api/data/ratios/{}"),
        ("GET", "/api/strategy-catalog"),
        ("GET", "/api/research-status"),
        ("GET", "/api/learning-dashboard"),
        ("GET", "/api/decision-journal"),
        ("GET", "/api/forward-soak"),
        ("POST", "/api/forward-soak"),
        ("GET", "/api/decision-simulator"),
        ("POST", "/api/decision-simulator"),
        ("GET", "/api/system-health-contract"),
        ("GET", "/api/scan-audit"),
        ("GET", "/api/product-contract"),
    }
    missing = sorted(expected - routes)
    assert not missing, f"Visible QuantTerm surface(s) are not wired to the backend: {missing}"


def test_frontend_operation_controls_are_recognized_by_the_worker_control_plane():
    expected = {
        "RUN_SCAN_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_MARKET_REPORT_NOW",
        "REFRESH_FNO_NOW",
        "REFRESH_DATA_NOW",
    }
    assert expected <= core._ALLOWED_CONTROLS


def test_live_money_mutation_is_not_exposed_through_terminal_controls():
    forbidden = {
        "LIVE_BUY",
        "LIVE_SELL",
        "BROKER_BUY",
        "BROKER_SELL",
        "UNLOCK_LIVE_MONEY",
        "DISABLE_RISK",
    }
    assert forbidden.isdisjoint(core._ALLOWED_CONTROLS)
