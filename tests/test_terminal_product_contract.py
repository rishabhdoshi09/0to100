from __future__ import annotations

import terminal_product_api_parallel as api


PRIMARY_PATHS = {
    "/api/dashboard",
    "/api/operations/{operation_id}",
    "/api/controls/{control_name}",
    "/api/recommendations-workspace",
    "/api/market-reports-workspace",
    "/api/stock-intelligence/{symbol}",
    "/api/due-diligence/{symbol}",
    "/api/due-diligence/{symbol}/acquire",
    "/api/operator-health",
    "/api/product-contract",
}


def _paths() -> set[str]:
    return {
        str(getattr(route, "path", ""))
        for route in api.app.routes
        if getattr(route, "path", None)
    }


def test_primary_react_product_routes_are_registered():
    missing = PRIMARY_PATHS - _paths()
    assert not missing, f"React product calls missing API route(s): {sorted(missing)}"


def test_product_contract_separates_wiring_from_data_availability(monkeypatch):
    monkeypatch.setattr(api.core, "_scan_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_long_term_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_operations_payload", lambda: {"running": False})
    monkeypatch.setattr(
        api.core,
        "_autonomy_payload",
        lambda: {"running": True, "learning_status": "WAITING_FOR_FRESH_EOD_DATA"},
    )

    payload = api.product_contract()
    assert payload["wired"] is True
    assert payload["checks"]["recommendations"]["route_registered"] is True
    assert payload["checks"]["recommendations"]["data_available"] is False
    assert payload["checks"]["market_scan"]["worker_running"] is False
    assert payload["checks"]["learning"]["status"] == "WAITING_FOR_FRESH_EOD_DATA"


def test_recommendations_route_returns_honest_empty_workspace(monkeypatch):
    monkeypatch.setattr(
        api.core,
        "_scan_payload",
        lambda: {"available": False, "records": [], "scanned_at": "", "records_status": ""},
    )
    monkeypatch.setattr(
        api.core,
        "_long_term_payload",
        lambda: {"available": False, "records": [], "scanned_at": ""},
    )

    payload = api.recommendations_workspace()
    assert payload["schema_version"] >= 1
    assert isinstance(payload["categories"], list)
    assert payload["ensemble"]["high_conviction_count"] == 0
    assert payload["from_saved_market_scan"] is True


def test_market_reports_route_returns_structured_empty_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api.core, "_scan_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_news_payload", lambda: {"available": False, "articles": []})

    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path / "market_reports")

    payload = api.market_reports_workspace()
    assert payload["schema_version"] >= 1
    assert isinstance(payload["reports"], list)
    assert isinstance(payload["missing_lanes"], list)
    assert payload["needs_refresh"] is True
    assert "invent" in (payload.get("empty_detail") or "").lower()
