from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_product_api_exposes_scan_audit_and_coverage_state():
    src = (ROOT / "terminal_product_api_parallel.py").read_text(encoding="utf-8")
    assert '@product.app.get("/api/scan-audit")' in src
    assert '"audit_route_registered": "/api/scan-audit" in paths' in src
    assert '"coverage_state": str(scan.get("coverage_state") or "UNKNOWN")' in src


def test_market_scan_persists_stock_by_stock_coverage():
    src = (ROOT / "scan" / "market_scan_service.py").read_text(encoding="utf-8")
    assert "with observe_scanner(scanner, symbols) as probe" in src
    assert 'payload["coverage"] = coverage' in src
    assert "save_audit(audit)" in src
    assert "universe_size=universe_n" in src
