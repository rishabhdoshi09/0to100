from __future__ import annotations

from pathlib import Path

import report_api


def test_report_api_has_no_broker_or_order_routes():
    paths = {route.path for route in report_api.app.routes}
    assert "/reports/equity/{symbol}" in paths
    assert "/reports/basket/long-term" in paths
    assert not any("broker" in path.lower() or "order" in path.lower() for path in paths)


def test_pdf_response_rejects_non_pdf(tmp_path: Path):
    path = tmp_path / "bad.txt"
    path.write_text("not a pdf", encoding="utf-8")
    try:
        report_api._pdf_response(path)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 500
    else:
        raise AssertionError("non-PDF artifact was accepted")
