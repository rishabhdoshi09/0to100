from __future__ import annotations

from pathlib import Path

import report_api


def test_report_api_has_no_broker_or_order_routes():
    paths = {route.path for route in report_api.app.routes}
    assert "/reports/equity/{symbol}" in paths
    assert "/reports/basket/long-term" in paths
    assert "/evidence/{symbol}" in paths
    assert "/evidence/{symbol}/actions/auto-acquire" in paths
    assert "/evidence/{symbol}/{kind}" in paths
    assert "/evidence/templates/{kind}.csv" in paths
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


def test_template_endpoint_returns_csv():
    response = report_api.evidence_template("financial_history")
    assert response.media_type == "text/csv"
    assert b"period_end" in response.body


def test_acquire_result_does_not_call_structured_failure_success():
    summary = report_api._acquire_result_summary(
        {
            "acquired_at": "2026-09-03T12:00:00+00:00",
            "steps": [
                {"id": "nse_filings", "ok": False, "error": "HTTP 403"},
                {"id": "screener", "ok": False, "error": "HTTP 503"},
                {"id": "option_chain", "ok": False, "skipped": True},
            ],
        },
        {
            "requirements": [
                {"id": "exchange_filings", "acquisition": "AUTOMATION_FAILED"},
                {"id": "quarterly_results", "acquisition": "AUTOMATION_FAILED"},
            ]
        },
    )

    assert summary["status"] == "FAILED"
    assert summary["items_attempted"] == 2
    assert summary["items_succeeded"] == 0
    assert summary["items_failed"] == 2
    assert summary["automation_failed"] == 2


def test_acquire_result_reports_partial_when_some_evidence_arrived():
    summary = report_api._acquire_result_summary(
        {
            "steps": [
                {"id": "screener", "ok": True},
                {"id": "nse_filings", "ok": False, "error": "temporary outage"},
            ]
        },
        {
            "requirements": [
                {"id": "quarterly_results", "acquisition": "AUTO_SOURCED"},
                {"id": "exchange_filings", "acquisition": "AUTOMATION_FAILED"},
            ]
        },
    )

    assert summary["status"] == "PARTIAL"
    assert summary["items_attempted"] == 2
    assert summary["items_succeeded"] == 1
    assert summary["items_failed"] == 1


def test_acquire_result_succeeds_when_attempt_has_no_failures():
    summary = report_api._acquire_result_summary(
        {"steps": [{"id": "screener", "ok": True}]},
        {"requirements": [{"id": "quarterly_results", "acquisition": "AUTO_SOURCED"}]},
    )

    assert summary["status"] == "SUCCEEDED"
    assert summary["items_succeeded"] == 1
    assert summary["automation_failed"] == 0
