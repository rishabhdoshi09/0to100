from __future__ import annotations

from pathlib import Path
import asyncio
import json

from fastapi import Request
from fastapi.responses import JSONResponse

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



def test_report_api_public_mode_blocks_evidence_mutation_before_route(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)
    downstream_called = False

    async def downstream(_request):
        nonlocal downstream_called
        downstream_called = True
        return JSONResponse({"ok": True})

    request = Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/evidence/TCS/actions/auto-acquire",
        "raw_path": b"/evidence/TCS/actions/auto-acquire",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 44000),
        "server": ("127.0.0.1", 8766),
    })
    response = asyncio.run(report_api._public_mutation_boundary(request, downstream))

    assert response.status_code == 403
    assert downstream_called is False
    assert json.loads(response.body)["code"] == "PUBLIC_READ_ONLY"


def test_report_health_exposes_public_access_without_token_value(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.setenv("QT_OPERATOR_TOKEN", "secret-report-operator-token")

    payload = report_api.health()
    assert payload["access"]["public_read_only"] is True
    assert payload["access"]["mutation_policy"] == "OPERATOR_TOKEN_REQUIRED"
    assert payload["access"]["operator_token_configured"] is True
    assert "secret-report-operator-token" not in json.dumps(payload)
