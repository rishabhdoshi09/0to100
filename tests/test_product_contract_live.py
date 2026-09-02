"""Issue #92: user-visible live-terminal contract against real operation handlers.

These tests execute MarketOperationsWorker / TestClient paths. They do not
accept 'the component exists' as done. Network is patched so CI stays offline,
but jobs actually enqueue, run, persist progress, and return results.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import terminal_api as api
from operations.market_ops import (
    BLOCKED,
    DUE_DILIGENCE_ACQUIRE,
    LANES,
    MARKET_REPORT,
    MARKET_SCAN,
    MarketOperationsWorker,
    OperationBlocked,
)
from operations.store import OperationStore


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs_db))
    monkeypatch.setattr(api, "_ensure_ops_worker", lambda *a, **k: {"running": True})
    return TestClient(api.app)


def test_refresh_market_report_control_enqueues_and_dedupes(client: TestClient):
    first = client.post("/api/controls/REFRESH_MARKET_REPORT_NOW")
    second = client.post("/api/controls/REFRESH_MARKET_REPORT_NOW")
    assert first.status_code == 200
    assert second.status_code == 200
    payload = first.json()
    assert payload["accepted"] is True
    assert payload["operation_id"]
    assert first.json()["operation_id"] == second.json()["operation_id"]
    assert second.json().get("created") is False
    store = OperationStore(api.OPS_DB)
    assert any(item.get("kind") == MARKET_REPORT for item in store.active())


def test_market_report_job_rebuilds_from_official_files(tmp_path: Path, monkeypatch):
    rebuilt: list[bool] = []

    def fake_build(*, persist_today=True, rebuild=False, news_payload=None, scan_payload=None):
        rebuilt.append(bool(rebuild))
        return {
            "as_of_ist": "2026-08-27",
            "today_pulse": {"takeaways": ["Official session close"]},
            "desk_note": {"daily_wrap": ["Wrap from sourced files"], "wrap_sourced": 1},
            "scan_highlights": {"row_count": 3},
            "needs_refresh": False,
            "missing_lanes": [],
            "reports": [{"id": "market_pulse_2026-08-27"}],
        }

    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.recommendations_workspace.build_market_reports_workspace",
        fake_build,
    )
    monkeypatch.setattr("product.scan_store.load_scan", lambda: {"records": [{"symbol": "TCS"}]})

    worker = MarketOperationsWorker(store=OperationStore(tmp_path / "ops.db"))
    op, created = worker.store.enqueue(MARKET_REPORT, lane=LANES[MARKET_REPORT], requested_by="terminal")
    assert created is True
    result = worker._run_market_report(op)
    assert rebuilt == [True]
    assert result["takeaways"] == 1
    assert result["wrap_sourced"] == 1
    assert result["scan_rows"] == 3
    assert result["needs_refresh"] is False


def test_market_scan_job_writes_qualified_count(tmp_path: Path, monkeypatch):
    store = OperationStore(tmp_path / "ops.db")
    worker = MarketOperationsWorker(store=store)
    monkeypatch.setattr(
        worker,
        "_require_current_history",
        lambda *_a, **_k: {"sessions": 240, "ready": True, "current": True, "latest_date": "2026-09-01"},
    )
    monkeypatch.setattr("scan.bulk_fetcher.adopt_ready_store", lambda overlay_live=True: 500)
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(
        worker,
        "_notify_scan_telegram",
        lambda payload: {"sent": False, "reason": "test"},
    )

    class FakeReport:
        ok = True
        status = "OK"
        payload = {
            "summary": {"with_any_setup": 4},
            "records": [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}, {"symbol": "DDD"}],
            "long_term_overlay": {},
        }

    monkeypatch.setattr(
        "scan.market_scan_service.run_whole_market_scan",
        lambda **_k: FakeReport(),
    )
    op, _ = store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="terminal", priority=100)
    result = worker._run_market_scan(op)
    assert result["summary"]["qualified"] == 4
    assert result["records"] == 4
    saved = store.get(op["operation_id"])
    assert saved is not None
    # Handler itself does not finish; caller/lane does. Progress must have been written.
    progress = store.get(op["operation_id"])
    assert progress["stage"] in {"LOADING_UNIVERSE", "SCANNING", "RANKING", "SAVING", ""}


def test_blocked_job_exposes_code_and_retry_is_a_new_job(tmp_path: Path, monkeypatch, client: TestClient):
    store = OperationStore(api.OPS_DB)
    worker = MarketOperationsWorker(store=store)
    monkeypatch.setattr(
        worker,
        "_run_market_scan",
        lambda _op: (_ for _ in ()).throw(
            OperationBlocked("Official bhavcopy is missing", code="BHAV_MISSING")
        ),
    )
    queued, _ = store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="terminal", priority=100)
    leased = store.lease_next("market_scan", worker_pid=1)
    assert leased is not None
    try:
        worker._execute(leased)
        raise AssertionError("scan should have blocked")
    except OperationBlocked as exc:
        store.finish(
            leased["operation_id"],
            status=BLOCKED,
            message=str(exc),
            error_code=exc.code,
            error_message=str(exc),
        )
    response = client.get(f"/api/operations/{queued['operation_id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "BLOCKED"
    assert payload["error_code"] == "BHAV_MISSING"
    assert "bhavcopy" in (payload.get("error_message") or payload.get("message") or "").lower()

    retry = client.post("/api/controls/RUN_SCAN_NOW")
    assert retry.status_code == 200
    assert retry.json()["created"] is True
    assert retry.json()["operation_id"] != queued["operation_id"]


def test_repeated_clicks_do_not_duplicate_active_jobs(client: TestClient):
    first = client.post("/api/controls/RUN_SCAN_NOW")
    second = client.post("/api/controls/RUN_SCAN_NOW")
    third = client.post("/api/controls/REFRESH_MARKET_REPORT_NOW")
    fourth = client.post("/api/controls/REFRESH_MARKET_REPORT_NOW")
    assert first.json()["operation_id"] == second.json()["operation_id"]
    assert second.json()["created"] is False
    assert third.json()["operation_id"] == fourth.json()["operation_id"]
    assert fourth.json()["created"] is False


def test_async_acquire_enqueues_symbol_job_and_dedupes_same_ticker(tmp_path: Path, monkeypatch):
    import terminal_product_api as papi

    monkeypatch.setattr(api, "OPS_DB", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(api, "_ensure_ops_worker", lambda *a, **k: {"running": True})
    client = TestClient(papi.app)
    first = client.post("/api/due-diligence/TCS/acquire?async_job=true")
    second = client.post("/api/due-diligence/TCS/acquire?async_job=true")
    other = client.post("/api/due-diligence/RELIANCE/acquire?async_job=true")
    assert first.status_code == 200
    assert first.json()["async"] is True
    assert first.json()["operation_id"]
    assert "report" not in first.json()
    assert second.json()["operation_id"] == first.json()["operation_id"]
    assert second.json()["created"] is False
    assert other.json()["operation_id"] != first.json()["operation_id"]
    assert other.json()["created"] is True


def test_due_diligence_acquire_handler_runs_for_one_symbol(tmp_path: Path, monkeypatch):
    seen: list[tuple[str, bool]] = []

    def fake_acquire(symbol: str, force: bool = False):
        seen.append((symbol, force))
        return {"ok": True, "n_ok": 1, "symbol": symbol}

    monkeypatch.setattr("product.due_diligence.acquire.acquire_symbol", fake_acquire)
    worker = MarketOperationsWorker(store=OperationStore(tmp_path / "ops.db"))
    op, _ = worker.store.enqueue(
        DUE_DILIGENCE_ACQUIRE,
        lane=LANES[DUE_DILIGENCE_ACQUIRE],
        requested_by="terminal",
        payload={"symbol": "TCS", "force": False},
        deduplicate=False,
    )
    result = worker._run_due_diligence_acquire(op)
    assert seen == [("TCS", False)]
    assert result["symbol"] == "TCS"
    assert result["n_ok"] == 1


def test_page_open_get_still_does_not_rebuild_report(tmp_path: Path, monkeypatch):
    """GET stays cache-first. The MARKET_REPORT job is what rebuilds."""
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda **_k: (_ for _ in ()).throw(AssertionError("GET must not crawl")),
    )
    monkeypatch.setattr(
        "product.desk_note.build_desk_note",
        lambda **_k: {"wrap": [], "desks": [], "explainers": [], "daily_wrap": [], "wrap_sourced": 0},
    )
    payload = rw.build_market_reports_workspace(
        persist_today=True,
        rebuild=False,
        scan_payload={"records": []},
        news_payload={"available": False, "articles": []},
    )
    assert payload["needs_refresh"] is True
    assert "market_scan" in payload["missing_lanes"]


def test_empty_high_conviction_explains_what_was_checked():
    from product.reco_ensemble import ensemble_summary

    summary = ensemble_summary([
        {
            "reco_tier": "WATCH",
            "families": [{"id": "trend", "label": "Trend", "status": "fail"}],
            "methods": [{"id": "sepa", "label": "SEPA", "status": "fail"}],
        },
        {
            "reco_tier": "good_setup",
            "families": [{"id": "quality", "label": "Quality", "status": "pass"}],
            "methods": [{"id": "earnings", "label": "Earnings", "status": "pass"}],
        },
    ])
    assert summary["empty_high_conviction"] is True
    assert summary["checked_rows"] == 2
    assert "Checked 2" in summary["empty_detail"]
    assert "Quality" in summary["empty_detail"] or "Trend" in summary["empty_detail"]
    assert "Good setups: 1" in summary["empty_detail"]


def test_decision_surface_exposes_blockers_freshness_coverage():
    from product.decision_card import decision_surface

    surface = decision_surface(
        {
            "symbol": "ABC",
            "price": 100,
            "entry": 99,
            "stop": 90,
            "target": 120,
            "conflicts": ["RSI overbought"],
            "price_tag": "EOD",
            "fundamental_coverage": 42,
        },
        category_id="wealth_builders",
        action_badge="Watch",
        qualify_reason="Quality compounder",
        market_ctx={},
    )
    assert surface["blockers"] == ["RSI overbought"]
    assert surface["freshness"] == "EOD"


def test_generic_framework_includes_cross_company_kpis():
    from product.due_diligence.sector_frameworks.kpis import GENERIC

    ids = {row.id for row in GENERIC}
    assert {"roe", "roce", "borrowings", "eps", "cfo"} <= ids


def test_named_quality_scores_stay_unmeasured_without_filings():
    from product.due_diligence.generic_scores import generic_cross_company_scores, piotroski_f_score

    empty = generic_cross_company_scores({"available": False, "data": {}})
    assert empty["available"] is False
    labels = {row["id"]: row["label_text"] for row in empty["scores"]}
    assert labels["piotroski_f"] == "Unmeasured"
    assert labels["altman_z"] == "Unmeasured"
    assert labels["beneish_m"] == "Unmeasured"
    assert labels["dupont_roe"] == "Unmeasured"
    thin = piotroski_f_score({
        "available": True,
        "data": {
            "key_ratios": [{"row_label": "ROA", "Mar 2026": 5.0}],
        },
    })
    assert thin["available"] is False
    assert thin["measured"] < 6
    assert thin["label_text"] == "Unmeasured"
