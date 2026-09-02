"""Phase-13 burn-in: API-before-frontend startup and official-history freshness."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from operations.market_ops import (
    DATA_PREPARE,
    LANES,
    MARKET_SCAN,
    MarketOperationsWorker,
    OperationBlocked,
)
from operations.store import FAILED, SUCCEEDED, OperationStore
from product.home_os import PREPARING, build_home_os


IST_OPEN = datetime(2026, 9, 2, 10, 45)
IST_WEEKEND = datetime(2026, 9, 5, 10, 45)
STALE = {
    "ready": True,
    "sessions": 1796,
    "symbols": 2200,
    "latest_date": "2026-08-28",
}
CURRENT = {
    "ready": True,
    "sessions": 1796,
    "symbols": 2200,
    "latest_date": "2026-09-01",
}


def _freshness(history, *, current: bool, expected="2026-09-01", reason="HISTORY_STALE"):
    snap = dict(history)
    latest = str(snap.get("latest_date") or "")
    return {
        **snap,
        "current": current,
        "expected_latest_completed_session": expected,
        "available_session": latest,
        "stale_sessions": 0 if current else 2,
        "reason_code": "HISTORY_CURRENT" if current else reason,
        "history": snap,
    }


def test_expected_session_skips_weekend_and_holiday():
    from data.bhavcopy_runtime import expected_latest_completed_session, official_history_freshness

    saturday = expected_latest_completed_session(now=IST_WEEKEND, holidays=set())
    assert saturday.isoformat() == "2026-09-04"

    holiday_monday = expected_latest_completed_session(
        now=datetime(2026, 9, 7, 10, 0),
        holidays={"2026-09-07"},
    )
    assert holiday_monday.isoformat() == "2026-09-04"

    before_publish = expected_latest_completed_session(now=IST_OPEN, holidays=set())
    assert before_publish.isoformat() == "2026-09-01"

    stale = official_history_freshness(STALE, now=IST_OPEN, holidays=set(), load_cache=False)
    assert stale["current"] is False
    assert stale["reason_code"] == "HISTORY_STALE"
    assert stale["expected_latest_completed_session"] == "2026-09-01"
    assert stale["available_session"] == "2026-08-28"
    assert int(stale["stale_sessions"] or 0) >= 2

    current = official_history_freshness(CURRENT, now=IST_OPEN, holidays=set(), load_cache=False)
    assert current["current"] is True
    assert current["reason_code"] == "HISTORY_CURRENT"


def test_large_stale_history_is_not_current():
    from data.bhavcopy_runtime import official_history_freshness
    from product.desk_pipeline import prices_kind_due

    freshness = official_history_freshness(STALE, now=IST_OPEN, holidays=set(), load_cache=False)
    assert freshness["sessions"] == 1796
    assert freshness["current"] is False
    assert prices_kind_due.__doc__


def test_prices_kind_due_requests_prepare_when_latest_is_stale(monkeypatch):
    from data import bhavcopy_runtime
    from operations.market_ops import DATA_PREPARE
    from product.desk_pipeline import prices_kind_due

    monkeypatch.setattr(
        bhavcopy_runtime,
        "official_history_freshness",
        lambda **_k: _freshness(STALE, current=False),
    )
    assert prices_kind_due() == DATA_PREPARE


def test_scan_does_not_proceed_on_stale_large_history(tmp_path: Path, monkeypatch):
    scanned = []
    worker = MarketOperationsWorker(store=OperationStore(tmp_path / "ops.db"))
    monkeypatch.setattr(worker, "_history_ready", lambda snapshot=None: (False, _freshness(STALE, current=False)))
    monkeypatch.setattr(worker.store, "get", lambda _oid: {"status": SUCCEEDED})
    monkeypatch.setattr("scan.market_scan_service.run_whole_market_scan", lambda **_k: scanned.append(1))
    op, _ = worker.store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="terminal", priority=100)
    try:
        worker._run_market_scan(op)
        raise AssertionError("stale history must not scan as current")
    except OperationBlocked as exc:
        assert exc.code == "HISTORY_STALE"
        assert exc.result.get("non_actionable") is True
    assert scanned == []
    kinds = [row["kind"] for row in worker.store.active() + worker.store.recent()]
    assert DATA_PREPARE in kinds
    assert kinds.count(MARKET_SCAN) == 1


def test_current_history_allows_canonical_scan(tmp_path: Path, monkeypatch):
    worker = MarketOperationsWorker(store=OperationStore(tmp_path / "ops.db"))
    monkeypatch.setattr(
        worker,
        "_require_current_history",
        lambda _op: {**CURRENT, "current": True, "available_session": "2026-09-01"},
    )
    monkeypatch.setattr("scan.bulk_fetcher.adopt_ready_store", lambda overlay_live=True: 500)
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: ["AAA"])

    class FakeReport:
        ok = True
        status = "OK"
        payload = {"summary": {"with_any_setup": 1}, "records": [{"symbol": "AAA"}]}

    monkeypatch.setattr("scan.market_scan_service.run_whole_market_scan", lambda **_k: FakeReport())
    monkeypatch.setattr(worker, "_notify_scan_telegram", lambda _p: {})
    op, _ = worker.store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="pipeline")
    result = worker._run_market_scan(op)
    assert result["records"] == 1
    assert result["as_of_session"] == "2026-09-01"


def test_persisted_market_scan_waits_for_data_prepare(tmp_path: Path, monkeypatch):
    store = OperationStore(tmp_path / "ops.db")
    data, _ = store.enqueue(DATA_PREPARE, lane=LANES[DATA_PREPARE], requested_by="bootstrap")
    scan, _ = store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="pipeline")
    store.lease_next("data", worker_pid=1)
    leased = store.lease_next("market_scan", worker_pid=1)
    assert leased["operation_id"] == scan["operation_id"]
    assert store.get(data["operation_id"])["status"] == "RUNNING"

    worker = MarketOperationsWorker(store=store)
    hits = {"ready": 0}

    def fake_ready(snapshot=None):
        hits["ready"] += 1
        if hits["ready"] >= 3:
            return True, _freshness(CURRENT, current=True, reason="HISTORY_CURRENT")
        return False, _freshness(STALE, current=False)

    monkeypatch.setattr(worker, "_history_ready", fake_ready)
    monkeypatch.setattr(worker.store, "get", lambda _oid: {"status": SUCCEEDED, "operation_id": data["operation_id"]})
    monkeypatch.setattr(worker, "HISTORY_WAIT_POLL_S", 0.01, raising=False)

    class FakeReport:
        ok = True
        status = "OK"
        payload = {"summary": {"with_any_setup": 0}, "records": []}

    scanned = []

    def fake_scan(**_k):
        scanned.append(1)
        return FakeReport()

    monkeypatch.setattr("scan.bulk_fetcher.adopt_ready_store", lambda overlay_live=True: 500)
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: ["AAA"])
    monkeypatch.setattr("scan.market_scan_service.run_whole_market_scan", fake_scan)
    monkeypatch.setattr(worker, "_notify_scan_telegram", lambda _p: {})

    result = worker._run_market_scan(leased)
    assert scanned == [1]
    assert result["records"] == 0
    active_scans = [row for row in store.active() if row["kind"] == MARKET_SCAN]
    assert len(active_scans) <= 1
    prepare_ops = [row for row in store.recent() if row["kind"] == DATA_PREPARE]
    assert len(prepare_ops) == 1


def test_scan_now_with_stale_history_enqueues_refresh_then_scans(tmp_path: Path, monkeypatch):
    import terminal_api as api
    from fastapi.testclient import TestClient

    jobs = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs))
    monkeypatch.setattr(api, "_ensure_ops_worker", lambda *a, **k: {"running": True})
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *a, **k: _freshness(STALE, current=False),
    )
    client = TestClient(api.app)
    first = client.post("/api/controls/RUN_SCAN_NOW")
    second = client.post("/api/controls/RUN_SCAN_NOW")
    assert first.status_code == 200
    assert first.json()["created"] is True
    assert second.json()["created"] is False
    assert first.json()["operation_id"] == second.json()["operation_id"]
    store = OperationStore(jobs)
    kinds = [row["kind"] for row in store.active()]
    assert kinds.count(MARKET_SCAN) == 1
    assert kinds.count(DATA_PREPARE) == 1

    worker = MarketOperationsWorker(store=store)
    hits = {"ready": 0}

    def fake_ready(snapshot=None):
        hits["ready"] += 1
        if hits["ready"] >= 3:
            return True, _freshness(CURRENT, current=True, reason="HISTORY_CURRENT")
        return False, _freshness(STALE, current=False)

    monkeypatch.setattr(worker, "_history_ready", fake_ready)
    monkeypatch.setattr(worker.store, "get", lambda _oid: {"status": SUCCEEDED})
    monkeypatch.setattr("scan.bulk_fetcher.adopt_ready_store", lambda overlay_live=True: 500)
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: ["AAA"])

    class FakeReport:
        ok = True
        status = "OK"
        payload = {"summary": {"with_any_setup": 1}, "records": [{"symbol": "AAA"}]}

    scanned = []
    monkeypatch.setattr(
        "scan.market_scan_service.run_whole_market_scan",
        lambda **_k: scanned.append(1) or FakeReport(),
    )
    monkeypatch.setattr(worker, "_notify_scan_telegram", lambda _p: {})
    op = store.latest(MARKET_SCAN)
    result = worker._run_market_scan(op)
    assert scanned == [1]
    assert result["records"] == 1
    assert [row["kind"] for row in store.recent() if row["kind"] == MARKET_SCAN].count(MARKET_SCAN) == 1


def test_failed_data_refresh_does_not_fake_a_fresh_scan(tmp_path: Path, monkeypatch):
    worker = MarketOperationsWorker(store=OperationStore(tmp_path / "ops.db"))
    monkeypatch.setattr(worker, "_history_ready", lambda snapshot=None: (False, _freshness(STALE, current=False)))
    monkeypatch.setattr(worker.store, "get", lambda _oid: {"status": FAILED})
    scanned = []
    monkeypatch.setattr("scan.market_scan_service.run_whole_market_scan", lambda **_k: scanned.append(1))
    op, _ = worker.store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="terminal", priority=100)
    try:
        worker._run_market_scan(op)
        raise AssertionError("failed refresh must fail closed")
    except OperationBlocked as exc:
        assert exc.code == "HISTORY_STALE"
        assert exc.result.get("non_actionable") is True
    assert scanned == []


def test_home_hides_stale_scan_and_explains_preparing():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {
                "ready": True,
                "bhavcopy": {
                    "ready": True,
                    "sessions": 1796,
                    "latest_date": "2026-08-28",
                    "current": False,
                    "expected_latest_completed_session": "2026-09-01",
                    "available_session": "2026-08-28",
                    "stale_sessions": 2,
                    "reason_code": "HISTORY_STALE",
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-02T03:00:00+00:00", "records": [{"symbol": "TCS"}]},
        radar={"best_of_best": [{"symbol": "TCS", "verdict": "BUY"}]},
        now=datetime(2026, 9, 2, 10, 45),
    )
    assert os["state"] == PREPARING
    assert "still being prepared" in os["headline"].lower() or "latest market data" in os["subtext"].lower()
    assert "getting the latest market data" in os["subtext"].lower()
    assert os["opportunities"] == []
    assert os["history_freshness"]["expected_latest_completed_session"] == "2026-09-01"
    assert os["history_freshness"]["available_session"] == "2026-08-28"
    assert os["history_freshness"]["stale_sessions"] == 2
    assert os["history_freshness"]["reason_code"] == "HISTORY_STALE"
    assert os["today"]["data_fresh"] is False
    assert os["live_locked"] is True


def test_home_no_trade_and_live_lock_unchanged_when_history_current():
    from product.home_os import NO_TRADE

    os = build_home_os(
        dashboard={
            "autonomy": {"state": "RUNNING", "running": True},
            "data": {"ready": True, "bhavcopy": {"ready": True, "latest_date": "2026-09-01", "current": True}},
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={
            "available": True,
            "headline": "No paper trade taken: ENTRY_TOO_EXTENDED",
            "decision": "NO_TRADE",
            "reasons": ["ENTRY_TOO_EXTENDED"],
            "rejections": [{"symbol": "TCS", "reason_code": "ENTRY_TOO_EXTENDED"}],
            "taken": [],
        },
        soak={"real_forward_observations": 0, "insufficient_evidence": True, "FORWARD_SOAK_STATUS": "COLLECTING"},
        scan={"scanned_at": "2026-09-01T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=datetime(2026, 9, 1, 10, 45),
    )
    assert os["state"] == NO_TRADE
    assert os["live_locked"] is True
    assert os["need_me"] is False
