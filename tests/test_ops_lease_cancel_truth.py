"""Release-blocking lease, cancellation, FD, and operations-status truth."""
from __future__ import annotations

import inspect
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from operations.store import FAILED, KIND_RUNNING_LEASE_S, PENDING, RUNNING, SUCCEEDED, OperationStore
from product.process_resources import (
    RESOURCE_OK,
    RESOURCE_UNKNOWN,
    classify_fd_pressure,
    count_open_fds,
    resource_diagnostics,
)
from operations.status_snapshot import load_operations_snapshot, persist_operations_snapshot, slim_operations_status
from product.runtime_lifecycle import OPERATION_STATE_DIVERGED, READY, STARTING, inspect_runtime, lifecycle_reason


def test_retry_resets_attempt_deadline(tmp_path: Path):
    store = OperationStore(tmp_path / "jobs.db")
    item, _ = store.enqueue("DUE_DILIGENCE_ACQUIRE", lane="due_diligence")
    first = store.lease_next("due_diligence", worker_pid=os.getpid())
    assert first is not None
    first_started = float(first["first_started_at"])
    first_attempt = float(first["attempt_started_at"])
    assert first["attempt"] == 1
    assert abs(first_started - first_attempt) < 0.05
    store.recover_orphans()
    time.sleep(0.05)
    second = store.lease_next("due_diligence", worker_pid=os.getpid())
    assert second is not None
    assert second["operation_id"] == item["operation_id"]
    assert second["attempt"] == 2
    assert float(second["attempt_started_at"]) > first_attempt
    assert float(second["started_at"]) == float(second["attempt_started_at"])
    assert abs(float(second["first_started_at"]) - first_started) < 0.05
    now = float(second["attempt_started_at"]) + 10
    assert store.overdue_running(now=now) == []


def test_first_started_at_does_not_poison_later_attempts(tmp_path: Path):
    store = OperationStore(tmp_path / "jobs.db")
    store.enqueue("MARKET_SCAN", lane="market_scan")
    first = store.lease_next("market_scan", worker_pid=os.getpid())
    old = float(first["started_at"]) - (4 * 60 * 60)
    with store._connect() as con:
        con.execute(
            "UPDATE operations SET first_started_at=?, started_at=?, attempt_started_at=? "
            "WHERE operation_id=?",
            (old, old, old, first["operation_id"]),
        )
    store.recover_orphans()
    later = time.time()
    second = store.lease_next("market_scan", worker_pid=os.getpid())
    assert second is not None
    assert float(second["first_started_at"]) == old
    assert float(second["attempt_started_at"]) >= later - 1
    assert float(second["started_at"]) == float(second["attempt_started_at"])
    poison_now = float(second["attempt_started_at"]) + 30
    assert store.overdue_running(now=poison_now) == []
    assert store.overdue_running(now=old + (21 * 60)) == []
    assert store.overdue_running(now=float(second["attempt_started_at"]) + (21 * 60))


def test_stale_deadline_stops_actual_execution(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO

    monkeypatch.setitem(KIND_RUNNING_LEASE_S, "NEWS_REFRESH", 0.4)
    monkeypatch.setenv("QT_JOB_TEST_SLEEP", "8")
    store = OperationStore(tmp_path / "jobs.db")
    store.enqueue("NEWS_REFRESH", lane="news")
    leased = store.lease_next("news", worker_pid=os.getpid())
    worker = MO.MarketOperationsWorker(store)
    before = count_open_fds()
    started = time.monotonic()
    worker._run_leased_operation("news", leased)
    elapsed = time.monotonic() - started
    row = store.get(leased["operation_id"])
    assert row["status"] == FAILED
    assert row["error_code"] == "DEADLINE_EXCEEDED"
    assert elapsed < 6
    assert worker._active == {}
    assert leased["operation_id"] not in worker._children
    after = count_open_fds()
    assert before is not None and after is not None
    assert after - before <= 8


def test_terminal_db_operation_cannot_remain_runtime_active(tmp_path: Path):
    from operations import market_ops as MO

    store = OperationStore(tmp_path / "jobs.db")
    store.enqueue("NEWS_REFRESH", lane="news")
    leased = store.lease_next("news", worker_pid=os.getpid())
    worker = MO.MarketOperationsWorker(store)
    worker._set_active("news", leased)
    store.finish(leased["operation_id"], status=FAILED, message="forced terminal", error_code="DEADLINE_EXCEEDED")
    worker._clear_ghost_active()
    assert "news" not in worker._active
    runtime = worker._runtime_payload(running=True)
    assert leased["operation_id"] not in json.dumps(runtime.get("active") or {})


def test_ghost_market_scan_lane_recovers_and_replacement_leases(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO
    from operations.status_snapshot import persist_operations_snapshot

    monkeypatch.setitem(KIND_RUNNING_LEASE_S, "MARKET_SCAN", 0.5)
    monkeypatch.setenv("QT_JOB_TEST_SLEEP", "12")
    monkeypatch.setenv("QT_SCAN_PATH", str(tmp_path / "latest_momentum_scan.json"))
    store = OperationStore(tmp_path / "jobs.db")
    old, _ = store.enqueue("MARKET_SCAN", lane="market_scan")
    leased = store.lease_next("market_scan", worker_pid=os.getpid())
    worker = MO.MarketOperationsWorker(store)
    thread = threading.Thread(target=worker._run_leased_operation, args=("market_scan", leased), daemon=True)
    thread.start()
    deadline = time.time() + 8
    while time.time() < deadline and store.get(old["operation_id"])["status"] == RUNNING:
        time.sleep(0.05)
    thread.join(timeout=4)
    assert store.get(old["operation_id"])["status"] == FAILED
    assert worker._active == {}
    replacement, created = store.enqueue("MARKET_SCAN", lane="market_scan")
    assert created
    assert replacement["status"] == PENDING
    nxt = store.lease_next("market_scan", worker_pid=os.getpid())
    assert nxt is not None
    assert nxt["operation_id"] == replacement["operation_id"]
    monkeypatch.setitem(KIND_RUNNING_LEASE_S, "MARKET_SCAN", 20 * 60)
    monkeypatch.delenv("QT_JOB_TEST_SLEEP", raising=False)
    monkeypatch.setenv("QT_JOB_TEST_SCAN", "1")
    monkeypatch.setenv("QT_JOB_TEST_SESSION", "2026-09-04")
    worker._cancel_ids.clear()
    worker._run_leased_operation("market_scan", nxt)
    done = store.get(replacement["operation_id"])
    assert done["status"] == SUCCEEDED
    assert done["stage"] in {SUCCEEDED, "SCANNING"} or done["progress_total"] == 4
    artifact = json.loads((tmp_path / "latest_momentum_scan.json").read_text(encoding="utf-8"))
    assert artifact["as_of_session"] == "2026-09-04"
    assert artifact["scanned_at"]
    persist_operations_snapshot(store.compact_status(kinds=["MARKET_SCAN"]))


def test_runtime_db_divergence_degrades_health(tmp_path: Path, monkeypatch):
    runtime_path = tmp_path / "runtime.json"
    snap_path = tmp_path / "operations_status.json"
    monkeypatch.setenv("QT_MARKET_OPS_RUNTIME", str(runtime_path))
    monkeypatch.setenv("QT_OPERATIONS_SNAPSHOT", str(snap_path))
    op_id = "deadbeefghost"
    now = time.time()
    runtime_path.write_text(
        json.dumps(
            {
                "process_running": True,
                "worker_pid": os.getpid(),
                "heartbeat_epoch": now,
                "heartbeat": "now",
                "fd_count": 40,
                "active": {
                    "due_diligence": {
                        "operation_id": op_id,
                        "kind": "DUE_DILIGENCE_ACQUIRE",
                        "started_at": now - 30,
                        "attempt_started_at": now - 30,
                        "attempt": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    snap_path.write_text(
        json.dumps(
            {
                "available": True,
                "generated_at": now,
                "freshness": "CURRENT",
                "active": [],
                "recent": [
                    {
                        "operation_id": op_id,
                        "kind": "DUE_DILIGENCE_ACQUIRE",
                        "status": FAILED,
                        "attempt": 3,
                    }
                ],
                "latest": {},
                "counts": {FAILED: 1},
            }
        ),
        encoding="utf-8",
    )
    runtime = inspect_runtime(api_serving=True)
    assert runtime["lifecycle"] != READY
    assert runtime["integrity"]["state"] == OPERATION_STATE_DIVERGED
    assert runtime["integrity"]["diverged"]


def test_macos_self_fd_count_without_proc(monkeypatch):
    from pathlib import Path as PathCls

    real_is_dir = PathCls.is_dir

    def fake_is_dir(self):
        if str(self).startswith("/proc"):
            return False
        return real_is_dir(self)

    monkeypatch.setattr(PathCls, "is_dir", fake_is_dir)
    n = count_open_fds(os.getpid())
    assert isinstance(n, int) and n > 0
    assert count_open_fds(9_999_999) is None


def test_unknown_fd_count_is_not_ok():
    assert classify_fd_pressure(None, 256) == RESOURCE_UNKNOWN
    assert classify_fd_pressure(None, None) == RESOURCE_UNKNOWN
    payload = resource_diagnostics(api_pid=os.getpid(), market_ops_pid=9_999_999)
    assert payload["state"] == RESOURCE_UNKNOWN
    assert payload["state"] != RESOURCE_OK
    assert "not a safe-band" in payload["reason"]
    assert "within the safe band" not in payload["reason"]


def test_operations_endpoint_stays_bounded_under_sqlite_lock(tmp_path: Path, monkeypatch):
    import terminal_api as api

    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs_db))
    monkeypatch.setattr(api, "_ensure_ops_worker", lambda *a, **k: {"running": True})
    store = OperationStore(jobs_db)
    store.enqueue("MARKET_SCAN", lane="market_scan")
    store._drop_cached()
    locker = sqlite3.connect(str(jobs_db), timeout=5.0, isolation_level=None)
    locker.execute("PRAGMA locking_mode=EXCLUSIVE")
    locker.execute("BEGIN EXCLUSIVE")
    locker.execute("UPDATE operations SET message=?", ("held",))
    client = TestClient(api.app)
    started = time.monotonic()
    response = client.get("/api/operations")
    elapsed = time.monotonic() - started
    try:
        locker.execute("COMMIT")
    finally:
        locker.close()
    assert elapsed < 1.0
    payload = response.json()
    assert payload.get("available") is False or payload.get("freshness") in {"UNAVAILABLE", "STALE", "CURRENT"}
    if payload.get("available") is False:
        assert payload.get("freshness") == "UNAVAILABLE"


def test_lane_loop_does_not_run_hot_recovery():
    from operations.market_ops import MarketOperationsWorker

    source = inspect.getsource(MarketOperationsWorker._lane_loop)
    assert "recover_dead_running" not in source
    assert "recover_stale_running" not in source
    sweep = inspect.getsource(MarketOperationsWorker._supervisor_sweep)
    assert "recover_dead_running" in sweep
    assert "recover_stale_running" in sweep


def test_replacement_leases_after_timeout(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO

    monkeypatch.setitem(KIND_RUNNING_LEASE_S, "NEWS_REFRESH", 0.35)
    monkeypatch.setenv("QT_JOB_TEST_SLEEP", "6")
    store = OperationStore(tmp_path / "jobs.db")
    first, _ = store.enqueue("NEWS_REFRESH", lane="news")
    leased = store.lease_next("news", worker_pid=os.getpid())
    worker = MO.MarketOperationsWorker(store)
    worker._run_leased_operation("news", leased)
    assert store.get(first["operation_id"])["status"] == FAILED
    second, created = store.enqueue("NEWS_REFRESH", lane="news")
    assert created
    nxt = store.lease_next("news", worker_pid=os.getpid())
    assert nxt is not None
    assert nxt["operation_id"] == second["operation_id"]
    assert nxt["status"] == RUNNING


def test_no_fd_growth_after_timeout_cancellation(tmp_path: Path, monkeypatch):
    from operations import market_ops as MO

    monkeypatch.setitem(KIND_RUNNING_LEASE_S, "NEWS_REFRESH", 0.3)
    monkeypatch.setenv("QT_JOB_TEST_SLEEP", "5")
    store = OperationStore(tmp_path / "jobs.db")
    worker = MO.MarketOperationsWorker(store)
    warm = count_open_fds()
    assert warm is not None
    for _ in range(3):
        store.enqueue("NEWS_REFRESH", lane="news", deduplicate=False)
        leased = store.lease_next("news", worker_pid=os.getpid())
        worker._run_leased_operation("news", leased)
    after = count_open_fds()
    assert after is not None
    assert after - warm <= 8


def test_compact_status_strips_embedded_scan_payload(tmp_path: Path, monkeypatch):
    store = OperationStore(tmp_path / "jobs.db")
    item, _ = store.enqueue("MARKET_SCAN", lane="market_scan")
    leased = store.lease_next("market_scan", worker_pid=os.getpid())
    assert leased is not None
    fat_records = [{"symbol": f"S{i}", "why": "x" * 80} for i in range(400)]
    store.finish(
        item["operation_id"],
        status=SUCCEEDED,
        message="done",
        result={
            "records": 400,
            "summary": {"qualified": 12},
            "payload": {"records": fat_records, "universe": list(range(2000))},
        },
    )
    compact = store.compact_status(kinds=["MARKET_SCAN"])
    latest = compact["latest"]["MARKET_SCAN"]
    result = latest["result"]
    assert result["records"] == 400
    assert result["summary"]["qualified"] == 12
    assert result["payload"]["n_keys"] == 2
    encoded = json.dumps(compact)
    assert len(encoded) < 20_000
    assert "S199" not in encoded
    snap = tmp_path / "operations_status.json"
    persist_operations_snapshot(compact, path=snap)
    loaded, freshness = load_operations_snapshot(path=snap)
    assert freshness == "CURRENT"
    assert loaded["latest"]["MARKET_SCAN"]["result"]["records"] == 400
    assert "S199" not in json.dumps(loaded)


def test_lifecycle_reason_cites_the_blocker_not_generic_starting():
    components = [
        {"name": "api", "status": READY, "detail": "serving"},
        {"name": "official_history", "status": STARTING, "detail": "HISTORY_STALE"},
    ]
    assert "official_history" in lifecycle_reason(STARTING, [], components)
    assert "HISTORY_STALE" in lifecycle_reason(STARTING, [], components)
    assert lifecycle_reason(STARTING, [], components) != "Desk is still coming up"
    assert lifecycle_reason(READY, [], components).startswith("Required services")
    assert lifecycle_reason(STARTING, ["OPERATION_STATE_DIVERGED"], components) == "OPERATION_STATE_DIVERGED"
    slim = slim_operations_status(
        {
            "recent": [
                {
                    "kind": "MARKET_SCAN",
                    "result": {"payload": {"records": [{"symbol": "X"}] * 50}, "records": 50},
                }
            ],
            "latest": {},
            "active": [],
        }
    )
    assert slim["recent"][0]["result"]["records"] == 50
    assert slim["recent"][0]["result"]["payload"]["n_keys"] == 1
