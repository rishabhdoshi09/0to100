"""Market-ops worker must come ONLINE before heavy bootstrap work."""
from __future__ import annotations

import json
import time
from pathlib import Path

from operations.store import OperationStore, PENDING


def test_enqueue_refreshes_pending_message_on_reclick(tmp_path: Path):
    store = OperationStore(tmp_path / "ops.db")
    first, created = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        message="Queued and waiting for the dedicated market-operations worker",
    )
    assert created is True
    second, created = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="terminal",
        message="Queued, but market-ops worker is OFFLINE — restart stack",
    )
    assert created is False
    assert second["operation_id"] == first["operation_id"]
    assert second["status"] == PENDING
    assert "OFFLINE" in second["message"]


def test_bootstrap_does_not_require_full_pkl_load(tmp_path: Path, monkeypatch):
    from data import bhavcopy_runtime
    from operations import market_ops as MO

    monkeypatch.setattr(MO, "ROOT", tmp_path)
    monkeypatch.setattr(MO, "LOCK_PATH", tmp_path / "market_ops" / "worker.lock")
    calls = {"load_cache": []}

    def status(*, load_cache: bool = False):
        calls["load_cache"].append(load_cache)
        return {
            "ready": False,
            "sessions": 0,
            "symbols": 0,
            "cache_exists": True,
            "csv_files": 200,
        }

    monkeypatch.setattr(bhavcopy_runtime, "status", status)
    worker = MO.MarketOperationsWorker(OperationStore(tmp_path / "jobs.db"))
    queued = worker._bootstrap()
    assert True not in calls["load_cache"], "bootstrap must not block on full pkl load"
    # History present via cache/csv → no DATA_PREPARE; other product inputs still due.
    assert MO.DATA_PREPARE not in queued
    assert MO.MARKET_SCAN in queued


def test_dead_lock_can_be_reclaimed(tmp_path: Path):
    from operations.market_ops import SingleWorkerLock

    lock_path = tmp_path / "worker.lock"
    lock_path.write_text("999999", encoding="utf-8")  # almost certainly dead
    lock = SingleWorkerLock(lock_path)
    assert lock.reclaim_if_dead() is True
    assert lock.acquire() is True
    lock.release()


def test_reclaim_stale_ops_lock_terminates_stale_holder(tmp_path: Path, monkeypatch):
    import operations.market_ops as market_ops
    import terminal_api as api

    ops_root = tmp_path / "market_ops"
    ops_root.mkdir(parents=True, exist_ok=True)
    lock_path = ops_root / "worker.lock"
    runtime_path = ops_root / "runtime.json"
    lock_path.write_text("999991\n", encoding="utf-8")
    runtime_path.write_text(
        json.dumps(
            {
                "worker_id": "stale",
                "worker_pid": 999991,
                "status": "LIVE",
                "heartbeat_epoch": time.time() - 999.0,
                "process_running": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(market_ops, "LOCK_PATH", lock_path)
    monkeypatch.setattr(market_ops, "RUNTIME_PATH", runtime_path)
    monkeypatch.setattr(api, "OPS_RUNTIME", runtime_path)

    calls = {"n": 0}

    def fake_terminate(self, *, reason: str = ""):
        calls["n"] += 1
        assert reason == "stale_heartbeat"
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass
        return True

    monkeypatch.setattr(market_ops.SingleWorkerLock, "terminate_holder", fake_terminate)
    monkeypatch.setattr(market_ops.SingleWorkerLock, "reclaim_if_dead", lambda self: False)
    monkeypatch.setattr(
        market_ops.SingleWorkerLock,
        "holder_pid",
        lambda self: 999991,
    )

    note = api._reclaim_stale_ops_lock(max_heartbeat_age_s=20.0)
    assert note.startswith("terminated_stale_holder:")
    assert calls["n"] == 1


def test_ensure_ops_worker_force_bypasses_throttle(tmp_path: Path, monkeypatch):
    import terminal_api as api

    monkeypatch.setattr(api, "OPS_RUNTIME", tmp_path / "runtime.json")
    monkeypatch.setattr(api, "_ops_process", None)
    api._ops_ensure_last_attempt = time.time()
    pops: list[list[str]] = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            pops.append(list(args))
            self.pid = 4242

        def poll(self):
            return 1  # exited immediately so ensure does not wait forever

    monkeypatch.setattr(api.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(api, "_reclaim_stale_ops_lock", lambda **kwargs: "lock_held_or_clear")
    monkeypatch.setattr(
        api,
        "_ops_runtime_payload",
        lambda: {"running": False, "process_running": False, "active": {}},
    )

    blocked = api._ensure_ops_worker(wait_s=0.01, force=False)
    assert blocked.get("ensure_ok") is False
    assert pops == []

    forced = api._ensure_ops_worker(wait_s=0.01, force=True)
    assert forced.get("ensure_attempted") is True
    assert len(pops) >= 1
    assert pops[0][-1] == "operations.market_ops"


def test_bhavcopy_status_fast_never_loads_full_pickle(monkeypatch):
    import terminal_api as api

    api._bhav_status_cache = {"ts": 0.0, "payload": None}
    calls = {"load_cache": []}

    def status(*, load_cache: bool = False):
        calls["load_cache"].append(load_cache)
        return {
            "ready": False,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_files": 200,
            "cache_exists": True,
            "minimum_sessions": 60,
        }

    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        status,
    )
    # Also patch the import path used inside the helper.
    import data.bhavcopy_runtime as bhavcopy_runtime

    monkeypatch.setattr(bhavcopy_runtime, "status", status)

    out = api._bhavcopy_status_fast()
    assert True not in calls["load_cache"]
    assert out["disk_ready"] is True
    assert "market-ops" in str(out.get("message") or "").lower() or out["cache_exists"] is True


def test_data_payload_uses_fast_bhav_status(monkeypatch):
    import terminal_api as api

    api._bhav_status_cache = {"ts": 0.0, "payload": None}
    monkeypatch.setattr(
        api,
        "_bhavcopy_status_fast",
        lambda: {
            "ready": False,
            "disk_ready": True,
            "cache_exists": True,
            "csv_files": 200,
            "sessions": 0,
            "minimum_sessions": 60,
            "message": "fast",
        },
    )
    monkeypatch.setattr(
        api,
        "_snapshot_payload",
        lambda: {"ready": False, "path": "", "as_of": "", "equity_count": 0},
    )
    out = api._data_payload(
        scan={"available": False, "records": []},
        long_term={"available": False, "records": []},
        operations={"running": True},
        fno={"available": True},
        news={"available": True},
    )
    assert out["bhavcopy"]["message"] == "fast"
    assert out["ready"] is True
    assert any("not loaded in the API process" in b for b in out["blockers"])
