"""A live worker must never be reported dead because SQLite was busy.

Observed on the real Mac: market_ops kept pid 77437, process_running stayed
true, starts stayed 1, the worker kept doing real scan/fundamentals work -- and
the supervisor still flipped it to healthy=False because the heartbeat had aged
to 19.3s. _heartbeat_loop ran recover_dead_running/recover_stale_running BEFORE
writing the heartbeat, so ordinary database contention aged liveness.

The contract is now three separate facts:

  liveness   -- the process is there and beating (restart decisions use this)
  operational-- the operation store is doing its upkeep (reported, never fatal)
  failure    -- neither of the above, for long enough

These tests use a fake clock and a fake store; nothing sleeps for 15 seconds.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

import operations.market_ops as MO
import product.host_supervisor as HS


# ── helpers ────────────────────────────────────────────────────────────────
class _BlockingStore:
    """An operation store whose maintenance calls block until released."""

    def __init__(self, hold: threading.Event):
        self.hold = hold
        self.calls: list[str] = []

    def recover_dead_running(self, keep_pid=None):
        self.calls.append("recover_dead_running")
        self.hold.wait(30)

    def recover_stale_running(self):
        self.calls.append("recover_stale_running")

    def recover_orphans(self):
        return 0


class _FailingStore(_BlockingStore):
    def recover_dead_running(self, keep_pid=None):
        self.calls.append("recover_dead_running")
        raise RuntimeError("database is locked")

    def recover_stale_running(self):
        self.calls.append("recover_stale_running")
        raise RuntimeError("database is locked")


def _worker(tmp_path: Path, store) -> MO.MarketOperationsWorker:
    worker = MO.MarketOperationsWorker.__new__(MO.MarketOperationsWorker)
    worker.store = store
    worker.stop_event = threading.Event()
    worker._active_lock = threading.Lock()
    worker._active = {}
    worker._threads = []
    worker._last_rss_sample = time.time()
    worker._last_pipeline_snapshot = time.time()
    worker._maint_lock = threading.Lock()
    worker._maint_last_ok = time.time()
    worker._maint_last_error = ""
    worker._maint_in_flight = False
    worker._maint_started_at = 0.0
    return worker


def _runtime(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "market_ops" / "runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(MO, "RUNTIME_PATH", path)
    monkeypatch.setattr(HS, "logs_path", lambda *parts: tmp_path.joinpath(*parts))
    return path


def _publish(path: Path, **overrides):
    payload = {
        "process_running": True,
        "worker_pid": os.getpid(),
        "heartbeat_epoch": time.time(),
        "maintenance_ok": True,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


# ── A: blocking maintenance must not starve liveness ───────────────────────
def test_A_blocking_store_maintenance_does_not_starve_the_heartbeat(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(MO, "HEARTBEAT_EVERY_S", 0.01)
    monkeypatch.setattr(MO, "MAINTENANCE_EVERY_S", 0.01)
    hold = threading.Event()
    store = _BlockingStore(hold)
    worker = _worker(tmp_path, store)

    beats = threading.Thread(target=worker._heartbeat_loop, daemon=True)
    maint = threading.Thread(target=worker._maintenance_loop, daemon=True)
    beats.start()
    maint.start()
    try:
        # Maintenance is wedged in the store for the whole window.
        deadline = time.time() + 1.0
        ages = []
        while time.time() < deadline:
            time.sleep(0.05)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                ages.append(time.time() - float(payload["heartbeat_epoch"]))
        assert "recover_dead_running" in store.calls, "maintenance must actually be blocked"
        assert ages, "the heartbeat must keep publishing"
        assert max(ages) < 0.5, f"heartbeat starved while the store blocked: {max(ages)}"
    finally:
        hold.set()
        worker.stop_event.set()
        beats.join(2)
        maint.join(2)


def test_A_heartbeat_loop_never_touches_the_operation_store(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(MO, "HEARTBEAT_EVERY_S", 0.01)
    store = _BlockingStore(threading.Event())
    worker = _worker(tmp_path, store)

    beats = threading.Thread(target=worker._heartbeat_loop, daemon=True)
    beats.start()
    try:
        time.sleep(0.2)
        assert path.exists()
        assert store.calls == [], f"heartbeat path touched the store: {store.calls}"
    finally:
        worker.stop_event.set()
        beats.join(2)


# ── B: a busy worker stays healthy ─────────────────────────────────────────
def test_B_worker_running_long_work_stays_healthy(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    # A scan in flight and a maintenance pass that has been running 40s: slow,
    # not broken. This is the real-Mac shape.
    _publish(
        path,
        active={"data": {"operation_id": "abc", "kind": "MARKET_SCAN"}},
        maintenance_ok=True,
        maintenance_in_flight=True,
        maintenance_in_flight_for_s=40.0,
    )
    assert HS._market_ops_health() is True
    assert HS._market_ops_liveness() is True


def test_B_in_flight_maintenance_is_never_reported_as_failure(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    worker = _worker(tmp_path, _BlockingStore(threading.Event()))
    with worker._maint_lock:
        worker._maint_in_flight = True
        worker._maint_started_at = time.time() - 40
    snapshot = worker._maintenance_snapshot()
    assert snapshot["maintenance_ok"] is True
    assert snapshot["maintenance_in_flight"] is True
    assert snapshot["maintenance_in_flight_for_s"] >= 40


def test_B_a_single_store_error_does_not_flip_operational_health(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    worker = _worker(tmp_path, _FailingStore(threading.Event()))
    worker._run_maintenance_pass()
    snapshot = worker._maintenance_snapshot()
    # The label and exception type are recorded; the raw driver message is not,
    # so a database error cannot leak store internals into runtime.json.
    assert snapshot["maintenance_last_error"].startswith("recover_dead_running: RuntimeError")
    assert "database is locked" not in snapshot["maintenance_last_error"]
    assert snapshot["maintenance_ok"] is True, "one busy timeout is not a degraded store"


def test_B_sustained_store_failure_does_degrade_operational_health(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    worker = _worker(tmp_path, _FailingStore(threading.Event()))
    worker._maint_last_ok = time.time() - (MO.MAINTENANCE_DEGRADED_AFTER_S + 5)
    worker._run_maintenance_pass()
    assert worker._maintenance_snapshot()["maintenance_ok"] is False


# ── C + D: real failure is still failure ───────────────────────────────────
def test_C_genuinely_stale_heartbeat_is_unhealthy(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    _publish(path, heartbeat_epoch=time.time() - (HS.MARKET_OPS_HEARTBEAT_MAX_AGE_S + 1))
    assert HS._market_ops_health() is False
    assert HS._market_ops_liveness() is False


def test_C_threshold_was_not_quietly_raised():
    assert HS.MARKET_OPS_HEARTBEAT_MAX_AGE_S == 15.0


def test_D_dead_pid_fails_health(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    _publish(path, worker_pid=0)
    assert HS._market_ops_health() is False
    assert HS._market_ops_liveness() is False

    _publish(path)
    monkeypatch.setattr(HS.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    assert HS._market_ops_health() is False
    assert HS._market_ops_liveness() is False


def test_D_process_running_false_fails_health(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    _publish(path, process_running=False)
    assert HS._market_ops_health() is False


def test_D_missing_runtime_file_fails_closed(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    assert HS._market_ops_health() is False
    assert HS._market_ops_liveness() is False


# ── E + F: restart policy, and only when it should ─────────────────────────
class _FakeChild:
    def __init__(self, spec):
        self.spec = spec
        self.alive = True
        self.last_health_ok = True
        self.health_failures = 0
        self.health_bad_since = None
        self.last_start = 0.0
        self.restarts = 0
        self.terminated = 0

    def terminate(self):
        self.terminated += 1

    def close_log(self):
        pass


def _supervisor_with(spec):
    sup = HS.HostSupervisor.__new__(HS.HostSupervisor)
    child = _FakeChild(spec)
    sup.children = {spec.name: child}
    started: list[str] = []
    sup.start_child = lambda name: started.append(name)
    return sup, child, started


def _market_ops_spec():
    return next(s for s in HS.child_specs() if s.name == "market_ops")


def test_E_prolonged_real_failure_still_restarts(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    _publish(path, heartbeat_epoch=time.time() - 600)      # genuinely stuck
    spec = _market_ops_spec()
    sup, child, started = _supervisor_with(spec)
    child.health_bad_since = time.time() - (HS.HEALTH_FAILURE_WINDOW_S + 5)

    sup._supervise_child(child)

    assert child.last_health_ok is False
    assert child.terminated == 1
    assert started == ["market_ops"]


def test_F_degraded_store_on_a_live_worker_never_restarts(tmp_path, monkeypatch):
    path = _runtime(tmp_path, monkeypatch)
    # The exact defect shape: alive, beating, but the store is unhappy.
    _publish(path, maintenance_ok=False, maintenance_last_error="recover_dead_running: OperationalError")
    spec = _market_ops_spec()
    sup, child, started = _supervisor_with(spec)
    child.health_bad_since = time.time() - (HS.HEALTH_FAILURE_WINDOW_S + 5)

    sup._supervise_child(child)

    # Reported truthfully as unhealthy...
    assert child.last_health_ok is False
    # ...but never killed, and never duplicated.
    assert child.terminated == 0
    assert started == []
    assert child.restarts == 0


def test_F_market_ops_declares_a_liveness_probe_distinct_from_health():
    spec = _market_ops_spec()
    assert spec.health is HS._market_ops_health
    assert spec.liveness is HS._market_ops_liveness
    assert spec.health is not spec.liveness


def test_F_worker_start_runs_heartbeat_and_maintenance_on_separate_threads():
    import inspect
    src = inspect.getsource(MO.MarketOperationsWorker.run)
    assert "self._heartbeat_loop" in src
    assert "self._maintenance_loop" in src
    # Check the CODE, not the docstring, which names the old behaviour on purpose.
    heartbeat_src = inspect.getsource(MO.MarketOperationsWorker._heartbeat_loop)
    body = heartbeat_src.split('"""', 2)[-1]
    for banned in ("self.store", "recover_dead_running", "recover_stale_running",
                   "refresh_desk_pipeline_snapshot"):
        assert banned not in body, f"liveness path must not touch {banned}"


def test_F_single_worker_lock_still_guards_duplicate_workers():
    import inspect
    src = inspect.getsource(MO.MarketOperationsWorker.run)
    assert "self.lock" in src
