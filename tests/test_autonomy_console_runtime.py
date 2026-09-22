from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy.console_runtime import run_visible_loop
from research.autonomy.supervisor import Supervisor


class _Deps:
    def now_ist(self):
        return datetime(2026, 7, 31, 19, 30)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return None


class _ExplodingSupervisor(Supervisor):
    def __init__(self, root):
        super().__init__(root, deps=_Deps())
        self.calls = 0

    def tick(self, now_ist=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("one bad tick")
        self.stop()
        self.heartbeat()
        return None


class _SingleJobSupervisor(Supervisor):
    """Execute exactly the pre-enqueued job without involving the market schedule."""

    def tick(self, now_ist=None):
        job = self.jobs.lease_due(self.owner)
        if job is not None:
            self._execute(job)
        self.stop()
        self.heartbeat()
        return job


class _BlockingSupervisor(Supervisor):
    """Hold one tick open long enough to prove runtime liveness is independent of tick completion."""

    def __init__(self, root):
        super().__init__(root, deps=_Deps())
        self.entered = threading.Event()
        self.release = threading.Event()

    def tick(self, now_ist=None):
        self.entered.set()
        # The test owns release. Keep a generous fail-safe so a slow CI runner cannot
        # finish the synthetic tick and mark runtime offline before liveness is asserted.
        self.release.wait(timeout=30.0)
        self.stop()
        return None


class _RepeatedPollingSupervisor(Supervisor):
    """Poll the same background job repeatedly with elapsed-time-only summary churn."""

    def __init__(self, root, *, deps=None):
        super().__init__(root, deps=deps or _Deps())
        self.calls = 0

    def tick(self, now_ist=None):
        job = self.jobs.lease_due(self.owner)
        if job is not None:
            self._execute(job)
        self.calls += 1
        if self.calls >= 3:
            self.stop()
        self.heartbeat()
        return job

    def _execute(self, job):
        elapsed = 45 + (self.calls * 15)
        self.jobs.reschedule_poll(
            job.job_id,
            when=self.jobs.clock(),
            error_code="DATA_REFRESH_IN_PROGRESS",
            error_message="snapshot refresh is still running; supervisor remains available",
            result_summary=f"data refresh running in background · historical_sync · {elapsed}s",
        )


class _PollingJobSupervisor(Supervisor):
    """Lease one background-poll row and keep it pending without inventing a new attempt."""

    def tick(self, now_ist=None):
        job = self.jobs.lease_due(self.owner)
        if job is not None:
            self._execute(job)
        self.stop()
        self.heartbeat()
        return job

    def _execute(self, job):
        self.jobs.reschedule_poll(
            job.job_id,
            when=self.jobs.clock(),
            error_code="DATA_REFRESH_IN_PROGRESS",
            error_message="snapshot refresh is still running; supervisor remains available",
            result_summary="data refresh running in background · historical_sync · 45s",
        )


def test_visible_loop_recovers_from_tick_exception(tmp_path, capsys):
    sup = _ExplodingSupervisor(tmp_path / "auto")
    assert sup.start()
    try:
        run_visible_loop(
            sup,
            interval_s=0,
            max_iterations=2,
            sleep_fn=lambda _seconds: None,
            heartbeat_s=0,
        )
        assert sup.calls == 2
        output = capsys.readouterr().out
        assert "LOOP ERROR" in output
        assert "one bad tick" in output
        assert "HEARTBEAT" in output
    finally:
        sup.shutdown()


def test_visible_loop_reports_completed_job(tmp_path, capsys):
    sup = _SingleJobSupervisor(tmp_path / "auto", deps=_Deps())
    assert sup.start()
    job = sup.jobs.enqueue("UNKNOWN_JOB_FOR_CONSOLE_TEST", idempotency_key="console-test")
    try:
        run_visible_loop(
            sup,
            interval_s=0,
            max_iterations=1,
            sleep_fn=lambda _seconds: None,
            heartbeat_s=0,
        )
        final = sup.jobs.get(job.job_id)
        assert final is not None and final.status == JS.PERMANENT_FAILED
        output = capsys.readouterr().out
        assert "UNKNOWN_JOB_FOR_CONSOLE_TEST" in output
        assert "PERMANENT_FAILED" in output
    finally:
        sup.shutdown()


def test_background_poll_is_reported_as_progress_not_restarted_job(tmp_path, capsys):
    sup = _PollingJobSupervisor(tmp_path / "auto", deps=_Deps())
    assert sup.start()
    queued = sup.jobs.enqueue("data_refresh", idempotency_key="console-poll-test")
    seeded = sup.jobs.lease_due(sup.owner)
    assert seeded is not None and seeded.job_id == queued.job_id
    sup.jobs.reschedule_poll(
        seeded.job_id,
        when=sup.jobs.clock(),
        error_code="DATA_REFRESH_IN_PROGRESS",
        error_message="snapshot refresh is still running; supervisor remains available",
        result_summary="data refresh running in background · historical_sync · 30s",
    )
    try:
        run_visible_loop(
            sup,
            interval_s=0,
            max_iterations=1,
            sleep_fn=lambda _seconds: None,
            heartbeat_s=0,
        )
        output = capsys.readouterr().out
        assert "JOB POLL" in output
        assert "JOB PROGRESS" in output
        assert "historical_sync" in output
        assert "JOB START" not in output
        assert "JOB DONE" not in output
    finally:
        sup.shutdown()


def test_fresh_runtime_heartbeat_overrides_stale_durable_snapshot(tmp_path):
    root = tmp_path / "auto"
    root.mkdir()
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()
    (root / "status.json").write_text(json.dumps({
        "state": "DATA_REFRESHING",
        "process_running": True,
        "heartbeat_ist": stale,
    }), encoding="utf-8")
    live_pid = os.getpid()
    (root / "runtime.json").write_text(json.dumps({
        "process_running": True,
        "heartbeat_ist": fresh,
        "scheduler_owner_pid": live_pid,
        "active_job": {"job_type": "data_refresh"},
    }), encoding="utf-8")

    status = H.read_status(state_path=root / "status.json")
    assert status["supervisor_running"] is True
    assert status["heartbeat_ist"] == fresh
    assert status["scheduler_owner_pid"] == live_pid
    assert status["active_job"]["job_type"] == "data_refresh"


def test_runtime_heartbeat_advances_while_tick_is_blocked(tmp_path):
    root = tmp_path / "auto"
    sup = _BlockingSupervisor(root)
    assert sup.start()
    runner = threading.Thread(
        target=run_visible_loop,
        kwargs={"supervisor": sup, "interval_s": 0, "heartbeat_s": 0.2},
        daemon=True,
    )
    try:
        runner.start()
        assert sup.entered.wait(timeout=2.0)
        runtime_path = root / "runtime.json"
        first = json.loads(runtime_path.read_text(encoding="utf-8"))["heartbeat_ist"]
        time.sleep(1.25)  # runtime worker minimum interval is one second
        second_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        assert second_payload["heartbeat_ist"] != first
        assert second_payload["process_running"] is True
        status = H.read_status(state_path=root / "status.json")
        assert status["supervisor_running"] is True
    finally:
        sup.release.set()
        runner.join(timeout=3.0)
        sup.shutdown()


def test_repeated_background_poll_suppresses_elapsed_only_console_churn(tmp_path, capsys):
    sup = _RepeatedPollingSupervisor(tmp_path / "auto", deps=_Deps())
    assert sup.start()
    queued = sup.jobs.enqueue("data_refresh", idempotency_key="console-dedupe-test")
    seeded = sup.jobs.lease_due(sup.owner)
    assert seeded is not None and seeded.job_id == queued.job_id
    sup.jobs.reschedule_poll(
        seeded.job_id,
        when=sup.jobs.clock(),
        error_code="DATA_REFRESH_IN_PROGRESS",
        error_message="snapshot refresh is still running; supervisor remains available",
        result_summary="data refresh running in background · historical_sync · 30s",
    )
    try:
        run_visible_loop(
            sup,
            interval_s=0,
            max_iterations=3,
            sleep_fn=lambda _seconds: None,
            heartbeat_s=0,
        )
        output = capsys.readouterr().out
        assert output.count("JOB POLL") == 1
        assert output.count("JOB PROGRESS") == 1
        # The initial heartbeat plus a genuine activity/state transition may both be
        # visible; elapsed-only polling must not emit one heartbeat per iteration.
        assert output.count("HEARTBEAT") < 3
        # Liveness remains a file-level pulse even though console lines collapse.
        runtime = json.loads((tmp_path / "auto" / "runtime.json").read_text(encoding="utf-8"))
        assert runtime["process_running"] is False
    finally:
        sup.shutdown()
