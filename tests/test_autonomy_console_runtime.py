from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import threading
import time

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.console_runtime import _format_active_text, run_visible_loop
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


def test_heartbeat_lists_scan_and_news_together():
    text = _format_active_text(
        {
            "n": {"job_type": SCH.NEWS_REFRESH, "attempt": 3, "started_monotonic": 100.0},
            "s": {"job_type": SCH.MARKET_SCAN, "attempt": 0, "started_monotonic": 100.0},
        },
        now=1040.0,
    )
    assert "market_scan (940s, attempt 0)" in text
    assert "news_refresh (940s, attempt 3)" in text
    assert " + " in text


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
    (root / "runtime.json").write_text(json.dumps({
        "process_running": True,
        "heartbeat_ist": fresh,
        "scheduler_owner_pid": 1234,
        "active_job": {"job_type": "data_refresh"},
    }), encoding="utf-8")

    status = H.read_status(state_path=root / "status.json")
    assert status["supervisor_running"] is True
    assert status["heartbeat_ist"] == fresh
    assert status["scheduler_owner_pid"] == 1234
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
