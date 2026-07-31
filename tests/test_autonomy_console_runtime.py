from __future__ import annotations

from datetime import datetime

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
    sup = Supervisor(tmp_path / "auto", deps=_Deps())
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
