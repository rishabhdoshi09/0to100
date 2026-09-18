from __future__ import annotations

import time
from datetime import datetime

from product import historical_replay as HR
from research.auto_research.scheduler import AutoResearchBrain
from research.autonomy import jobs as JOBS
from research.autonomy import job_store as JS
from research.autonomy.console_runtime import run_visible_loop
from research.autonomy.supervisor import Supervisor
from research.intelligence.runtime.cycle_context import CycleContext


class _Spec:
    strategy_id = "vcp-1"
    version = 3

    def config_hash(self):
        return "rules-abc"


def test_insample_cache_survives_restart_and_is_rules_versioned(tmp_path):
    path = tmp_path / "insample.json"
    first = AutoResearchBrain(insample_cache_path=path)
    provider = type("P", (), {"snapshot_id": "snap-1"})()
    key = first._insample_cache_key(provider, _Spec())
    first._insample_cache[key] = (0.42, 137)
    first._save_insample_cache()

    second = AutoResearchBrain(insample_cache_path=path)
    assert second._insample_cache[key] == (0.42, 137)

    changed = type(
        "Changed",
        (),
        {
            "strategy_id": "vcp-1",
            "version": 4,
            "config_hash": lambda self: "rules-def",
        },
    )()
    assert second._insample_cache_key(provider, changed) != key


def test_completed_intelligence_cycle_skips_historical_evidence_rebuild(tmp_path, monkeypatch):
    brain = AutoResearchBrain(
        event_store_path=tmp_path / "events.jsonl",
        runtime_state_path=tmp_path / "runtime.json",
        intel_book_path=tmp_path / "book.json",
        insample_cache_path=tmp_path / "cache.json",
    )
    ctx = CycleContext(
        as_of_date="2026-09-18",
        cycle_type="paper_session",
        mode="PAPER_AUTO",
        data_ok=True,
        data_snapshot_id="snap-1",
        market_regime="RISK_ON",
        config_hash="cfg",
        registry_version="reg",
        strategies=[_Spec()],
        data={},
        clusters={},
    )
    brain.runtime_state.mark_cycle_done(ctx.cycle_id())
    monkeypatch.setattr(brain, "_build_intel_ctx", lambda day: (ctx, object()))

    def should_not_run(*args, **kwargs):
        raise AssertionError("completed cycle must not rebuild in-sample evidence")

    monkeypatch.setattr(brain, "_insample_evidence", should_not_run)
    out = brain.run_intelligence_cycle_day(date="2026-09-18")
    assert out["status"] == "ALREADY_DONE"
    assert out["eligibility"] == "NO_ELIGIBLE_TRADE"


def test_async_replay_does_not_start_when_inputs_are_unchanged(monkeypatch):
    monkeypatch.setattr(
        HR,
        "load_latest",
        lambda directory=None: {
            "status": "SUCCEEDED",
            "run_id": "same",
            "live_locked": True,
        },
    )
    monkeypatch.setattr(HR, "replay_is_current", lambda **kwargs: True)

    class _ForbiddenThread:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unchanged replay must not start a thread")

    monkeypatch.setattr(HR.threading, "Thread", _ForbiddenThread)
    out = HR.start_replay_async(sessions=8, universe_limit=40)
    assert out["skipped"] is True
    assert out["reason"] == "replay_inputs_unchanged"
    assert out["cache_hit"] is True


def test_forced_replay_can_still_run_even_when_cache_is_current(monkeypatch):
    monkeypatch.setattr(HR, "load_latest", lambda directory=None: {"status": "SUCCEEDED"})
    monkeypatch.setattr(HR, "replay_is_current", lambda **kwargs: True)
    calls = []

    class _Thread:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
        def start(self):
            calls.append("started")

    monkeypatch.setattr(HR.threading, "Thread", _Thread)
    out = HR.start_replay_async(force=True, sessions=8, universe_limit=40)
    assert out["status"] == "RUNNING"
    assert "started" in calls


class _Deps:
    def now_ist(self):
        return datetime(2026, 9, 18, 12, 30)
    def holidays(self):
        return set()
    def active_snapshot_id(self):
        return None


class _LeaseSupervisor(Supervisor):
    def tick(self, now_ist=None):
        job = self.jobs.lease_due(self.owner, lease_seconds=0.4)
        if job is not None:
            self._execute(job)
        self.stop()
        return job


def test_runtime_heartbeat_renews_long_job_lease(tmp_path):
    sup = _LeaseSupervisor(tmp_path / "auto", deps=_Deps())
    assert sup.start()
    job_type = "LEASE_RENEWAL_TEST"
    original = JOBS.HANDLERS.get(job_type)
    renewals = []
    real_renew = sup.jobs.renew_lease

    def tracked_renew(job_id, owner, *, lease_seconds=300.0):
        renewals.append((job_id, owner, lease_seconds))
        return real_renew(job_id, owner, lease_seconds=lease_seconds)

    sup.jobs.renew_lease = tracked_renew
    JOBS.HANDLERS[job_type] = lambda ctx: (
        time.sleep(1.25)
        or JOBS.JobResult(JS.SUCCEEDED, "long work complete")
    )
    job = sup.jobs.enqueue(job_type, idempotency_key="lease-renewal-test")
    try:
        run_visible_loop(
            sup,
            interval_s=0,
            max_iterations=1,
            sleep_fn=lambda _seconds: None,
            heartbeat_s=0.2,
        )
        final = sup.jobs.get(job.job_id)
        assert final is not None
        assert final.status == JS.SUCCEEDED
        assert renewals
        assert all(item[0] == job.job_id for item in renewals)
    finally:
        if original is None:
            JOBS.HANDLERS.pop(job_type, None)
        else:
            JOBS.HANDLERS[job_type] = original
        sup.shutdown()


def test_autonomous_replay_cache_hit_is_not_recorded_as_new_cycle(tmp_path, monkeypatch):
    from product import autonomous_learning as AL

    control_path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(control_path))
    AL.save_control({
        "enabled": True,
        "mode": AL.MODE_HISTORICAL_REPLAY,
        "last_cycle_at": "2026-09-17T20:00:00+00:00",
        "last_replay_at": "2026-09-17T20:00:00+00:00",
    })
    monkeypatch.setattr(
        HR,
        "load_latest",
        lambda: {"status": "SUCCEEDED"},
    )
    monkeypatch.setattr(
        HR,
        "start_replay_async",
        lambda **kwargs: {
            "status": "SUCCEEDED",
            "skipped": True,
            "reason": "replay_inputs_unchanged",
            "cache_hit": True,
        },
    )
    out = AL.maybe_run_closed_market_replay(force=False)
    after = AL.load_control()
    assert out["skipped"] is True
    assert out["next_action"] == "WAIT_FOR_NEW_EVIDENCE"
    assert after["last_cycle_at"] == "2026-09-17T20:00:00+00:00"
    assert after["last_replay_at"] == "2026-09-17T20:00:00+00:00"


def test_automatic_historical_replay_is_deferred_while_market_is_open(tmp_path, monkeypatch):
    from product import autonomous_learning as AL
    from zoneinfo import ZoneInfo

    control_path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(control_path))
    AL.save_control({"enabled": True, "mode": AL.MODE_HISTORICAL_REPLAY})

    called = []
    monkeypatch.setattr(
        HR,
        "start_replay_async",
        lambda **kwargs: called.append(kwargs) or {"status": "RUNNING"},
    )
    market_open = datetime(2026, 9, 18, 12, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    out = AL.maybe_run_closed_market_replay(now=market_open, force=False)
    assert out["skipped"] is True
    assert out["reason"] == "market_open_replay_deferred"
    assert out["next_action"] == "WAIT_FOR_MARKET_CLOSE"
    assert called == []


def test_snapshot_pipeline_runs_once_then_stops(tmp_path):
    from tests.test_autonomy import FakeDeps

    now = datetime(2026, 7, 31, 10, 0)
    root = tmp_path / "auto"
    sup = Supervisor(root, deps=FakeDeps(now=now, data_ok=True))
    assert sup.start() is True
    try:
        for _ in range(12):
            sup.tick(now)

        rows = sup.jobs.list(limit=200)
        by_type = {}
        for row in rows:
            by_type.setdefault(row.job_type, []).append(row)

        assert len(by_type.get(JOBS.SCH.AUTH_HEALTH, [])) == 1
        assert len(by_type.get(JOBS.SCH.DATA_REFRESH, [])) == 1
        assert len(by_type.get(JOBS.SCH.MARKET_SCAN, [])) == 1
        assert len(by_type.get(JOBS.SCH.PAPER_CYCLE, [])) == 1
        assert len(by_type.get(JOBS.SCH.NEWS_REFRESH, [])) == 0

        assert by_type[JOBS.SCH.DATA_REFRESH][0].status == JS.SUCCEEDED
        assert by_type[JOBS.SCH.MARKET_SCAN][0].status == JS.SUCCEEDED
        assert by_type[JOBS.SCH.PAPER_CYCLE][0].status == JS.SUCCEEDED
        assert sup.owner_state["completed_snapshot_id"] == "snap1"

        before = [(j.job_id, j.job_type, j.status) for j in sup.jobs.list(limit=200)]
        for _ in range(8):
            assert sup.tick(now) is None
        after = [(j.job_id, j.job_type, j.status) for j in sup.jobs.list(limit=200)]
        assert after == before
    finally:
        sup.shutdown()


def test_completed_snapshot_does_not_restart_pipeline_after_supervisor_restart(tmp_path):
    from tests.test_autonomy import FakeDeps

    now = datetime(2026, 7, 31, 10, 0)
    root = tmp_path / "auto"
    first = Supervisor(root, deps=FakeDeps(now=now, data_ok=True))
    assert first.start() is True
    for _ in range(12):
        first.tick(now)
    first_rows = first.jobs.list(limit=200)
    first_paper_ids = [j.job_id for j in first_rows if j.job_type == JOBS.SCH.PAPER_CYCLE]
    assert len(first_paper_ids) == 1
    assert first.owner_state["completed_snapshot_id"] == "snap1"
    first.shutdown()

    second = Supervisor(root, deps=FakeDeps(now=now, data_ok=True))
    assert second.start() is True
    try:
        for _ in range(8):
            assert second.tick(now) is None
        rows = second.jobs.list(limit=200)
        paper_ids = [j.job_id for j in rows if j.job_type == JOBS.SCH.PAPER_CYCLE]
        assert paper_ids == first_paper_ids
        assert second.owner_state["completed_snapshot_id"] == "snap1"
    finally:
        second.shutdown()


def test_market_ops_scan_cannot_invoke_autonomous_paper_loop():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "operations" / "market_ops.py").read_text(
        encoding="utf-8"
    )
    assert "from product.autonomous_loop import advance_loop" not in source
    assert "autonomous loop · candidates=" not in source
    assert "MARKET_SCAN complete · handed off to autonomy supervisor" in source


def test_reclaimed_legacy_recurring_job_is_cancelled_before_it_can_run(tmp_path):
    from tests.test_autonomy import FakeDeps

    clock = [1000.0]
    now = datetime(2026, 7, 31, 10, 0)
    root = tmp_path / "auto"
    sup = Supervisor(
        root,
        deps=FakeDeps(now=now, data_ok=True),
        clock=lambda: clock[0],
    )
    assert sup.start() is True
    try:
        legacy = sup.jobs.enqueue(
            JOBS.SCH.PAPER_CYCLE,
            idempotency_key="paper_cycle:snap1:2026-07-31:intraday-1000",
            critical=True,
        )
        leased = sup.jobs.lease_due("dead-old-owner", lease_seconds=1.0)
        assert leased is not None and leased.job_id == legacy.job_id
        clock[0] += 10.0

        # tick() reclaims the dead lease, then retires the legacy recurring row
        # before lease_due() can execute it again.
        executed = sup.tick(now)
        final = sup.jobs.get(legacy.job_id)
        assert final is not None
        assert final.status == JS.CANCELLED
        assert executed is None or executed.job_id != legacy.job_id
    finally:
        sup.shutdown()


def test_parallel_runtime_cannot_prelaunch_scan_before_data():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "research" / "autonomy" / "parallel_runtime.py"
    ).read_text(encoding="utf-8")
    assert "Supervisor.enqueue_due = enqueue_due_parallel" not in source
    assert "def enqueue_due_parallel(" not in source
    assert "Launch the market scan before" not in source
    assert "DATA_REFRESH -> MARKET_SCAN -> PAPER_CYCLE -> OBSERVING" in source


def test_snapshot_bound_market_operation_never_reuses_other_snapshot(monkeypatch):
    from research.autonomy import parallel_runtime as PR

    created = []
    class _Store:
        def recent_full(self, limit=250):
            return [{
                "operation_id": "old-op",
                "kind": "MARKET_SCAN",
                "status": "SUCCEEDED",
                "payload": {"snapshot_id": "snap-old"},
            }]
        def enqueue(self, kind, **kwargs):
            created.append((kind, kwargs))
            return {
                "operation_id": "new-op",
                "kind": kind,
                "status": "PENDING",
                "payload": kwargs.get("payload") or {},
            }, True
        def latest(self, kind):
            raise AssertionError("snapshot-bound operation must not use latest(kind)")

    monkeypatch.setattr(PR, "_ops_store", lambda: _Store())
    monkeypatch.setattr(PR, "_ensure_ops_worker", lambda: None)
    op = PR._queue_operation(
        "MARKET_SCAN",
        requested_by="autonomy",
        identity="snap-new",
    )
    assert op["operation_id"] == "new-op"
    assert created[0][1]["payload"]["snapshot_id"] == "snap-new"
    assert created[0][1]["deduplicate"] is False


def test_snapshot_bound_market_operation_reuses_exact_identity(monkeypatch):
    from research.autonomy import parallel_runtime as PR

    class _Store:
        def recent_full(self, limit=250):
            return [{
                "operation_id": "same-op",
                "kind": "MARKET_SCAN",
                "status": "SUCCEEDED",
                "payload": {"snapshot_id": "snap-1"},
            }]
        def enqueue(self, *args, **kwargs):
            raise AssertionError("exact completed snapshot operation must be reused")

    monkeypatch.setattr(PR, "_ops_store", lambda: _Store())
    monkeypatch.setattr(PR, "_ensure_ops_worker", lambda: None)
    op = PR._queue_operation(
        "MARKET_SCAN",
        requested_by="autonomy",
        identity="snap-1",
    )
    assert op["operation_id"] == "same-op"


def test_live_ready_data_without_broker_snapshot_gets_stable_identity():
    from research.autonomy.jobs import _kite_live_ready_result

    class _D:
        def live_market_ready(self):
            return {
                "ready": True,
                "session_date": "2026-09-18",
                "source": "kite_quotes",
                "symbols": 200,
            }

    ctx = type("Ctx", (), {"deps": _D()})()
    result = _kite_live_ready_result(ctx, sid=None)
    assert result is not None
    assert result.status == JS.SUCCEEDED
    assert result.output_snapshot_id == "market:kite_quotes:2026-09-18"
    assert result.metadata["data_identity"] == result.output_snapshot_id
