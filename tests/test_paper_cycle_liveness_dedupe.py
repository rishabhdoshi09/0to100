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
