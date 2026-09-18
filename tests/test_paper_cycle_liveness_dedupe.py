from __future__ import annotations

import time

from product import historical_replay as HR
from research.auto_research.scheduler import AutoResearchBrain
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
