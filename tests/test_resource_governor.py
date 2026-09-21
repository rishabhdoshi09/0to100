from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import resource_governor as RG


def _job(job_type, *, status=JS.PENDING, scheduled_for=0.0, key="", critical=False):
    return SimpleNamespace(
        job_id=f"{job_type}-{key or 'x'}",
        job_type=job_type,
        status=status,
        scheduled_for=scheduled_for,
        idempotency_key=key,
        critical=critical,
    )


def test_due_critical_data_preempts_historical_replay():
    out = RG.assess(
        [_job(SCH.DATA_REFRESH, status=JS.PENDING, scheduled_for=99, key="data_refresh:2026-09-21", critical=True)],
        now_epoch=100,
    )
    assert out["historical_replay_allowed"] is False
    assert out["decision"] == "DEFER_HISTORICAL_REPLAY"
    assert out["blocking_jobs"][0]["priority_lane"] == "CRITICAL_DATA"
    assert out["learning_allowed"] is True
    assert out["research_allowed"] is True


def test_running_data_refresh_preempts_even_when_scheduled_in_past():
    out = RG.assess(
        [_job(SCH.DATA_REFRESH, status=JS.RUNNING, scheduled_for=1)],
        now_epoch=100,
    )
    assert out["historical_replay_allowed"] is False
    assert "DATA_REFRESH".lower() in out["reason"].lower()


def test_future_or_blocked_data_work_does_not_starve_history():
    jobs = [
        _job(SCH.DATA_REFRESH, status=JS.PENDING, scheduled_for=200),
        _job(SCH.DATA_REFRESH, status=JS.BLOCKED, scheduled_for=1),
    ]
    out = RG.assess(jobs, now_epoch=100)
    assert out["historical_replay_allowed"] is True
    assert out["blocking_jobs"] == []


def test_current_scan_forward_paper_and_settlement_rank_ahead_of_history():
    jobs = [
        _job(SCH.OUTCOME_RESOLUTION, status=JS.PENDING, scheduled_for=1, key="forward_outcome:2026-09-21"),
        _job(SCH.PAPER_CYCLE, status=JS.PENDING, scheduled_for=1, key="snapshot_paper:s1"),
        _job(SCH.MARKET_SCAN, status=JS.RUNNING, scheduled_for=1, key="startup_discovery_scan:x:s1"),
    ]
    out = RG.assess(jobs, now_epoch=100)
    assert out["historical_replay_allowed"] is False
    assert [r["priority_lane"] for r in out["blocking_jobs"]] == [
        "CURRENT_SCAN",
        "FORWARD_PAPER",
        "SETTLEMENT",
    ]


def test_historical_poll_does_not_preempt_itself():
    jobs = [
        _job(SCH.HISTORICAL_PAPER_CYCLE, status=JS.PENDING, scheduled_for=1, key="hist_paper:b1"),
        _job(SCH.LEARNING_CYCLE, status=JS.PENDING, scheduled_for=1, key="hist_learning:b1"),
        _job(SCH.RESEARCH_CYCLE, status=JS.PENDING, scheduled_for=1, key="hist_research:b1"),
    ]
    out = RG.assess(jobs, now_epoch=100)
    assert out["historical_replay_allowed"] is True


def test_priority_order_is_explicit_and_live_authority_unchanged():
    out = RG.assess([], now_epoch=100)
    assert out["priority_order"] == [
        "CRITICAL_DATA",
        "CURRENT_SCAN",
        "FORWARD_PAPER",
        "SETTLEMENT",
        "HISTORICAL_REPLAY",
        "RESEARCH",
    ]
    assert out["live_money_unchanged"] is True
