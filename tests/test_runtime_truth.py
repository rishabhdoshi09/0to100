from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import runtime_truth as RT
from research.autonomy import schedules as SCH


def _job(job_type, *, status=JS.PENDING, scheduled_for=0.0, key=""):
    return SimpleNamespace(
        job_id=f"{job_type}:{key or 'x'}",
        job_type=job_type,
        status=status,
        scheduled_for=scheduled_for,
        started_at=0.0,
        created_at=0.0,
        idempotency_key=key,
        attempt=0,
        result_summary="",
        error_code="",
        error_message="",
    )


def test_future_pending_work_is_queue_not_current_activity():
    out = RT.derive_activity(
        [_job(SCH.DATA_REFRESH, scheduled_for=200.0, key="data:future")],
        now_epoch=100.0,
    )
    assert out["activity"] == RT.ACTIVITY_IDLE
    assert out["busy"] is False
    assert out["next_pending"]["job_type"] == SCH.DATA_REFRESH


def test_running_work_is_current_even_when_higher_priority_work_is_due():
    out = RT.derive_activity(
        [
            _job(SCH.RESEARCH_CYCLE, status=JS.RUNNING, scheduled_for=1.0, key="hist_research:b1"),
            _job(SCH.DATA_REFRESH, status=JS.PENDING, scheduled_for=1.0, key="data:due"),
        ],
        now_epoch=100.0,
    )
    assert out["activity"] == RT.ACTIVITY_RESEARCH
    assert out["primary_job"]["job_type"] == SCH.RESEARCH_CYCLE
    assert out["current_count"] == 2


def test_running_historical_replay_is_research_activity():
    out = RT.derive_activity(
        [_job(SCH.HISTORICAL_PAPER_CYCLE, status=JS.RUNNING, key="hist_paper:b1")],
        now_epoch=100.0,
    )
    assert out["activity"] == RT.ACTIVITY_HISTORY
    assert out["research_activity"] is True


def test_snapshot_decision_is_not_reported_as_forward_paper_or_research_learning():
    out = RT.derive_activity(
        [_job(SCH.PAPER_CYCLE, status=JS.RUNNING, key="snapshot_decision:s1:t1")],
        now_epoch=100.0,
    )
    assert out["activity"] == RT.ACTIVITY_DECISION
    assert out["activity"] != RT.ACTIVITY_PAPER
    assert out["research_activity"] is False


def test_stale_researching_state_is_detected_but_idle_paper_policy_is_not():
    idle = {"activity": RT.ACTIVITY_IDLE}
    stale = RT.transient_state_mismatch("RESEARCHING", idle)
    paper = RT.transient_state_mismatch("PAPER_ACTIVE", idle)

    assert stale["mismatch"] is True
    assert stale["recommended_transient_state"] == "OBSERVING"
    assert paper["mismatch"] is False
    assert paper["recommended_transient_state"] == "PAPER_ACTIVE"


def test_researching_with_real_learning_job_is_not_a_mismatch():
    truth = RT.derive_activity(
        [_job(SCH.LEARNING_CYCLE, status=JS.RUNNING, key="hist_learning:b1")],
        now_epoch=100.0,
    )
    out = RT.transient_state_mismatch("RESEARCHING", truth)
    assert out["mismatch"] is False


def test_discovery_refresh_is_reported_as_decision_simulation_not_maintenance():
    out = RT.derive_activity(
        [_job(SCH.DISCOVERY_REFRESH, status=JS.RUNNING, key="discovery_refresh:s1:t1")],
        now_epoch=100.0,
    )
    assert out["activity"] == RT.ACTIVITY_DECISION
    assert out["primary_job"]["job_type"] == SCH.DISCOVERY_REFRESH
    assert out["research_activity"] is False
