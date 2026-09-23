"""Authoritative current-work projection for the autonomy supervisor.

Policy state and current activity are different facts. PAPER_ACTIVE can be a valid
policy state while no job is executing; RESEARCHING and DATA_REFRESHING are
transient activity labels and must not remain latched after their durable work ends.

This module derives activity only from the durable job ledger:
RUNNING work first, then work that is PENDING and due now. Future recurring rows
are queue, not current activity.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH

SCHEMA_VERSION = 1

ACTIVITY_IDLE = "IDLE"
ACTIVITY_DATA = "DATA_REFRESH"
ACTIVITY_SCAN = "CURRENT_SCAN"
ACTIVITY_PAPER = "FORWARD_PAPER"
ACTIVITY_DECISION = "DECISION_SIMULATION"
ACTIVITY_SETTLEMENT = "SETTLEMENT"
ACTIVITY_HISTORY = "HISTORICAL_REPLAY"
ACTIVITY_LEARNING = "LEARNING"
ACTIVITY_RESEARCH = "RESEARCH"
ACTIVITY_MAINTENANCE = "MAINTENANCE"

_ACTIVITY_PRIORITY = (
    ACTIVITY_DATA,
    ACTIVITY_SCAN,
    ACTIVITY_PAPER,
    ACTIVITY_DECISION,
    ACTIVITY_SETTLEMENT,
    ACTIVITY_HISTORY,
    ACTIVITY_LEARNING,
    ACTIVITY_RESEARCH,
    ACTIVITY_MAINTENANCE,
)

_RESEARCH_ACTIVITIES = {ACTIVITY_HISTORY, ACTIVITY_LEARNING, ACTIVITY_RESEARCH}

_JOB_ACTIVITY = {
    SCH.DATA_REFRESH: ACTIVITY_DATA,
    SCH.BHAVCOPY_UPDATE: ACTIVITY_DATA,
    SCH.UNIVERSE_HISTORY: ACTIVITY_DATA,
    SCH.INDEX_WARMUP: ACTIVITY_DATA,
    SCH.MARKET_SCAN: ACTIVITY_SCAN,
    SCH.DISCOVERY_REFRESH: ACTIVITY_DECISION,
    SCH.LONG_TERM_SCAN: ACTIVITY_SCAN,
    SCH.LONG_TERM_REFRESH: ACTIVITY_SCAN,
    SCH.PAPER_CYCLE: ACTIVITY_PAPER,
    SCH.OUTCOME_RESOLUTION: ACTIVITY_SETTLEMENT,
    SCH.HISTORICAL_PAPER_CYCLE: ACTIVITY_HISTORY,
    SCH.LEARNING_CYCLE: ACTIVITY_LEARNING,
    SCH.RESEARCH_CYCLE: ACTIVITY_RESEARCH,
    SCH.AUTH_HEALTH: ACTIVITY_MAINTENANCE,
    SCH.INSTRUMENT_REFRESH: ACTIVITY_MAINTENANCE,
    SCH.CORPORATE_ACTIONS: ACTIVITY_MAINTENANCE,
    SCH.NEWS_REFRESH: ACTIVITY_MAINTENANCE,
}


def _row(job: Any) -> dict[str, Any]:
    return {
        "job_id": str(getattr(job, "job_id", "") or ""),
        "job_type": str(getattr(job, "job_type", "") or ""),
        "status": str(getattr(job, "status", "") or ""),
        "idempotency_key": str(getattr(job, "idempotency_key", "") or ""),
        "scheduled_for": float(getattr(job, "scheduled_for", 0.0) or 0.0),
        "started_at": float(getattr(job, "started_at", 0.0) or 0.0),
        "created_at": float(getattr(job, "created_at", 0.0) or 0.0),
        "attempt": int(getattr(job, "attempt", 0) or 0),
        "result_summary": str(getattr(job, "result_summary", "") or ""),
        "error_code": str(getattr(job, "error_code", "") or ""),
        "error_message": str(getattr(job, "error_message", "") or ""),
    }


def _activity_for(row: Mapping[str, Any]) -> str:
    job_type = str(row.get("job_type") or "")
    key = str(row.get("idempotency_key") or "")
    if job_type == SCH.PAPER_CYCLE:
        if key.startswith(("hist_", "hist-paper:", "historical:")):
            return ACTIVITY_HISTORY
        if key.startswith("snapshot_decision:"):
            return ACTIVITY_DECISION
    return _JOB_ACTIVITY.get(job_type, ACTIVITY_MAINTENANCE)


def _current(row: Mapping[str, Any], now_epoch: float) -> bool:
    status = str(row.get("status") or "")
    if status == JS.RUNNING:
        return True
    return status == JS.PENDING and float(row.get("scheduled_for") or 0.0) <= float(now_epoch)


def derive_activity(jobs: Iterable[Any], *, now_epoch: float) -> dict[str, Any]:
    current: list[dict[str, Any]] = []
    future_due: list[dict[str, Any]] = []
    for job in jobs:
        row = _row(job)
        row["activity"] = _activity_for(row)
        if _current(row, now_epoch):
            current.append(row)
        elif row["status"] == JS.PENDING:
            future_due.append(row)

    def rank(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
        activity = str(row.get("activity") or ACTIVITY_MAINTENANCE)
        try:
            priority = _ACTIVITY_PRIORITY.index(activity)
        except ValueError:
            priority = len(_ACTIVITY_PRIORITY)
        # "What is running now?" and "what should run next?" are different
        # questions. A genuinely RUNNING job is the primary current activity;
        # priority only orders peers with the same running/due status.
        running_rank = 0 if row.get("status") == JS.RUNNING else 1
        return (
            running_rank,
            priority,
            float(row.get("scheduled_for") or 0.0),
            str(row.get("job_id") or ""),
        )

    current.sort(key=rank)
    future_due.sort(key=lambda r: (float(r.get("scheduled_for") or 0.0), str(r.get("job_id") or "")))
    primary = current[0] if current else {}
    activity = str(primary.get("activity") or ACTIVITY_IDLE)
    return {
        "schema_version": SCHEMA_VERSION,
        "activity": activity,
        "busy": bool(current),
        "primary_job": primary,
        "current_jobs": current[:20],
        "current_count": len(current),
        "next_pending": future_due[0] if future_due else {},
        "research_activity": activity in _RESEARCH_ACTIVITIES,
        "source": "durable_job_ledger",
    }


def transient_state_mismatch(policy_state: str, activity_truth: Mapping[str, Any]) -> dict[str, Any]:
    """Explain only transient-state contradictions; never reinterpret PAPER_ACTIVE."""
    state = str(policy_state or "")
    activity = str((activity_truth or {}).get("activity") or ACTIVITY_IDLE)
    mismatch = False
    expected = state
    reason = ""

    if state == "RESEARCHING" and activity not in _RESEARCH_ACTIVITIES:
        mismatch = True
        expected = "DATA_REFRESHING" if activity == ACTIVITY_DATA else "OBSERVING"
        reason = f"policy state says RESEARCHING while durable activity is {activity}"
    elif state == "DATA_REFRESHING" and activity != ACTIVITY_DATA:
        mismatch = True
        expected = "OBSERVING" if activity == ACTIVITY_IDLE else state
        reason = f"policy state says DATA_REFRESHING while durable activity is {activity}"

    return {
        "mismatch": mismatch,
        "policy_state": state,
        "activity": activity,
        "recommended_transient_state": expected,
        "reason": reason,
    }


def research_activities() -> set[str]:
    return set(_RESEARCH_ACTIVITIES)
