"""Priority-aware resource governor for autonomous QuantTerm work.

The supervisor remains productive while protecting current-market and official-data
work from background historical replay. This first governor is intentionally simple:
it prevents *new/heavy* historical replay from starting when a higher-priority job is
already due or running. It does not weaken any data/risk gate and it does not cancel
learning/research that can consume already-produced evidence.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH

SCHEMA_VERSION = 1

# Ordered highest -> lowest. Historical replay is the first lane we actively defer.
PRIORITY_ORDER = (
    "CRITICAL_DATA",
    "CURRENT_SCAN",
    "FORWARD_PAPER",
    "SETTLEMENT",
    "HISTORICAL_REPLAY",
    "RESEARCH",
)

_HEAVY_PREEMPTORS = {
    SCH.DATA_REFRESH: "CRITICAL_DATA",
    SCH.BHAVCOPY_UPDATE: "CRITICAL_DATA",
    SCH.MARKET_SCAN: "CURRENT_SCAN",
    SCH.PAPER_CYCLE: "FORWARD_PAPER",
    SCH.OUTCOME_RESOLUTION: "SETTLEMENT",
}


def _row(job: Any) -> dict[str, Any]:
    return {
        "job_id": str(getattr(job, "job_id", "") or ""),
        "job_type": str(getattr(job, "job_type", "") or ""),
        "status": str(getattr(job, "status", "") or ""),
        "idempotency_key": str(getattr(job, "idempotency_key", "") or ""),
        "scheduled_for": float(getattr(job, "scheduled_for", 0.0) or 0.0),
        "critical": bool(getattr(job, "critical", False)),
    }


def _is_due_or_running(row: Mapping[str, Any], now_epoch: float) -> bool:
    status = str(row.get("status") or "")
    if status == JS.RUNNING:
        return True
    if status != JS.PENDING:
        return False
    return float(row.get("scheduled_for") or 0.0) <= float(now_epoch)


def _is_forward_paper_key(key: str) -> bool:
    # Do not let an old historical poll classify itself as a preemptor.
    return not str(key or "").startswith(("hist_", "hist-paper:", "historical:"))


def assess(
    jobs: Iterable[Any],
    *,
    now_epoch: float,
) -> dict[str, Any]:
    """Return a deterministic resource decision from durable job truth."""
    blockers: list[dict[str, Any]] = []
    for job in jobs:
        row = _row(job)
        lane = _HEAVY_PREEMPTORS.get(row["job_type"])
        if not lane or not _is_due_or_running(row, now_epoch):
            continue
        if row["job_type"] == SCH.PAPER_CYCLE and not _is_forward_paper_key(
            row["idempotency_key"]
        ):
            continue
        blockers.append({**row, "priority_lane": lane})

    blockers.sort(
        key=lambda r: (
            PRIORITY_ORDER.index(r["priority_lane"])
            if r["priority_lane"] in PRIORITY_ORDER
            else len(PRIORITY_ORDER),
            0 if r["status"] == JS.RUNNING else 1,
            r["scheduled_for"],
            r["job_id"],
        )
    )
    top = blockers[0] if blockers else {}
    allowed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "historical_replay_allowed": allowed,
        "decision": "ALLOW_HISTORICAL_REPLAY" if allowed else "DEFER_HISTORICAL_REPLAY",
        "reason": (
            "no higher-priority due/running work"
            if allowed
            else f"{top.get('priority_lane')}:{top.get('job_type')}:{top.get('status')}"
        ),
        "priority_order": list(PRIORITY_ORDER),
        "blocking_jobs": blockers,
        "blocking_count": len(blockers),
        "learning_allowed": True,
        "research_allowed": True,
        "live_money_unchanged": True,
    }
