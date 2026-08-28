"""Operator-facing health projection for the one-terminal QuantTerm product.

The durable autonomy ledger intentionally keeps historical jobs forever.  Raw totals
therefore answer an audit question, not the operator's immediate question: "is the
system healthy today?"  This module separates current-session health from historical
ledger counts without deleting evidence or changing scheduler semantics.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None

_FAILURE = {"PERMANENT_FAILED", "FAILED"}
_ACTIVE = {"PENDING", "RUNNING", "RETRYABLE_FAILED"}


def _today() -> str:
    now = datetime.now(_IST) if _IST is not None else datetime.now()
    return now.date().isoformat()


def _job_day(job: dict[str, Any]) -> str:
    stamps = (
        job.get("finished_at"),
        job.get("started_at"),
        job.get("scheduled_for"),
    )
    for raw in stamps:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        dt = datetime.fromtimestamp(value, tz=_IST) if _IST is not None else datetime.fromtimestamp(value)
        return dt.date().isoformat()
    return ""


def _counts(jobs: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(job.get("status") or "UNKNOWN") for job in jobs)
    return dict(sorted(counter.items()))


def _latest(jobs: list[dict[str, Any]], job_type: str) -> dict[str, Any]:
    return next((job for job in jobs if str(job.get("job_type") or "") == job_type), {})


def enrich_autonomy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add truthful current-session health while preserving the full audit ledger."""
    out = dict(payload or {})
    jobs = [dict(job) for job in list(out.get("jobs_recent") or []) if isinstance(job, dict)]
    today = _today()
    current = [job for job in jobs if _job_day(job) == today]
    historical = [job for job in jobs if _job_day(job) and _job_day(job) != today]

    current_counts = _counts(current)
    historical_counts = _counts(historical)
    current_failed = [job for job in current if str(job.get("status") or "") in _FAILURE]
    current_blocked_critical = [
        job for job in current
        if str(job.get("status") or "") == "BLOCKED" and bool(job.get("critical"))
    ]
    current_active = [job for job in current if str(job.get("status") or "") in _ACTIVE]

    refresh = _latest(current, "data_refresh")
    refresh_in_progress = (
        str(refresh.get("error_code") or "") == "DATA_REFRESH_IN_PROGRESS"
        or str((out.get("active_job") or {}).get("job_type") or "") == "data_refresh"
    )
    outcome = _latest(current, "outcome_resolution")
    learning = _latest(current, "learning_cycle")
    research = _latest(current, "research_cycle")

    if str(learning.get("status") or "") == "SUCCEEDED":
        learning_status = "CURRENT"
    elif refresh_in_progress:
        learning_status = "WAITING_FOR_FRESH_EOD_DATA"
    elif str(outcome.get("status") or "") in {"PENDING", "RUNNING", "BLOCKED", "RETRYABLE_FAILED"}:
        learning_status = "WAITING_FOR_OUTCOMES"
    elif str(outcome.get("status") or "") == "SUCCEEDED":
        learning_status = "LEARNING_DUE"
    else:
        learning_status = "NO_EOD_LEARNING_YET"

    capability_failures = [str(item) for item in list(out.get("active_failures") or []) if str(item)]
    job_failures = [
        f"JOB_FAILED:{job.get('job_type')}:{job.get('error_code') or 'UNKNOWN'}"
        for job in current_failed
    ]
    job_failures.extend(
        f"JOB_BLOCKED:{job.get('job_type')}:{job.get('blocked_on') or 'DEPENDENCY'}"
        for job in current_blocked_critical
    )
    out["active_failures"] = list(dict.fromkeys(capability_failures + job_failures))

    if not out.get("running"):
        operator_state = "OFFLINE"
        plain = "Autonomy supervisor is offline. Market Operations may still be available separately."
    elif current_failed or current_blocked_critical or capability_failures:
        operator_state = "DEGRADED"
        plain = "QuantTerm is running, but a current-session capability or critical job needs attention."
    elif refresh_in_progress:
        operator_state = "WORKING"
        plain = "Terminal is available. Market data is refreshing in the background; EOD learning waits for fresh data."
    elif current_active:
        operator_state = "WORKING"
        plain = "QuantTerm is operational and processing current-session work."
    else:
        operator_state = "HEALTHY"
        plain = "QuantTerm is operational; no current-session failure is active."

    historical_failed = int(historical_counts.get("PERMANENT_FAILED", 0) + historical_counts.get("FAILED", 0))
    historical_blocked = int(historical_counts.get("BLOCKED", 0))
    history_note = (
        f" Historical ledger: {historical_failed} failed and {historical_blocked} blocked job(s); "
        "these are retained for audit and are not counted as today's health."
        if historical_failed or historical_blocked else ""
    )

    out["operator_state"] = operator_state
    out["plain_state"] = plain
    out["explanation"] = plain + history_note
    out["current_job_counts"] = current_counts
    out["historical_job_counts"] = historical_counts
    out["current_failed_jobs"] = current_failed[:12]
    out["current_blocked_critical_jobs"] = current_blocked_critical[:12]
    out["learning_status"] = learning_status
    out["learning_current"] = str(learning.get("status") or "") == "SUCCEEDED"
    out["research_current"] = str(research.get("status") or "") == "SUCCEEDED"
    out["data_refresh_background"] = refresh_in_progress
    return out
