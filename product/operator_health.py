"""Operator-facing health projection for the one-terminal QuantTerm product.

The durable autonomy ledger intentionally keeps historical jobs forever. Raw totals
therefore answer an audit question, not the operator's immediate question: "is the
system healthy today?" This module separates current-session health from historical
ledger counts without deleting evidence or changing scheduler semantics.

Broker login is projected as a separate execution/live-data lane. Missing Zerodha
authentication must not make official-data research or the autonomy supervisor look
failed when they are otherwise healthy.

Important request-path rule: rendering Home must never make a synchronous broker
network call. The autonomy ``auth_health`` job owns the live Zerodha probe. This
module only projects that persisted result plus the local active snapshot so a slow
broker/provider cannot hang ``/api/dashboard``.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import sqlite3
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


def _cached_auth_job() -> dict[str, Any]:
    """Read the latest persisted auth probe without contacting Zerodha."""
    try:
        from research.autonomy import default_root

        db_path = default_root() / "jobs.db"
        if not db_path.exists():
            return {}
        connection = sqlite3.connect(str(db_path), timeout=0.25)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT job_id,job_type,status,attempt,critical,scheduled_for,started_at,finished_at,"
                "result_summary,error_code,error_message,blocked_on,blocked_reason "
                "FROM jobs WHERE job_type = ? ORDER BY created_at DESC LIMIT 1",
                ("auth_health",),
            ).fetchone()
            return dict(row) if row is not None else {}
        finally:
            connection.close()
    except Exception:
        return {}


def _broker_lane() -> dict[str, Any]:
    """Project broker readiness from persisted auth evidence, never a request-time probe."""
    job = _cached_auth_job()
    status = str(job.get("status") or "").upper()
    result = str(job.get("result_summary") or "").upper()
    error_code = str(job.get("error_code") or "").upper()
    error_message = str(job.get("error_message") or "").strip()

    try:
        from product.readiness import kite_snapshot_id
        snapshot = str(kite_snapshot_id() or "")
    except Exception:
        snapshot = ""

    auth_ready = status == "SUCCEEDED" and (
        "AUTH_READY" in result or "SESSION_VALID" in result or result == "READY"
    )
    token_problem = (
        "TOKEN" in error_code
        or "SESSION_EXPIRED" in error_code
        or "LOGIN_REQUIRED" in result
        or "TOKEN_MISSING" in result
        or "SESSION_EXPIRED" in result
    )
    provider_problem = any(
        marker in error_code
        for marker in ("TIMEOUT", "CONNECTION", "NETWORK", "PROVIDER_UNAVAILABLE")
    )

    if auth_ready and snapshot:
        state = "READY"
        detail = "Latest scheduled Zerodha auth check passed and an active broker snapshot is available."
        auth_status = "SESSION_VALID"
    elif auth_ready:
        state = "SNAPSHOT_REQUIRED"
        detail = "Latest scheduled Zerodha auth check passed; broker snapshot is still being prepared."
        auth_status = "SESSION_VALID"
    elif token_problem:
        state = "LOGIN_REQUIRED"
        detail = error_message or "Daily Zerodha login is required for broker-dependent work."
        auth_status = "TOKEN_MISSING" if "TOKEN" in error_code or "TOKEN_MISSING" in result else "SESSION_EXPIRED"
    elif status in {"PENDING", "RUNNING", "RETRYABLE_FAILED"}:
        state = "CHECKING"
        detail = "Scheduled Zerodha auth check is still running; the desk remains available."
        auth_status = "CHECKING"
    elif provider_problem:
        state = "UNAVAILABLE"
        detail = error_message or "The latest Zerodha auth check could not reach the provider."
        auth_status = "PROVIDER_UNAVAILABLE"
    elif status in _FAILURE:
        state = "NOT_READY"
        detail = error_message or "The latest scheduled Zerodha auth check did not establish readiness."
        auth_status = error_code or "NOT_READY"
    else:
        state = "CHECKING"
        detail = "Waiting for the scheduled Zerodha auth-health result."
        auth_status = "UNKNOWN"

    live_data_ready = bool(state == "READY")
    return {
        "state": state,
        "ready": live_data_ready,
        "live_data_ready": live_data_ready,
        "execution_ready": live_data_ready,
        "auth_ready": bool(auth_ready),
        "login_required": state == "LOGIN_REQUIRED",
        "auth_status": auth_status,
        "reason_code": error_code or ("AUTH_READY" if auth_ready else auth_status),
        "detail": detail,
        "snapshot_id": snapshot,
        "source": "persisted_auth_health",
        "auth_job_id": str(job.get("job_id") or ""),
        "auth_checked_at": job.get("finished_at") or job.get("started_at") or job.get("scheduled_for"),
    }


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
    out["broker"] = _broker_lane()
    if refresh_in_progress:
        out["next_check_at"] = refresh.get("next_retry_at") or refresh.get("scheduled_for")
        out["data_refresh_next_poll_at"] = refresh.get("next_retry_at") or refresh.get("scheduled_for")
    return out
