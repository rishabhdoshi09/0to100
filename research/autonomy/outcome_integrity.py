"""Fail-closed integrity guard for the lightweight EOD outcome lane.

`outcome_liveness` deliberately settles paper positions and frozen counterfactuals separately. A
partial settlement must not be translated into an overall SUCCEEDED job, because that would unblock
LEARNING while some matured decisions were silently unresolved. Successful rows are already
persisted idempotently, so the correct behaviour is to retry the same logical outcome job; on the
retry only unresolved rows are attempted again.

The legacy forward-evidence pass historically swallowed resolver exceptions with ``continue``. Until
that old loop is removed, this guard re-probes only still-unsettled forward rows. A normal
not-yet-mature horizon returns ``None`` and remains pending; an exception is explicit retryable
failure rather than invisible evidence loss.
"""
from __future__ import annotations

from typing import Any

_INSTALLED = False
_BASE_HANDLER = None


def _failure_summary(failures: list[Any]) -> str:
    bits: list[str] = []
    for item in failures[:4]:
        if isinstance(item, dict):
            symbol = str(item.get("symbol") or item.get("decision_id") or "").strip()
            error = str(item.get("error") or item.get("reason") or "").strip()
            bits.append(": ".join(x for x in (symbol, error) if x))
        else:
            bits.append(str(item))
    return "; ".join(bit for bit in bits if bit)[:240]


def _unreported_forward_failures() -> list[dict[str, str]]:
    """Find resolver errors that the legacy forward-ledger loop can otherwise hide."""
    failures: list[dict[str, str]] = []
    try:
        from core.outcome_resolver import session_close_return
        from product.forward_evidence import load_ledger
    except Exception:
        return failures

    for row in load_ledger() or []:
        if not isinstance(row, dict) or row.get("later_outcome") or not row.get("decision_id"):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        day = str(row.get("market_timestamp") or (row.get("pit_proof") or {}).get("as_of") or "")[:10]
        if not symbol or not day:
            continue
        try:
            # None is an expected not-yet-mature horizon and must NOT fail the job.
            session_close_return(symbol, day, horizon=5)
        except Exception as exc:
            failures.append({
                "symbol": symbol,
                "decision_id": str(row.get("decision_id") or ""),
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "source": "forward_evidence_reprobe",
            })
    return failures


def strict_outcome_resolution(ctx):
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as J

    if _BASE_HANDLER is None:
        return J.JobResult(
            JS.RETRYABLE_FAILED,
            "outcome integrity handler is not installed",
            error_code="OUTCOME_INTEGRITY_NOT_INSTALLED",
        )

    result = _BASE_HANDLER(ctx)
    if getattr(result, "status", None) != JS.SUCCEEDED:
        return result

    metadata = dict(getattr(result, "metadata", {}) or {})
    official = dict(metadata.get("official_settlement") or {})
    failures = list(official.get("failed") or [])
    failures.extend(_unreported_forward_failures())
    if not failures:
        return result

    official["failed"] = failures
    metadata["official_settlement"] = official
    return J.JobResult(
        JS.RETRYABLE_FAILED,
        f"official outcome settlement partial · {len(failures)} unresolved error(s)",
        error_code="OFFICIAL_SETTLEMENT_PARTIAL",
        error_message=_failure_summary(failures) or "one or more official outcomes failed to resolve",
        failures=set(getattr(result, "failures", set()) or set()),
        state_hint=getattr(result, "state_hint", None),
        new_entries_allowed=bool(getattr(result, "new_entries_allowed", True)),
        metadata=metadata,
    )


def install_outcome_integrity() -> None:
    """Wrap the already-installed lightweight handler exactly once."""
    global _INSTALLED, _BASE_HANDLER
    if _INSTALLED:
        return
    from research.autonomy import jobs as J
    from research.autonomy import schedules as SCH

    base = J.HANDLERS.get(SCH.OUTCOME_RESOLUTION)
    if base is None or base is strict_outcome_resolution:
        return
    _BASE_HANDLER = base
    J.run_outcome_resolution = strict_outcome_resolution
    J.HANDLERS[SCH.OUTCOME_RESOLUTION] = strict_outcome_resolution
    _INSTALLED = True
