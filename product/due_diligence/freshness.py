"""Evidence-aware freshness for autonomous due-diligence acquisition.

A recent download *attempt* is not the same thing as current research. This
module reads the dataset-level coverage contract and answers two separate
questions:

1. Is the shortlisted research actually current enough to use?
2. If not, is another provider attempt due now, or should we respect a retry
   cooldown?

The split prevents both false-green research and tight retry loops when an
upstream provider is temporarily unavailable.

This module also acts as a defensive truth boundary around older acquisition
artifacts. A provider refresh can fail while a cached dataset remains present.
That cache may still be useful evidence, but the failed refresh must never be
converted into a green/current research state merely because values exist on
disk.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

RETRY_COOLDOWN_S = 20 * 60
_UNRESOLVED = frozenset({
    "stale",
    "not_yet_acquired",
    "acquisition_failed",
    "source_unavailable",
    "missing",
    "refresh_failed",
})
_META_FAILURES = frozenset({"acquisition_failed", "source_unavailable", "refresh_failed"})


def _utc_now(value: float | datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if value is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_attempt(facts: Mapping[str, Any]) -> datetime | None:
    """Latest real provider/check attempt recorded for a symbol."""
    stamps: list[datetime] = []
    meta = facts.get("dataset_meta")
    if isinstance(meta, Mapping):
        for raw in meta.values():
            if not isinstance(raw, Mapping):
                continue
            stamp = _parse_iso(raw.get("checked_at") or raw.get("fetched_at"))
            if stamp is not None:
                stamps.append(stamp)
    for key in ("inspected_at", "acquired_at"):
        stamp = _parse_iso(facts.get(key))
        if stamp is not None:
            stamps.append(stamp)
    return max(stamps) if stamps else None


def _required_problems(coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for raw in list(coverage.get("datasets") or []):
        if not isinstance(raw, Mapping) or not bool(raw.get("required")):
            continue
        status = str(raw.get("status") or "")
        if status not in _UNRESOLVED:
            continue
        problems.append({
            "id": str(raw.get("id") or ""),
            "label": str(raw.get("label") or raw.get("id") or "dataset"),
            "status": status,
            "checked_at": raw.get("checked_at"),
            "age_label": raw.get("age_label"),
            "cached_data_present": bool(raw.get("present")),
            "truth_source": "coverage",
        })
    return problems


def _metadata_refresh_problems(
    facts: Mapping[str, Any],
    coverage: Mapping[str, Any],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Catch failed refresh attempts that a present cache might otherwise hide.

    Some legacy acquisition paths recorded ``status=current`` when a live
    refresh failed but an older cache was available. The error was persisted in
    the same metadata row. Treat that as unresolved research until a later
    successful provider refresh clears the error. This is intentionally
    conservative: stale-but-usable evidence can still be read, but it cannot
    certify research freshness.
    """
    meta = facts.get("dataset_meta")
    if not isinstance(meta, Mapping):
        return []

    coverage_rows = {
        str(row.get("id") or ""): row
        for row in list(coverage.get("datasets") or [])
        if isinstance(row, Mapping)
    }
    required = {
        ds_id
        for ds_id, row in coverage_rows.items()
        if ds_id and bool(row.get("required"))
    }
    already = {str(row.get("id") or "") for row in existing}
    problems: list[dict[str, Any]] = []

    for ds_id in sorted(required):
        if ds_id in already:
            continue
        raw = meta.get(ds_id)
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or "").strip().lower()
        error = str(raw.get("error") or "").strip()
        if status not in _META_FAILURES and not error:
            continue

        coverage_row = coverage_rows.get(ds_id) or {}
        problem_status = status if status in _META_FAILURES else "refresh_failed"
        problems.append({
            "id": ds_id,
            "label": str(coverage_row.get("label") or ds_id),
            "status": problem_status,
            "checked_at": raw.get("checked_at") or raw.get("fetched_at"),
            "age_label": "Refresh failed; cached evidence retained" if coverage_row.get("present") else "Refresh failed",
            "cached_data_present": bool(coverage_row.get("present")),
            "refresh_error": error or None,
            "provider": raw.get("provider") or None,
            "truth_source": "dataset_meta",
        })
    return problems


def research_freshness(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    now: float | datetime | None = None,
    retry_cooldown_s: float = RETRY_COOLDOWN_S,
) -> dict[str, Any]:
    """Return truthful shortlist research state without hitting the network.

    ``fresh`` means every required dataset that the coverage contract can still
    acquire is resolved/current (``metric_not_reported`` is a legitimate resolved
    state). ``retry_due`` is independent: unresolved evidence checked recently
    remains *not fresh*, but is not immediately hammered again.
    """
    from product.due_diligence.acquire import (
        inspect_symbol_coverage,
        load_autonomy_facts,
        shortlist_symbols,
    )

    current = _utc_now(now)
    cooldown = max(0.0, float(retry_cooldown_s))
    shortlist = shortlist_symbols(scan_payload=scan_payload)
    if not shortlist:
        return {
            "schema_version": 2,
            "fresh": True,
            "retry_due": False,
            "state": "NO_SHORTLIST",
            "checked_at": current.isoformat(),
            "symbols": [],
            "unresolved_symbols": [],
            "unresolved_datasets": [],
            "next_retry_at": None,
            "reason": "No shortlisted recommendation requires autonomous research.",
        }

    symbols: list[dict[str, Any]] = []
    unresolved_symbols: list[str] = []
    unresolved_datasets: list[dict[str, Any]] = []
    retry_due = False
    next_retry: datetime | None = None

    for symbol in shortlist:
        facts = load_autonomy_facts(symbol)
        inspection_error = ""
        try:
            coverage = inspect_symbol_coverage(symbol, now=current)
            problems = _required_problems(coverage)
            problems.extend(_metadata_refresh_problems(facts, coverage, problems))
        except Exception as exc:  # local inspection failure must never become green
            coverage = {}
            problems = [{
                "id": "coverage_inspection",
                "label": "Research coverage inspection",
                "status": "inspection_failed",
                "checked_at": None,
                "age_label": "Inspection failed",
                "cached_data_present": False,
                "truth_source": "inspection",
            }]
            inspection_error = f"{type(exc).__name__}: {exc}"[:240]

        symbol_due = False
        symbol_next: datetime | None = None
        fallback_attempt = _latest_attempt(facts)
        for problem in problems:
            checked = _parse_iso(problem.get("checked_at")) or fallback_attempt
            if checked is None:
                due = True
                candidate_next = current
            else:
                candidate_next = checked + timedelta(seconds=cooldown)
                due = current >= candidate_next
            symbol_due = symbol_due or due
            if not due and (symbol_next is None or candidate_next < symbol_next):
                symbol_next = candidate_next
            unresolved_datasets.append({
                "symbol": symbol,
                **problem,
                "retry_due": due,
                "retry_at": candidate_next.isoformat(),
            })

        fresh = not problems
        if not fresh:
            unresolved_symbols.append(symbol)
            retry_due = retry_due or symbol_due
            if symbol_next is not None and (next_retry is None or symbol_next < next_retry):
                next_retry = symbol_next

        symbols.append({
            "symbol": symbol,
            "fresh": fresh,
            "retry_due": symbol_due,
            "coverage_pct": coverage.get("coverage_pct"),
            "summary": coverage.get("summary") or "Coverage unavailable",
            "unresolved": [row.get("id") for row in problems],
            "inspection_error": inspection_error or None,
            "last_attempt_at": fallback_attempt.isoformat() if fallback_attempt else None,
            "next_retry_at": symbol_next.isoformat() if symbol_next else None,
        })

    fresh = not unresolved_symbols
    if fresh:
        state = "CURRENT"
        reason = "Required research datasets are current for the active shortlist."
    elif retry_due:
        state = "RETRY_DUE"
        reason = "Required research evidence is unresolved and at least one provider retry is due."
    else:
        state = "RETRY_COOLDOWN"
        reason = "Required research evidence is unresolved; recent provider attempts are cooling down."

    return {
        "schema_version": 2,
        "fresh": fresh,
        "retry_due": bool(retry_due and not fresh),
        "state": state,
        "checked_at": current.isoformat(),
        "symbols": symbols,
        "unresolved_symbols": unresolved_symbols,
        "unresolved_datasets": unresolved_datasets,
        "next_retry_at": next_retry.isoformat() if next_retry else None,
        "reason": reason,
    }
