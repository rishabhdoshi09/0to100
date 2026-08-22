"""Canonical company-result event timeline.

Uses official AVAILABLE_AT (exchange broadcast). Never reconstructs old
announcement dates from today's calendar. Does not compute earnings surprise
unless a genuine historical consensus series is supplied (none is, today).
"""
from __future__ import annotations

from typing import Any

from data.pit_events import (
    EVENT_TYPES,
    get_events,
    ledger_path,
    ledger_status,
    validate_rows,
)

RESULT_TYPES = (
    "EARNINGS_RESULT",
    "FINANCIAL_RESULT_UPDATE",
)
CANONICAL_TYPES = RESULT_TYPES + (
    "ANNUAL_RESULT",
    "QUARTERLY_RESULT",
    "BOARD_RESULT_MEETING",
    "GUIDANCE",
    "EARNINGS_ANNOUNCEMENT",
)

# IST cash session
_OPEN_MIN = 9 * 60 + 15
_CLOSE_MIN = 15 * 60 + 30


def classify_session(available_at_ts: str | None) -> str:
    """known-before-market / during-market / after-market / unknown."""
    if not available_at_ts:
        return "unknown"
    try:
        import pandas as pd
        from core.market_clock import IST
        ts = pd.Timestamp(available_at_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        minutes = ts.hour * 60 + ts.minute
        if minutes < _OPEN_MIN:
            return "known_before_market"
        if minutes <= _CLOSE_MIN:
            return "during_market"
        return "after_market"
    except Exception:
        return "unknown"


def _event_type(row: dict[str, Any]) -> str:
    et = str(row.get("event_type") or "OTHER").upper()
    period = str(row.get("period") or "").lower()
    if et == "EARNINGS_RESULT":
        if "annual" in period:
            return "ANNUAL_RESULT"
        if "quarter" in period:
            return "QUARTERLY_RESULT"
        return "EARNINGS_ANNOUNCEMENT"
    return et


def _quality(row: dict[str, Any]) -> str:
    if row.get("available_at") and (row.get("source") or "").startswith("nse"):
        return "PIT_STRONG"
    if row.get("available_at"):
        return "PIT_DEGRADED"
    return "UNUSABLE"


def normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    ts = row.get("available_at_ts")
    return {
        "symbol": row.get("symbol"),
        "event_type": _event_type(row),
        "announced_date": row.get("available_at"),
        "timestamp": ts,
        "fiscal_period": row.get("period"),
        "period_start": row.get("period_start"),
        "period_end": row.get("period_end"),
        "source": row.get("source"),
        "filing_id": row.get("seq_id") or row.get("event_id") or row.get("xbrl_url"),
        "session_class": classify_session(ts if isinstance(ts, str) else None),
        "data_quality": _quality(row),
        "event_id": row.get("event_id"),
        "not_earnings_surprise": True,
        "raw_type": row.get("event_type"),
    }


def timeline(
    symbol: str | None,
    as_of,
    *,
    path=None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Events publicly knowable as of ``as_of``. Future filings are excluded."""
    rows = get_events(symbol, as_of, path=path, since=since)
    return [normalize_event(r) for r in rows]


def post_result_study_rows(
    symbol: str | None,
    as_of,
    *,
    path=None,
) -> list[dict[str, Any]]:
    """Raw infrastructure for a future post-result study. Not an EDGE experiment.

    Growth / result-strength fields may be attached later from PIT fundamentals.
    There is no ``surprise`` key.
    """
    out = []
    for ev in timeline(symbol, as_of, path=path):
        if ev["event_type"] not in {
            "QUARTERLY_RESULT", "ANNUAL_RESULT", "EARNINGS_ANNOUNCEMENT",
            "EARNINGS_RESULT", "FINANCIAL_RESULT_UPDATE",
        } and ev.get("raw_type") not in RESULT_TYPES:
            continue
        out.append({
            **ev,
            "study_ready": ev["data_quality"] in {"PIT_STRONG", "PIT_DEGRADED"}
            and ev["announced_date"] is not None,
            "label_family": "result_strength_or_growth",
            "forbidden_label": "earnings_surprise",
        })
    return out


def coverage(path=None) -> dict[str, Any]:
    st = ledger_status(path)
    st["consensus_series"] = False
    st["may_compute_surprise"] = False
    st["event_schema"] = list(CANONICAL_TYPES)
    st["status"] = (
        "RESEARCH_READY_WITH_LIMITATIONS" if st.get("rows")
        else "DESCRIPTIVE_ONLY"
    )
    return st
