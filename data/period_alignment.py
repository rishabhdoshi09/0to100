"""Fiscal-period alignment. Never treat YTD / 9-month as a quarter."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

QUARTER = "quarter"
ANNUAL = "annual"
NINE_MONTH = "nine_month"
YTD = "year_to_date"
IRREGULAR = "irregular"
UNKNOWN = "unknown"


def _d(raw: Any) -> date | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    s = str(raw).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(raw).strip()[:32], fmt).date()
        except ValueError:
            continue
    return None


def span_days(start, end) -> int | None:
    a, b = _d(start), _d(end)
    if not a or not b or b < a:
        return None
    return (b - a).days + 1


def classify_period(
    *,
    period: str | None = None,
    period_start=None,
    period_end=None,
    cumulative: str | None = None,
    relating_to: str | None = None,
) -> dict[str, Any]:
    """Return period_kind + whether the row may be used as a single quarter."""
    days = span_days(period_start, period_end)
    per = str(period or "").lower()
    rel = str(relating_to or "").lower()
    cum = str(cumulative or "").lower()
    kind = UNKNOWN
    quarterly_usable = False
    note = ""

    if "annual" in per or rel in {"annual", "year ended", "yearly"}:
        kind = ANNUAL
        quarterly_usable = False
        note = "annual_period"
    elif days is not None and 330 <= days <= 400:
        kind = ANNUAL
        quarterly_usable = False
        note = "span_looks_annual"
    elif days is not None and 200 <= days < 330:
        kind = NINE_MONTH
        quarterly_usable = False
        note = "nine_month_or_three_quarter_span"
    elif "non-cumulative" in cum or cum in {"", "no", "false"}:
        if days is not None and 70 <= days <= 110:
            kind = QUARTER
            quarterly_usable = True
            note = "quarter_span"
        elif days is not None and 28 <= days <= 69:
            kind = QUARTER
            quarterly_usable = True
            note = "short_quarter_span"
        elif "quarter" in per:
            if days is not None and days > 200:
                kind = YTD if days < 240 else ANNUAL
                quarterly_usable = False
                note = "labelled_quarterly_but_span_too_long"
            else:
                kind = QUARTER
                quarterly_usable = days is None or days <= 120
                note = "labelled_quarterly"
        else:
            kind = IRREGULAR
            quarterly_usable = False
            note = "unrecognised_span"
    else:
        # cumulative / YTD
        if days is not None and 200 <= days < 240:
            kind = NINE_MONTH
        elif days is not None and 70 <= days <= 200:
            kind = YTD
        else:
            kind = YTD
        quarterly_usable = False
        note = "cumulative_or_ytd"

    return {
        "period_kind": kind,
        "span_days": days,
        "quarterly_usable": quarterly_usable,
        "note": note,
        "canonical_preference": "CONSOLIDATED",
    }


def consol_label(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s.startswith("consolid"):
        return "CONSOLIDATED"
    if s.startswith("non-consolid") or s.startswith("standalone") or s == "non-consolidated":
        return "STANDALONE"
    return "UNKNOWN"
