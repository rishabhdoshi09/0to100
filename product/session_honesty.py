"""First-class NSE session banner — weekend / last official session, never inferred live.

Retry on F&O must not look dead: a 5-minute NSE backoff after a fail is expected.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def format_session_label(value: date | datetime | str | None) -> str:
    parsed = _as_date(value)
    if parsed is None:
        return ""
    return f"{_WEEKDAYS[parsed.weekday()]} {parsed.day} {_MONTHS[parsed.month - 1]} {parsed.year}"


def _as_date(value: date | datetime | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except Exception:
        return None


def _last_official_session() -> str:
    try:
        from data.bhavcopy_runtime import status as bhav_status

        snap = bhav_status(load_cache=True)
        return str(snap.get("latest_date") or snap.get("required_session") or "")
    except Exception:
        return ""


def session_payload(
    *,
    now: datetime | None = None,
    last_session: str | date | None = None,
    market_open: bool | None = None,
) -> dict[str, Any]:
    """Pure enough to unit-test: inject now / last_session / market_open."""
    if now is None:
        from core.market_clock import now_ist

        now = now_ist()
    today = now.date() if isinstance(now, datetime) else now
    weekday = int(today.weekday())
    is_weekend = weekday >= 5
    if market_open is None:
        try:
            from core.market_session import in_market_open

            market_open = bool(in_market_open(now if isinstance(now, datetime) else None))
        except Exception:
            market_open = False

    last = str(last_session or _last_official_session() or "")
    last_label = format_session_label(last) or (last or "unknown")
    today_label = format_session_label(today)

    if market_open:
        banner = f"NSE open · session {today_label}"
        state = "open"
    elif is_weekend:
        banner = f"NSE closed (weekend) · last session {last_label}"
        state = "weekend"
    else:
        banner = f"NSE closed · last session {last_label}"
        state = "closed"

    return {
        "available": True,
        "state": state,
        "market_open": bool(market_open),
        "is_weekend": is_weekend,
        "ist_date": today.isoformat(),
        "weekday": _WEEKDAYS[weekday],
        "last_session": last,
        "last_session_label": last_label,
        "banner": banner,
        "retry_note": (
            "Refresh live chain forces a new NSE fetch. After a fail the API waits "
            "up to 5 minutes so we do not hammer the exchange — that wait is not a dead button."
        ),
    }
