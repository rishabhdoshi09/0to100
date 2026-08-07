"""
Earnings Calendar — auto-detects upcoming earnings and shows historical reactions.
Silently reduces position size recommendations for stocks reporting soon.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

# Quarter end dates (month numbers)
_QUARTER_END_MONTHS = {3, 6, 9, 12}

# Days after quarter end when earnings typically land
_DAYS_AFTER_QUARTER_END = 45


def _next_quarter_end(today: date) -> date:
    """Returns the next quarter-end date (Mar 31, Jun 30, Sep 30, Dec 31)."""
    month = today.month
    if month <= 3:
        return date(today.year, 3, 31)
    elif month <= 6:
        return date(today.year, 6, 30)
    elif month <= 9:
        return date(today.year, 9, 30)
    else:
        return date(today.year, 12, 31)


def _prev_quarter_end(today: date) -> date:
    """Returns the most recent quarter-end date."""
    month = today.month
    if month < 4:
        return date(today.year - 1, 12, 31)
    elif month < 7:
        return date(today.year, 3, 31)
    elif month < 10:
        return date(today.year, 6, 30)
    else:
        return date(today.year, 9, 30)


def _estimated_earnings_date(today: date) -> date:
    """
    Heuristic: earnings land ~45 days after quarter end.
    Returns the next expected earnings date.
    """
    prev_qe   = _prev_quarter_end(today)
    est_date  = prev_qe + timedelta(days=_DAYS_AFTER_QUARTER_END)

    # If estimated date is in the past, advance to next quarter
    if est_date <= today:
        next_qe  = _next_quarter_end(today)
        est_date = next_qe + timedelta(days=_DAYS_AFTER_QUARTER_END)

    return est_date


def _fetch_nse_earnings(symbol: str) -> Optional[date]:
    """
    Attempt to scrape NSE website for upcoming earnings.
    Returns date on success, None on any failure.
    """
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        url = (
            f"https://www.nseindia.com/api/corporate-announcements"
            f"?index=equities&symbol={symbol.upper()}&subject=Results"
        )
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        today = date.today()
        # Look for future or very recent announcement
        for item in data:
            bm_date_str = item.get("bm_date") or item.get("date") or ""
            if not bm_date_str:
                continue
            try:
                from datetime import datetime
                bm_date = datetime.strptime(bm_date_str[:10], "%d-%b-%Y").date()
                if bm_date >= today:
                    return bm_date
            except Exception:
                try:
                    from datetime import datetime
                    bm_date = datetime.strptime(bm_date_str[:10], "%Y-%m-%d").date()
                    if bm_date >= today:
                        return bm_date
                except Exception:
                    continue
    except Exception as exc:
        log.debug("nse_earnings_fetch_failed", symbol=symbol, error=str(exc))
    return None


def get_earnings_date(symbol: str) -> Optional[date]:
    """
    Returns next expected earnings date for a symbol.
    Tries NSE first (CONFIRMED), falls back to heuristic (ESTIMATED).
    Returns None only if heuristic also fails (shouldn't happen).
    """
    # 1. Try NSE
    nse_date = _fetch_nse_earnings(symbol)
    if nse_date is not None:
        return nse_date

    # 2. Heuristic fallback
    try:
        return _estimated_earnings_date(date.today())
    except Exception:
        return None


def _get_confidence(symbol: str) -> str:
    """Returns CONFIRMED if NSE API succeeded, ESTIMATED otherwise."""
    nse_date = _fetch_nse_earnings(symbol)
    if nse_date is not None:
        return "CONFIRMED"
    return "ESTIMATED"


def get_earnings_risk(symbol: str) -> dict:
    """
    Returns earnings proximity risk for the given symbol.

    Keys:
      days_to_earnings : int | None
      date             : str | None   (ISO format)
      confidence       : CONFIRMED | ESTIMATED | UNKNOWN
      risk_level       : HIGH | MEDIUM | LOW | NONE
      size_multiplier  : float  (1.0 = normal, 0.5 = halve size)
      note             : str
    """
    today = date.today()

    # Try NSE first
    nse_date = _fetch_nse_earnings(symbol)
    if nse_date is not None:
        confidence = "CONFIRMED"
        earnings_date = nse_date
    else:
        # Heuristic
        try:
            earnings_date = _estimated_earnings_date(today)
            confidence    = "ESTIMATED"
        except Exception:
            return {
                "days_to_earnings": None,
                "date":             None,
                "confidence":       "UNKNOWN",
                "risk_level":       "NONE",
                "size_multiplier":  1.0,
                "note":             "Earnings date unknown",
            }

    days = (earnings_date - today).days

    if days is not None and days <= 3:
        risk_level      = "HIGH"
        size_multiplier = 0.5
        note            = f"Earnings in {days} day{'s' if days != 1 else ''} — half size or skip"
    elif days is not None and days <= 7:
        risk_level      = "MEDIUM"
        size_multiplier = 0.75
        note            = f"Earnings next week — reduce size"
    elif days is not None and days <= 14:
        risk_level      = "LOW"
        size_multiplier = 1.0
        note            = f"Earnings in {days} days — normal size"
    else:
        risk_level      = "NONE"
        size_multiplier = 1.0
        note            = "No earnings risk in next 14 days"

    return {
        "days_to_earnings": days,
        "date":             earnings_date.isoformat(),
        "confidence":       confidence,
        "risk_level":       risk_level,
        "size_multiplier":  size_multiplier,
        "note":             note,
    }


def render_earnings_chip(symbol: str) -> str:
    """
    Returns an HTML chip showing earnings proximity.
    Only shows if days_to_earnings <= 14. Returns empty string otherwise.

    Red chip for HIGH, yellow for MEDIUM, gray for LOW.
    """
    try:
        info = get_earnings_risk(symbol)
    except Exception:
        return ""

    days       = info.get("days_to_earnings")
    risk_level = info.get("risk_level", "NONE")
    confidence = info.get("confidence", "UNKNOWN")

    if days is None or days > 14:
        return ""

    label = f"📅 Earnings in {days} day{'s' if days != 1 else ''}"
    if confidence == "ESTIMATED":
        label += " (est.)"

    if risk_level == "HIGH":
        bg     = "#2d0a0a"
        border = "#ff4b4b"
        color  = "#ff4b4b"
    elif risk_level == "MEDIUM":
        bg     = "#1f1500"
        border = "#f59e0b"
        color  = "#f59e0b"
    else:  # LOW
        bg     = "#111827"
        border = "#4a5568"
        color  = "#8892a4"

    return (
        f"<span style='display:inline-flex;align-items:center;"
        f"background:{bg};border:1px solid {border}44;border-radius:6px;"
        f"padding:3px 8px;font-size:.65rem;color:{color};font-weight:600'>"
        f"{label}</span>"
    )
