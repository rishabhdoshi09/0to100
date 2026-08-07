"""
Market session — which market is "active" right now.

The terminal covers two markets in different time zones. During Indian
hours you want to see NSE; when India closes and the US opens (evening
IST) you want to see US. This module is the single source of truth for
that decision, with a manual override for when the user wants to look at
the other market off-hours.

  auto_market()      → "IN" if NSE is open, else "US" if US is open, else
                       the home market ("IN") as the resting default.
  resolve_market()   → honours a manual pref ("IN"/"US"), else auto.
"""
from __future__ import annotations

MARKETS = {
    "IN": {"name": "NSE", "flag": "🇮🇳", "cur": "₹", "tz": "Asia/Kolkata",
           "open": (9, 15), "close": (15, 30)},
    "US": {"name": "US", "flag": "🇺🇸", "cur": "$", "tz": "America/New_York",
           "open": (9, 30), "close": (16, 0)},
}


def _is_open(market: str, now=None) -> bool:
    m = MARKETS[market]
    try:
        import pytz
        from datetime import datetime as _dt
        tz = pytz.timezone(m["tz"])
        now = now.astimezone(tz) if now is not None else _dt.now(tz)
        if now.weekday() >= 5:
            return False
        hm = now.hour * 60 + now.minute
        return (m["open"][0] * 60 + m["open"][1]) <= hm <= \
               (m["close"][0] * 60 + m["close"][1])
    except Exception:
        return False


def in_market_open(now=None) -> bool:
    return _is_open("IN", now)


def us_market_open(now=None) -> bool:
    return _is_open("US", now)


def auto_market(now=None) -> str:
    """IN if NSE open, else US if US open, else the home default (IN)."""
    if _is_open("IN", now):
        return "IN"
    if _is_open("US", now):
        return "US"
    return "IN"


def resolve_market(pref: str = "AUTO", now=None) -> str:
    """Manual pref wins ('IN'/'US'); 'AUTO' (or anything else) → auto."""
    p = (pref or "AUTO").upper()
    return p if p in ("IN", "US") else auto_market(now)


def market_meta(market: str) -> dict:
    return MARKETS.get(market, MARKETS["IN"])


def status_line(now=None) -> str:
    """'🇮🇳 NSE OPEN' / '🇺🇸 US OPEN' / '🔴 Both closed → NSE (home)'."""
    if _is_open("IN", now):
        return "🇮🇳 NSE OPEN"
    if _is_open("US", now):
        return "🇺🇸 US OPEN"
    return "🔴 Dono band → NSE (home) default"
