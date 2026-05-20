"""
F&O Securities Ban List — NSE publishes this daily.

Stocks in the F&O ban period cannot have fresh positions opened.
Showing them as ELITE setups is dangerous — they're untradable for new entries.

Sources tried in order:
  1. NSE archives CSV (most reliable)
  2. NSE web API (requires cookie)
  3. Empty set fallback (never block the app)
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {"symbols": set(), "date": None, "fetched_at": 0.0}
_TTL = 4 * 60 * 60   # refresh every 4 hours (ban list changes once daily at market open)

_NSE_CSV_URL = "https://archives.nseindia.com/web/sites/default/files/content/fo_secban.csv"
_NSE_API_URL  = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O%20BAN%20PERIOD"
_NSE_HOME     = "https://www.nseindia.com/"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}


def _fetch_via_csv() -> set[str]:
    resp = requests.get(_NSE_CSV_URL, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    symbols: set[str] = set()
    for line in resp.text.splitlines():
        s = line.strip().strip('"').upper()
        if s and not s.startswith("SYMBOL") and len(s) <= 20:
            symbols.add(s)
    return symbols


def _fetch_via_api() -> set[str]:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_NSE_HOME, timeout=8)
    resp = session.get(_NSE_API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    symbols: set[str] = set()
    for row in data.get("data", []):
        sym = row.get("symbol", "").upper()
        if sym:
            symbols.add(sym)
    return symbols


def get_fno_ban_list() -> set[str]:
    """
    Return today's F&O ban list as a set of NSE symbols.
    Cached for up to 4 hours. Returns empty set on any failure.
    """
    now = time.time()
    today = date.today().isoformat()

    if _CACHE["date"] == today and (now - _CACHE["fetched_at"]) < _TTL:
        return _CACHE["symbols"]

    symbols: set[str] = set()
    for fetcher, name in [(_fetch_via_csv, "CSV"), (_fetch_via_api, "API")]:
        try:
            symbols = fetcher()
            if symbols:
                logger.info("F&O ban list loaded via %s: %d symbols", name, len(symbols))
                break
        except Exception as exc:
            logger.warning("F&O ban fetch failed (%s): %s", name, exc)

    _CACHE["symbols"] = symbols
    _CACHE["date"] = today
    _CACHE["fetched_at"] = now
    return symbols


def is_in_fno_ban(symbol: str) -> bool:
    """True if this symbol is currently in the F&O ban period."""
    return symbol.upper() in get_fno_ban_list()


def annotate_setups(setups: list[dict]) -> list[dict]:
    """
    Add fno_banned=True to any setup dict whose symbol is in the ban list.
    Downgrades banned ELITE_A_PLUS → A tier (can't build fresh position).
    """
    try:
        ban = get_fno_ban_list()
        if not ban:
            return setups
        annotated = []
        for s in setups:
            sym = s.get("symbol", "").upper()
            if sym in ban:
                s = dict(s)
                s["fno_banned"] = True
                # Downgrade tier — can't take fresh F&O position
                if s.get("quality_tier") == "ELITE_A_PLUS":
                    s["quality_tier"] = "A"
                    s["fno_ban_note"] = "In F&O ban — no fresh positions allowed"
            annotated.append(s)
        return annotated
    except Exception:
        return setups
