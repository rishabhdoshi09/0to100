"""Lazy fundamentals sync — scrape Screener.in per symbol on demand, not bulk upfront."""
from __future__ import annotations

import time
from typing import Any, Dict

from fundamentals.cache import FundamentalsCache
from fundamentals.fetcher import get_deep_fundamentals
from logger import get_logger

log = get_logger(__name__)

_MIN_GAP_S = 1.0
_FAIL_BACKOFF_S = 600
_last_scrape_s = 0.0
_fail_until: dict[str, float] = {}


def _throttle() -> None:
    global _last_scrape_s
    now = time.time()
    wait = _MIN_GAP_S - (now - _last_scrape_s)
    if wait > 0:
        time.sleep(wait)
    _last_scrape_s = time.time()


def ensure_deep_fundamentals(
    symbol: str,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return fundamentals for one symbol, fetching from Screener.in only when needed.

    - Fresh cache hit → instant return
    - Missing or stale (when force_refresh) → single polite scrape (~1s)
    - Scrape fail → return stale cache if any exists; otherwise raise
    """
    symbol = symbol.upper().strip()
    cache = FundamentalsCache()

    if not force_refresh:
        cached = cache.get(symbol)
        if cached is not None:
            return cached
        if not cache.has(symbol):
            fail_ts = _fail_until.get(symbol)
            if fail_ts and time.time() < fail_ts:
                stale = cache.get_any(symbol)
                if stale:
                    return stale
                raise RuntimeError(f"Fundamentals fetch for {symbol} is in backoff after a recent failure")

    _throttle()
    try:
        data = get_deep_fundamentals(symbol, force_refresh=force_refresh)
        _fail_until.pop(symbol, None)
        return data
    except Exception as exc:
        _fail_until[symbol] = time.time() + _FAIL_BACKOFF_S
        stale = cache.get_any(symbol)
        if stale:
            log.warning("fundamentals_ensure_stale_fallback", symbol=symbol, error=type(exc).__name__)
            return stale
        raise


def cache_status() -> dict[str, Any]:
    return FundamentalsCache().stats()
