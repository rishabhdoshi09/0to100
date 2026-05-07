"""
Unified fundamentals entry point.

Checks SQLite cache first; scrapes screener.in only when the cache
is empty or stale (>1 day).  Always writes results back to cache.
"""

from __future__ import annotations

from typing import Any, Dict

from fundamentals.cache import FundamentalsCache
from fundamentals.screener_deep import ScreenerDeepFetcher
from logger import get_logger

log = get_logger(__name__)

_cache   = FundamentalsCache()
_scraper = ScreenerDeepFetcher()


def get_deep_fundamentals(
    symbol: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return full fundamentals dict for *symbol*.

    Flow
    ----
    1. If force_refresh=False, check SQLite cache.
       Return cached data if fresh (< 1 day old).
    2. Otherwise scrape screener.in (1-second polite delay).
    3. Store result in cache.
    4. Return data.

    Raises
    ------
    ValueError  – symbol not found on screener.in (HTTP 404)
    RuntimeError – unexpected HTTP status
    """
    symbol = symbol.upper().strip()

    if not force_refresh:
        cached = _cache.get(symbol)
        if cached is not None:
            log.info("fundamentals_served_from_cache", symbol=symbol)
            return cached

    log.info("fundamentals_scraping", symbol=symbol, force=force_refresh)
    data = _scraper.fetch_all(symbol)

    _cache.set(symbol, data)
    _cache.clear_old()   # housekeeping — remove stale entries

    return data
