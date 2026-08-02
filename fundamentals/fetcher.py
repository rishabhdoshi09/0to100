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

    Prefer ``fundamentals.lazy.ensure_deep_fundamentals`` for API/UI paths
  (rate-limited, backoff). This function is the direct scrape + cache write.
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


def ensure_deep_fundamentals(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Lazy per-symbol sync (see fundamentals.lazy)."""
    from fundamentals.lazy import ensure_deep_fundamentals as _ensure

    return _ensure(symbol, force_refresh=force_refresh)
