"""
Unified fundamentals entry point.

Checks SQLite cache first; scrapes screener.in only when the cache
is empty or stale (>1 day).  Always writes results back to cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

    last_good = _cache.get(symbol, allow_stale=True)

    if not force_refresh:
        cached = _cache.get(symbol, allow_stale=False)
        if cached is not None:
            cached.setdefault("source_label", "cache")
            cached.setdefault("source_tier", "cache")
            log.info("fundamentals_served_from_cache", symbol=symbol)
            return cached

    log.info("fundamentals_scraping", symbol=symbol, force=force_refresh)
    try:
        data = _scraper.fetch_all(symbol)
    except Exception:
        if last_good:
            last_good["stale"] = True
            last_good["source_label"] = "last_good_snapshot"
            last_good["source_tier"] = "last_good"
            last_good["official"] = False
            log.info("fundamentals_served_last_good", symbol=symbol)
            return last_good
        raise

    if isinstance(data, dict):
        data.setdefault("source_label", "secondary_public")
        data.setdefault("source_tier", "secondary")
        data["official"] = False
        data["retrieved_at"] = data.get("retrieved_at") or datetime.now(timezone.utc).isoformat()
    _cache.set(symbol, data)
    return data
