"""
Unified fundamentals entry point.

Checks SQLite cache first; scrapes screener.in only when the cache
is empty or stale (>1 day).  Always writes results back to cache.
"""

from __future__ import annotations

from typing import Any, Dict

from fundamentals.cache import FundamentalsCache
from logger import get_logger

log = get_logger(__name__)

_cache = FundamentalsCache()


def get_deep_fundamentals(
    symbol: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return full fundamentals dict for *symbol*.

    Uses the yielding multi-source resolver (Screener.in → Yahoo Finance →
    stale cache → user uploads). Prefer ``fundamentals.lazy.ensure_deep_fundamentals``
    for API/UI paths (rate-limited, backoff, trail).
    """
    symbol = symbol.upper().strip()

    if not force_refresh:
        cached = _cache.get(symbol)
        if cached is not None:
            log.info("fundamentals_served_from_cache", symbol=symbol)
            return cached

    from fundamentals.resolver import resolve

    log.info("fundamentals_resolving", symbol=symbol, force=force_refresh)
    data, steps = resolve(symbol, force_refresh=True, write_cache=True)
    if data is None:
        detail = steps[-1]["message"] if steps else "all sources exhausted"
        raise RuntimeError(f"Fundamentals unavailable for {symbol}: {detail}")
    _cache.clear_old()
    return data


def ensure_deep_fundamentals(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Lazy per-symbol sync (see fundamentals.lazy)."""
    from fundamentals.lazy import ensure_deep_fundamentals as _ensure

    return _ensure(symbol, force_refresh=force_refresh)
