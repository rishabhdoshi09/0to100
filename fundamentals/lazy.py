"""Lazy fundamentals sync — multi-source resolver with per-step yield trail."""
from __future__ import annotations

import time
from typing import Any, Dict

from fundamentals.cache import FundamentalsCache
from logger import get_logger

log = get_logger(__name__)

_MIN_GAP_S = 1.0
_FAIL_BACKOFF_S = 600
_last_scrape_s = 0.0
_fail_until: dict[str, float] = {}
_last_trail: dict[str, list[dict[str, Any]]] = {}


def _throttle() -> None:
    global _last_scrape_s
    now = time.time()
    wait = _MIN_GAP_S - (now - _last_scrape_s)
    if wait > 0:
        time.sleep(wait)
    _last_scrape_s = time.time()


def last_resolve_trail(symbol: str) -> list[dict[str, Any]]:
    return list(_last_trail.get(symbol.upper().strip(), []))


def ensure_deep_fundamentals(
    symbol: str,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return fundamentals for one symbol via the yielding multi-source resolver.

    Order: fresh cache → Screener.in → Yahoo Finance → stale cache → uploads.
    Every attempt is recorded in ``last_resolve_trail(symbol)``.
    """
    symbol = symbol.upper().strip()
    cache = FundamentalsCache()

    if not force_refresh:
        cached = cache.get(symbol)
        if cached is not None:
            _last_trail[symbol] = [{
                "step": 1, "source": "local_cache_fresh", "status": "OK",
                "message": "Today's IST cache hit — no network fetch (once per day)",
                "elapsed_ms": 0, "sections": {}, "reputed": True,
                "official": False, "coverage": 100,
            }]
            return cached
        if not cache.has(symbol):
            fail_ts = _fail_until.get(symbol)
            if fail_ts and time.time() < fail_ts:
                raise RuntimeError(
                    f"Fundamentals fetch for {symbol} is in backoff after a recent failure"
                )

    _throttle()
    from fundamentals.resolver import resolve

    data, steps = resolve(symbol, force_refresh=force_refresh, write_cache=True)
    _last_trail[symbol] = steps
    if data is not None:
        _fail_until.pop(symbol, None)
        if isinstance(data, dict):
            data = {**data, "_qt_cache_status": "TODAY"}
        return data

    _fail_until[symbol] = time.time() + _FAIL_BACKOFF_S
    stale = cache.get_any(symbol)
    if stale:
        log.warning("fundamentals_ensure_stale_fallback", symbol=symbol)
        if isinstance(stale, dict):
            stale = {**stale, "_qt_cache_status": "STALE"}
        _last_trail[symbol] = (steps or []) + [{
            "step": 99, "source": "local_cache_stale", "status": "STALE",
            "message": "Network failed — serving prior-day cache marked STALE (not current)",
            "elapsed_ms": 0, "sections": {}, "reputed": False,
            "official": False, "coverage": 50,
        }]
        return stale
    detail = steps[-1]["message"] if steps else "all sources exhausted"
    raise RuntimeError(f"Fundamentals unavailable for {symbol}: {detail}")


def cache_status() -> dict[str, Any]:
    return FundamentalsCache().stats()
