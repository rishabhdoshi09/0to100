"""
Unified fundamentals entry point.

Delivery policy:
1. Fresh SQLite evidence when available.
2. Reputable public Screener.in scrape when refresh is needed.
3. Persisted last-good evidence, explicitly marked stale, when the internet or
   provider fails.

A provider outage must not turn an already researched stock into an empty page.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fundamentals.cache import FundamentalsCache
from fundamentals.screener_deep import ScreenerDeepFetcher
from logger import get_logger

log = get_logger(__name__)

_cache = FundamentalsCache()
_scraper = ScreenerDeepFetcher()


def _delivery(
    data: Dict[str, Any],
    *,
    state: str,
    source: str,
    source_tier: str,
    stale: bool,
    error: str = "",
) -> Dict[str, Any]:
    out = dict(data)
    meta = dict(out.get("_delivery") or {})
    meta.update({
        "state": state,
        "source": source,
        "source_tier": source_tier,
        "stale": bool(stale),
        "served_at": datetime.now(timezone.utc).isoformat(),
        "source_url": f"https://www.screener.in/company/{str(out.get('symbol') or '').upper()}/" if out.get("symbol") else "https://www.screener.in/",
    })
    if error:
        meta["refresh_error"] = error[:400]
    out["_delivery"] = meta
    return out


def get_deep_fundamentals(
    symbol: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Return full fundamentals for *symbol* without blanking on transient failures.

    Screener.in is a public secondary source and is labelled as such. Missing
    fields remain missing; stale fallback is never presented as fresh.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("symbol required")

    if not force_refresh:
        cached = _cache.get(symbol)
        if cached is not None:
            log.info("fundamentals_served_from_cache", symbol=symbol)
            return _delivery(
                cached,
                state="FRESH_CACHE",
                source="Screener.in via persisted cache",
                source_tier="reputable_secondary",
                stale=False,
            )

    last_good = _cache.get_any(symbol)
    log.info("fundamentals_scraping", symbol=symbol, force=force_refresh)
    try:
        data = _scraper.fetch_all(symbol)
        if not isinstance(data, dict) or not data:
            raise RuntimeError("Screener.in returned an empty fundamentals payload")
        data = dict(data)
        data.setdefault("symbol", symbol)
        data = _delivery(
            data,
            state="FRESH_SECONDARY",
            source="Screener.in",
            source_tier="reputable_secondary",
            stale=False,
        )
        _cache.set(symbol, data)
        # Do NOT clear stale rows here. They are the last-good safety net for
        # symbols whose next internet refresh fails.
        return data
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if last_good is not None:
            log.warning("fundamentals_refresh_failed_serving_last_good", symbol=symbol, error=error)
            stale = _delivery(
                last_good,
                state="STALE_LAST_GOOD",
                source=str((last_good.get("_delivery") or {}).get("source") or "Screener.in persisted last-good"),
                source_tier=str((last_good.get("_delivery") or {}).get("source_tier") or "reputable_secondary"),
                stale=True,
                error=error,
            )
            stale["_delivery"]["cache_age_seconds"] = _cache.age_seconds(symbol)
            return stale
        # No invented values and no hidden success when every usable source is
        # absent. Preserve the original error type/meaning for callers.
        log.error("fundamentals_refresh_failed_no_fallback", symbol=symbol, error=error)
        raise
