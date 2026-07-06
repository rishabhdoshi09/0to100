"""
Bulk OHLCV prefetcher — replaces thousands of individual yfinance requests
with chunked yf.download() batch calls. Shared in-memory cache, 5-min TTL.

Usage:
    prefetch(symbols)            # call once before pipeline stages
    df = get_cached("RELIANCE")  # returns pd.DataFrame (lowercase cols) or None
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import pandas as pd

_cache: dict[str, pd.DataFrame] = {}
_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 300           # seconds
_CHUNK = 200         # symbols per yf.download() call


def prefetch(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Batch-download OHLCV for all symbols in chunked yf.download() calls.
    Returns the number of symbols loaded. Safe to call repeatedly — no-op
    while the cache is fresh and already covers the requested symbols.
    """
    global _cache, _cache_ts

    now = time.time()
    with _lock:
        if now - _cache_ts < _TTL and _cache:
            missing = [s for s in symbols if s not in _cache]
            # Fresh cache already covers (almost) everything requested
            if len(missing) < max(10, len(symbols) * 0.05):
                return len(_cache)

    wanted = [s for s in symbols if not s.startswith("^")]
    if not wanted:
        return 0

    import yfinance as yf
    log = __import__("logger").get_logger(__name__)
    t0 = time.time()
    log.info("bulk_fetch_start", count=len(wanted), chunks=(len(wanted) + _CHUNK - 1) // _CHUNK)

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(wanted), _CHUNK):
        chunk = wanted[i:i + _CHUNK]
        ns_syms = [f"{s}.NS" for s in chunk]
        try:
            raw = yf.download(
                ns_syms,
                period=period,
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            log.warning("bulk_fetch_chunk_failed", chunk=i // _CHUNK, error=str(exc))
            continue
        if raw is None or raw.empty:
            continue

        single = len(ns_syms) == 1
        for sym, ns in zip(chunk, ns_syms):
            try:
                df = raw.copy() if single else raw[ns].copy()
                df.columns = [str(c).lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 30:
                    result[sym] = df
            except Exception:
                pass

        if progress:
            try:
                progress(min(i + _CHUNK, len(wanted)), len(wanted))
            except Exception:
                pass

    with _lock:
        _cache = result
        _cache_ts = time.time()

    elapsed = round(time.time() - t0, 1)
    log.info("bulk_fetch_done", loaded=len(result), of=len(wanted), elapsed_s=elapsed)
    return len(result)


def get_cached(symbol: str) -> Optional[pd.DataFrame]:
    """Return a copy of the cached OHLCV DataFrame, or None if not available."""
    with _lock:
        df = _cache.get(symbol)
    return df.copy() if df is not None else None


def cached_symbols() -> list[str]:
    """All symbols currently in the cache."""
    with _lock:
        return list(_cache.keys())


def is_warm() -> bool:
    """True if the cache is populated and not expired."""
    with _lock:
        return bool(_cache) and (time.time() - _cache_ts < _TTL)
