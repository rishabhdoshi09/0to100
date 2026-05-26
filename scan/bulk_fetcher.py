"""
Bulk OHLCV prefetcher — replaces 370 individual yfinance requests with one
batched yf.download() call. Shared in-memory cache lives for 5 minutes.

Usage:
    prefetch(symbols)          # call once before pipeline stages
    df = get_cached("RELIANCE") # returns pd.DataFrame or None
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import pandas as pd

_cache: dict[str, pd.DataFrame] = {}
_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 300  # seconds


def prefetch(symbols: list[str], period: str = "260d") -> None:
    """Batch-download OHLCV for all symbols in one yf.download() call."""
    global _cache, _cache_ts

    now = time.time()
    with _lock:
        if now - _cache_ts < _TTL and _cache:
            return  # still fresh — skip download

    ns_syms = [f"{s}.NS" for s in symbols if not s.startswith("^")]
    if not ns_syms:
        return

    import yfinance as yf
    log = __import__("logger").get_logger(__name__)
    t0 = time.time()
    log.info("bulk_fetch_start", count=len(ns_syms))

    try:
        raw = yf.download(
            ns_syms,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            show_errors=False,
            auto_adjust=True,
        )
    except Exception as exc:
        log.warning("bulk_fetch_failed", error=str(exc))
        return

    result: dict[str, pd.DataFrame] = {}
    single = len(ns_syms) == 1

    for sym, ns in zip(symbols, ns_syms):
        try:
            df = raw.copy() if single else raw[ns].copy()
            df.columns = [str(c).lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if len(df) >= 30:
                result[sym] = df
        except Exception:
            pass

    with _lock:
        _cache = result
        _cache_ts = time.time()

    elapsed = round(time.time() - t0, 1)
    log.info("bulk_fetch_done", loaded=len(result), of=len(ns_syms), elapsed_s=elapsed)


def get_cached(symbol: str) -> Optional[pd.DataFrame]:
    """Return a copy of the cached OHLCV DataFrame, or None if not available."""
    with _lock:
        df = _cache.get(symbol)
    return df.copy() if df is not None else None


def is_warm() -> bool:
    """True if the cache is populated and not expired."""
    with _lock:
        return bool(_cache) and (time.time() - _cache_ts < _TTL)
