"""
Bulk OHLCV prefetcher — NSE official bhavcopy first, Yahoo only as backup.

Primary : data/bhavcopy_store.py — one NSE file per day covers the WHOLE
          market. First run downloads history once to logs/bhav/; after
          that only the newest day is fetched.
Backup  : chunked yf.download() — used only when the bhavcopy store
          can't be built (e.g. NSE archive unreachable).

Usage:
    prefetch(symbols)            # call once before pipeline stages
    df = get_cached("RELIANCE")  # returns pd.DataFrame (lowercase cols) or None
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import pandas as pd

_yf_cache: dict[str, pd.DataFrame] = {}
_yf_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 300           # seconds (yfinance backup cache)
_CHUNK = 200         # symbols per yf.download() call
_bhav_ok: bool = False


def prefetch(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Make OHLCV available for `symbols`. Tries the NSE bhavcopy store
    first (whole market, official). Falls back to yfinance chunks only
    if the store can't be built. Returns number of symbols covered.
    """
    global _bhav_ok

    # ── Primary: NSE bhavcopy (no Yahoo) ──────────────────────────────────────
    try:
        from data.bhavcopy_store import build_store, store_symbols
        n = build_store(days=260, progress=progress)
        if n >= 200:                      # sane whole-market build
            # During market hours: overlay TODAY's live bar so scans see
            # today's move, not yesterday's close (bhavcopy is EOD).
            try:
                from data.nse_live import apply_live_to_store
                apply_live_to_store()
            except Exception:
                pass
            with _lock:
                _bhav_ok = True
            covered = set(store_symbols())
            return sum(1 for s in symbols if s in covered)
    except Exception as exc:
        __import__("logger").get_logger(__name__).warning(
            "bhav_prefetch_failed", error=str(exc))
    with _lock:
        _bhav_ok = False

    # ── Backup: yfinance chunked download ─────────────────────────────────────
    return _prefetch_yf(symbols, period=period, progress=progress)


def get_cached(symbol: str) -> Optional[pd.DataFrame]:
    """Cached OHLCV for one symbol — bhavcopy store first, then yf cache."""
    with _lock:
        use_bhav = _bhav_ok
    if use_bhav:
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(symbol)
            if df is not None:
                return df
        except Exception:
            pass
    with _lock:
        df = _yf_cache.get(symbol)
    return df.copy() if df is not None else None


def cached_symbols() -> list[str]:
    with _lock:
        use_bhav = _bhav_ok
    if use_bhav:
        try:
            from data.bhavcopy_store import store_symbols
            syms = store_symbols()
            if syms:
                return syms
        except Exception:
            pass
    with _lock:
        return list(_yf_cache.keys())


def is_warm() -> bool:
    with _lock:
        if _bhav_ok:
            return True
        return bool(_yf_cache) and (time.time() - _yf_cache_ts < _TTL)


# ── yfinance backup path ──────────────────────────────────────────────────────

def _prefetch_yf(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    global _yf_cache, _yf_cache_ts

    now = time.time()
    with _lock:
        if now - _yf_cache_ts < _TTL and _yf_cache:
            missing = [s for s in symbols if s not in _yf_cache]
            if len(missing) < max(10, len(symbols) * 0.05):
                return len(_yf_cache)

    wanted = [s for s in symbols if not s.startswith("^")]
    if not wanted:
        return 0

    import yfinance as yf
    log = __import__("logger").get_logger(__name__)
    t0 = time.time()
    log.info("yf_bulk_fetch_start", count=len(wanted))

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(wanted), _CHUNK):
        chunk = wanted[i:i + _CHUNK]
        ns_syms = [f"{s}.NS" for s in chunk]
        try:
            raw = yf.download(
                ns_syms, period=period, interval="1d", group_by="ticker",
                threads=8, progress=False, auto_adjust=True,
            )
        except Exception as exc:
            log.warning("yf_bulk_chunk_failed", chunk=i // _CHUNK, error=str(exc))
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
        _yf_cache = result
        _yf_cache_ts = time.time()
    log.info("yf_bulk_fetch_done", loaded=len(result), of=len(wanted),
             elapsed_s=round(time.time() - t0, 1))
    return len(result)
