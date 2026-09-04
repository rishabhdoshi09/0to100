"""
Bulk OHLCV prefetcher — NSE official bhavcopy first, Kite targeted repair second,
Yahoo only as a last-resort whole-store backup.

Primary : data/bhavcopy_store.py — one NSE file per day covers the whole market.
Repair  : the EXISTING data-only Zerodha session fetches only requested NSE EQ
          symbols that are still absent from the official store. This closes the
          old hole where a generally-ready store caused missing symbols to vanish
          from the scanner without a repair attempt.
Backup  : chunked yf.download() — used only when the bhavcopy store itself cannot
          be built.

Usage:
    prefetch(symbols)            # warm the whole-market store
    backfill_missing(symbols)    # repair current-master names absent from it
    df = get_cached("RELIANCE")  # bhavcopy -> Kite repair -> Yahoo backup
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd

_yf_cache: dict[str, pd.DataFrame] = {}
_kite_cache: dict[str, pd.DataFrame] = {}
_yf_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 300           # seconds (yfinance backup cache)
_CHUNK = 200         # symbols per yf.download() call
_bhav_ok: bool = False


def _overlay_live_quiet() -> None:
    try:
        from data.nse_live import apply_live_to_store
        apply_live_to_store()
    except Exception:
        pass


def _bhav_symbols() -> set[str]:
    try:
        from data.bhavcopy_store import store_symbols
        return {str(s).strip().upper() for s in (store_symbols() or []) if str(s).strip()}
    except Exception:
        return set()


def adopt_ready_store(*, overlay_live: bool = True) -> int:
    """Use the official bhavcopy already in this process. Do not block a scan on a rebuild.

    Live Kite/NSE overlay runs in the background so the parallel stock walk can start
    immediately on the bars already loaded.
    """
    global _bhav_ok
    try:
        from data.bhavcopy_runtime import status as history_status

        info = history_status(load_cache=True)
        n = int(info.get("symbols") or 0)
        sessions = int(info.get("sessions") or 0)
        if not info.get("ready") or n < 200 or sessions < 60:
            return 0
        with _lock:
            _bhav_ok = True
        if overlay_live:
            threading.Thread(
                target=_overlay_live_quiet, name="live-overlay", daemon=True,
            ).start()
        covered = _bhav_symbols()
        return len(covered) if covered else n
    except Exception:
        return 0


def prefetch(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Make OHLCV available for ``symbols`` with official whole-market data first."""
    global _bhav_ok

    ready = adopt_ready_store(overlay_live=True)
    if ready >= 200:
        have = _bhav_symbols()
        return sum(1 for s in symbols if str(s).upper() in have) or ready

    # ── Primary: NSE bhavcopy (no Yahoo) ──────────────────────────────────────
    try:
        from data.bhavcopy_store import build_store
        n = build_store(days=260, progress=progress)
        if n >= 200:                      # sane whole-market build
            try:
                from data.nse_live import apply_live_to_store
                apply_live_to_store()
            except Exception:
                pass
            with _lock:
                _bhav_ok = True
            covered = _bhav_symbols()
            return sum(1 for s in symbols if str(s).upper() in covered)
    except Exception as exc:
        __import__("logger").get_logger(__name__).warning(
            "bhav_prefetch_failed", error=str(exc))
    with _lock:
        _bhav_ok = False

    # ── Backup: yfinance chunked download ─────────────────────────────────────
    return _prefetch_yf(symbols, period=period, progress=progress)


def _frame_from_kite(candles: list[dict]) -> Optional[pd.DataFrame]:
    rows = []
    for candle in candles or []:
        try:
            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            v = float(candle.get("volume", 0) or 0)
            stamp = pd.to_datetime(candle.get("date"))
        except Exception:
            continue
        if min(o, h, l, c) <= 0 or l > o or o > h or l > c or c > h or v < 0:
            continue
        rows.append((stamp, o, h, l, c, v))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date").set_index("date")
    return frame


def backfill_missing(symbols: list[str], *, client=None, now: datetime | None = None) -> dict:
    """Targeted repair for current NSE EQ symbols absent from the bhavcopy store.

    This is DATA ONLY. It uses the same authenticated Kite data facade as PAPER_AUTO
    and exposes no order/GTT surface. Every missing requested symbol is attempted
    when a valid daily Zerodha session exists; unresolved/failed names remain missing
    so the scan coverage ledger can report them honestly.
    """
    requested = list(dict.fromkeys(
        str(s).strip().upper() for s in (symbols or []) if str(s).strip()
    ))
    with _lock:
        fallback_have = set(_kite_cache) | set(_yf_cache)
    have = _bhav_symbols() | fallback_have
    missing = [s for s in requested if s not in have]
    if not missing:
        return {"requested": len(requested), "missing": 0, "attempted": 0, "loaded": 0, "unresolved": 0, "failed": 0}

    try:
        if client is None:
            from research.intelligence.data.kite_activation import KiteDataClient
            client = KiteDataClient.from_config()
        # profile is the cheapest explicit session validity check and contains no
        # secret in our logs; only success/failure counts are returned below.
        if not client.profile():
            raise RuntimeError("empty Kite profile")
        instruments = list(client.instruments("NSE"))
    except Exception as exc:
        return {
            "requested": len(requested), "missing": len(missing), "attempted": 0,
            "loaded": 0, "unresolved": len(missing), "failed": 0,
            "error_code": "KITE_HISTORY_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}",
        }

    token_by_symbol: dict[str, object] = {}
    for row in instruments:
        if not isinstance(row, dict):
            continue
        if str(row.get("exchange") or "NSE").upper() != "NSE":
            continue
        if str(row.get("instrument_type") or "").upper() != "EQ":
            continue
        sym = str(row.get("tradingsymbol") or "").strip().upper()
        token = row.get("instrument_token")
        if sym and token is not None:
            token_by_symbol[sym] = token

    resolvable = [s for s in missing if s in token_by_symbol]
    unresolved = len(missing) - len(resolvable)
    if not resolvable:
        return {"requested": len(requested), "missing": len(missing), "attempted": 0,
                "loaded": 0, "unresolved": unresolved, "failed": 0}

    try:
        from research.intelligence.data import nse_calendar as CAL
        from research.intelligence.data.kite_source import RateLimiter
        required = CAL.latest_required_session(now or CAL._now_ist(), CAL.load_holidays())
        want_to = required.isoformat()
        want_from = (required - timedelta(days=400)).isoformat()
        limiter = RateLimiter(max_per_sec=3.0)
    except Exception:
        end = (now or datetime.now()).date()
        want_to = end.isoformat()
        want_from = (end - timedelta(days=400)).isoformat()
        from research.intelligence.data.kite_source import RateLimiter
        limiter = RateLimiter(max_per_sec=3.0)

    loaded: dict[str, pd.DataFrame] = {}
    failed = 0
    for symbol in resolvable:
        try:
            limiter.acquire()
            candles = list(client.historical(token_by_symbol[symbol], want_from, want_to, "day"))
            frame = _frame_from_kite(candles)
            if frame is None:
                failed += 1
                continue
            loaded[symbol] = frame
        except Exception:
            failed += 1

    if loaded:
        with _lock:
            _kite_cache.update(loaded)
            overflow = len(_kite_cache) - 256
            if overflow > 0:
                for key in list(_kite_cache)[:overflow]:
                    _kite_cache.pop(key, None)
    return {
        "requested": len(requested),
        "missing": len(missing),
        "attempted": len(resolvable),
        "loaded": len(loaded),
        "unresolved": unresolved,
        "failed": failed,
        "source": "zerodha_kite_data_only",
    }


def get_cached(symbol: str) -> Optional[pd.DataFrame]:
    """Cached OHLCV for one symbol — bhavcopy, Kite repair, then Yahoo backup."""
    clean = str(symbol or "").strip().upper()
    with _lock:
        use_bhav = _bhav_ok
    if use_bhav:
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(clean)
            if df is not None:
                return df
        except Exception:
            pass
    with _lock:
        df = _kite_cache.get(clean)
        if df is None:
            df = _yf_cache.get(clean)
    return df.copy() if df is not None else None


def cached_symbols() -> list[str]:
    symbols: set[str] = set()
    with _lock:
        use_bhav = _bhav_ok
        symbols.update(_kite_cache)
        symbols.update(_yf_cache)
    if use_bhav:
        symbols.update(_bhav_symbols())
    return sorted(symbols)


def is_warm() -> bool:
    with _lock:
        if _bhav_ok:
            return True
        return bool(_kite_cache) or (bool(_yf_cache) and (time.time() - _yf_cache_ts < _TTL))


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
