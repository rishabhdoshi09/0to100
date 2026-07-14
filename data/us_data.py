"""
US daily data — yfinance-backed OHLCV for the US scanner.

Kite serves no US data, so per the data policy yfinance is the
legitimate primary here (there is no better free EOD source for US
equities). Daily candles power the setup scan; the last bar is honest
about whether it is a settled close or an intraday snapshot.

Everything is cached and threaded so a whole-universe scan stays cheap.
"""
from __future__ import annotations

import threading
import time

from logger import get_logger

log = get_logger(__name__)

# yfinance prints its own '401 Invalid Crumb / possibly delisted' spam via
# stdout + its logger — muzzle the logger so the console stays clean.
try:
    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)
except Exception:
    pass

_cache: dict[str, tuple[float, object]] = {}       # symbol -> (ts, DataFrame)
_lock = threading.Lock()
_TTL = 3600.0                                       # 1h — daily data barely moves


def get_us_daily(symbol: str, lookback_days: int = 400):
    """Daily OHLCV DataFrame (open/high/low/close/volume, lowercase) for a
    US ticker, or None. Cached 1h."""
    now = time.time()
    with _lock:
        hit = _cache.get(symbol)
        if hit and now - hit[0] < _TTL:
            return hit[1]
    try:
        import yfinance as yf
        raw = yf.download(symbol, period=f"{lookback_days}d", interval="1d",
                          progress=False, auto_adjust=True, threads=False)
        if raw is None or raw.empty:
            return None
        import pandas as pd
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        with _lock:
            _cache[symbol] = (now, df)
        return df
    except Exception as exc:
        log.debug("us_daily_failed", symbol=symbol, error=str(exc)[:80])
        return None


def get_us_daily_batch(symbols: list[str], lookback_days: int = 400) -> dict:
    """{symbol: DataFrame} for many tickers in ONE yfinance call — 100
    symbols/request instead of 100 requests, so the full US listing is
    actually fetchable without hammering Yahoo. Per-symbol cached; only
    the cache-misses hit the network. Falls back to per-symbol on error."""
    now = time.time()
    out: dict = {}
    misses: list[str] = []
    with _lock:
        for s in symbols:
            hit = _cache.get(s)
            if hit and now - hit[0] < _TTL:
                out[s] = hit[1]
            else:
                misses.append(s)
    if not misses:
        return out
    try:
        import yfinance as yf
        import pandas as pd
        raw = yf.download(misses, period=f"{lookback_days}d", interval="1d",
                          progress=False, auto_adjust=True, threads=True,
                          group_by="ticker")
        for s in misses:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.rename(columns={c: str(c).lower() for c in df.columns})
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                if len(df) >= 60:
                    out[s] = df
                    with _lock:
                        _cache[s] = (now, df)
            except Exception:
                continue
    except Exception as exc:
        log.debug("us_batch_failed", n=len(misses), error=str(exc)[:80])
        for s in misses:                     # fallback: per-symbol
            df = get_us_daily(s, lookback_days)
            if df is not None:
                out[s] = df
    return out


def sp500_return_30d() -> float:
    """S&P 500 30-session return (%) — the US relative-strength benchmark
    (the role Nifty plays for the NSE scanner)."""
    try:
        df = get_us_daily("^GSPC", lookback_days=90)
        if df is None or len(df) < 31:
            return 0.0
        c = df["close"].values
        return float((c[-1] / c[-31] - 1) * 100)
    except Exception:
        return 0.0


def us_live_prices(symbols: list[str]) -> dict[str, dict]:
    """{symbol: {price}} via yfinance fast_info. Near-real-time for
    indices, can be delayed for individual stocks (free-feed limit).
    Missing symbol simply absent — never a fake zero."""
    out: dict[str, dict] = {}
    if not symbols:
        return out
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                fi = yf.Ticker(sym).fast_info
                px = float(getattr(fi, "last_price", 0) or 0)
                if px > 0:
                    out[sym] = {"price": px}
            except Exception:
                continue
    except Exception:
        pass
    return out


def us_market_open(now=None) -> bool:
    """NYSE/Nasdaq regular session — 09:30–16:00 America/New_York."""
    try:
        import pytz
        from datetime import datetime as _dt
        ny = pytz.timezone("America/New_York")
        now = now.astimezone(ny) if now is not None else _dt.now(ny)
        if now.weekday() >= 5:
            return False
        hm = now.hour * 60 + now.minute
        return 9 * 60 + 30 <= hm <= 16 * 60
    except Exception:
        return False
