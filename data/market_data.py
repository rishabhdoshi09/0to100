"""
MarketDataProvider — Kite-first, yfinance fallback.

All UI modules that need live prices, % change, volume, or OHLC should
import from here instead of calling yfinance directly.

Priority:
  1. Zerodha Kite — real NSE tick data, no delay, accurate
  2. yfinance     — fallback when KITE_ACCESS_TOKEN is not set

Usage:
    from data.market_data import get_provider, get_historical_data
    mdp = get_provider()

    # Single quote
    q = mdp.quote("RELIANCE")
    # {"price": 2450.0, "prev_close": 2420.0, "chg_pct": 1.24, "volume": 1234567}

    # Batch quotes (one API call for Kite)
    quotes = mdp.quotes(["RELIANCE", "TCS", "INFY"])

    # Historical OHLCV (Kite-first with yfinance fallback)
    df = get_historical_data("RELIANCE", interval="day", from_date="2025-01-01", to_date="2025-03-01")
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)


def _kite_available() -> bool:
    return bool(os.getenv("KITE_API_KEY", "") and os.getenv("KITE_ACCESS_TOKEN", ""))


# ── Kite historical data ──────────────────────────────────────────────────────

_kite_instruments_cache: Optional[pd.DataFrame] = None


def _get_kite_raw():
    from data.kite_client import KiteClient
    return KiteClient()._kite


def _get_instrument_token(symbol: str) -> int:
    global _kite_instruments_cache
    kite = _get_kite_raw()
    if _kite_instruments_cache is None:
        _kite_instruments_cache = pd.DataFrame(kite.instruments("NSE"))
    row = _kite_instruments_cache[_kite_instruments_cache["tradingsymbol"] == symbol]
    if row.empty:
        raise ValueError(f"Instrument {symbol} not found in NSE instrument list")
    return int(row.iloc[0]["instrument_token"])


def get_historical_data_kite(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV from Kite Connect. Raises on failure."""
    if not _kite_available():
        raise ValueError("Kite credentials not set. Add KITE_API_KEY + KITE_ACCESS_TOKEN to .env")
    kite = _get_kite_raw()
    token = _get_instrument_token(symbol)
    from_dt = from_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_dt = to_date or datetime.today().strftime("%Y-%m-%d")
    data = kite.historical_data(token, from_dt, to_dt, interval)
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No data returned from Kite for {symbol}")
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    return df


def _init_yf_cache() -> None:
    """Point yfinance tz-cache at /tmp to avoid SQLite lock errors under concurrent threads."""
    try:
        import os, tempfile, yfinance as yf
        cache_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except Exception:
        pass


_init_yf_cache()


def get_historical_data_yfinance(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV from yfinance (NSE suffix auto-appended)."""
    import yfinance as yf
    _interval_map = {
        "day": "1d", "minute": "1m", "5minute": "5m",
        "15minute": "15m", "60minute": "1h", "week": "1wk", "month": "1mo",
    }
    yf_interval = _interval_map.get(interval, "1d")
    # Index symbols (^NSEI etc.) don't get .NS suffix
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        ticker = symbol
    else:
        ticker = symbol + ".NS"
    from_dt = from_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_dt = to_date or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=from_dt, end=to_dt, interval=yf_interval,
                     progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def get_historical_data(
    symbol: str,
    interval: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Unified OHLCV entry point.
    source = "auto"    → try Kite first, fall back to yfinance
    source = "kite"    → Kite only (raises on failure)
    source = "yfinance" → yfinance only
    """
    if source == "yfinance":
        return get_historical_data_yfinance(symbol, interval, from_date, to_date)
    if source == "kite":
        return get_historical_data_kite(symbol, interval, from_date, to_date)
    # auto
    if _kite_available():
        try:
            return get_historical_data_kite(symbol, interval, from_date, to_date)
        except Exception as e:
            log.debug("kite_hist_fallback_yf", symbol=symbol, error=str(e)[:80])
    return get_historical_data_yfinance(symbol, interval, from_date, to_date)


# ── Live quote providers ──────────────────────────────────────────────────────

def _empty_quote() -> dict:
    return {"price": 0.0, "prev_close": 0.0, "chg_pct": 0.0, "volume": 0, "avg_volume": 1}


def _yf_history_closes(symbol: str, days: int) -> list[float]:
    try:
        import yfinance as yf
        h = yf.Ticker(symbol + ".NS").history(period=f"{max(days + 5, 30)}d")
        if h is None or h.empty:
            return []
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]
        return [float(x) for x in h["Close"].dropna().tolist()[-days:]]
    except Exception:
        return []


_KITE_INDEX_SYMBOLS = {
    "NIFTY50": "NSE:NIFTY 50",
    "NIFTY 50": "NSE:NIFTY 50",
    "^NSEI": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "NIFTY BANK": "NSE:NIFTY BANK",
    "^NSEBANK": "NSE:NIFTY BANK",
}


class _KiteProvider:
    def __init__(self):
        from data.kite_client import KiteClient
        self._kite_client = KiteClient()
        self._exchange = os.getenv("KITE_EXCHANGE", "NSE")

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        try:
            # Separate indices from equities
            index_syms = [s for s in symbols if s.upper() in _KITE_INDEX_SYMBOLS or s.startswith("^")]
            equity_syms = [s for s in symbols if s not in index_syms]

            out: dict[str, dict] = {}

            # Fetch equities via get_ohlcv
            if equity_syms:
                raw = self._kite_client.get_ohlcv(equity_syms)
                for sym in equity_syms:
                    key = f"{self._exchange}:{sym}"
                    if key not in raw:
                        out[sym] = _empty_quote()
                        continue
                    d = raw[key]
                    price = float(d.get("last_price", 0))
                    ohlc  = d.get("ohlc", {})
                    prev  = float(ohlc.get("close", price))
                    vol   = float(d.get("volume_traded", 0))
                    chg   = (price - prev) / prev * 100 if prev else 0.0
                    out[sym] = {
                        "price": price, "prev_close": prev, "chg_pct": chg,
                        "volume": vol, "avg_volume": vol or 1,
                        "open": float(ohlc.get("open", price)),
                        "high": float(ohlc.get("high", price)),
                        "low":  float(ohlc.get("low", price)),
                    }

            # Fetch indices via ltp
            if index_syms:
                kite_index_keys = [_KITE_INDEX_SYMBOLS.get(s.upper(), f"NSE:{s}") for s in index_syms]
                raw_ltp = self._kite_client._kite.ltp(kite_index_keys)
                for sym, kite_key in zip(index_syms, kite_index_keys):
                    d = raw_ltp.get(kite_key, {})
                    price = float(d.get("last_price", 0))
                    ohlc  = d.get("ohlc", {})
                    prev  = float(ohlc.get("close", price))
                    chg   = (price - prev) / prev * 100 if prev else 0.0
                    out[sym] = {
                        "price": price, "prev_close": prev, "chg_pct": chg,
                        "volume": 0, "avg_volume": 1,
                        "open": float(ohlc.get("open", price)),
                        "high": float(ohlc.get("high", price)),
                        "low":  float(ohlc.get("low", price)),
                    }

            return out
        except Exception as e:
            log.debug("kite_provider_quotes_failed", error=str(e)[:80])
            return {s: _empty_quote() for s in symbols}

    def live_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Fast LTP fetch — one call, returns {symbol: price}."""
        try:
            instruments = [f"{self._exchange}:{s}" for s in symbols]
            raw = self._kite_client._kite.ltp(instruments)
            return {
                k.split(":")[1]: float(v.get("last_price", 0))
                for k, v in raw.items()
            }
        except Exception:
            return {}

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)

    @property
    def source(self) -> str:
        return "kite"


class _GoogleFinanceProvider:
    """Non-Kite fallback provider — routes through the unified quote
    chain (data.live_quotes: NSE snapshot → Google), yfinance backup.
    Name kept for backwards compat; Google is no longer the first hop."""

    def __init__(self):
        self._yf_backup = _YFinanceProvider()

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        try:
            from data.live_quotes import get_live_quotes
            for sym, q in get_live_quotes(symbols).items():
                price = float(q["price"])
                chg = float(q.get("chg_pct") or 0)
                prev = price / (1 + chg / 100) if chg > -100 else 0.0
                out[sym] = {
                    "price": price, "prev_close": round(prev, 2),
                    "chg_pct": chg, "volume": 0.0, "avg_volume": 1.0,
                }
        except Exception:
            pass
        missing = [s for s in symbols if s not in out]
        if missing:
            out.update(self._yf_backup.quotes(missing))
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        # Prefer the official bhav store; yfinance only as last resort
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(symbol)
            if df is not None and len(df) >= days:
                return [float(x) for x in df["close"].values[-days:]]
        except Exception:
            pass
        return _yf_history_closes(symbol, days)

    @property
    def source(self) -> str:
        return "unified_fallback"


class _YFinanceProvider:
    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out = {}
        import yfinance as yf
        for sym in symbols:
            try:
                ticker_sym = sym + ".NS" if not sym.endswith(".NS") else sym
                t = yf.Ticker(ticker_sym)

                # fast_info gives real-time last price (no API key needed)
                fi = getattr(t, "fast_info", None)
                price = float(fi.last_price) if fi and getattr(fi, "last_price", None) else 0.0
                prev  = float(fi.previous_close) if fi and getattr(fi, "previous_close", None) else 0.0
                vol   = float(fi.three_month_average_volume or 0) if fi else 0.0
                avg_v = vol or 1.0

                # Fallback if fast_info unavailable
                if price == 0.0:
                    h = t.history(period="5d", progress=False)
                    if h is None or len(h) < 1:
                        out[sym] = _empty_quote()
                        continue
                    if isinstance(h.columns, pd.MultiIndex):
                        h.columns = [c[0] for c in h.columns]
                    price = float(h["Close"].iloc[-1])
                    prev  = float(h["Close"].iloc[-2]) if len(h) >= 2 else price
                    vol   = float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                    avg_v = float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns and len(h) > 1 else 1

                chg = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {
                    "price": price, "prev_close": prev,
                    "chg_pct": chg, "volume": vol,
                    "avg_volume": avg_v or 1,
                }
            except Exception:
                out[sym] = _empty_quote()
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)

    @property
    def source(self) -> str:
        return "yfinance"


_provider: Optional[_KiteProvider | _GoogleFinanceProvider | _YFinanceProvider] = None


def get_provider() -> _KiteProvider | _GoogleFinanceProvider | _YFinanceProvider:
    """Return a cached provider. Kite if token is set, else Google Finance
    (live quotes, with yfinance as per-symbol backup)."""
    global _provider
    if _provider is None:
        if _kite_available():
            try:
                _provider = _KiteProvider()
            except Exception:
                _provider = _GoogleFinanceProvider()
        else:
            _provider = _GoogleFinanceProvider()
    return _provider


def reset_provider():
    """Call this after updating .env at runtime to pick up new credentials."""
    global _provider, _kite_instruments_cache
    _provider = None
    _kite_instruments_cache = None
