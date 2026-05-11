"""
MarketDataProvider — Kite-first, yfinance fallback.

All UI modules that need live prices, % change, volume, or OHLC should
import from here instead of calling yfinance directly.

Priority:
  1. Zerodha Kite — real NSE tick data, no MultiIndex weirdness, accurate
  2. yfinance     — fallback when KITE_ACCESS_TOKEN is not set

Usage:
    from data.market_data import get_provider
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
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    from_dt = from_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_dt = to_date or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=from_dt, end=to_dt, interval=yf_interval, progress=False, auto_adjust=True)
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
    source = "auto"  → try Kite first, fall back to yfinance
    source = "kite"  → Kite only (raises on failure)
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
            print(f"[market_data] Kite failed for {symbol}: {e}. Falling back to yfinance.")
    return get_historical_data_yfinance(symbol, interval, from_date, to_date)


# ── Live quote providers ──────────────────────────────────────────────────────

class _KiteProvider:
    def __init__(self):
        from data.kite_client import KiteClient
        self._kite = KiteClient()

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        try:
            instruments = [f"NSE:{s}" for s in symbols]
            raw = self._kite.raw.ohlc(instruments)
            out = {}
            for sym in symbols:
                key = f"NSE:{sym}"
                if key not in raw:
                    out[sym] = _empty_quote()
                    continue
                d = raw[key]
                price = float(d.get("last_price", 0))
                prev  = float(d.get("ohlc", {}).get("close", price))
                chg   = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {"price": price, "prev_close": prev, "chg_pct": chg, "volume": 0}
            return out
        except Exception:
            return {s: _empty_quote() for s in symbols}

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)


class _YFinanceProvider:
    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out = {}
        import yfinance as yf
        for sym in symbols:
            try:
                h = yf.Ticker(sym + ".NS").history(period="5d")
                if h is None or len(h) < 2:
                    out[sym] = _empty_quote()
                    continue
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = [c[0] for c in h.columns]
                price = float(h["Close"].iloc[-1])
                prev  = float(h["Close"].iloc[-2])
                vol   = float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                avg_v = float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns else 1
                chg   = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {
                    "price": price, "prev_close": prev,
                    "chg_pct": chg, "volume": vol,
                    "avg_volume": avg_v,
                }
            except Exception:
                out[sym] = _empty_quote()
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)


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


def _empty_quote() -> dict:
    return {"price": 0.0, "prev_close": 0.0, "chg_pct": 0.0, "volume": 0, "avg_volume": 1}


_provider: Optional[_KiteProvider | _YFinanceProvider] = None


def get_provider() -> _KiteProvider | _YFinanceProvider:
    """Return a cached provider. Kite if token is set, else yfinance."""
    global _provider
    if _provider is None:
        if _kite_available():
            try:
                _provider = _KiteProvider()
            except Exception:
                _provider = _YFinanceProvider()
        else:
            _provider = _YFinanceProvider()
    return _provider


def reset_provider():
    """Call this after updating .env at runtime to pick up new credentials."""
    global _provider, _kite_instruments_cache
    _provider = None
    _kite_instruments_cache = None


def _kite_available() -> bool:
    return bool(os.getenv("KITE_API_KEY", "") and os.getenv("KITE_ACCESS_TOKEN", ""))


class _KiteProvider:
    def __init__(self):
        from data.kite_client import KiteClient
        self._kite = KiteClient()

    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        try:
            instruments = [f"NSE:{s}" for s in symbols]
            raw = self._kite.raw.ohlc(instruments)
            out = {}
            for sym in symbols:
                key = f"NSE:{sym}"
                if key not in raw:
                    out[sym] = _empty_quote()
                    continue
                d = raw[key]
                price = float(d.get("last_price", 0))
                prev  = float(d.get("ohlc", {}).get("close", price))
                chg   = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {"price": price, "prev_close": prev, "chg_pct": chg, "volume": 0}
            return out
        except Exception:
            return {s: _empty_quote() for s in symbols}

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        # Kite historical requires instrument tokens — fall back to yfinance for sparklines
        return _yf_history_closes(symbol, days)


class _YFinanceProvider:
    def quote(self, symbol: str) -> dict:
        return self.quotes([symbol]).get(symbol, _empty_quote())

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        out = {}
        import yfinance as yf
        for sym in symbols:
            try:
                h = yf.Ticker(sym + ".NS").history(period="5d")
                if h is None or len(h) < 2:
                    out[sym] = _empty_quote()
                    continue
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = [c[0] for c in h.columns]
                price = float(h["Close"].iloc[-1])
                prev  = float(h["Close"].iloc[-2])
                vol   = float(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                avg_v = float(h["Volume"].iloc[:-1].mean()) if "Volume" in h.columns else 1
                chg   = (price - prev) / prev * 100 if prev else 0.0
                out[sym] = {
                    "price": price, "prev_close": prev,
                    "chg_pct": chg, "volume": vol,
                    "avg_volume": avg_v,
                }
            except Exception:
                out[sym] = _empty_quote()
        return out

    def history_closes(self, symbol: str, days: int = 20) -> list[float]:
        return _yf_history_closes(symbol, days)


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


def _empty_quote() -> dict:
    return {"price": 0.0, "prev_close": 0.0, "chg_pct": 0.0, "volume": 0, "avg_volume": 1}


_provider: Optional[_KiteProvider | _YFinanceProvider] = None


def get_provider() -> _KiteProvider | _YFinanceProvider:
    """Return a cached provider. Kite if token is set, else yfinance."""
    global _provider
    if _provider is None:
        if _kite_available():
            try:
                _provider = _KiteProvider()
            except Exception:
                _provider = _YFinanceProvider()
        else:
            _provider = _YFinanceProvider()
    return _provider


def reset_provider():
    """Call this after updating .env at runtime to pick up new credentials."""
    global _provider
    _provider = None
