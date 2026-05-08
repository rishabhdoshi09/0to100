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

    # Historical closes (for sparklines / charts)
    closes = mdp.history_closes("RELIANCE", days=20)  # list[float]
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


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
