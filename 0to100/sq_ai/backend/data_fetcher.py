"""Market-data fetchers.

Minimal, dependency-light wrappers around:
  • KiteConnect REST  (live OHLC + LTP)
  • yfinance          (fallback / NSE500 history)
  • NewsAPI           (top headlines for sentiment)

All public functions are synchronous-friendly; FastAPI routes wrap them
with ``run_in_executor`` to avoid blocking the event loop.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests


# ── yfinance --------------------------------------------------------------
def fetch_yf_history(symbol: str, period: str = "1y",
                     interval: str = "1d") -> pd.DataFrame:
    """Yahoo daily OHLCV. Returns columns: open, high, low, close, volume."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed") from exc
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


# ── KiteConnect ----------------------------------------------------------
class KiteFetcher:
    """Thin Kite REST wrapper.  Falls back gracefully if SDK is absent."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("KITE_API_KEY", "")
        self.access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
        self._kite = None
        if self.api_key and self.access_token:
            try:
                from kiteconnect import KiteConnect  # noqa: WPS433
                self._kite = KiteConnect(api_key=self.api_key)
                self._kite.set_access_token(self.access_token)
            except Exception as exc:  # pragma: no cover
                print(f"[KiteFetcher] init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._kite is not None

    def ltp(self, instruments: list[str]) -> dict[str, float]:
        """instruments are like ['NSE:RELIANCE','NSE:TCS']."""
        if not self.available:
            return {}
        try:
            data = self._kite.ltp(instruments)
            return {k: float(v["last_price"]) for k, v in data.items()}
        except Exception as exc:  # pragma: no cover
            print(f"[KiteFetcher.ltp] {exc}")
            return {}

    def historical(self, instrument_token: int, frm: datetime,
                   to: datetime, interval: str = "day") -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()
        try:
            data = self._kite.historical_data(instrument_token, frm, to, interval)
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("date", inplace=True)
            return df
        except Exception as exc:  # pragma: no cover
            print(f"[KiteFetcher.historical] {exc}")
            return pd.DataFrame()


# ── NewsAPI --------------------------------------------------------------
def fetch_news(query: str, top_n: int = 3,
               api_key: str | None = None) -> list[dict[str, Any]]:
    api_key = api_key or os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": top_n,
        "apiKey": api_key,
    }
    backoff = 1.0
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            arts = r.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                    "url": a.get("url", ""),
                    "publishedAt": a.get("publishedAt", ""),
                }
                for a in arts[:top_n]
            ]
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return []


# Liquid NSE large-caps for default watchlist (no external call needed).
DEFAULT_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
]


def nse500_symbols() -> list[str]:
    """Pre-baked, hardcoded list (top liquid names).  For the full NSE500
    list, the Colab notebook downloads it via yfinance / NSE archives."""
    return DEFAULT_WATCHLIST + [
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    ]
