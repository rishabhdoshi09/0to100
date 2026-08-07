"""
Relative Strength — computes RS score vs Nifty and vs sector.
RS 94 = top 6% of universe. Silent signal of institutional accumulation.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

# ── Module-level Nifty cache (1-hour TTL) ─────────────────────────────────────
_nifty_cache: dict = {"df": None, "ts": 0.0}
_NIFTY_CACHE_TTL = 3600  # seconds

# Sector index proxies (Yahoo Finance NSE index format)
_SECTOR_ETF = {
    "BANK":   "^NSEBANK",
    "IT":     "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "AUTO":   "^CNXAUTO",
    "FMCG":   "^CNXFMCG",
    "METAL":  "^CNXMETAL",
    "INFRA":  "^CNXINFRA",
    "ENERGY": "^CNXENERGY",
}


def _get_nifty_df() -> Optional[pd.DataFrame]:
    """Fetch Nifty50 daily OHLCV. Kite primary (token 256265), yfinance fallback."""
    global _nifty_cache
    now = time.time()
    if _nifty_cache["df"] is not None and (now - _nifty_cache["ts"]) < _NIFTY_CACHE_TTL:
        return _nifty_cache["df"]

    df = None

    # Primary: Kite
    try:
        import os
        from datetime import date, timedelta
        if os.getenv("KITE_API_KEY") and os.getenv("KITE_ACCESS_TOKEN"):
            from data.kite_client import KiteClient
            kite = KiteClient()
            to_dt   = date.today().strftime("%Y-%m-%d")
            from_dt = (date.today() - timedelta(days=150)).strftime("%Y-%m-%d")
            df = kite.get_historical(256265, from_dt, to_dt, interval="day")
            if df is not None and len(df) >= 20:
                df.columns = [c.lower() for c in df.columns]
    except Exception:
        df = None

    # Fallback: yfinance
    if df is None or len(df) < 20:
        try:
            import yfinance as yf
            raw = yf.Ticker("^NSEI").history(period="100d", interval="1d")
            if raw is not None and len(raw) >= 20:
                raw.columns = [c.lower() for c in raw.columns]
                df = raw
        except Exception:
            pass

    if df is not None and len(df) >= 20:
        _nifty_cache["df"] = df
        _nifty_cache["ts"] = now

    return _nifty_cache["df"]  # return stale cache rather than None


def _pct_return(series: pd.Series, lookback: int) -> float:
    """Compute percentage return over last `lookback` bars."""
    arr = series.values if isinstance(series, pd.Series) else series
    if len(arr) < lookback + 1:
        return 0.0
    return float(arr[-1]) / float(arr[-lookback]) - 1.0


def compute_rs_score(
    symbol: str,
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    universe_returns: Optional[list[float]] = None,
) -> float:
    """
    RS score 0-100 (percentile rank within universe).

    Steps:
    1. 3-month return of stock (~63 trading days)
    2. 3-month return of Nifty (fetched via yfinance if nifty_df is None, cached 1hr)
    3. RS_raw = stock_return / nifty_return  (>1.0 = outperforming)
    4. Normalise across universe_returns list to 0-100 percentile.
       If no universe_returns provided, uses a fixed heuristic mapping.

    Returns: float in [0, 100]
    """
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        stock_ret = _pct_return(close, 63)

        if nifty_df is None:
            nifty_df = _get_nifty_df()

        if nifty_df is not None:
            nc = nifty_df["close"] if "close" in nifty_df.columns else nifty_df["Close"]
            nifty_ret = _pct_return(nc, 63)
        else:
            nifty_ret = 0.0

        # RS_raw: ratio > 1 means outperforming
        if nifty_ret == 0.0:
            rs_raw = 1.0 + stock_ret
        else:
            rs_raw = (1.0 + stock_ret) / (1.0 + nifty_ret)

        # Percentile rank
        if universe_returns:
            try:
                from scipy.stats import percentileofscore
                score = percentileofscore(universe_returns, rs_raw, kind="rank")
            except ImportError:
                # Simple rank fallback
                below = sum(1 for r in universe_returns if r < rs_raw)
                score = below / len(universe_returns) * 100.0
        else:
            # Heuristic: rs_raw ~ 0.85–1.20 maps to 0–100
            score = (rs_raw - 0.85) / 0.35 * 100.0
            score = max(0.0, min(100.0, score))

        return round(float(score), 1)

    except Exception:
        return 50.0  # neutral fallback


def rs_vs_sector(symbol: str, sector: str, df: pd.DataFrame) -> float:
    """
    RS score 0-100 vs sector ETF proxy.
    Same logic as compute_rs_score but benchmarks against sector instead of Nifty.
    """
    try:
        import yfinance as yf
        ticker = _SECTOR_ETF.get(sector.upper())
        if not ticker:
            return 50.0

        sector_df = yf.Ticker(ticker).history(period="100d", interval="1d")
        if sector_df is None or len(sector_df) < 20:
            return 50.0
        sector_df.columns = [c.lower() for c in sector_df.columns]
        return compute_rs_score(symbol, df, nifty_df=sector_df)
    except Exception:
        return 50.0


def rs_trend(symbol: str, df: pd.DataFrame) -> str:
    """
    Is RS improving? Compares 1-month RS vs 3-month RS.
    Returns 'RISING' / 'FLAT' / 'FALLING'.
    """
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        nifty_df = _get_nifty_df()
        if nifty_df is None:
            return "FLAT"
        nc = nifty_df["close"] if "close" in nifty_df.columns else nifty_df["Close"]

        ret_3m_stock  = _pct_return(close, 63)
        ret_1m_stock  = _pct_return(close, 21)
        ret_3m_nifty  = _pct_return(nc, 63)
        ret_1m_nifty  = _pct_return(nc, 21)

        rs_3m = (1.0 + ret_3m_stock) / (1.0 + ret_3m_nifty) if ret_3m_nifty != 0 else 1.0 + ret_3m_stock
        rs_1m = (1.0 + ret_1m_stock) / (1.0 + ret_1m_nifty) if ret_1m_nifty != 0 else 1.0 + ret_1m_stock

        diff = rs_1m - rs_3m
        if diff > 0.02:
            return "RISING"
        elif diff < -0.02:
            return "FALLING"
        else:
            return "FLAT"
    except Exception:
        return "FLAT"


def get_rs_badge(score: float) -> str:
    """
    Returns an HTML badge colored by RS score.
    score >= 90: green star badge
    score >= 70: amber badge
    score <  50: red badge
    """
    score_int = int(round(score))
    if score >= 90:
        return (
            f"<span style='color:#00d4a0;font-weight:700'>"
            f"RS {score_int} &#11088;</span>"
        )
    elif score >= 70:
        return (
            f"<span style='color:#f59e0b;font-weight:700'>"
            f"RS {score_int}</span>"
        )
    else:
        return (
            f"<span style='color:#ff4b4b'>"
            f"RS {score_int}</span>"
        )
