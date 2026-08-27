"""
Sector Rotation Acceleration — not which sector is leading,
but which is ACCELERATING. Early rotation = bigger opportunity.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

# Kite instrument tokens for NSE sector indices
_SECTOR_KITE_TOKENS: dict[str, int] = {
    "Banking":  260105,   # NIFTY BANK
    "IT":       259849,   # NIFTY IT
    "Auto":     258049,   # NIFTY AUTO
    "Pharma":   261249,   # NIFTY PHARMA
    "FMCG":     257801,   # NIFTY FMCG
    "Metal":    234249,   # NIFTY METAL
    "Realty":   261633,   # NIFTY REALTY
    "Energy":   258921,   # NIFTY ENERGY
}

# yfinance fallback symbols
_SECTOR_ETFS: dict[str, str] = {
    "Banking":  "^NSEBANK",
    "IT":       "^CNXIT",
    "Auto":     "^CNXAUTO",
    "Pharma":   "^CNXPHARMA",
    "FMCG":     "^CNXFMCG",
    "Metal":    "^CNXMETAL",
    "Realty":   "^CNXREALTY",
    "Energy":   "^CNXENERGY",
}


def _fetch_sector_kite(sector: str, weeks: int) -> Optional[pd.Series]:
    """Fetch weekly sector index closes from Kite. Returns None on failure."""
    token = _SECTOR_KITE_TOKENS.get(sector)
    if not token:
        return None
    try:
        import os
        if not (os.getenv("KITE_API_KEY") and os.getenv("KITE_ACCESS_TOKEN")):
            return None
        from data.kite_client import KiteClient
        kite = KiteClient()
        to_dt   = date.today().strftime("%Y-%m-%d")
        from_dt = (date.today() - timedelta(weeks=weeks + 4)).strftime("%Y-%m-%d")
        df = kite.get_historical(token, from_dt, to_dt, interval="week")
        if df is None or df.empty or len(df) < 2:
            return None
        col = "close" if "close" in df.columns else df.columns[-1]
        return df[col]
    except Exception:
        return None


def _fetch_sector_data(lookback_weeks: int = 4) -> dict[str, pd.Series]:
    """Fetch weekly closing prices for each sector. Kite primary, yfinance fallback."""
    import yfinance as yf

    result: dict[str, pd.Series] = {}
    period = f"{lookback_weeks + 2}wk"

    for sector, yf_ticker in _SECTOR_ETFS.items():
        # Primary: Kite
        series = _fetch_sector_kite(sector, lookback_weeks)
        if series is not None and len(series) >= 2:
            result[sector] = series
            continue
        # Fallback: yfinance
        try:
            hist = yf.Ticker(yf_ticker).history(period=period, interval="1wk")
            if hist is not None and not hist.empty and len(hist) >= 2:
                result[sector] = hist["Close"]
        except Exception:
            pass

    return result


def _classify_stage(ret_1w: float, ret_4w: float, acceleration: float) -> str:
    """
    Classify the rotation stage based on returns and acceleration.

    EARLY:    Accelerating from negative (turning the corner)
    TRENDING: Accelerating from positive (momentum building)
    MATURE:   Decelerating from positive (momentum fading)
    TOPPING:  Decelerating from high positive (distribution likely)
    """
    avg_4w_weekly = ret_4w / 4.0

    if acceleration > 0:
        # Accelerating
        if avg_4w_weekly <= 0:
            return "EARLY"       # Was negative, now turning up
        else:
            return "TRENDING"    # Was positive, getting stronger
    else:
        # Decelerating
        if ret_4w > 5.0:
            return "TOPPING"     # Was very strong, now fading — distribution risk
        else:
            return "MATURE"      # Moderate, losing steam


def compute_rotation_matrix(lookback_weeks: int = 4) -> dict:
    """
    Compute sector rotation acceleration matrix.

    Returns a dict keyed by sector with:
        return_1w, return_4w, acceleration, stage
    Sorted by acceleration descending.
    """
    sector_data = _fetch_sector_data(lookback_weeks)
    results: dict[str, dict] = {}

    for sector, closes in sector_data.items():
        try:
            closes = closes.dropna()
            if len(closes) < 2:
                continue

            # Last available close
            last = float(closes.iloc[-1])

            # 1-week return: last close vs one-period-ago
            prev_1w = float(closes.iloc[-2]) if len(closes) >= 2 else last
            ret_1w = ((last - prev_1w) / prev_1w * 100.0) if prev_1w else 0.0

            # 4-week return: last close vs earliest available in window
            oldest_idx = max(0, len(closes) - (lookback_weeks + 1))
            prev_4w = float(closes.iloc[oldest_idx])
            ret_4w = ((last - prev_4w) / prev_4w * 100.0) if prev_4w else 0.0

            # Acceleration: 1w return vs average weekly return over 4w
            acceleration = ret_1w - (ret_4w / lookback_weeks)

            stage = _classify_stage(ret_1w, ret_4w, acceleration)

            results[sector] = {
                "sector": sector,
                "return_1w": round(ret_1w, 2),
                "return_4w": round(ret_4w, 2),
                "acceleration": round(acceleration, 2),
                "stage": stage,
            }
        except Exception:
            continue

    # Sort by acceleration descending
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["acceleration"], reverse=True)
    )
    return sorted_results


def get_rotation_leaders() -> list[dict]:
    """Returns top 3 accelerating sectors with stage and return data."""
    matrix = compute_rotation_matrix()
    leaders = list(matrix.values())[:3]
    return leaders


def render_rotation_matrix() -> None:
    """
    Streamlit component: compact sector rotation acceleration table.

    Color coding:
        EARLY/TRENDING → green (buy zones)
        MATURE         → yellow
        TOPPING        → red (avoid)
    """
    return
