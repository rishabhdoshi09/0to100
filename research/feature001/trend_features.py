"""Canonical Trend Template feature vector (trend_features_v1).

Same SMA / 52-week arithmetic as research.sepa.trend.evaluate_trend.
Exposes atomic flags and continuous distances. Does not collapse to one
Stage-2 bit. RS is not part of this vector.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from research.feature001.constants import TREND_VERSION, trend_bucket
from research.sepa.config import DEFAULT_CONFIG, SepaConfig
from research.sepa.frames import close_series, sma


def _sma_slope_pct(close: pd.Series, window: int, lookback: int) -> float | None:
    if close is None or len(close) < window + lookback:
        return None
    now = sma(close, window)
    prev = sma(close.iloc[:-lookback], window)
    if now is None or prev is None or prev == 0:
        return None
    return (float(now) / float(prev) - 1.0) * 100.0


def _pct_above(price: float, level: float | None) -> float | None:
    if level is None or level == 0:
        return None
    return (price / float(level) - 1.0) * 100.0


def compute_trend_features(frame, config: SepaConfig | None = None) -> dict[str, Any]:
    """PIT feature vector from bars already sliced to as-of (no future)."""
    cfg = config or DEFAULT_CONFIG
    empty = {
        "version": TREND_VERSION,
        "available": False,
        "price_gt_sma50": None,
        "price_gt_sma150": None,
        "price_gt_sma200": None,
        "sma50_gt_sma150": None,
        "sma50_gt_sma200": None,
        "sma150_gt_sma200": None,
        "sma200_rising": None,
        "dist_above_52w_low_pct": None,
        "dist_from_52w_high_pct": None,
        "structure_pass": None,
        "n_structure_passed": None,
        "pct_above_sma50": None,
        "pct_above_sma150": None,
        "pct_above_sma200": None,
        "sma50_slope_pct": None,
        "sma150_slope_pct": None,
        "sma200_slope_pct": None,
        "ma_spread_50_200_pct": None,
        "trend_bucket": None,
        "price": None,
        "sma50": None,
        "sma150": None,
        "sma200": None,
        "high_52w": None,
        "low_52w": None,
    }
    close = close_series(frame)
    if close is None or frame is None or len(frame) == 0:
        return empty

    price = float(close.iloc[-1])
    high_col = frame["high"] if "high" in frame.columns else close
    low_col = frame["low"] if "low" in frame.columns else close
    win = min(int(cfg.high_low_lookback), len(frame))
    high_52w = float(pd.to_numeric(high_col, errors="coerce").tail(win).max())
    low_52w = float(pd.to_numeric(low_col, errors="coerce").tail(win).min())
    s50 = sma(close, cfg.sma50)
    s150 = sma(close, cfg.sma150)
    s200 = sma(close, cfg.sma200)
    s200_prev = None
    need_slope = cfg.sma200 + cfg.sma200_slope_lookback
    if len(close) >= need_slope:
        s200_prev = sma(close.iloc[: -cfg.sma200_slope_lookback], cfg.sma200)

    dist_low = ((price / low_52w - 1.0) * 100.0) if low_52w > 0 else None
    dist_high = ((1.0 - price / high_52w) * 100.0) if high_52w > 0 else None

    price_gt_50 = None if s50 is None else bool(price > s50)
    price_gt_150 = None if s150 is None else bool(price > s150)
    price_gt_200 = None if s200 is None else bool(price > s200)
    sma50_gt_150 = None if s50 is None or s150 is None else bool(s50 > s150)
    sma50_gt_200 = None if s50 is None or s200 is None else bool(s50 > s200)
    sma150_gt_200 = None if s150 is None or s200 is None else bool(s150 > s200)
    sma200_rising = None if s200 is None or s200_prev is None else bool(s200 > s200_prev)
    off_low = None if dist_low is None else bool(dist_low >= cfg.off_52w_low_pct)
    near_high = None if dist_high is None else bool(dist_high <= cfg.near_52w_high_pct)

    # Original 7-rule structure AND-gate (RS excluded) — same as evaluate_trend.
    structure = [
        None if price_gt_150 is None or price_gt_200 is None else (price_gt_150 and price_gt_200),
        sma150_gt_200,
        sma200_rising,
        None if sma50_gt_150 is None or sma50_gt_200 is None else (sma50_gt_150 and sma50_gt_200),
        price_gt_50,
        off_low,
        near_high,
    ]
    known = [x for x in structure if x is not None]
    n_passed = sum(1 for x in structure if x is True)
    structure_pass = len(structure) == 7 and all(x is True for x in structure)

    spread = None
    if s50 is not None and s200 is not None and s200 != 0:
        spread = (float(s50) / float(s200) - 1.0) * 100.0
    sl50 = _sma_slope_pct(close, cfg.sma50, cfg.sma200_slope_lookback)
    sl150 = _sma_slope_pct(close, cfg.sma150, cfg.sma200_slope_lookback)
    sl200 = _sma_slope_pct(close, cfg.sma200, cfg.sma200_slope_lookback)
    pa50 = _pct_above(price, s50)
    pa150 = _pct_above(price, s150)
    pa200 = _pct_above(price, s200)

    return {
        "version": TREND_VERSION,
        "available": len(known) >= 5,
        "price_gt_sma50": price_gt_50,
        "price_gt_sma150": price_gt_150,
        "price_gt_sma200": price_gt_200,
        "sma50_gt_sma150": sma50_gt_150,
        "sma50_gt_sma200": sma50_gt_200,
        "sma150_gt_sma200": sma150_gt_200,
        "sma200_rising": sma200_rising,
        "dist_above_52w_low_pct": None if dist_low is None else round(dist_low, 4),
        "dist_from_52w_high_pct": None if dist_high is None else round(dist_high, 4),
        "structure_pass": structure_pass,
        "n_structure_passed": int(n_passed),
        "pct_above_sma50": None if pa50 is None else round(pa50, 4),
        "pct_above_sma150": None if pa150 is None else round(pa150, 4),
        "pct_above_sma200": None if pa200 is None else round(pa200, 4),
        "sma50_slope_pct": None if sl50 is None else round(sl50, 4),
        "sma150_slope_pct": None if sl150 is None else round(sl150, 4),
        "sma200_slope_pct": None if sl200 is None else round(sl200, 4),
        "ma_spread_50_200_pct": None if spread is None else round(spread, 4),
        "trend_bucket": trend_bucket(structure_pass, n_passed),
        "price": round(price, 4),
        "sma50": None if s50 is None else round(s50, 4),
        "sma150": None if s150 is None else round(s150, 4),
        "sma200": None if s200 is None else round(s200, 4),
        "high_52w": round(high_52w, 4) if high_52w == high_52w else None,
        "low_52w": round(low_52w, 4) if low_52w == low_52w else None,
        "off_52w_low_ok": off_low,
        "near_52w_high_ok": near_high,
    }
