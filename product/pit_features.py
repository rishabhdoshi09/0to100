"""Point-in-time technical features from official bars, for the learning record.

The challenger's feature schema (research.feature_schema) names 25 features,
but recommendation cards and saved scan records carry only rsi, volume_ratio
and quality_score (reward_risk is derived). The other features had no producer
anywhere, so every frozen DECISION observation stored them as missing: owner's
Mac 2026-11-18, 21 of 25 challenger features were constant zero after median
imputation, and the model learned from four inputs.

This module computes the features that are defined by a stock's own official
OHLCV, using ONLY bars that existed at decision time:

* decided at or after the 15:30 IST close  -> bars up to the decision date;
* decided before the close                  -> bars up to the previous session.

A missing or short history yields no feature (never an invented value). The
values follow research.feature_schema's definitions. Cross-sectional and
external features (rel_strength, rs/sector/volatility percentiles, delivery,
breadth, VIX, base geometry) are NOT computed here.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping

_IST = timezone(timedelta(hours=5, minutes=30))
_CLOSE = time(15, 30)
MIN_BARS = 21

PRODUCED = (
    "atr_pct",
    "adr_pct",
    "dist_from_high_pct",
    "clv",
    "volume_z",
    "liquidity_cr",
    "dist_20ema_pct",
    "dist_50dma_pct",
)


def cutoff_session(decided_at: Any) -> date | None:
    """Last session whose bar was complete when the decision was made."""
    text = str(decided_at or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)  # naive timestamps in this codebase are IST
    local = dt.astimezone(_IST)
    return local.date() if local.time() >= _CLOSE else local.date() - timedelta(days=1)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def technical_features(symbol: str, decided_at: Any, frame=None) -> dict[str, float]:
    """PIT features for ``symbol`` as of ``decided_at``. {} when not computable."""
    cut = cutoff_session(decided_at)
    if cut is None:
        return {}
    if frame is None:
        try:
            from data.bhavcopy_store import get_ohlcv
            frame = get_ohlcv(str(symbol or "").upper())
        except Exception:
            return {}
    if frame is None or getattr(frame, "empty", True):
        return {}
    import pandas as pd

    cols = {c.lower(): c for c in frame.columns}
    if not {"high", "low", "close"} <= set(cols):
        return {}
    cutoff = pd.Timestamp(cut)
    tz = getattr(frame.index, "tz", None)
    if tz is not None:
        cutoff = cutoff.tz_localize(tz)
    df = frame[frame.index <= cutoff + pd.Timedelta(hours=23, minutes=59)]
    if len(df) < MIN_BARS:
        return {}
    close = df[cols["close"]].astype(float)
    high = df[cols["high"]].astype(float)
    low = df[cols["low"]].astype(float)
    volume = df[cols["volume"]].astype(float) if "volume" in cols else None
    last_c, last_h, last_l = float(close.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
    if last_c <= 0:
        return {}
    out: dict[str, Any] = {}
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr_pct"] = float(true_range.iloc[-14:].mean()) / last_c * 100.0
    out["adr_pct"] = float(((high - low) / close).iloc[-20:].mean()) * 100.0
    hi = float(high.iloc[-252:].max())
    if hi > 0:
        out["dist_from_high_pct"] = max(0.0, (hi - last_c) / hi * 100.0)
    rng = last_h - last_l
    out["clv"] = ((last_c - last_l) - (last_h - last_c)) / rng if rng > 0 else 0.0
    if volume is not None:
        hist = volume.iloc[-21:-1]
        sd = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0
        if sd > 0:
            out["volume_z"] = (float(volume.iloc[-1]) - float(hist.mean())) / sd
        turnover = (close * volume).iloc[-20:]
        out["liquidity_cr"] = float(turnover.median()) / 1e7
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    if ema20 > 0:
        out["dist_20ema_pct"] = (last_c / ema20 - 1.0) * 100.0
    if len(close) >= 50:
        sma50 = float(close.iloc[-50:].mean())
        if sma50 > 0:
            out["dist_50dma_pct"] = (last_c / sma50 - 1.0) * 100.0
    return {k: round(v, 6) for k, v in ((k, _finite(v)) for k, v in out.items()) if v is not None}


def fill_missing(features: Mapping[str, Any], symbol: str, decided_at: Any, frame=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill only features the decision did not carry. Returns (features, provenance)."""
    merged = dict(features)
    missing = [k for k in PRODUCED if merged.get(k) is None]
    if not missing:
        return merged, {"filled": [], "cutoff_session": None}
    computed = technical_features(symbol, decided_at, frame=frame)
    filled = [k for k in missing if k in computed]
    for k in filled:
        merged[k] = computed[k]
    cut = cutoff_session(decided_at)
    return merged, {"filled": filled, "cutoff_session": None if cut is None else cut.isoformat(),
                    "source": "official_bhavcopy_pit"}
