"""Declarative dynamic screener.

Filter schema (all keys optional)::

    {
      # Technical
      "rsi": {"min": 0, "max": 100},
      "price_vs_sma20": "above" | "below",
      "price_vs_sma50": "above" | "below",
      "price_vs_sma200": "above" | "below",
      "macd": "bullish" | "bearish",
      "volume": "above_avg" | "below_avg",
      "atr_pct": {"min": 0.0, "max": 0.10},
      "from_52w_high_pct": {"min": -0.5, "max": 0.0},

      # Fundamental
      "pe":  {"min": 0,  "max": 100},
      "pb":  {"min": 0,  "max": 20},
      "roe": {"min": 0.10},
      "debt_to_equity": {"max": 1.5},
      "market_cap_min_cr": 5000,
      "dividend_yield_min": 0.005,

      # Momentum
      "ret_1w_min":  0.0,
      "ret_1m_min":  0.0,
      "ret_3m_min":  0.0
    }
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from sq_ai.backend.data_fetcher import fetch_yf_history
from sq_ai.backend.financials import get_ratios
from sq_ai.signals.composite_signal import atr, rsi


def _technical_features(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 60:
        raise ValueError("need >=60 bars")
    close = df["close"]
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    rsi14 = float(rsi(close, 14).iloc[-1])
    atr14 = float(atr(df, 14).iloc[-1])
    price = float(close.iloc[-1])
    vol_avg = float(df["volume"].rolling(20).mean().iloc[-1] or 0)
    vol = float(df["volume"].iloc[-1])
    macd = (close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean())
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_state = "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish"
    high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
    return {
        "price": price, "sma_20": sma20, "sma_50": sma50, "sma_200": sma200,
        "rsi": rsi14, "atr": atr14, "atr_pct": atr14 / price if price else 0,
        "volume": vol, "volume_avg_20": vol_avg,
        "vol_ratio": vol / vol_avg if vol_avg else 1.0,
        "macd_state": macd_state,
        "high_52w": high_52w, "low_52w": low_52w,
        "from_52w_high_pct": (price - high_52w) / high_52w if high_52w else 0,
        "ret_1w": float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0,
        "ret_1m": float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else 0,
        "ret_3m": float(close.iloc[-1] / close.iloc[-66] - 1) if len(close) >= 66 else 0,
    }


def _passes_range(value: Any, rng: dict[str, float] | None) -> bool:
    if rng is None or value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return False
    if "min" in rng and value < rng["min"]:
        return False
    if "max" in rng and value > rng["max"]:
        return False
    return True


def matches(features: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True when ``features`` passes every filter."""
    if not _passes_range(features.get("rsi"), filters.get("rsi")):
        return False
    if not _passes_range(features.get("atr_pct"), filters.get("atr_pct")):
        return False
    if not _passes_range(features.get("from_52w_high_pct"),
                         filters.get("from_52w_high_pct")):
        return False
    for k in ("ret_1w", "ret_1m", "ret_3m"):
        m = filters.get(f"{k}_min")
        if m is not None and (features.get(k) or 0) < m:
            return False

    pvs = filters.get("price_vs_sma20")
    if pvs == "above" and features["price"] <= features["sma_20"]:
        return False
    if pvs == "below" and features["price"] >= features["sma_20"]:
        return False
    pvs = filters.get("price_vs_sma50")
    if pvs == "above" and features["price"] <= features["sma_50"]:
        return False
    if pvs == "below" and features["price"] >= features["sma_50"]:
        return False
    pvs = filters.get("price_vs_sma200")
    if pvs == "above" and not (features["price"] > features["sma_200"]):
        return False
    if pvs == "below" and not (features["price"] < features["sma_200"]):
        return False

    if filters.get("macd") and filters["macd"] != features["macd_state"]:
        return False
    vf = filters.get("volume")
    if vf == "above_avg" and features["vol_ratio"] <= 1.0:
        return False
    if vf == "below_avg" and features["vol_ratio"] >= 1.0:
        return False

    # Fundamentals
    if not _passes_range(features.get("pe"), filters.get("pe")):
        return False
    if not _passes_range(features.get("pb"), filters.get("pb")):
        return False
    roe_rng = filters.get("roe")
    if roe_rng and (features.get("roe") is None or features["roe"] < roe_rng.get("min", 0)):
        return False
    de = filters.get("debt_to_equity")
    if de and features.get("debt_to_equity") is not None and \
       features["debt_to_equity"] > de.get("max", float("inf")):
        return False
    if filters.get("market_cap_min_cr") and \
       (features.get("market_cap_cr") or 0) < filters["market_cap_min_cr"]:
        return False
    if filters.get("dividend_yield_min") and \
       (features.get("dividend_yield") or 0) < filters["dividend_yield_min"]:
        return False
    return True


def _score(features: dict[str, Any]) -> float:
    """Composite ranking score (higher = better)."""
    s = 0.0
    if features["price"] > features["sma_20"] > features["sma_50"]:
        s += 1.0
    if features["macd_state"] == "bullish":
        s += 0.5
    s += min(max(features["vol_ratio"] - 1, 0), 1.0) * 0.5
    s += (features.get("ret_1m") or 0) * 2
    s += min(max((features.get("rsi", 50) - 40) / 30, 0), 1) * 0.5
    return float(s)


def run_screener(symbols: list[str], filters: dict[str, Any] | None = None,
                 include_fundamentals: bool = False,
                 max_results: int = 50,
                 fetch_fn=None) -> list[dict]:
    """Run filters across ``symbols``. Returns ranked passers (top N)."""
    filters = filters or {}
    fetch_fn = fetch_fn or (lambda s: fetch_yf_history(s, period="1y", interval="1d"))
    results: list[dict] = []
    for sym in symbols:
        try:
            df = fetch_fn(sym)
            if df is None or len(df) < 60:
                continue
            feat = _technical_features(df)
            feat["symbol"] = sym
            if include_fundamentals:
                ratios = get_ratios(sym) or {}
                feat.update({
                    "pe": ratios.get("pe"),
                    "pb": ratios.get("pb"),
                    "roe": ratios.get("roe"),
                    "debt_to_equity": ratios.get("debt_to_equity"),
                    "dividend_yield": ratios.get("dividend_yield"),
                    "market_cap_cr": (ratios.get("market_cap") or 0) / 1e7,
                    "sector": ratios.get("sector"),
                    "name": ratios.get("name"),
                })
            if not matches(feat, filters):
                continue
            feat["score"] = _score(feat)
            results.append(feat)
        except Exception:
            continue
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:max_results]


__all__ = ["run_screener", "matches", "_technical_features"]
