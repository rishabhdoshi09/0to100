"""Stock-research aggregator.

Pulls every section needed by the research-hub UI (header, technicals,
fundamentals, financials, estimates, shareholding, actions, news, peers)
and returns a single dict.  Each upstream call is cached and resilient.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from sq_ai.backend.analyst_estimates import get_estimates
from sq_ai.backend.cache import cached
from sq_ai.backend.corporate_actions import get_actions
from sq_ai.backend.data_fetcher import fetch_news, fetch_yf_history
from sq_ai.backend.financials import get_financials, get_quarterly, get_ratios
from sq_ai.backend.screener_engine import _technical_features
from sq_ai.backend.shareholding import get_shareholding
from sq_ai.signals.composite_signal import CompositeSignal


# Coarse sector → peers mapping (extend as needed)
DEFAULT_PEERS: dict[str, list[str]] = {
    "Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS",
                           "KOTAKBANK.NS", "SBIN.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
                           "BRITANNIA.NS", "DABUR.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
             "EICHERMOT.NS"],
}


def _signal_label(score: float) -> str:
    if score >= 0.5:
        return "STRONG BUY"
    if score >= 0.15:
        return "BUY"
    if score >= -0.15:
        return "HOLD"
    if score >= -0.5:
        return "SELL"
    return "STRONG SELL"


def _peers_for(sector: str | None, exclude: str) -> list[str]:
    if not sector:
        return []
    return [s for s in DEFAULT_PEERS.get(sector, []) if s != exclude][:4]


@cached("price_history", ttl_seconds=1800)
def _history(symbol: str, period: str = "1y") -> list[dict]:
    df = fetch_yf_history(symbol, period=period, interval="1d")
    if df is None or df.empty:
        return []
    df = df.tail(260)
    return [
        {"date": str(idx.date() if hasattr(idx, "date") else idx),
         "open": float(row["open"]), "high": float(row["high"]),
         "low": float(row["low"]), "close": float(row["close"]),
         "volume": float(row["volume"])}
        for idx, row in df.iterrows()
    ]


def header(symbol: str) -> dict[str, Any]:
    df = fetch_yf_history(symbol, period="1y", interval="1d")
    if df is None or df.empty:
        return {"symbol": symbol, "error": "no data"}
    feat = _technical_features(df)
    ratios = get_ratios(symbol) or {}
    last_close_prev = float(df["close"].iloc[-2]) if len(df) >= 2 else feat["price"]
    return {
        "symbol": symbol,
        "name": ratios.get("name"),
        "sector": ratios.get("sector"),
        "price": feat["price"],
        "change": feat["price"] - last_close_prev,
        "change_pct": (feat["price"] - last_close_prev) / last_close_prev
                      if last_close_prev else 0,
        "market_cap": ratios.get("market_cap"),
        "high_52w": feat["high_52w"], "low_52w": feat["low_52w"],
        "volume": feat["volume"], "volume_avg_20": feat["volume_avg_20"],
    }


def technicals(symbol: str,
               composite: CompositeSignal | None = None) -> dict[str, Any]:
    composite = composite or CompositeSignal()
    df = fetch_yf_history(symbol, period="1y", interval="1d")
    if df is None or len(df) < 60:
        return {"error": "no data"}
    feats = composite.compute_indicators(df)
    sig = composite.compute(feats)
    atr_v = feats["atr"]
    price = feats["close"]
    return {
        "indicators": feats,
        "signal": sig,
        "label": _signal_label(sig["signal"]),
        "key_levels": {
            "support": float(df["low"].tail(20).min()),
            "resistance": float(df["high"].tail(20).max()),
            "stop": price - 2 * atr_v,
            "target": price + 3 * atr_v,
        },
    }


def news_section(symbol: str, top_n: int = 10) -> list[dict]:
    name_query = symbol.replace(".NS", "")
    return fetch_news(name_query, top_n=top_n)


def peers_compare(symbol: str) -> list[dict]:
    sector = (get_ratios(symbol) or {}).get("sector")
    syms = _peers_for(sector, exclude=symbol)
    out: list[dict] = []
    for s in [symbol, *syms]:
        r = get_ratios(s) or {}
        out.append({
            "symbol": s,
            "name": r.get("name"),
            "pe": r.get("pe"),
            "pb": r.get("pb"),
            "roe": r.get("roe"),
            "debt_to_equity": r.get("debt_to_equity"),
            "market_cap": r.get("market_cap"),
            "dividend_yield": r.get("dividend_yield"),
        })
    return out


def full_profile(symbol: str) -> dict[str, Any]:
    """Single-shot blob used by ``GET /api/stock/profile/{symbol}``."""
    return {
        "symbol": symbol,
        "header": header(symbol),
        "technicals": technicals(symbol),
        "history": _history(symbol),
        "ratios": get_ratios(symbol),
        "financials": get_financials(symbol),
        "quarterly": get_quarterly(symbol),
        "estimates": get_estimates(symbol),
        "shareholding": get_shareholding(symbol),
        "actions": get_actions(symbol),
        "news": news_section(symbol),
        "peers": peers_compare(symbol),
    }


__all__ = ["full_profile", "header", "technicals", "news_section",
           "peers_compare", "DEFAULT_PEERS"]
