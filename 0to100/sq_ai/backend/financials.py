"""Fundamentals + financials.

Source order: Alpha Vantage → yfinance → deterministic synthesised fallback.
All results cached for 24 h via ``cache.kv_cache``.
"""
from __future__ import annotations

import os
from typing import Any

from sq_ai.backend.cache import cached


_AV_BASE = "https://www.alphavantage.co/query"


def _av_key() -> str:
    return os.environ.get("ALPHA_VANTAGE_KEY", "") or os.environ.get(
        "ALPHA_VANTAGE_API_KEY", ""
    )


# ─── Alpha Vantage helpers ──────────────────────────────────────────────────
def _av_request(function: str, symbol: str) -> dict | None:
    key = _av_key()
    if not key or "REPLACE" in key:
        return None
    try:
        import requests  # noqa: WPS433
        r = requests.get(_AV_BASE,
                         params={"function": function, "symbol": symbol,
                                 "apikey": key},
                         timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or "Note" in data or "Information" in data:
            return None
        return data
    except Exception:
        return None


# ─── yfinance helpers ───────────────────────────────────────────────────────
def _yf_info(symbol: str) -> dict[str, Any] | None:
    try:
        import yfinance as yf  # noqa: WPS433
        t = yf.Ticker(symbol)
        info = t.info or {}
        return info if info else None
    except Exception:
        return None


def _yf_financials(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {"income": [], "balance": [], "cashflow": []}
    try:
        import yfinance as yf  # noqa: WPS433
        t = yf.Ticker(symbol)
        for attr, key in (("income_stmt", "income"),
                          ("balance_sheet", "balance"),
                          ("cashflow", "cashflow")):
            df = getattr(t, attr, None)
            if df is None or getattr(df, "empty", True):
                continue
            df = df.fillna(0)
            for col in df.columns[:5]:                  # last 5 yrs
                row = {"period": str(col.date() if hasattr(col, "date") else col)}
                for idx in df.index:
                    row[str(idx)] = float(df[col][idx])
                out[key].append(row)
    except Exception:
        pass
    return out


# ─── public API ────────────────────────────────────────────────────────────
@cached("ratios", ttl_seconds=86400)
def get_ratios(symbol: str) -> dict[str, Any]:
    """Key valuation/profitability ratios."""
    av = _av_request("OVERVIEW", symbol.replace(".NS", ""))
    if av:
        return {
            "source": "alpha_vantage",
            "symbol": symbol,
            "name": av.get("Name"),
            "sector": av.get("Sector"),
            "market_cap": _to_float(av.get("MarketCapitalization")),
            "pe": _to_float(av.get("PERatio")),
            "pb": _to_float(av.get("PriceToBookRatio")),
            "roe": _to_float(av.get("ReturnOnEquityTTM")),
            "debt_to_equity": _to_float(av.get("DebtEquityRatio")) or _to_float(av.get("DebtToEquityRatio")),
            "ev_ebitda": _to_float(av.get("EVToEBITDA")),
            "dividend_yield": _to_float(av.get("DividendYield")),
            "eps": _to_float(av.get("EPS")),
            "52w_high": _to_float(av.get("52WeekHigh")),
            "52w_low": _to_float(av.get("52WeekLow")),
        }
    info = _yf_info(symbol) or {}
    return {
        "source": "yfinance" if info else "fallback",
        "symbol": symbol,
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "dividend_yield": info.get("dividendYield"),
        "eps": info.get("trailingEps"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }


@cached("financials", ttl_seconds=86400)
def get_financials(symbol: str) -> dict[str, Any]:
    """Annual P&L, balance sheet, cash flow."""
    return {"symbol": symbol, **_yf_financials(symbol)}


@cached("quarterly", ttl_seconds=86400)
def get_quarterly(symbol: str) -> list[dict]:
    """Last 8 quarterly results."""
    out: list[dict] = []
    try:
        import yfinance as yf  # noqa: WPS433
        t = yf.Ticker(symbol)
        df = t.quarterly_income_stmt
        if df is None or df.empty:
            return []
        df = df.fillna(0)
        for col in df.columns[:8]:
            row: dict[str, Any] = {"period": str(col.date() if hasattr(col, "date") else col)}
            for idx in df.index:
                row[str(idx)] = float(df[col][idx])
            out.append(row)
    except Exception:
        pass
    return out


def _to_float(x: Any) -> float | None:
    try:
        return float(x) if x not in (None, "None", "-", "") else None
    except (TypeError, ValueError):
        return None


__all__ = ["get_ratios", "get_financials", "get_quarterly"]
