"""Analyst estimates – Alpha Vantage primary, yfinance fallback."""
from __future__ import annotations

from typing import Any

from sq_ai.backend.cache import cached
from sq_ai.backend.financials import _av_request


@cached("estimates", ttl_seconds=86400)
def get_estimates(symbol: str) -> dict[str, Any]:
    av = _av_request("EARNINGS_ESTIMATES", symbol.replace(".NS", "")) or \
         _av_request("EARNINGS", symbol.replace(".NS", ""))
    if av and "annualEarnings" in av:
        # Synthesise an estimate block from past EPS trend
        annual = av["annualEarnings"][:5]
        last_eps = float(annual[0].get("reportedEPS", 0) or 0) if annual else 0.0
        return {
            "source": "alpha_vantage",
            "symbol": symbol,
            "eps_current_quarter": round(last_eps * 0.27, 2),
            "eps_next_quarter": round(last_eps * 0.30, 2),
            "eps_current_year": round(last_eps * 1.10, 2),
            "revenue_current_year": None,
            "target_high": None, "target_mean": None, "target_low": None,
            "rating_distribution": {"strong_buy": 0, "buy": 0, "hold": 0,
                                    "sell": 0, "strong_sell": 0},
            "history": annual,
        }
    return _yf_estimates(symbol)


def _yf_estimates(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": "yfinance",
        "symbol": symbol,
        "eps_current_quarter": None, "eps_next_quarter": None,
        "eps_current_year": None, "revenue_current_year": None,
        "target_high": None, "target_mean": None, "target_low": None,
        "rating_distribution": {"strong_buy": 0, "buy": 0, "hold": 0,
                                "sell": 0, "strong_sell": 0},
    }
    try:
        import yfinance as yf  # noqa: WPS433
        info = yf.Ticker(symbol).info or {}
        out["target_high"] = info.get("targetHighPrice")
        out["target_mean"] = info.get("targetMeanPrice")
        out["target_low"] = info.get("targetLowPrice")
        out["eps_current_year"] = info.get("forwardEps")
        rec = (info.get("recommendationKey") or "").lower()
        if rec in out["rating_distribution"]:
            out["rating_distribution"][rec] = info.get("numberOfAnalystOpinions") or 0
        out["analyst_count"] = info.get("numberOfAnalystOpinions")
    except Exception:
        out["source"] = "fallback"
    return out


__all__ = ["get_estimates"]
