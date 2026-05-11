"""
Tools that agents can invoke – wrappers around existing 0to100 modules.
All functions return plain dicts so they can be JSON-serialised for LLM prompts.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

from features.indicators import IndicatorEngine

_ie = IndicatorEngine()


def _fetch_ohlcv(symbol: str, days: int = 100) -> pd.DataFrame | None:
    """Fetch OHLCV via yfinance (Kite-free fallback for agent context)."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{days}d")
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def get_technical_indicators(symbol: str, days: int = 100) -> Dict[str, Any]:
    """Compute SMA/EMA, RSI, ATR, volatility, momentum, volume for a symbol."""
    df = _fetch_ohlcv(symbol, days)
    if df is None:
        return {"error": f"No data for {symbol}", "symbol": symbol}
    return _ie.compute(df, symbol=symbol)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Return PE, PB, ROE, market cap.  Tries screener cache, falls back to yfinance."""
    try:
        from fundamentals.fetcher import get_deep_fundamentals
        data = get_deep_fundamentals(symbol, force_refresh=False)
        if data:
            return {"symbol": symbol, **data}
    except Exception:
        pass

    try:
        info = yf.Ticker(f"{symbol}.NS").info
        return {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap_cr": round(info.get("marketCap", 0) / 1e7, 1),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception as exc:
        return {"error": str(exc), "symbol": symbol}


def get_recent_news(symbol: str, days: int = 7) -> List[Dict]:
    """Fetch recent news headlines filtered for the symbol."""
    try:
        from news.fetcher import NewsFetcher
        fetcher = NewsFetcher()
        articles = fetcher.fetch_all(max_age_hours=min(days * 24, 168))
        sym_lower = symbol.lower()
        relevant = [
            a.to_dict() for a in articles
            if sym_lower in a.headline.lower() or sym_lower in a.summary.lower()
        ]
        # Fall back to top general news if nothing symbol-specific
        if not relevant:
            relevant = [a.to_dict() for a in articles[:5]]
        return relevant[:10]
    except Exception as exc:
        return [{"error": str(exc), "headline": "News unavailable"}]


def get_portfolio_state() -> Dict[str, Any]:
    """Return current open positions and P&L from portfolio state."""
    try:
        from portfolio.state import PortfolioState
        ps = PortfolioState()
        return {
            "positions": getattr(ps, "positions", {}),
            "total_pnl": getattr(ps, "total_pnl", 0.0),
            "open_positions_count": len(getattr(ps, "positions", {})),
        }
    except Exception:
        return {"positions": {}, "total_pnl": 0.0, "open_positions_count": 0}
