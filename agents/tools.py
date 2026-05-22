"""
Tools that agents can invoke – wrappers around existing 0to100 modules.
All functions return plain dicts so they can be JSON-serialised for LLM prompts.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from features.indicators import IndicatorEngine

_ie = IndicatorEngine()


def _fetch_ohlcv(symbol: str, days: int = 100) -> pd.DataFrame | None:
    """Fetch OHLCV. Kite primary, yfinance fallback."""
    try:
        from data.market_data import get_historical_data
        from datetime import datetime, timedelta
        from_date = (datetime.today() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
        df = get_historical_data(symbol, interval="day", from_date=from_date)
        if df is not None and not df.empty:
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[needed].dropna()
    except Exception:
        pass

    # Last resort: yfinance
    try:
        import yfinance as yf
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


def _get_live_price(symbol: str) -> float | None:
    """Return live LTP from Kite or yfinance fast_info. Never returns stale historical close."""
    try:
        from data.market_data import get_provider
        q = get_provider().quote(symbol)
        price = q.get("price", 0.0)
        if price > 0:
            return price
    except Exception:
        pass
    try:
        import yfinance as yf
        fi = getattr(yf.Ticker(f"{symbol}.NS"), "fast_info", None)
        if fi and getattr(fi, "last_price", None):
            return float(fi.last_price)
    except Exception:
        pass
    return None


def get_technical_indicators(symbol: str, days: int = 100) -> Dict[str, Any]:
    """Compute SMA/EMA, RSI, ATR, volatility, momentum, volume for a symbol.
    Live LTP is injected so the LLM sees the actual current price, not yesterday's close.
    """
    df = _fetch_ohlcv(symbol, days)
    if df is None:
        return {"error": f"No data for {symbol}", "symbol": symbol}

    result = _ie.compute(df, symbol=symbol)

    # Override price with live LTP so JARVIS never quotes stale prices
    live = _get_live_price(symbol)
    if live and live > 0:
        result["price"] = live
        result["ltp"] = live
        result["price_source"] = "live"
    else:
        result["price_source"] = "historical_close"

    return result


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Return PE, PB, ROE, market cap. Tries screener cache, falls back to yfinance."""
    try:
        from fundamentals.fetcher import get_deep_fundamentals
        data = get_deep_fundamentals(symbol, force_refresh=False)
        if data:
            return {"symbol": symbol, **data}
    except Exception:
        pass

    try:
        import yfinance as yf
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
