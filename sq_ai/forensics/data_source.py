"""
Fundamental data fetcher for the forensics module.

Uses yfinance (free, no credentials) so the analyst works standalone —
Kite is intraday-only and provides no financial statements.

NSE symbols from the SimpleQuant universe are resolved by appending `.NS`;
explicit tickers (containing a dot or a known US symbol) pass through.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from logger import get_logger
from forensics.models import FundamentalData

log = get_logger("forensics.data")

NSE_BENCHMARK = "^NSEI"   # NIFTY 50
US_BENCHMARK = "^GSPC"


def resolve_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    if "." in s or s.startswith("^"):
        return s
    # Default exchange for this platform is NSE
    return f"{s}.NS"


def _safe(getter, default=None):
    try:
        v = getter()
        if isinstance(v, pd.DataFrame) and v.empty:
            return default
        return v
    except Exception:
        return default


def fetch_fundamentals(symbol: str, price_years: int = 5) -> FundamentalData:
    """Fetch all raw data needed by the analysis layers. Raises on total failure."""
    import yfinance as yf

    ticker = resolve_ticker(symbol)
    log.info("fetching_fundamentals", symbol=symbol, ticker=ticker)
    tk = yf.Ticker(ticker)

    income = _safe(lambda: tk.income_stmt, pd.DataFrame())
    # NSE default didn't resolve — fall back to the raw symbol (US listings etc.)
    if (income is None or income.empty) and ticker != symbol.strip().upper():
        ticker = symbol.strip().upper()
        log.info("falling_back_to_raw_ticker", ticker=ticker)
        tk = yf.Ticker(ticker)
        income = _safe(lambda: tk.income_stmt, pd.DataFrame())
    balance = _safe(lambda: tk.balance_sheet, pd.DataFrame())
    cashflow = _safe(lambda: tk.cashflow, pd.DataFrame())
    q_income = _safe(lambda: tk.quarterly_income_stmt, pd.DataFrame())
    info = _safe(lambda: tk.info, {}) or {}

    prices = _safe(lambda: tk.history(period=f"{price_years}y", auto_adjust=True),
                   pd.DataFrame())
    if prices is None:
        prices = pd.DataFrame()

    if (income is None or income.empty) and prices.empty:
        raise RuntimeError(
            f"No data available for '{ticker}'. Check the symbol "
            "(NSE symbols are auto-suffixed with .NS; pass e.g. AAPL for US)."
        )

    bench_sym = NSE_BENCHMARK if ticker.endswith(".NS") else US_BENCHMARK
    benchmark = _safe(
        lambda: yf.Ticker(bench_sym).history(period=f"{price_years}y", auto_adjust=True),
        pd.DataFrame(),
    )
    if benchmark is None:
        benchmark = pd.DataFrame()

    return FundamentalData(
        symbol=symbol.upper(),
        ticker=ticker,
        info=info,
        income=income if income is not None else pd.DataFrame(),
        balance=balance if balance is not None else pd.DataFrame(),
        cashflow=cashflow if cashflow is not None else pd.DataFrame(),
        quarterly_income=q_income if q_income is not None else pd.DataFrame(),
        prices=prices,
        benchmark=benchmark,
        insider_transactions=_safe(lambda: tk.insider_transactions),
        institutional_holders=_safe(lambda: tk.institutional_holders),
        major_holders=_safe(lambda: tk.major_holders),
    )
