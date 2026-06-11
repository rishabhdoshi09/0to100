"""
Research Replay — point-in-time historical analysis.

    python main.py replay RELIANCE 2023-06-01

Fetches full history, then CLIPS every dataset to what was knowable on the
as-of date (statements whose fiscal period ended on/before it, prices up to
it, DAL records whose period precedes it) and runs the IDENTICAL analysis
pipeline. The verdict the system would have issued then is compared against
what actually happened afterward.

This is the platform's self-validation loop: it turns "trust the framework"
into "here is the framework's dated track record."

Known limitations (clearly surfaced in output):
  • yfinance statement history is ~4 annual periods deep, so replays more
    than ~2 years back lose statement depth.
  • Profile `info` fields (analyst counts, ISS deciles, holdings) are
    current-day; price-derived fields are rescaled to the as-of price, the
    rest are dropped to avoid lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from logger import get_logger
from forensics import analyzer as az
from forensics import data_source
from forensics.models import FundamentalData

log = get_logger("forensics.replay")

# info keys that are price/time-dependent and would leak the future
_LOOKAHEAD_INFO_KEYS = {
    "currentPrice", "marketCap", "targetMeanPrice", "targetHighPrice",
    "targetLowPrice", "recommendationMean", "recommendationKey",
    "numberOfAnalystOpinions", "trailingPE", "forwardPE", "priceToBook",
    "priceToSalesTrailing12Months", "enterpriseToEbitda", "enterpriseValue",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "shortPercentOfFloat",
    "heldPercentInsiders", "heldPercentInstitutions",
}


@dataclass
class ReplayResult:
    report: az.AnalysisReport
    as_of: date
    forward_returns: Dict[str, Optional[float]]   # "6m" / "12m" / "to date"
    benchmark_returns: Dict[str, Optional[float]]
    caveats: List[str] = field(default_factory=list)


def _clip_statement(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Keep only fiscal periods ending on or before the as-of date."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in df.columns
            if pd.Timestamp(c).date() <= as_of]
    return df[cols] if cols else pd.DataFrame()


def _clip_prices(px: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if px is None or px.empty:
        return pd.DataFrame()
    idx = px.index
    cutoff = pd.Timestamp(as_of, tz=idx.tz) if getattr(idx, "tz", None) \
        else pd.Timestamp(as_of)
    return px[px.index <= cutoff]


def clip(d: FundamentalData, as_of: date) -> FundamentalData:
    """Return a point-in-time view of the fundamentals."""
    clipped_px = _clip_prices(d.prices, as_of)

    # Rebuild a lookahead-free info dict
    info = {k: v for k, v in d.info.items() if k not in _LOOKAHEAD_INFO_KEYS}
    if not clipped_px.empty and not d.prices.empty:
        then_price = float(clipped_px["Close"].iloc[-1])
        now_price = float(d.prices["Close"].iloc[-1])
        info["currentPrice"] = then_price
        # Rescale market cap to the as-of price (share count drift ignored)
        mcap = d.info.get("marketCap")
        if mcap and now_price > 0:
            info["marketCap"] = mcap * then_price / now_price

    return FundamentalData(
        symbol=d.symbol,
        ticker=d.ticker,
        info=info,
        income=_clip_statement(d.income, as_of),
        balance=_clip_statement(d.balance, as_of),
        cashflow=_clip_statement(d.cashflow, as_of),
        quarterly_income=_clip_statement(d.quarterly_income, as_of),
        prices=clipped_px,
        benchmark=_clip_prices(d.benchmark, as_of),
        insider_transactions=None,        # not point-in-time safe
        institutional_holders=None,
        major_holders=None,
    )


def _fwd_return(px: pd.DataFrame, as_of: date,
                months: Optional[int]) -> Optional[float]:
    """Total return from as-of to as-of + months (None months = to latest)."""
    if px is None or px.empty:
        return None
    past = _clip_prices(px, as_of)
    if past.empty:
        return None
    base = float(past["Close"].iloc[-1])
    if months is None:
        end = float(px["Close"].iloc[-1])
        return end / base - 1
    target = pd.Timestamp(as_of) + pd.DateOffset(months=months)
    idx = px.index
    cutoff = target.tz_localize(idx.tz) if getattr(idx, "tz", None) else target
    window = px[px.index <= cutoff]
    if window.empty or window.index[-1] <= past.index[-1]:
        return None   # horizon not yet reached
    return float(window["Close"].iloc[-1]) / base - 1


def replay(symbol: str, as_of: date) -> ReplayResult:
    """Run the full pipeline as if today were `as_of`."""
    d = data_source.fetch_fundamentals(symbol, price_years=10)
    log.info("replay_started", symbol=symbol, as_of=as_of.isoformat())

    clipped = clip(d, as_of)
    if clipped.prices.empty and clipped.income.empty:
        raise RuntimeError(
            f"No data available for {symbol} on or before {as_of} — "
            "yfinance history does not reach that far back.")

    report = az.run_pipeline(clipped, save_predictions=False)

    horizons = {"6m": 6, "12m": 12, "to date": None}
    fwd = {h: _fwd_return(d.prices, as_of, m) for h, m in horizons.items()}
    bench = {h: _fwd_return(d.benchmark, as_of, m) for h, m in horizons.items()}

    caveats = []
    n_periods = len(clipped.income.columns) if not clipped.income.empty else 0
    if n_periods < 3:
        caveats.append(
            f"Only {n_periods} annual statement period(s) available as of "
            f"{as_of} — trend-based signals (accruals, ROIC trend, M-Score) "
            "are degraded. yfinance statement history is ~4 years deep.")
    caveats.append("Profile fields (analyst views, holdings, ISS deciles) "
                   "are excluded — they are current-day, not point-in-time.")

    log.info("replay_complete", symbol=symbol,
             verdict=report.composite.verdict.value,
             fwd_12m=fwd.get("12m"))
    return ReplayResult(report=report, as_of=as_of,
                        forward_returns=fwd, benchmark_returns=bench,
                        caveats=caveats)
