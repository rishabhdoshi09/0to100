"""
Momentum / breakout screener.

Scans a curated NSE universe (tagged by market-cap bucket), pulls daily
price data via Kite Connect — the same broker connection the live engine
and backtester use — and scores every stock on momentum + breakout
strength using the same IndicatorEngine. No fundamentals fetch — this is
a fast price/volume scan, not a forensic check (pair it with
`python main.py <SYMBOL>` for that, which still uses yfinance since Kite
has no fundamentals/financial-statements API).

Requires a valid KITE_ACCESS_TOKEN in .env (run `python main.py login`
each morning before market open — Kite tokens expire daily).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from logger import get_logger
from features.indicators import IndicatorEngine
from screener.nse_universe import NSE_UNIVERSE, UniverseStock, by_bucket

log = get_logger("screener")

_engine = IndicatorEngine()


@dataclass
class ScreenResult:
    symbol: str
    name: str
    bucket: str
    sector: str
    last_close: float
    momentum_score: float          # 0-100, higher = stronger winning setup
    breakout: bool                 # near 52w high + volume surge
    pct_from_52w_high: Optional[float]
    return_1m_pct: Optional[float]
    return_3m_pct: Optional[float]
    rsi_14: Optional[float]
    volume_ratio: Optional[float]
    verdict: str                   # one-line plain-English take


def _verdict(score: float, breakout: bool, rsi: Optional[float]) -> str:
    if breakout and score >= 70:
        return "Breaking out on volume — the classic winning setup"
    if score >= 75:
        return "Strong uptrend with healthy momentum"
    if score >= 55:
        return "Building momentum, worth watching"
    if rsi is not None and rsi > 75:
        return "Hot — extended, chase with caution"
    return "No clear edge right now"


def _score_one(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty or len(df) < 30:
        return None
    df = df.rename(columns=str.lower).dropna(subset=["close"])
    if df.empty or len(df) < 30:
        return None

    ind = _engine.compute(df, symbol=symbol)
    if "error" in ind:
        return None

    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    high_52w = float(close.max())
    pct_from_high = round((last / high_52w - 1) * 100, 2) if high_52w else None

    ret_1m = ind.get("momentum_20d_pct")
    ret_3m = None
    if len(close) > 60:
        ret_3m = round((last / float(close.iloc[-61]) - 1) * 100, 2)

    rsi = ind.get("rsi_14")
    vol_ratio = ind.get("volume_ratio")
    pct_above_sma50 = ind.get("pct_above_sma50")
    golden_cross = ind.get("golden_cross")

    breakout = bool(
        pct_from_high is not None and pct_from_high >= -3.0
        and vol_ratio is not None and vol_ratio >= 1.5
    )

    # Composite momentum score (0-100): trend position + recent return +
    # volume confirmation + RSI sweet spot (strong but not blown-off-top).
    score = 50.0
    if pct_above_sma50 is not None:
        score += np.clip(pct_above_sma50, -15, 15)
    if ret_1m is not None:
        score += np.clip(ret_1m / 2, -15, 15)
    if golden_cross:
        score += 8
    if vol_ratio is not None:
        score += np.clip((vol_ratio - 1) * 10, -10, 15)
    if rsi is not None:
        if 55 <= rsi <= 75:
            score += 10
        elif rsi > 80:
            score -= 10
        elif rsi < 40:
            score -= 10
    if breakout:
        score += 10
    score = float(np.clip(score, 0, 100))

    return dict(
        last_close=last, momentum_score=round(score, 1), breakout=breakout,
        pct_from_52w_high=pct_from_high, return_1m_pct=ret_1m, return_3m_pct=ret_3m,
        rsi_14=rsi, volume_ratio=vol_ratio,
        verdict=_verdict(score, breakout, rsi),
    )


def scan(buckets: List[str] = ("Large", "Mid", "Small"),
         min_score: float = 0.0,
         breakout_only: bool = False,
         historical=None) -> List[ScreenResult]:
    """Scan the universe, filtered by market-cap bucket, and rank by momentum.

    Pulls daily candles from Kite Connect (instrument-by-instrument, rate
    limited). Pass an existing HistoricalDataFetcher to reuse a session;
    otherwise one is built from .env credentials.
    """
    stocks = by_bucket(list(buckets))
    if not stocks:
        return []
    log.info("screener_scan_started", n=len(stocks), buckets=buckets)

    if historical is None:
        from data.kite_client import KiteClient
        from data.instruments import InstrumentManager
        historical = _HistoricalAdapter(KiteClient(), InstrumentManager())

    to_date = date.today()
    from_date = to_date - timedelta(days=280)  # ~9 months of trading days

    results: List[ScreenResult] = []
    for stock in stocks:
        df = historical.fetch(
            symbol=stock.symbol,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval="day",
        )
        scored = _score_one(stock.symbol, df)
        if scored is None:
            continue
        if scored["momentum_score"] < min_score:
            continue
        if breakout_only and not scored["breakout"]:
            continue
        results.append(ScreenResult(
            symbol=stock.symbol, name=stock.name, bucket=stock.bucket,
            sector=stock.sector, **scored,
        ))

    results.sort(key=lambda r: r.momentum_score, reverse=True)
    log.info("screener_scan_complete", matched=len(results))
    return results


class _HistoricalAdapter:
    """Thin wrapper so `scan()` doesn't import data.historical at module
    load time (keeps the screener importable without Kite credentials)."""

    def __init__(self, kite, instruments) -> None:
        from data.historical import HistoricalDataFetcher
        self._fetcher = HistoricalDataFetcher(kite, instruments)

    def fetch(self, **kwargs) -> pd.DataFrame:
        return self._fetcher.fetch(**kwargs)
