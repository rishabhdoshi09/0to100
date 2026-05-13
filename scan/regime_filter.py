"""
Regime Filter — Stage 2 of the scan pipeline.

Applies regime-aware filters to the liquid universe.
Prevents scanning for momentum setups in bear markets,
or mean-reversion setups in trending environments.

Each stock is evaluated for individual trend stage (Stage 1-4)
and filtered based on what the current market regime favours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StockRegimeProfile:
    symbol: str
    price: float
    stage: str              # "Stage 1" | "Stage 2" | "Stage 3" | "Stage 4" | "Unknown"
    above_sma200: bool
    above_sma50: bool
    sma50_rising: bool
    rs_vs_nifty_5d: float   # relative performance vs Nifty, 5d
    passes: bool
    reject_reason: str = ""


# Which stock stages are eligible per market regime
_REGIME_STAGE_ALLOW: dict[str, list[str]] = {
    "TRENDING_BULL":  ["Stage 2"],
    "EXPANSION":      ["Stage 2"],
    "CHOPPY":         ["Stage 1", "Stage 2"],  # early leaders in Stage 1→2 transition
    "COMPRESSION":    ["Stage 1", "Stage 2"],
    "DISTRIBUTION":   ["Stage 3", "Stage 4"],  # only failed-breakout / short setups
    "TRENDING_BEAR":  ["Stage 3", "Stage 4"],
}


class RegimeFilter:
    """
    Filter stocks to those in regime-compatible trend stages.
    """

    def __init__(self, market_regime: str, max_workers: int = 16):
        self._regime      = market_regime
        self._max_workers = max_workers
        self._allowed     = set(_REGIME_STAGE_ALLOW.get(market_regime, ["Stage 2"]))

    def filter(self, symbols: list[str]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        passed: list[str] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._profile, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    p = fut.result()
                    if p and p.passes:
                        passed.append(p.symbol)
                except Exception:
                    pass
        return passed

    def _profile(self, symbol: str) -> Optional[StockRegimeProfile]:
        try:
            import yfinance as yf
            import numpy as np
            h = yf.Ticker(f"{symbol}.NS").history(period="250d", interval="1d")
            if h is None or len(h) < 50:
                return None
            close = h["Close"].values
            price = float(close[-1])
            sma50  = float(np.mean(close[-50:])) if len(close) >= 50 else price
            sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else price
            sma50_prev = float(np.mean(close[-55:-5])) if len(close) >= 55 else sma50

            above_200 = price > sma200
            above_50  = price > sma50
            sma50_rising = sma50 > sma50_prev

            # Weinstein stage classification
            if above_50 and above_200 and sma50_rising:
                stage = "Stage 2"
            elif not above_50 and not above_200 and not sma50_rising:
                stage = "Stage 4"
            elif not above_50 and above_200:
                stage = "Stage 3"  # topping / consolidation after Stage 2
            else:
                stage = "Stage 1"  # basing

            # RS vs Nifty
            try:
                nifty = yf.Ticker("^NSEI").history(period="7d")
                nifty_ret = (float(nifty["Close"].iloc[-1]) / float(nifty["Close"].iloc[-6]) - 1) if len(nifty) >= 6 else 0.0
            except Exception:
                nifty_ret = 0.0
            stock_ret = (price / float(close[-6]) - 1) if len(close) >= 6 else 0.0
            rs = (stock_ret - nifty_ret) * 100

            passes = stage in self._allowed
            reason = "" if passes else f"stage_{stage.lower().replace(' ','')}_not_eligible"
            return StockRegimeProfile(symbol, price, stage, above_200, above_50, sma50_rising, rs, passes, reason)
        except Exception:
            return None
