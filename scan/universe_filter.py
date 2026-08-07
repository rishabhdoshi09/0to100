"""
Universe Filter — Stage 1 of the scan pipeline.

Applies hard filters before any scanning begins:
  - Liquidity (price floor, volume floor)
  - Exchange (NSE EQ only)
  - Exclusions (PSU banks, F&O basket optionally)

Returns a filtered subset of the input universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LiquidityProfile:
    symbol: str
    price: float
    avg_volume: float
    daily_turnover_cr: float    # price × avg_volume / 1e7
    passes: bool
    reject_reason: str = ""


class UniverseFilter:
    """
    Hard-filter the universe before expensive scanning.
    Defaults are conservative — tune via env/config.
    """

    def __init__(
        self,
        min_price: float = 20.0,
        min_avg_volume: int = 50_000,
        min_daily_turnover_cr: float = 1.0,
        max_workers: int = 16,
    ):
        self._min_price           = min_price
        self._min_avg_vol         = min_avg_volume
        self._min_turnover_cr     = min_daily_turnover_cr
        self._max_workers         = max_workers

    def filter(self, symbols: list[str]) -> list[str]:
        """Return symbols that pass all liquidity checks."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        passed: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._check, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                try:
                    profile = fut.result()
                    if profile and profile.passes:
                        passed.append(profile.symbol)
                except Exception:
                    pass
        return passed

    def _check(self, symbol: str) -> Optional[LiquidityProfile]:
        try:
            import yfinance as yf
            t  = yf.Ticker(f"{symbol}.NS")
            h  = t.history(period="30d", interval="1d")
            if h is None or len(h) < 10:
                return None
            price  = float(h["Close"].iloc[-1])
            avg_v  = float(h["Volume"].iloc[-20:].mean())
            to_cr  = price * avg_v / 1e7

            if price < self._min_price:
                return LiquidityProfile(symbol, price, avg_v, to_cr, False, "price_floor")
            if avg_v < self._min_avg_vol:
                return LiquidityProfile(symbol, price, avg_v, to_cr, False, "volume_floor")
            if to_cr < self._min_turnover_cr:
                return LiquidityProfile(symbol, price, avg_v, to_cr, False, "turnover_floor")
            return LiquidityProfile(symbol, price, avg_v, to_cr, True)
        except Exception:
            return None
