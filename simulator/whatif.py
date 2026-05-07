"""
What-If Trade Simulator.

Estimates the distribution of outcomes for a proposed trade by sampling
historical rolling windows of the same holding-period length.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_DEFAULT_HOLDING_DAYS = 5
_MIN_SAMPLES = 30


class WhatIfSimulator:
    """
    Simulate trade outcomes using historical rolling-window analysis.

    Usage
    -----
    sim = WhatIfSimulator()
    result = sim.simulate("RELIANCE", quantity=100, entry_price=2500.0, holding_days=5)
    """

    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher

    def simulate(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        holding_days: int = _DEFAULT_HOLDING_DAYS,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        symbol       : NSE symbol
        quantity     : number of shares
        entry_price  : intended entry price (may differ from current market price)
        holding_days : intended holding period in trading days

        Returns
        -------
        dict with probabilistic outcome estimates
        """
        df = self._fetch_data(symbol, holding_days)

        if df is None or len(df) < holding_days + _MIN_SAMPLES:
            log.warning(
                "whatif_insufficient_data",
                symbol=symbol,
                rows=len(df) if df is not None else 0,
            )
            return self._empty_result(symbol, quantity, entry_price, holding_days)

        close = df["close"].astype(float)

        # Build the distribution of all forward returns for this holding period
        # using historical rolling windows (no lookahead — each window uses only
        # prices known at window-start)
        forward_returns = []
        for i in range(len(close) - holding_days):
            entry = close.iloc[i]
            exit_price = close.iloc[i + holding_days]
            if entry > 0:
                forward_returns.append((exit_price - entry) / entry)

        if len(forward_returns) < _MIN_SAMPLES:
            return self._empty_result(symbol, quantity, entry_price, holding_days)

        fwd = np.array(forward_returns)

        # If entry_price != current price, scale using historical vol
        current_price = float(close.iloc[-1])
        entry_bias = 0.0
        if abs(entry_price - current_price) > 0.01 * current_price:
            # Shift distribution by the implied gain/loss from entry vs current
            entry_bias = (current_price - entry_price) / entry_price
            log.debug("whatif_applying_entry_bias", symbol=symbol, bias=round(entry_bias, 4))
            fwd = fwd + entry_bias

        prob_profit = float(np.mean(fwd > 0))
        prob_loss_gt_2pct = float(np.mean(fwd < -0.02))
        expected_return_pct = float(np.mean(fwd) * 100)

        # 1% VaR of the trade in INR (99th percentile worst case)
        var_99_pct = float(np.percentile(fwd, 1))
        var_99_inr = var_99_pct * entry_price * quantity

        result = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "holding_days": holding_days,
            "current_price": round(current_price, 2),
            "entry_bias_pct": round(entry_bias * 100, 3),
            "prob_profit": round(prob_profit, 3),
            "prob_loss_gt_2pct": round(prob_loss_gt_2pct, 3),
            "expected_return_pct": round(expected_return_pct, 3),
            "var_99": round(var_99_inr, 2),
            "var_99_pct": round(var_99_pct * 100, 3),
            "return_p10_pct": round(float(np.percentile(fwd, 10)) * 100, 3),
            "return_p50_pct": round(float(np.percentile(fwd, 50)) * 100, 3),
            "return_p90_pct": round(float(np.percentile(fwd, 90)) * 100, 3),
            "simulation_samples": len(forward_returns),
            "timestamp": datetime.utcnow().isoformat(),
        }

        log.info(
            "whatif_simulation_complete",
            symbol=symbol,
            prob_profit=result["prob_profit"],
            expected_return_pct=result["expected_return_pct"],
            var_99=result["var_99"],
        )
        return result

    # ── Data fetching ─────────────────────────────────────────────────────

    def _fetch_data(self, symbol: str, holding_days: int) -> Optional[pd.DataFrame]:
        # Need 2+ years of daily data for a good distribution
        years = 2
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

        # Kite
        try:
            fetcher = self._ensure_fetcher()
            df = fetcher.fetch(symbol, from_d, to_d, "day")
            if df is not None and len(df) > holding_days + _MIN_SAMPLES:
                return df
        except Exception as exc:
            log.warning("whatif_kite_failed", symbol=symbol, error=str(exc))

        # yfinance fallback
        try:
            import yfinance as yf
            df = yf.download(f"{symbol}.NS", start=from_d, end=to_d, progress=False)
            if df is not None and len(df) > holding_days + _MIN_SAMPLES:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as exc:
            log.warning("whatif_yf_failed", symbol=symbol, error=str(exc))

        return None

    def _ensure_fetcher(self):
        if self._fetcher is None:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            self._fetcher = HistoricalDataFetcher(KiteClient(), InstrumentManager())
        return self._fetcher

    @staticmethod
    def _empty_result(
        symbol: str, quantity: int, entry_price: float, holding_days: int
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "holding_days": holding_days,
            "current_price": None,
            "entry_bias_pct": 0.0,
            "prob_profit": None,
            "prob_loss_gt_2pct": None,
            "expected_return_pct": None,
            "var_99": None,
            "var_99_pct": None,
            "return_p10_pct": None,
            "return_p50_pct": None,
            "return_p90_pct": None,
            "simulation_samples": 0,
            "error": "insufficient_data",
        }
