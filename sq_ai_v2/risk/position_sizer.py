"""
Position sizing engine.

Methods implemented:
  1. Fractional Kelly Criterion — optimal size given edge and win-rate.
  2. Volatility targeting — scale total exposure to hit target annual vol.
  3. Hard caps from config (max_position_size_pct, max_capital_exposure).

Kelly formula:  f* = (bp - q) / b
  where b = avg_win / avg_loss, p = win_rate, q = 1 - p.
  We use a fractional Kelly: f = kelly_fraction * f*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class SizeDecision:
    symbol: str
    fraction: float      # fraction of portfolio to allocate (0–1)
    shares: int          # whole shares to trade
    trade_value: float   # INR value
    reason: str          # diagnostic string


class PositionSizer:
    def __init__(
        self,
        kelly_fraction: float = settings.kelly_fraction,
        vol_target: float = settings.vol_target,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.vol_target = vol_target

    # ── Kelly sizing ───────────────────────────────────────────────────────

    def kelly_size(
        self,
        probability: float,
        avg_win_pct: float = 0.012,
        avg_loss_pct: float = 0.008,
    ) -> float:
        """
        Returns optimal fraction of capital for one position.
        probability: P(up), from calibrated model.
        avg_win_pct: expected % gain on a winning trade.
        avg_loss_pct: expected % loss on a losing trade (positive number).
        """
        p = float(np.clip(probability, 0.0, 1.0))
        q = 1.0 - p

        if avg_loss_pct <= 0 or avg_win_pct <= 0:
            return 0.0

        b = avg_win_pct / avg_loss_pct   # odds ratio
        kelly_full = (b * p - q) / b

        # Fractional Kelly caps the bet
        kelly_frac = self.kelly_fraction * kelly_full

        # Negative Kelly → no position (expected loss)
        return float(np.clip(kelly_frac, 0.0, settings.max_position_size_pct))

    # ── Volatility-targeted sizing ────────────────────────────────────────

    def vol_scaled_size(
        self,
        base_fraction: float,
        realised_vol: float,          # annualised vol of this symbol
        portfolio_vol: float,         # current portfolio annualised vol
    ) -> float:
        """
        Scale position size so total portfolio vol stays near vol_target.
        """
        if portfolio_vol <= 0 or realised_vol <= 0:
            return base_fraction

        # How much residual vol budget is left?
        vol_budget = max(0, self.vol_target - portfolio_vol)
        # Marginal contribution of this position
        marginal_vol = base_fraction * realised_vol
        if marginal_vol <= 0:
            return 0.0

        scaled = base_fraction * min(1.0, vol_budget / marginal_vol)
        return float(np.clip(scaled, 0.0, settings.max_position_size_pct))

    # ── Combined size ─────────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        probability: float,
        last_price: float,
        portfolio_value: float,
        portfolio_vol: float,         # current portfolio annualised vol
        symbol_vol: float,            # annualised vol of this symbol
        regime_multiplier: float = 1.0,
        correlation_penalty: float = 1.0,  # [0,1] reduced by CorrelationManager
    ) -> SizeDecision:
        """
        Full position sizing for a BUY decision.
        Returns integer shares and diagnostic info.
        """
        if last_price <= 0 or portfolio_value <= 0:
            return SizeDecision(symbol, 0.0, 0, 0.0, "invalid_inputs")

        # 1. Kelly fraction
        kelly_f = self.kelly_size(probability, symbol_vol / np.sqrt(252), symbol_vol / np.sqrt(252) * 0.6)

        # 2. Vol-scale
        vol_f = self.vol_scaled_size(kelly_f, symbol_vol, portfolio_vol)

        # 3. Apply regime and correlation adjustments
        adjusted_f = vol_f * regime_multiplier * correlation_penalty

        # 4. Hard cap
        final_f = float(np.clip(adjusted_f, 0.0, settings.max_position_size_pct))

        trade_value = portfolio_value * final_f
        shares = max(0, int(trade_value / last_price))
        actual_trade_value = shares * last_price

        reason = (
            f"kelly={kelly_f:.3f} vol_scaled={vol_f:.3f} "
            f"regime_mult={regime_multiplier:.2f} corr_penalty={correlation_penalty:.2f} "
            f"final={final_f:.3f}"
        )
        logger.debug(f"PositionSizer: {symbol} {reason}")

        return SizeDecision(
            symbol=symbol,
            fraction=final_f,
            shares=shares,
            trade_value=actual_trade_value,
            reason=reason,
        )

    # ── Portfolio-level vol estimate ──────────────────────────────────────

    @staticmethod
    def estimate_portfolio_vol(
        positions: Dict[str, float],   # symbol → position value
        symbol_vols: Dict[str, float],  # symbol → annualised vol
        portfolio_value: float,
    ) -> float:
        """
        Naive estimate: assume zero correlation (conservative).
        Actual correlation is handled by CorrelationManager.
        """
        if portfolio_value <= 0:
            return 0.0
        var = sum(
            (val / portfolio_value) ** 2 * symbol_vols.get(sym, 0.20) ** 2
            for sym, val in positions.items()
        )
        return float(np.sqrt(var))

    @staticmethod
    def compute_symbol_vol(close: pd.Series, window: int = 20) -> float:
        """Annualised realised volatility from daily closes."""
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 5:
            return 0.20
        return float(log_ret.tail(window).std() * np.sqrt(252))
