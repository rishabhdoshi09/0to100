"""
Market impact / slippage model for backtesting.

Three tiers:
  1. Flat-rate slippage: constant % of trade value (default, fast).
  2. Volume-impact model: slippage grows with order_size / ADV (Almgren-style).
  3. Spread model: half-spread based on bid-ask approximation.

Reference: Almgren et al. (2005), "Direct Estimation of Equity Market Impact"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class SlippageEstimate:
    bps: float          # basis points of slippage
    pct: float          # fraction of price
    price_impact: float # absolute price adjustment


class SlippageModel:
    """
    Computes realistic slippage for a given order.
    """

    def __init__(
        self,
        flat_bps: float = 5.0,         # 5 bps flat slippage
        impact_factor: float = 0.1,    # Almgren impact coefficient
        spread_bps: float = 3.0,       # assumed half-spread
    ) -> None:
        self.flat_bps = flat_bps
        self.impact_factor = impact_factor
        self.spread_bps = spread_bps

    # ── Flat rate ───────────────────────────────���─────────────────────────

    def flat_slippage(self, price: float) -> SlippageEstimate:
        bps = self.flat_bps
        pct = bps / 10_000
        return SlippageEstimate(bps=bps, pct=pct, price_impact=price * pct)

    # ── Volume-impact model ───────────────────────────────────────────────

    def volume_impact(
        self,
        price: float,
        order_shares: int,
        adv_shares: int,       # average daily volume in shares
        volatility: float = 0.02,   # daily return volatility
    ) -> SlippageEstimate:
        """
        Almgren-style permanent + temporary market impact.
        Total impact ≈ impact_factor × σ × (order_size / ADV)^0.6
        """
        if adv_shares <= 0:
            return self.flat_slippage(price)

        participation_rate = order_shares / adv_shares
        impact_pct = self.impact_factor * volatility * (participation_rate ** 0.6)
        # Add spread cost
        spread_pct = self.spread_bps / 10_000
        total_pct = impact_pct + spread_pct

        return SlippageEstimate(
            bps=total_pct * 10_000,
            pct=total_pct,
            price_impact=price * total_pct,
        )

    # ── Adjusted fill price ──────────────────────────────────────────────���

    def adjusted_buy_price(
        self,
        price: float,
        order_shares: int = 0,
        adv_shares: int = 0,
        volatility: float = 0.02,
    ) -> float:
        """Return realistic fill price for a BUY order (price + slippage)."""
        if adv_shares > 0 and order_shares > 0:
            est = self.volume_impact(price, order_shares, adv_shares, volatility)
        else:
            est = self.flat_slippage(price)
        return price + est.price_impact

    def adjusted_sell_price(
        self,
        price: float,
        order_shares: int = 0,
        adv_shares: int = 0,
        volatility: float = 0.02,
    ) -> float:
        """Return realistic fill price for a SELL order (price - slippage)."""
        if adv_shares > 0 and order_shares > 0:
            est = self.volume_impact(price, order_shares, adv_shares, volatility)
        else:
            est = self.flat_slippage(price)
        return price - est.price_impact

    # ── Transaction cost ──────────────────────────────────────────────────

    @staticmethod
    def transaction_cost(trade_value: float, cost_pct: float = 0.001) -> float:
        """Brokerage + STT + exchange charges (default 10 bps round-trip)."""
        return trade_value * cost_pct
