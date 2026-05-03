"""
TradeSetup — ATR-based entry, stop-loss, target, quantity, and R:R calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeSetup:
    symbol: str
    direction: str          # BUY or SELL
    entry: float
    stop: float
    target: float
    quantity: int
    risk_amount: float      # capital at risk (₹)
    reward_amount: float    # expected reward (₹)
    rr_ratio: float         # reward / risk


def compute_trade_setup(
    symbol: str,
    direction: str,
    entry_price: float,
    atr: float,
    capital: float = 100_000.0,
    risk_pct: float = 0.01,     # 1% of capital per trade
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 3.0,
) -> Optional[TradeSetup]:
    """
    Compute ATR-based trade setup.

    stop   = entry ± atr * atr_stop_mult
    target = entry ± atr * atr_target_mult
    qty    = floor(risk_capital / stop_distance)
    """
    if atr <= 0 or entry_price <= 0:
        return None

    risk_capital = capital * risk_pct
    stop_dist = atr * atr_stop_mult
    target_dist = atr * atr_target_mult

    if direction == "BUY":
        stop = entry_price - stop_dist
        target = entry_price + target_dist
    elif direction == "SELL":
        stop = entry_price + stop_dist
        target = entry_price - target_dist
    else:
        return None

    if stop_dist <= 0:
        return None

    quantity = max(1, int(risk_capital / stop_dist))
    risk_amount = quantity * stop_dist
    reward_amount = quantity * target_dist
    rr_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0.0

    return TradeSetup(
        symbol=symbol,
        direction=direction,
        entry=round(entry_price, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        quantity=quantity,
        risk_amount=round(risk_amount, 2),
        reward_amount=round(reward_amount, 2),
        rr_ratio=rr_ratio,
    )
