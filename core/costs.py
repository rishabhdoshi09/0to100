"""
💸 Trading costs — the gap between a backtest number and a brokerage statement.

Every accuracy/expectancy number a retail system shows is a LIE until it subtracts
the cost of trading. This is the one source of truth for Indian-equity round-trip
cost, applied consistently to the backtest, the live R-multiples, and both equity
curves so the reported edge is what you'd actually keep — not the gross move.

Round-trip cost as a % of turnover (buy + sell), realistic and slightly
conservative:

  • DELIVERY (CNC): STT 0.1%×2 + stamp 0.015% + exchange/SEBI/GST bits ≈ 0.22%,
    brokerage ₹0 (Zerodha free equity delivery).
  • INTRADAY (MIS): brokerage ~0.06% + STT 0.025% (sell) + stamp 0.003% +
    exchange/GST bits ≈ 0.10%.
  • SLIPPAGE: entry/exit rarely at the exact price — a round-trip default of
    0.10% (0.05% each side) on top, env-tunable for illiquid names.

Expressed in R (`cost_in_r`) so it can be subtracted straight from an R-multiple:
a trade risking 4% that pays 0.32% round-trip gives up 0.08R to costs — small per
trade, decisive over a hundred of them. Turn off with QT_COSTS=0 for gross study.
"""
from __future__ import annotations

import os as _os

_ENABLED = (_os.getenv("QT_COSTS", "1") or "1") not in ("0", "false", "False")
# round-trip taxes + fees as % of turnover, by product
_BASE_PCT = {"CNC": float(_os.getenv("QT_COST_CNC_PCT", "0.22") or 0.22),
             "MIS": float(_os.getenv("QT_COST_MIS_PCT", "0.10") or 0.10)}
_SLIPPAGE_PCT = float(_os.getenv("QT_COST_SLIPPAGE_PCT", "0.10") or 0.10)


def round_trip_cost_pct(product: str = "CNC") -> float:
    """Total round-trip trading cost as a % of price (taxes + fees + slippage).
    0.0 when costs are disabled (QT_COSTS=0)."""
    if not _ENABLED:
        return 0.0
    base = _BASE_PCT.get((product or "CNC").upper(), _BASE_PCT["CNC"])
    return base + _SLIPPAGE_PCT


def cost_in_r(risk_frac: float, product: str = "CNC") -> float:
    """Round-trip cost expressed in R-multiples for a trade whose stop is
    `risk_frac` of entry (e.g. 0.04 = a 4% stop). cost = cost% ÷ risk%."""
    try:
        rf = float(risk_frac)
    except (TypeError, ValueError):
        return 0.0
    if rf <= 0:
        return 0.0
    return round_trip_cost_pct(product) / (rf * 100.0)


def net_r(gross_r: float, risk_frac: float, product: str = "CNC") -> float:
    """A gross R-multiple, net of round-trip trading costs. NO_FILL / no-trade
    callers should simply not call this (there's no cost without a trade)."""
    return float(gross_r) - cost_in_r(risk_frac, product)


def cost_rupees(qty: int, price: float, product: str = "CNC") -> float:
    """Round-trip cost in rupees for `qty` shares bought at `price`."""
    try:
        return abs(int(qty) * float(price)) * round_trip_cost_pct(product) / 100.0
    except (TypeError, ValueError):
        return 0.0
