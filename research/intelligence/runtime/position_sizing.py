"""Pure long-only cash-equity position sizing shared by portfolio planning and PAPER.

Risk inputs use percentage points: ``1.0`` means one percent of capital. This module
contains no book mutation, broker access, market-data access, or hidden fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingResult:
    ok: bool
    reason_code: str
    requested_risk_pct: float
    capped_risk_pct: float
    effective_entry: float
    risk_per_share: float
    quantity: int
    risk_amount: float
    actual_risk_pct: float
    capital_amount: float


def size_long_cash(
    *,
    capital: float,
    entry: float,
    stop: float,
    requested_risk_pct: float,
    max_risk_fraction: float,
    max_position_fraction: float,
    slippage_bps: float = 0.0,
    requested_quantity: int | None = None,
) -> SizingResult:
    """Return the maximum safe long-cash quantity under per-trade and name caps.

    Total portfolio risk, cash availability, sector and correlation constraints belong
    to the Target Portfolio constructor and are deliberately not hidden in this helper.
    """
    try:
        capital = float(capital)
        entry = float(entry)
        stop = float(stop)
        requested_risk_pct = float(requested_risk_pct)
        max_risk_fraction = float(max_risk_fraction)
        max_position_fraction = float(max_position_fraction)
        slippage_bps = float(slippage_bps)
    except (TypeError, ValueError):
        return _failed("INVALID_NUMERIC_INPUT")

    values = (capital, entry, stop, requested_risk_pct, max_risk_fraction,
              max_position_fraction, slippage_bps)
    if not all(math.isfinite(value) for value in values):
        return _failed("NON_FINITE_INPUT")
    if capital <= 0:
        return _failed("NON_POSITIVE_CAPITAL")
    if not (entry > stop > 0):
        return _failed("INVALID_ENTRY_STOP")
    if requested_risk_pct <= 0:
        return _failed("NON_POSITIVE_RISK")
    if max_risk_fraction <= 0 or max_position_fraction <= 0:
        return _failed("INVALID_HOUSE_LIMIT")

    max_risk_pct = max_risk_fraction * 100.0
    capped_risk_pct = min(requested_risk_pct, max_risk_pct)
    effective_entry = entry * (1.0 + slippage_bps / 10_000.0)
    risk_per_share = effective_entry - stop
    if risk_per_share <= 0:
        return _failed("INVALID_EFFECTIVE_RISK")

    risk_budget = capital * capped_risk_pct / 100.0
    quantity_by_risk = int(math.floor(risk_budget / risk_per_share))
    quantity_by_capital = int(math.floor(capital * max_position_fraction / effective_entry))
    if quantity_by_risk <= 0:
        return _failed("RISK_BUDGET_TOO_SMALL")
    if quantity_by_capital <= 0:
        return _failed("POSITION_CAP_TOO_SMALL")
    maximum_quantity = min(quantity_by_risk, quantity_by_capital)

    if requested_quantity is None:
        quantity = maximum_quantity
    else:
        try:
            quantity = int(requested_quantity)
        except (TypeError, ValueError):
            return _failed("INVALID_REQUESTED_QUANTITY")
        if quantity <= 0:
            return _failed("NON_POSITIVE_QUANTITY")
        if quantity > maximum_quantity:
            return _failed("QUANTITY_EXCEEDS_APPROVED_LIMIT")

    risk_amount = quantity * risk_per_share
    capital_amount = quantity * effective_entry
    actual_risk_pct = risk_amount / capital * 100.0
    return SizingResult(
        ok=True,
        reason_code="OK",
        requested_risk_pct=requested_risk_pct,
        capped_risk_pct=capped_risk_pct,
        effective_entry=effective_entry,
        risk_per_share=risk_per_share,
        quantity=quantity,
        risk_amount=risk_amount,
        actual_risk_pct=actual_risk_pct,
        capital_amount=capital_amount,
    )


def _failed(reason_code: str) -> SizingResult:
    return SizingResult(
        ok=False,
        reason_code=reason_code,
        requested_risk_pct=0.0,
        capped_risk_pct=0.0,
        effective_entry=0.0,
        risk_per_share=0.0,
        quantity=0,
        risk_amount=0.0,
        actual_risk_pct=0.0,
        capital_amount=0.0,
    )
