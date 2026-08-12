"""
💸 India cash-equity friction model — so paper cannot flatter itself.

A paper edge measured on gross, exact-price fills will look better than anything that ever
trades live, and that gap is exactly where overfit strategies get promoted. This is a
transparent, documented estimate of the real round-trip drag on an NSE delivery trade:
brokerage, STT, exchange transaction, SEBI, stamp duty, and GST on the taxable components.

It is intentionally an *estimate* (rates change; broker plans differ) and is labelled as
such — never presented as an exact ledger. Slippage is modelled separately (at the fill),
because it is a market-impact effect, not a statutory charge.
"""
from __future__ import annotations

# statutory / exchange rates (NSE equity delivery, approx — review annually)
_STT = 0.001           # 0.10% on BOTH buy and sell (delivery)
_EXCH_TXN = 0.0000297  # NSE ~0.00297% on turnover
_SEBI = 0.000001       # ₹10 per crore
_STAMP_BUY = 0.00015   # 0.015% on the buy leg only
_GST = 0.18            # 18% on (brokerage + exchange txn + SEBI)


def india_cash_costs(entry_price: float, exit_price: float, qty: int,
                     brokerage_per_side: float = 0.0) -> float:
    """Estimated round-trip cost in ₹ for buying `qty` at `entry_price` and selling at
    `exit_price`. Defaults to a zero-brokerage delivery plan (e.g. Zerodha equity delivery);
    pass `brokerage_per_side` for a different broker. Never negative."""
    if qty <= 0 or entry_price <= 0 or exit_price <= 0:
        return 0.0
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    turnover = buy_turnover + sell_turnover

    stt = _STT * turnover
    exch = _EXCH_TXN * turnover
    sebi = _SEBI * turnover
    stamp = _STAMP_BUY * buy_turnover
    brokerage = 2.0 * brokerage_per_side
    gst = _GST * (brokerage + exch + sebi)
    return round(stt + exch + sebi + stamp + brokerage + gst, 2)


def cost_in_R(entry_price: float, exit_price: float, qty: int, r_unit: float,
              brokerage_per_side: float = 0.0) -> float:
    """The same round-trip cost expressed in R (fraction of the trade's 1R risk), so it can
    be compared apples-to-apples with expectancy. r_unit is |entry − stop| per share."""
    if r_unit <= 0 or qty <= 0:
        return 0.0
    return india_cash_costs(entry_price, exit_price, qty, brokerage_per_side) / (qty * r_unit)
