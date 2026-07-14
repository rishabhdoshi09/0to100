"""
💰 Execution cost model — paper trades that tell the TRUTH.

Idealised paper fills (exact stop, exact target, zero cost) flatter the
Report Card and lie to the trader about what LIVE will feel like. This
module models what a real Zerodha round-trip actually costs, so the
system earns trust on honest numbers.

Two frictions, both real:

  1. Slippage — a market buy fills a little above your price; a stop
     sell fills a little BELOW the trigger (stops gap against you).
     Modelled in basis points, worse on stop-outs than on targets.

  2. Charges — Zerodha's actual statutory + exchange costs. Equity
     delivery (CNC) brokerage is ₹0, but STT, exchange txn, SEBI,
     stamp duty, DP charges and GST are not — they add up to ~0.25%
     of turnover on a round trip, before slippage.

Every rate is a Zerodha default and overridable via .env, so the model
tracks reality as fees change. Pure functions, no I/O, fully tested.
"""
from __future__ import annotations

import os

# ── Slippage (basis points; 1 bp = 0.01%) ─────────────────────────────────────
_SLIP_ENTRY_BPS = float(os.getenv("QT_SLIP_ENTRY_BPS", "5") or 5)     # market buy
_SLIP_TARGET_BPS = float(os.getenv("QT_SLIP_TARGET_BPS", "3") or 3)   # limit sell
_SLIP_STOP_BPS = float(os.getenv("QT_SLIP_STOP_BPS", "12") or 12)     # stop gaps

# ── Zerodha equity charges (fractions of turnover unless noted) ───────────────
_STT = float(os.getenv("QT_STT_PCT", "0.001") or 0.001)              # 0.1% each side (CNC)
_STT_MIS_SELL = float(os.getenv("QT_STT_MIS_SELL_PCT", "0.00025") or 0.00025)
_EXCH_TXN = float(os.getenv("QT_EXCH_TXN_PCT", "0.0000297") or 0.0000297)  # NSE
_SEBI = float(os.getenv("QT_SEBI_PCT", "0.000001") or 0.000001)      # ₹10/crore
_STAMP_BUY = float(os.getenv("QT_STAMP_BUY_PCT", "0.00015") or 0.00015)    # 0.015% buy (delivery)
_STAMP_BUY_MIS = float(os.getenv("QT_STAMP_BUY_MIS_PCT", "0.00003") or 0.00003)
_GST = float(os.getenv("QT_GST_PCT", "0.18") or 0.18)
_DP_SELL = float(os.getenv("QT_DP_SELL", "13.5") or 13.5)            # ₹/scrip sell (CNC)
_BROKERAGE_MIS = float(os.getenv("QT_BROKERAGE_MIS", "20") or 20)    # ₹20 or 0.03% cap/leg


def zerodha_charges(buy_price: float, sell_price: float, qty: int,
                    product: str = "CNC") -> dict:
    """Full statutory + exchange charge breakdown for one round trip.
    Returns {stt, txn, sebi, stamp, dp, brokerage, gst, total} in ₹."""
    if qty <= 0 or buy_price <= 0 or sell_price <= 0:
        return {"stt": 0.0, "txn": 0.0, "sebi": 0.0, "stamp": 0.0,
                "dp": 0.0, "brokerage": 0.0, "gst": 0.0, "total": 0.0}
    buy_val = buy_price * qty
    sell_val = sell_price * qty
    turnover = buy_val + sell_val
    is_mis = product.upper() == "MIS"

    if is_mis:
        stt = sell_val * _STT_MIS_SELL                 # sell side only
        stamp = buy_val * _STAMP_BUY_MIS
        brokerage = 2 * min(_BROKERAGE_MIS, 0.0003 * buy_val)  # both legs
        dp = 0.0                                        # no DP for intraday
    else:                                               # CNC delivery
        stt = (buy_val + sell_val) * _STT              # both sides
        stamp = buy_val * _STAMP_BUY
        brokerage = 0.0                                 # free delivery
        dp = _DP_SELL                                   # per scrip, sell

    txn = turnover * _EXCH_TXN
    sebi = turnover * _SEBI
    # GST applies to brokerage + transaction + SEBI + DP (not STT / stamp)
    gst = _GST * (brokerage + txn + sebi + dp)
    total = stt + txn + sebi + stamp + dp + brokerage + gst
    return {"stt": round(stt, 2), "txn": round(txn, 2), "sebi": round(sebi, 4),
            "stamp": round(stamp, 2), "dp": round(dp, 2),
            "brokerage": round(brokerage, 2), "gst": round(gst, 2),
            "total": round(total, 2)}


def simulate_fill(price: float, side: str, is_stop: bool = False) -> float:
    """Realistic fill for a PAPER order. Buys slip up, sells slip down;
    stop-loss sells slip the most (they gap through the trigger)."""
    if price <= 0:
        return price
    if side == "BUY":
        return round(price * (1 + _SLIP_ENTRY_BPS / 10000), 2)
    bps = _SLIP_STOP_BPS if is_stop else _SLIP_TARGET_BPS
    return round(price * (1 - bps / 10000), 2)


def net_result(entry: float, exit_price: float, qty: int,
               product: str = "CNC", exit_is_stop: bool = False,
               paper: bool = True) -> dict:
    """The honest bottom line for a closed trade.

    PAPER → slippage simulated on both fills (buy up, sell/stop down).
    LIVE  → entry/exit are already the real reconciled fills, so only
            charges are applied. Returns
    {gross, slippage, charges, net, net_pct} in ₹ (net_pct on entry value)."""
    if qty <= 0 or entry <= 0 or exit_price <= 0:
        return {"gross": 0.0, "slippage": 0.0, "charges": 0.0,
                "net": 0.0, "net_pct": 0.0}
    if paper:
        fill_entry = simulate_fill(entry, "BUY")
        fill_exit = simulate_fill(exit_price, "SELL", is_stop=exit_is_stop)
    else:
        fill_entry, fill_exit = entry, exit_price
    ideal_gross = (exit_price - entry) * qty
    real_gross = (fill_exit - fill_entry) * qty
    slippage = round(ideal_gross - real_gross, 2)      # what slippage cost
    ch = zerodha_charges(fill_entry, fill_exit, qty, product)["total"]
    net = round(real_gross - ch, 2)
    net_pct = round(net / (entry * qty) * 100, 2) if entry * qty else 0.0
    return {"gross": round(ideal_gross, 2), "slippage": slippage,
            "charges": ch, "net": net, "net_pct": net_pct}
