"""
Position Sizer — the discipline layer.

Turns a signal's entry/stop into an exact quantity using the 1% rule:
never risk more than `risk_per_trade_pct` of capital on a single trade,
and never put more than `max_position_size_pct` of capital into one name.

This is what separates a trading business from gambling — the loss on
any single trade is a fixed, survivable number decided in advance.
"""
from __future__ import annotations


def size_position(entry: float, stop: float,
                  capital: float | None = None,
                  risk_pct: float | None = None) -> dict:
    """
    Returns {qty, invested, max_loss, risk_pct_used, capped} for a trade.
    qty is 0 when the stop is invalid (>= entry) or capital can't afford 1 share.
    """
    try:
        from config import settings
        capital = capital if capital is not None else settings.trading_capital
        risk_pct = risk_pct if risk_pct is not None else settings.risk_per_trade_pct
        max_pos_pct = settings.max_position_size_pct
    except Exception:
        capital = capital or 100_000.0
        risk_pct = risk_pct or 0.01
        max_pos_pct = 0.10

    per_share_risk = entry - stop
    if entry <= 0 or per_share_risk <= 0 or capital <= 0:
        return {"qty": 0, "invested": 0.0, "max_loss": 0.0,
                "risk_pct_used": 0.0, "capped": False}

    risk_budget = capital * risk_pct
    qty = int(risk_budget / per_share_risk)

    # Cap by max position size (concentration limit)
    capped = False
    max_invest = capital * max_pos_pct
    if qty * entry > max_invest:
        qty = int(max_invest / entry)
        capped = True

    if qty < 1:
        return {"qty": 0, "invested": 0.0, "max_loss": 0.0,
                "risk_pct_used": 0.0, "capped": False}

    invested = qty * entry
    max_loss = qty * per_share_risk
    return {
        "qty": qty,
        "invested": round(invested, 0),
        "max_loss": round(max_loss, 0),
        "risk_pct_used": round(max_loss / capital * 100, 2),
        "capped": capped,
    }
