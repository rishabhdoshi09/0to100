"""Audit how entry / stop / target and paper size are actually produced.

Scanner plan (unified_scanner):
  entry = pivot if pivot > price else price
  stop  = entry - 2 * ATR(14)   (else 5% fallback)
  target= entry + 4 * ATR(14)   (else 10% fallback)

That is an ATR multiple, not a measured structure target. A 2.0R label
is tautological: reward is defined as 2× the ATR stop. Do not treat the
UI target as an independently discovered price objective.
"""
from __future__ import annotations

from typing import Any, Mapping

from research.intelligence.runtime.position_sizing import size_long_cash
from risk.position_sizer import size_position


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_levels(card: Mapping[str, Any]) -> dict[str, Any]:
    entry = _f(card.get("entry") or card.get("entry_price") or card.get("cmp"))
    stop = _f(card.get("stop") or card.get("stop_price"))
    target = _f(card.get("target") or card.get("target_price"))
    atr = _f(card.get("atr") or card.get("atr14"))
    atr_pct = _f(card.get("atr_pct"))
    if atr is None and atr_pct is not None and entry:
        atr = entry * atr_pct / 100.0

    risk_basis = "UNKNOWN"
    target_basis = "UNKNOWN"
    target_artificial = False
    if entry and stop and entry > stop:
        gap = entry - stop
        if atr and atr > 0:
            stop_mult = gap / atr
            if 1.7 <= stop_mult <= 2.3:
                risk_basis = "ATR_2X"
            elif 0.8 <= stop_mult <= 1.3:
                risk_basis = "ATR_1X"
            else:
                risk_basis = f"ATR_{stop_mult:.1f}X"
        elif abs(gap / entry - 0.05) < 0.005:
            risk_basis = "FIXED_PCT_5"
        else:
            risk_basis = "STRUCTURE_OR_OTHER"
        if target and target > entry:
            reward = target - entry
            rr = reward / gap
            if atr and atr > 0:
                tgt_mult = reward / atr
                if 3.6 <= tgt_mult <= 4.4:
                    target_basis = "ATR_4X"
                    target_artificial = True
                elif 1.8 <= rr <= 2.2 and risk_basis.startswith("ATR"):
                    target_basis = "FIXED_2R_FROM_ATR_STOP"
                    target_artificial = True
                else:
                    target_basis = "STRUCTURE_OR_OTHER"
            elif 1.8 <= rr <= 2.2:
                target_basis = "FIXED_2R"
                target_artificial = True
            else:
                target_basis = "STRUCTURE_OR_OTHER"

    entry_risk_pct = None
    reward_risk = None
    r_unit = None
    if entry and stop and entry > stop:
        entry_risk_pct = round((entry - stop) / entry * 100.0, 2)
        r_unit = round(entry - stop, 4)
        if target and target > entry:
            reward_risk = round((target - entry) / (entry - stop), 2)

    return {
        "symbol": str(card.get("symbol") or "").upper(),
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "entry_risk_pct": entry_risk_pct,
        "reward_risk": reward_risk,
        "r_unit": r_unit,
        "risk_basis": risk_basis,
        "target_basis": target_basis,
        "target_artificial": target_artificial,
        "note": (
            "Target is a fixed ATR/R multiple, not a measured structure objective."
            if target_artificial else
            "Risk/target basis taken from existing levels — stops are never invented."
        ),
    }


def r_multiple(
    *,
    entry: float | None,
    stop: float | None,
    exit_price: float | None,
) -> float | None:
    """R = (exit - entry) / (entry - stop). Requires a frozen stop. No invented stop."""
    if entry is None or stop is None or exit_price is None:
        return None
    risk = float(entry) - float(stop)
    if risk <= 0:
        return None
    return round((float(exit_price) - float(entry)) / risk, 3)


def audit_sizing(
    *,
    entry: float,
    stop: float,
    capital: float,
    risk_per_trade_pct: float = 0.01,
    max_position_pct: float = 0.10,
) -> dict[str, Any]:
    """shares = allowed_rupee_risk / (entry - stop), then 1% / 10% caps."""
    levels_ok = entry > stop > 0 and capital > 0
    house = size_position(entry, stop, capital=capital, risk_pct=risk_per_trade_pct)
    paper = size_long_cash(
        capital=capital,
        entry=entry,
        stop=stop,
        requested_risk_pct=risk_per_trade_pct * 100.0,
        max_risk_fraction=risk_per_trade_pct,
        max_position_fraction=max_position_pct,
    )
    per_share = entry - stop if levels_ok else None
    allowed = capital * risk_per_trade_pct if levels_ok else None
    naive = int(allowed / per_share) if allowed and per_share and per_share > 0 else 0
    return {
        "ok": bool(levels_ok and paper.ok and house.get("qty", 0) > 0),
        "entry": entry,
        "stop": stop,
        "per_share_risk": per_share,
        "allowed_rupee_risk": allowed,
        "naive_shares": naive,
        "house_qty": house.get("qty"),
        "house_capped": house.get("capped"),
        "paper_qty": paper.quantity,
        "paper_ok": paper.ok,
        "paper_reason": paper.reason_code,
        "actual_risk_pct": paper.actual_risk_pct,
        "uses_actual_stop_distance": True,
        "bypasses_1pct_limit": False,
        "note": "Wider stops receive fewer shares. Limits are not bypassed.",
    }
