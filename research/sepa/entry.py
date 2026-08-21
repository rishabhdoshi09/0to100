"""Canonical pivot, buy-zone, structural stop, reward context."""
from __future__ import annotations

from typing import Any

from research.sepa.config import SepaConfig


def evaluate_entry(
    *,
    price: float | None,
    vcp: dict[str, Any],
    atr: float | None,
    config: SepaConfig,
    buy_zone_above_pct: float | None = None,
) -> dict[str, Any]:
    above = config.buy_zone_above_pct if buy_zone_above_pct is None else float(buy_zone_above_pct)
    below = config.buy_zone_below_pct
    pivot = vcp.get("pivot")
    stop = vcp.get("stop")
    out: dict[str, Any] = {
        "pivot": pivot,
        "pivot_date": vcp.get("pivot_date"),
        "pivot_type": vcp.get("pivot_type"),
        "price": price,
        "distance_from_pivot_pct": None,
        "buy_zone_low": None,
        "buy_zone_high": None,
        "entry_valid": False,
        "entry_rejection": None,
        "extended": False,
        "proposed_entry": None,
        "structural_stop": stop,
        "stop_basis": "final_contraction_low" if stop else None,
        "stop_distance_pct": None,
        "atr": atr,
        "stop_atr_multiple": None,
        "stop_ok": False,
        "risk_r": None,
        "measured_move": None,
        "reward_price": None,
        "reward_risk": None,
        "reward_status": "UNKNOWN",
        "resistance": {},
    }
    if pivot is None:
        out["entry_rejection"] = "NO_PIVOT"
        return out
    if price is None:
        out["entry_rejection"] = "NO_PRICE"
        return out

    zone_lo = float(pivot) * (1.0 - below / 100.0)
    zone_hi = float(pivot) * (1.0 + above / 100.0)
    dist = (float(price) / float(pivot) - 1.0) * 100.0
    out["buy_zone_low"] = round(zone_lo, 4)
    out["buy_zone_high"] = round(zone_hi, 4)
    out["distance_from_pivot_pct"] = round(dist, 4)
    out["proposed_entry"] = round(float(price), 4)

    if float(price) > zone_hi:
        out["entry_rejection"] = "ENTRY_EXTENDED"
        out["extended"] = True
    elif float(price) < zone_lo:
        out["entry_rejection"] = "ENTRY_BELOW_PIVOT"
    else:
        out["entry_valid"] = True

    if stop is None or float(stop) <= 0 or float(stop) >= float(price):
        out["stop_ok"] = False
        if out["entry_rejection"] is None:
            out["entry_rejection"] = out["entry_rejection"] or "STOP_UNDEFINED"
        if stop is not None and float(stop) >= float(price):
            out["entry_rejection"] = "STOP_NOT_BELOW_ENTRY"
        return out

    risk = float(price) - float(stop)
    risk_pct = risk / float(price) * 100.0
    out["stop_distance_pct"] = round(risk_pct, 4)
    if atr and atr > 0:
        out["stop_atr_multiple"] = round(risk / float(atr), 4)
    wide = risk_pct > config.max_stop_pct
    atr_wide = (
        out["stop_atr_multiple"] is not None
        and out["stop_atr_multiple"] > config.max_stop_atr
    )
    # ATR stretch is a diagnostic. Hard reject is percentage risk vs the setup.
    # A 4% structural stop on a compressed-ATR name is not "too wide".
    out["stop_ok"] = (not wide) and risk > 0
    out["evidence_atr_wide"] = bool(atr_wide)
    if wide:
        out["entry_rejection"] = out["entry_rejection"] or "WIDE_STRUCTURAL_STOP"
        out["stop_ok"] = False
    out["risk_r"] = 1.0 if out["stop_ok"] else None

    depths = list(vcp.get("depths") or [])
    if depths and pivot:
        first = float(depths[0]) / 100.0
        measured = float(pivot) * (1.0 + first)
        out["measured_move"] = round(first * 100.0, 3)
        out["reward_price"] = round(measured, 4)
        out["resistance"] = {"kind": "measured_move_from_first_contraction", "price": round(measured, 4)}
        if risk > 0:
            reward = measured - float(price)
            if reward > 0:
                out["reward_risk"] = round(reward / risk, 3)
                out["reward_status"] = "MEASURED_MOVE"
            else:
                out["reward_status"] = "UNKNOWN"
        else:
            out["reward_status"] = "UNKNOWN"
    else:
        out["reward_status"] = "UNKNOWN"
    return out


FILL_VALID = "VALID_FILL"
FILL_MISSED = "MISSED"
FILL_GAP_THROUGH = "GAP_THROUGH"
FILL_EXTENDED = "EXTENDED"
FILL_INVALIDATED = "INVALIDATED"
FILL_NO_BAR = "NO_BAR"
FILL_CA_CENSORED = "CA_CENSORED_OUTCOME"


def classify_next_open_fill(
    *,
    open_px: float | None,
    zone_lo: float | None,
    zone_hi: float | None,
    stop: float | None,
) -> dict[str, Any]:
    """Next-session open vs the versioned buy-zone. Never chase.

    A print outside the zone is a classification, not a market fill at a
    worse price. `entry = last price` is not a fallback.
    """
    if open_px is None or open_px <= 0:
        return {"class": FILL_NO_BAR, "fill": None, "reason": "no executable open"}
    if zone_lo is None or zone_hi is None:
        return {"class": FILL_NO_BAR, "fill": None, "reason": "buy-zone undefined"}
    o = float(open_px)
    lo, hi = float(zone_lo), float(zone_hi)
    if o > hi:
        return {"class": FILL_GAP_THROUGH, "fill": None, "reason": "next open gapped through buy-zone"}
    if o < lo:
        return {"class": FILL_MISSED, "fill": None, "reason": "next open below buy-zone — not filled"}
    if stop is not None and o <= float(stop):
        return {"class": FILL_INVALIDATED, "fill": None, "reason": "next open at or through structural stop"}
    return {"class": FILL_VALID, "fill": round(o, 4), "reason": "next open inside buy-zone"}

