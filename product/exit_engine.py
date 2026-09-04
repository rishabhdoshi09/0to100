"""Connect actual exit policy for paper and shadow decisions.

Does not invent a structural target when the card only has a 2R ATR
multiple. Artificial targets stay labelled ARTIFICIAL.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.risk_audit import audit_levels

INITIAL_INVALIDATION = "INITIAL_INVALIDATION"
TRAILING_STOP = "TRAILING_STOP"
TARGET = "TARGET"
TIME_STOP = "TIME_STOP"
TECHNICAL_DETERIORATION = "TECHNICAL_DETERIORATION"
REGIME_CHANGE = "REGIME_CHANGE"
RISK_REDUCTION = "RISK_REDUCTION"
HOLD = "HOLD"

STRUCTURAL = "STRUCTURAL"
ATR = "ATR"
HYBRID = "HYBRID"
FALLBACK = "FALLBACK"


def classify_stop(levels: Mapping[str, Any]) -> str:
    basis = str(levels.get("risk_basis") or "")
    if basis.startswith("ATR"):
        return ATR
    if basis in {"STRUCTURE_OR_OTHER"}:
        return STRUCTURAL
    if basis.startswith("FIXED_PCT"):
        return FALLBACK
    return FALLBACK


def evaluate_exit(
    position: Mapping[str, Any],
    *,
    last_price: float | None,
    bars_held: int = 0,
    regime: str = "",
    technical_broken: bool = False,
    time_limit_sessions: int = 20,
    trail_after_r: float = 1.0,
) -> dict[str, Any]:
    """Which exit, if any, the existing policy would fire. No invented stops."""
    entry = position.get("entry") or position.get("entry_price")
    stop = position.get("stop") or position.get("stop_price")
    target = position.get("target") or position.get("target_price")
    levels = audit_levels(position)
    stop_kind = classify_stop(levels)
    reasons: list[str] = []
    try:
        px = float(last_price) if last_price is not None else None
        e = float(entry) if entry is not None else None
        s = float(stop) if stop is not None else None
        t = float(target) if target is not None else None
    except (TypeError, ValueError):
        px = e = s = t = None
    if px is not None and s is not None and px <= s:
        reasons.append(INITIAL_INVALIDATION)
    if px is not None and t is not None and px >= t:
        if levels.get("target_artificial"):
            reasons.append(TARGET)
        else:
            reasons.append(TARGET)
    if e is not None and s is not None and px is not None and (e - s) > 0:
        r = (px - e) / (e - s)
        if r >= trail_after_r:
            reasons.append(TRAILING_STOP)
    if bars_held >= time_limit_sessions:
        reasons.append(TIME_STOP)
    if technical_broken:
        reasons.append(TECHNICAL_DETERIORATION)
    if str(regime or "").upper() in {"RISK_OFF", "CRISIS", "DEFENSIVE"}:
        reasons.append(REGIME_CHANGE)
    action = reasons[0] if reasons else HOLD
    return {
        "action": action,
        "reasons": reasons,
        "stop_kind": stop_kind,
        "target_artificial": bool(levels.get("target_artificial")),
        "target_basis": levels.get("target_basis"),
        "risk_basis": levels.get("risk_basis"),
        "exercises_real_policy": True,
        "note": (
            "2R target is ARTIFICIAL when derived from the ATR stop. "
            "Fallback ATR stops are not structural intelligence."
            if levels.get("target_artificial") or stop_kind in {ATR, FALLBACK} else
            "Exit uses the frozen invalidation / labelled target."
        ),
    }


def paper_should_exit(position: Mapping[str, Any], **kwargs: Any) -> bool:
    return evaluate_exit(position, **kwargs)["action"] != HOLD
