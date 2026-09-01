"""Execution-aware evidence integrity for QuantTerm learning.

Paper fills remain intended-price. This module only decides what outcome evidence
is safe to feed into learning and promotion. It preserves gross R, derives a
labelled execution-adjusted R when the settled trade has enough fields, and uses
the more conservative value for policy learning.
"""
from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1


def _f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _first(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _f(mapping.get(key))
        if value is not None:
            return value
    return None


def settled_learning_result(
    trade: Mapping[str, Any],
    taken_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return gross + execution-adjusted R without changing the paper trade.

    Statutory/explicit transaction costs come from ExecutionRealityEngine. When
    bid/ask is unavailable the engine records a labelled spread estimate but does
    not apply it to the fill; for learning we conservatively subtract that
    estimated round-trip spread. Unknown slippage stays explicitly unknown.
    """
    row = dict(trade or {})
    evidence = dict(taken_evidence or {})
    gross_r = _first(row, "realized_R", "gross_realized_R")
    entry = _first(evidence, "entry_fill", "entry") or _first(row, "entry_price", "entry")
    stop = _first(evidence, "stop") or _first(row, "stop_price", "stop")
    exit_price = _first(row, "exit_price", "exit", "close_price")
    qty = _first(row, "qty", "quantity") or _first(evidence, "qty")

    base = {
        "schema_version": SCHEMA_VERSION,
        "gross_realized_R": gross_r,
        "execution_adjusted_R": None,
        "policy_realized_R": gross_r,
        "execution_adjusted_available": False,
        "execution_complete": False,
        "execution_coverage": 0.0,
        "quality": "GROSS_ONLY",
        "missing": [],
        "warnings": [],
        "paper_fill_unchanged": True,
        "affects_orders": False,
    }
    required = {"entry": entry, "stop": stop, "exit": exit_price, "qty": qty}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        base["missing"] = missing
        return base
    if entry is None or stop is None or exit_price is None or qty is None:
        return base
    risk_per_share = entry - stop
    risk_amount = risk_per_share * qty
    if entry <= 0 or qty <= 0 or risk_per_share <= 0 or risk_amount <= 0:
        base["missing"] = ["valid_risk_geometry"]
        return base

    try:
        from product.execution_reality import ExecutionRealityEngine

        result = ExecutionRealityEngine(shadow_mode=True).analyze_round_trip(
            qty=qty,
            entry_price=entry,
            exit_price=exit_price,
            stop_price=stop,
        ).to_dict()
    except Exception as exc:
        base["warnings"] = [f"execution_reality_error:{type(exc).__name__}"]
        return base

    adjusted = dict(result.get("execution_adjusted_result") or {})
    adjusted_pnl = _f(adjusted.get("pnl"))
    if adjusted_pnl is None:
        base["warnings"] = ["execution_adjusted_pnl_missing"]
        return base

    # The engine deliberately does not apply an estimated spread to the paper
    # fill. Learning is more conservative: subtract the labelled round-trip
    # estimate while keeping the actual paper fill untouched.
    estimated_spread_cost = 0.0
    spread_estimated = False
    fill = dict(result.get("fill") or {})
    for field in fill.get("fields") or []:
        if str(field.get("name") or "") != "bid_ask_spread":
            continue
        value = _f(field.get("value"))
        if value is None:
            continue
        if bool(field.get("estimated")):
            spread_estimated = True
            estimated_spread_cost += 2.0 * value * qty
    adjusted_pnl -= estimated_spread_cost
    adjusted_r = adjusted_pnl / risk_amount

    # Slippage remains unknown unless measured/supplied. We therefore label the
    # execution result conservative-but-partial rather than pretending perfect
    # microstructure knowledge.
    slippage_known = False
    for field in fill.get("fields") or []:
        if str(field.get("name") or "") == "slippage" and field.get("value") is not None:
            slippage_known = True
            break
    coverage = 1.0 if slippage_known else 0.8
    quality = "MEASURED_EXECUTION" if slippage_known and not spread_estimated else "CONSERVATIVE_PARTIAL_EXECUTION"
    policy_r = adjusted_r if gross_r is None else min(gross_r, adjusted_r)

    base.update({
        "execution_adjusted_R": round(adjusted_r, 6),
        "policy_realized_R": round(policy_r, 6),
        "execution_adjusted_available": True,
        "execution_complete": bool(slippage_known),
        "execution_coverage": coverage,
        "quality": quality,
        "estimated_round_trip_spread_cost": round(estimated_spread_cost, 6),
        "slippage_known": slippage_known,
        "execution_reality": result,
    })
    if not slippage_known:
        base["warnings"] = ["slippage_unknown_not_treated_as_zero"]
    return base
