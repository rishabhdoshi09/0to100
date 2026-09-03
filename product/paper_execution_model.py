"""Reasonable paper fill semantics. Not perfect backtest fills.

Uses the existing execution_reality engine. Shadow hypotheses never
enter these statistics.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.execution_reality import ExecutionRealityEngine
from product.shadow_execution import PAPER_ENTERED, SHADOW_NOT_EXECUTED


def model_fill(
    rec: Mapping[str, Any],
    *,
    last: float | None = None,
    open_px: float | None = None,
    spread_bps: float = 5.0,
    slippage_bps: float = 3.0,
    adv_shares: float | None = None,
) -> dict[str, Any]:
    if rec.get("status") == SHADOW_NOT_EXECUTED or rec.get("not_a_trade"):
        return {
            "status": SHADOW_NOT_EXECUTED,
            "paper_executed": False,
            "not_a_trade": True,
            "inflates_paper_stats": False,
        }
    engine = ExecutionRealityEngine()
    entry = rec.get("entry") or rec.get("entry_price")
    stop = rec.get("stop") or rec.get("stop_price")
    target = rec.get("target") or rec.get("target_price") or entry
    qty = rec.get("qty") or rec.get("quantity") or 1
    try:
        result = engine.analyze_round_trip(
            side="BUY",
            entry_price=float(entry),
            exit_price=float(target),
            qty=float(qty),
            slippage_bps=slippage_bps,
            bid=float(last or entry) * (1 - spread_bps / 2e4),
            ask=float(last or entry) * (1 + spread_bps / 2e4),
            open_price=open_px,
            stop_price=float(stop) if stop else None,
            bar_volume=adv_shares,
        )
        if hasattr(result, "__dict__"):
            result = {k: v for k, v in vars(result).items() if not k.startswith("_")}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:160]}
    return {
        "status": PAPER_ENTERED,
        "paper_executed": True,
        "inflates_paper_stats": False,
        "perfect_fill": False,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "engine": result,
        "note": "Paper fill is execution-adjusted. Gaps and spread can move the fill.",
    }
