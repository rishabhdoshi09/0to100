"""Composition seam for the paper-only NSE F&O directional/options lane."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from options.directional_selector import select_option_contracts
from product.fo_setup import score_fo_setup


def evaluate_fo_opportunity(
    *,
    underlying_features: Mapping[str, Any],
    direction: str,
    option_contracts: Sequence[Mapping[str, Any]],
    iv_percentile: float | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Evaluate underlying first, then options. Weak underlyings never leak through."""
    setup = score_fo_setup(underlying_features, direction)
    base = {
        "symbol": str(underlying_features.get("symbol") or ""),
        "direction": str(direction).upper(),
        "setup": setup,
        "paper_only": True,
        "live_execution_allowed": False,
        "evidence_state": "UNCALIBRATED",
    }
    if not setup["tradable"]:
        return {
            **base,
            "decision": "NO_TRADE",
            "reason": "UNDERLYING_SETUP_REJECTED",
            "options": {
                "eligible_count": 0,
                "best_contracts": [],
                "all_candidates": [],
                "skipped": True,
            },
        }

    expected = setup["expected_move"]
    options = select_option_contracts(
        option_contracts,
        direction=direction,
        spot=float(underlying_features.get("price") or 0.0),
        expected_move_pct=float(expected["mid_pct"]),
        horizon=str(expected["horizon"]),
        holding_days=int(expected["holding_days"]),
        iv_percentile=iv_percentile,
        limit=top_n,
    )
    if not options["best_contracts"]:
        return {
            **base,
            "decision": "NO_OPTION_TRADE",
            "reason": "UNDERLYING_VALID_BUT_NO_ELIGIBLE_OPTION",
            "options": options,
        }

    return {
        **base,
        "decision": "PAPER_OPTION_CANDIDATE",
        "reason": "UNDERLYING_AND_OPTION_GATES_PASSED",
        "options": options,
        "selected_contract": options["best_contracts"][0],
    }
