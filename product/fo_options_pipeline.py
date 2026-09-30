"""Composition seam for the paper-only NSE F&O directional/options lane.

This is the ONE place a real candidate's contract gets chosen -- the single
"contract-selection seam" the rest of the desk depends on. Everything
downstream (product.fo_paper_runtime's paper execution, product.fno_ranking's
per-cycle sort) reads whatever this function put in ``selected_contract``; it
never re-selects among alternatives, because by the time those layers run,
every alternative contract this scan considered has already been discarded.
That is precisely why contract-evidence learning MUST happen here, before
the alternatives are thrown away, rather than downstream as an after-the-
fact annotation of whatever raw score happened to pick first.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from options.directional_selector import select_option_contracts
from product.fno_contract_evidence import contract_ranking_adjustment, contract_selection_context
from product.fo_setup import score_fo_setup


def _rerank_by_learned_score(
    eligible: list[dict[str, Any]],
    *,
    setup: Mapping[str, Any],
    path: str | None,
) -> list[dict[str, Any]]:
    """Attach the learned contract-selection adjustment to EVERY eligible
    contract, then re-sort by learned_contract_score. Hard eligibility
    (liquidity/risk/expiry/theta/premium/spread/delta-band gates) has already
    been decided by options.directional_selector.score_option_contract before
    this ever runs -- evidence here can only re-order contracts that already
    cleared every one of those gates, never rescue one that did not.
    """
    context = contract_selection_context(setup)
    rows: list[dict[str, Any]] = []
    for row in eligible:
        row = dict(row)
        raw_score = float(row.get("score") or 0.0)
        evidence = contract_ranking_adjustment(
            row, path=path,
            holding_horizon=context.get("holding_horizon"),
            regime=context.get("regime"),
            setup_type=context.get("setup_type"),
        )
        adjustment = float(evidence.get("adjustment") or 0.0)
        row["raw_contract_score"] = raw_score
        row["contract_evidence"] = evidence
        row["learned_contract_score"] = round(raw_score + adjustment, 4)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["learned_contract_score"]),
            float(row.get("projected_return_at_expected_move_pct") or 0.0),
            float(row.get("gamma_delta_change_for_1pct_move") or 0.0),
            -float(row.get("vega_pct_of_premium_per_vol_point") or 999.0),
            -float(row.get("spread_pct") or 999.0),
        ),
        reverse=True,
    )
    return rows


def evaluate_fo_opportunity(
    *,
    underlying_features: Mapping[str, Any],
    direction: str,
    option_contracts: Sequence[Mapping[str, Any]],
    iv_percentile: float | None = None,
    top_n: int = 5,
    evidence_path: str | None = None,
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
        underlying_stop_price=float((setup.get("underlying_trade_plan") or {}).get("stop") or 0.0),
        iv_percentile=iv_percentile,
        limit=top_n,
    )
    # THE contract-selection seam: every contract that cleared
    # options.directional_selector's hard gates (liquidity/risk/expiry/theta/
    # premium/spread/delta-band/score floor -- see score_option_contract) is
    # eligible for learning to re-rank. Evidence never sees, and can never
    # rescue, a contract that did not already clear those gates on raw score
    # alone -- it can only re-order what "all_candidates" already marked
    # eligible=True.
    eligible_by_raw_score = [
        row for row in options.get("all_candidates") or [] if row.get("eligible")
    ]
    learned = _rerank_by_learned_score(eligible_by_raw_score, setup=setup, path=evidence_path)
    options = {**options, "best_contracts": learned[: max(1, int(top_n))]}
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
