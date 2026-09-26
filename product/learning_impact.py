"""Explain whether QuantTerm learning is actually changing future paper selection.

This is a read-only product projection. It distinguishes historical/shadow evidence
from forward-proven behavior that is allowed to affect PAPER selection. It never
changes a decision and never authorizes live money.
"""
from __future__ import annotations

from typing import Any, Mapping


def _f(value: Any) -> float:
    try:
        out = float(value or 0.0)
        return out if out == out else 0.0
    except (TypeError, ValueError):
        return 0.0


def _current_board() -> dict[str, Any]:
    try:
        from product.decision_discovery_store import load_current
        return dict(load_current() or {})
    except Exception:
        return {}


def build_learning_impact() -> dict[str, Any]:
    challenger: dict[str, Any] = {}
    try:
        from product.challenger_learning import dashboard
        challenger = dict(dashboard() or {})
    except Exception:
        challenger = {}
    current_model = dict(challenger.get("current") or {})

    policies: list[dict[str, Any]] = []
    try:
        from product.learning_policy_store import load_policies
        policies = [
            dict(p) for p in (load_policies() or {}).get("policies") or []
            if isinstance(p, Mapping)
        ]
    except Exception:
        policies = []

    production_policies = [
        p for p in policies
        if p.get("affects_selection") is not False
        and str(p.get("production_status") or "") in {"ACTIVE", "ELIGIBLE"}
    ]
    policy_effects: dict[str, int] = {}
    for p in production_policies:
        effect = str(p.get("effect") or p.get("final_effect") or "").upper() or "MEASURED"
        policy_effects[effect] = policy_effects.get(effect, 0) + 1

    board = _current_board()
    decisions = [
        dict(r) for r in (board.get("decisions") or [])
        if isinstance(r, Mapping)
    ]
    # Quantify the actual ordering effect rather than merely saying that a
    # learner was consulted. Compare the same current decision set with and
    # without measured evidence/learning adjustments. This is descriptive
    # attribution, not a claim that the learned ordering is economically better.
    indexed = list(enumerate(decisions))
    base_order = sorted(
        indexed,
        key=lambda item: (
            -_f(item[1].get("base_score")),
            str(item[1].get("symbol") or ""),
            item[0],
        ),
    )
    learned_order = sorted(
        indexed,
        key=lambda item: (
            -_f(item[1].get("ranking_score")),
            str(item[1].get("symbol") or ""),
            item[0],
        ),
    )
    base_rank = {idx: rank for rank, (idx, _row) in enumerate(base_order, start=1)}
    learned_rank = {idx: rank for rank, (idx, _row) in enumerate(learned_order, start=1)}

    influenced: list[dict[str, Any]] = []
    rank_up = 0
    rank_down = 0
    score_only = 0
    max_abs_score_delta = 0.0
    for idx, row in indexed:
        evidence_adj = _f(row.get("evidence_adjustment"))
        learning_adj = _f(row.get("learning_adjustment"))
        if not evidence_adj and not learning_adj:
            continue
        base_score = _f(row.get("base_score"))
        ranking_score = _f(row.get("ranking_score"))
        score_delta = round(ranking_score - base_score, 6)
        before = int(base_rank.get(idx) or 0)
        after = int(learned_rank.get(idx) or 0)
        rank_change = before - after  # positive = moved up after measured learning
        if rank_change > 0:
            rank_up += 1
            effect = "RANK_UP"
        elif rank_change < 0:
            rank_down += 1
            effect = "RANK_DOWN"
        else:
            score_only += 1
            effect = "SCORE_CHANGED"
        max_abs_score_delta = max(max_abs_score_delta, abs(score_delta))
        influenced.append({
            "symbol": str(row.get("symbol") or "").upper(),
            "state": str(row.get("state") or ""),
            "base_score": row.get("base_score"),
            "ranking_score": row.get("ranking_score"),
            "score_delta": score_delta,
            "rank_before_measured": before,
            "rank_after_measured": after,
            "rank_change": rank_change,
            "measured_effect": effect,
            "evidence_adjustment": evidence_adj,
            "learning_adjustment": learning_adj,
            "why": str(row.get("why") or ""),
            "learner": dict(row.get("learning") or {}),
            "evidence": dict(row.get("evidence") or {}),
        })

    real_forward_n = int(current_model.get("real_forward_n") or 0)
    historical_n = int(current_model.get("historical_n") or 0)
    counterfactual_n = int(current_model.get("counterfactual_n") or 0)
    model_status = str(current_model.get("status") or "OBSERVING")
    model_active = bool(
        model_status == "PAPER_ACTIVE"
        and current_model.get("affects_selection")
    )
    policy_active = bool(production_policies)
    selection_influenced = bool(influenced)
    forward_validation = dict(current_model.get("forward_validation") or {})
    improvement_raw = forward_validation.get("improvement")
    lower_raw = forward_validation.get("improvement_lower_95")
    try:
        forward_improvement = None if improvement_raw is None else float(improvement_raw)
    except (TypeError, ValueError):
        forward_improvement = None
    try:
        forward_lower_95 = None if lower_raw is None else float(lower_raw)
    except (TypeError, ValueError):
        forward_lower_95 = None

    sim: dict[str, Any] = {}
    try:
        from product.autonomous_learning import dashboard as autonomous_learning_dashboard
        sim = dict(autonomous_learning_dashboard() or {})
    except Exception:
        sim = {}
    counts = dict(sim.get("counts") or {})

    if model_active or policy_active or selection_influenced:
        status = "ACTIVE_IN_PAPER_SELECTION"
        plain = (
            "Measured learning is affecting PAPER selection/ranking now. "
            "Hard risk gates remain authoritative."
        )
    elif real_forward_n or historical_n or counterfactual_n or int(counts.get("historical_decisions_simulated") or 0):
        status = "LEARNING_BUT_NOT_PROMOTED"
        plain = (
            "Learning evidence exists, but it has not yet earned permission to "
            "change production PAPER ordering. Historical replay remains shadow evidence."
        )
    else:
        status = "COLLECTING"
        plain = "No promoted learning effect yet; QuantTerm is still collecting outcome evidence."

    if (
        model_active
        and forward_improvement is not None
        and forward_lower_95 is not None
        and forward_improvement > 0.0
        and forward_lower_95 > 0.0
    ):
        benefit_status = "PROVEN_BETTER_IN_FORWARD_PAPER"
        benefit_plain = (
            "The promoted challenger has beaten the prior paper-ranking champion "
            "on exact-version forward probability calibration with a positive 95% lower bound."
        )
    elif model_active or policy_active or selection_influenced:
        benefit_status = "CHANGING_SELECTION_NOT_YET_AGGREGATE_PROVEN"
        benefit_plain = (
            "Learning is changing PAPER ranking, but the current aggregate forward "
            "comparison does not yet prove a statistically robust improvement."
        )
    elif real_forward_n or historical_n or counterfactual_n or int(counts.get("historical_decisions_simulated") or 0):
        benefit_status = "LEARNING_NOT_PROMOTED"
        benefit_plain = (
            "Useful evidence is being collected, but it is not yet allowed to change "
            "production PAPER ranking."
        )
    else:
        benefit_status = "COLLECTING"
        benefit_plain = "There is not enough measured outcome evidence yet to judge benefit."

    return {
        "schema_version": 1,
        "status": status,
        "plain": plain,
        "benefit_status": benefit_status,
        "benefit_plain": benefit_plain,
        "forward_improvement_brier": forward_improvement,
        "forward_improvement_lower_95": forward_lower_95,
        "selection_is_currently_changed": selection_influenced,
        "current_decisions_influenced": len(influenced),
        "rank_impact": {
            "moved_up": rank_up,
            "moved_down": rank_down,
            "score_changed_without_rank_move": score_only,
            "max_abs_score_delta": round(max_abs_score_delta, 6),
        },
        "influenced_examples": influenced[:8],
        "challenger": {
            "status": model_status,
            "model_version": str(current_model.get("model_version") or ""),
            "trained_n": int(current_model.get("trained_n") or 0),
            "real_forward_n": real_forward_n,
            "historical_n": historical_n,
            "counterfactual_n": counterfactual_n,
            "affects_selection": model_active,
            "forward_validation": forward_validation,
            "promotion_reason": str(current_model.get("promotion_reason") or ""),
        },
        "policies": {
            "total": len(policies),
            "production_effective": len(production_policies),
            "effects": policy_effects,
        },
        "simulation": {
            "historical_decisions_simulated": int(counts.get("historical_decisions_simulated") or 0),
            "historical_virtual_paper_trades": int(counts.get("historical_virtual_paper_trades") or 0),
            "forward_paper_decisions": int(counts.get("forward_paper_decisions") or 0),
            "correct_rejects": int(counts.get("correct_rejects") or 0),
            "avoided_losers": int(counts.get("avoided_losers") or 0),
            "missed_winners": int(counts.get("missed_winners") or 0),
        },
        "contract": {
            "historical_only_can_promote": False,
            "forward_proof_required_for_model": True,
            "learning_may_relax_hard_risk_gates": False,
            "live_money_affected": False,
        },
        "live_locked": True,
    }
