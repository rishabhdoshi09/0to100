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
    influenced: list[dict[str, Any]] = []
    for row in decisions:
        evidence_adj = _f(row.get("evidence_adjustment"))
        learning_adj = _f(row.get("learning_adjustment"))
        if not evidence_adj and not learning_adj:
            continue
        influenced.append({
            "symbol": str(row.get("symbol") or "").upper(),
            "state": str(row.get("state") or ""),
            "base_score": row.get("base_score"),
            "ranking_score": row.get("ranking_score"),
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

    return {
        "schema_version": 1,
        "status": status,
        "plain": plain,
        "selection_is_currently_changed": selection_influenced,
        "current_decisions_influenced": len(influenced),
        "influenced_examples": influenced[:8],
        "challenger": {
            "status": model_status,
            "model_version": str(current_model.get("model_version") or ""),
            "trained_n": int(current_model.get("trained_n") or 0),
            "real_forward_n": real_forward_n,
            "historical_n": historical_n,
            "counterfactual_n": counterfactual_n,
            "affects_selection": model_active,
            "forward_validation": dict(current_model.get("forward_validation") or {}),
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
