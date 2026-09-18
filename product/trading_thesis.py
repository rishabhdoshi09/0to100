"""Canonical production trading thesis and best-trade objective.

The same manifest is stamped onto current recommendations, present paper
decisions and historical PIT simulations. A simulation whose thesis hash does
not match the approved production thesis is never silently treated as evidence
for that approval.

The objective deliberately does not maximize raw hit-rate alone. It prefers a
high calibrated probability of positive R only when reward/risk and lower-bound
expectancy remain defensible. Shadow/historical evidence may be displayed and
used to train challengers, but only a forward-proven PAPER_ACTIVE challenger may
change production ordering.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

OBJECTIVE_ID = "WIN_PROB_WITH_POSITIVE_EXPECTANCY"
OBJECTIVE_VERSION = "v1"


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if out != out else out
    except (TypeError, ValueError):
        return None


def manifest() -> dict[str, Any]:
    from product.pit_versions import current_versions
    from product.strategy_catalog import ensemble_identity

    ensemble = ensemble_identity()
    learner: dict[str, Any] = {}
    try:
        from product.challenger_learning import dashboard
        current = dict((dashboard() or {}).get("current") or {})
        if str(current.get("status") or "") == "PAPER_ACTIVE":
            learner = {
                "status": "PAPER_ACTIVE",
                "model_version": str(current.get("model_version") or ""),
                "promotion_reason": str(current.get("promotion_reason") or ""),
                "affects_selection": True,
            }
    except Exception:
        learner = {}

    payload = {
        "objective_id": OBJECTIVE_ID,
        "objective_version": OBJECTIVE_VERSION,
        "objective": (
            "Prefer the highest calibrated probability of positive R among "
            "hard-gate-eligible setups while preserving positive expectancy, "
            "reward/risk, uncertainty discipline and abstention."
        ),
        "strategy_id": ensemble.get("strategy_id"),
        "strategy_version": ensemble.get("strategy_version"),
        "rules_hash": ensemble.get("rules_hash"),
        "decision_versions": current_versions().as_dict(),
        "active_selection_learner": learner,
        "hard_invariants": [
            "hard risk and evidence vetoes remain authoritative",
            "historical-only evidence cannot promote production selection",
            "no trade is valid when uncertainty or expected payoff is weak",
            "live money remains locked",
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["thesis_hash"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return payload


def decision_quality(decision, *, ranking_score: float | None = None) -> dict[str, Any]:
    """Explain the win-probability/expectancy view without inventing a probability."""
    evidence = dict(
        (getattr(decision, "historical_evidence", {}) or {}).get("evidence_intelligence") or {}
    )
    active = dict(evidence.get("challenger_shadow") or {})
    active_p = _f(active.get("p_positive_R")) if active.get("affects_selection") else None
    measured_p = _f(evidence.get("calibrated_p_positive_R"))
    lower_p = _f(evidence.get("p_positive_R_lower_95"))
    lower_r = _f(evidence.get("lower_95_R"))
    mean_r = _f(evidence.get("shrunk_mean_R"))
    rr = _f(getattr(decision, "computed_expected_R", None))
    if callable(getattr(decision, "computed_expected_R", None)):
        rr = _f(decision.computed_expected_R)
    else:
        rr = _f(getattr(decision, "computed_expected_R", None))

    probability = active_p if active_p is not None else measured_p
    probability_source = (
        "ACTIVE_FORWARD_PROVEN_MODEL"
        if active_p is not None
        else "MEASURED_SHADOW_ONLY"
        if measured_p is not None
        else "INSUFFICIENT_EVIDENCE"
    )
    return {
        "thesis_hash": manifest()["thesis_hash"],
        "win_probability": None if probability is None else round(probability, 4),
        "win_probability_source": probability_source,
        "win_probability_lower_95": None if lower_p is None else round(lower_p, 4),
        "shrunk_expectancy_R": mean_r,
        "lower_95_R": lower_r,
        "reward_risk_R": rr,
        "effective_n": _f(evidence.get("effective_n")) or 0.0,
        "decision_confidence": _f(evidence.get("decision_confidence")),
        "setup_similarity": _f(evidence.get("setup_similarity")),
        "ranking_score": ranking_score,
        "active_learner_version": str(active.get("model_version") or "") if active_p is not None else "",
        "selection_can_be_changed_by_probability": active_p is not None,
        "abstention_is_valid": True,
    }


def active_learning_adjustment(decision) -> dict[str, Any]:
    """Bounded reorder-only adjustment for an already canonical decision."""
    try:
        from product.challenger_learning import score_decision

        scored = score_decision(decision)
    except Exception:
        scored = {}
    if not scored.get("available") or not scored.get("affects_selection"):
        return {
            **dict(scored or {}),
            "adjustment": 0.0,
            "reason": "no forward-proven active selection learner",
        }
    p = _f(scored.get("p_positive_R"))
    if p is None:
        return {**dict(scored), "adjustment": 0.0, "reason": "active learner probability unavailable"}
    # Same bounded scale as present paper-autopilot selection.
    adjustment = max(-5.0, min(3.0, (p - 0.50) * 10.0))
    return {
        **dict(scored),
        "adjustment": round(adjustment, 4),
        "reason": "forward-proven learner reorders only already-eligible names",
    }
