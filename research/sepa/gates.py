"""Separate statistical signal from deployment/paper readiness.

``research.harness.evaluate`` may return PROMOTE on PSR/DSR alone. That word
is unsafe in a research report when PIT/CA/OOS gates fail. R2 never surfaces
PROMOTE as a deployment label.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


_STAT_MAP = {
    "PROMOTE": "STATISTICAL_SIGNAL",
    "REJECT": "REJECT",
    "UNDERPOWERED": "UNDERPOWERED",
    "INCONCLUSIVE": "INCONCLUSIVE",
}


def statistical_gate(
    returns: Sequence[float],
    *,
    n_trials: int = 6,
    min_n: int = 30,
    require_block_ci: bool = True,
) -> dict[str, Any]:
    r = np.asarray(list(returns), dtype=float)
    packed: dict[str, Any] = {
        "n": int(r.size),
        "mean_r": None if r.size == 0 else float(np.mean(r)),
        "statistical_verdict": "UNDERPOWERED",
        "harness_verdict": None,
        "block_ci": None,
        "insight": "",
    }
    if r.size == 0:
        packed["insight"] = "No trades."
        return packed
    try:
        from research.harness import block_bootstrap_mean_ci, evaluate
        harness = evaluate(
            r, n_trials=n_trials, min_n=min_n, require_block_ci=require_block_ci,
        )
        packed["harness_verdict"] = getattr(harness, "verdict", None)
        packed["statistical_verdict"] = _STAT_MAP.get(
            str(packed["harness_verdict"]), "INCONCLUSIVE",
        )
        packed["insight"] = getattr(harness, "insight", "") or ""
        packed["psr"] = getattr(harness, "psr", None)
        packed["dsr"] = getattr(harness, "dsr", None)
        packed["p_value"] = getattr(harness, "p_value", None)
        if r.size >= 2:
            packed["block_ci"] = block_bootstrap_mean_ci(r, n_boot=800, seed=7)
    except Exception as exc:
        packed["statistical_verdict"] = "INCONCLUSIVE"
        packed["insight"] = str(exc)
    return packed


def deployment_eligible(
    *,
    statistical: Mapping[str, Any],
    pit_class: str,
    ca_complete: bool,
    n_post_warmup_years: float,
    has_unseen_block: bool,
    unseen_n: int,
    ci_lower_ok: bool,
    min_effective_n: int = 30,
    known_lookahead: bool = False,
    causality_ok: bool = True,
) -> dict[str, Any]:
    """Paper-shadow readiness. Fail closed. Never returns a PROMOTE string."""
    n = int(statistical.get("n") or 0)
    reasons = []
    if known_lookahead:
        reasons.append("known_universe_or_signal_lookahead")
    if not causality_ok:
        reasons.append("vcp_or_pivot_causality_defect")
    if pit_class != "PIT_STRONG" and not (pit_class == "PIT_DEGRADED" and ca_complete):
        # Degraded membership is allowed only with CA complete AND explicit
        # documentation; paper shadow still requires the other gates.
        if pit_class != "PIT_DEGRADED":
            reasons.append(f"pit_class={pit_class}")
    if not ca_complete:
        reasons.append("ca_verification_failed")
    if n_post_warmup_years < 5:
        reasons.append(f"post_warmup_years={n_post_warmup_years:.2f}<5")
    if not has_unseen_block:
        reasons.append("no_unseen_validation_block")
    if unseen_n < min_effective_n:
        reasons.append(f"unseen_n={unseen_n}<{min_effective_n}")
    if n < min_effective_n:
        reasons.append(f"n={n}<{min_effective_n}")
    if not ci_lower_ok:
        reasons.append("confidence_interval_not_acceptable")
    if statistical.get("statistical_verdict") != "STATISTICAL_SIGNAL":
        reasons.append(f"statistical={statistical.get('statistical_verdict')}")
    mean = statistical.get("mean_r")
    if mean is None or float(mean) <= 0:
        reasons.append("non_positive_expectancy")
    ok = len(reasons) == 0
    return {
        "deployment_eligible": ok,
        "paper_shadow": ok,
        "reasons": reasons,
        "label": "DEPLOYMENT_ELIGIBLE" if ok else "NOT_DEPLOYMENT_ELIGIBLE",
    }
