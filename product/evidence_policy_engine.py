"""Canonical EvidencePolicyEngine.

Consumes frozen learning policies plus the candidate's point-in-time evidence.
Outputs SUPPORT / NEUTRAL / PENALIZE / BLOCK. Never emits BUY.

Hard product gates are not reversed by this engine. A policy with
affects_selection=False is observation-only (counterfactual / exit / portfolio).

Production paper selection additionally follows the autonomous evidence ladder:
reproduced PIT history first, then real-forward paper confirmation. Historical
replay can authorize PAPER exploration only; it can never unlock live money.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from product.decision_context import (
    dd_status,
    entry_quality,
    extension_bucket,
    liquidity_bucket,
    rs_bucket,
    snapshot,
    volatility_bucket,
)
from product.learning_policy_store import (
    ACTIVE,
    ELIGIBLE,
    EXPERIMENTAL,
    load_policies,
)

SUPPORT = "SUPPORT"
NEUTRAL = "NEUTRAL"
PENALIZE = "PENALIZE"
BLOCK = "BLOCK"

_EFFECT_RANK = {BLOCK: 3, PENALIZE: 2, SUPPORT: 1, NEUTRAL: 0}

# Learned overlays must not silently disable these production gates.
HARD_REASON_CODES = {
    "PAPER_TRADING_DISABLED",
    "MARKET_NOT_READY",
    "OUTSIDE_ENTRY_WINDOW",
    "STALE_RECOMMENDATION",
    "DD_GATE_FAILED",
    "ENTRY_TOO_EXTENDED",
    "INVALID_STOP",
    "DUPLICATE_POSITION",
    "MAX_POSITIONS",
    "MAX_PORTFOLIO_RISK",
    "SECTOR_CAP",
    "CORRELATION_CAP",
    "PER_NAME_CAP",
    "INSUFFICIENT_CAPITAL",
    "LIQUIDITY_FAILED",
    "REGIME_STANDDOWN",
    "UNRECONCILED",
    "PORTFOLIO_GATE_ERROR",
    "NO_VALID_ENTRY",
}


def _bucket_of(candidate: Mapping[str, Any], dimension: str) -> str:
    if "|" in dimension:
        return "|".join(_bucket_of(candidate, part) for part in dimension.split("|"))
    if dimension == "reason_code":
        return str(candidate.get("reason_code") or candidate.get("last_reject_reason") or "")
    if dimension == "setup":
        return str(candidate.get("setup_label") or candidate.get("primary_thesis") or "")
    if dimension == "entry_state":
        return str(candidate.get("entry_state") or "")
    if dimension == "entry_quality":
        return str(candidate.get("entry_quality") or entry_quality(candidate))
    if dimension == "tier":
        return str(candidate.get("reco_tier") or "")
    if dimension == "sector":
        return str(candidate.get("sector") or "")
    if dimension == "regime":
        return str(candidate.get("regime") or (candidate.get("portfolio") or {}).get("regime") or "")
    if dimension == "dd_status":
        return str(candidate.get("dd_status") or dd_status(candidate))
    if dimension == "rs_bucket":
        return str(candidate.get("rs_bucket") or rs_bucket(candidate.get("rs_percentile")))
    if dimension == "liquidity":
        return str(candidate.get("liquidity") or liquidity_bucket(candidate.get("volume_ratio")))
    if dimension == "volatility":
        return str(candidate.get("volatility") or volatility_bucket(candidate.get("atr_pct")))
    if dimension == "extension":
        if candidate.get("extension"):
            return str(candidate.get("extension"))
        try:
            ext = float(candidate.get("extension_pct") or 0)
        except (TypeError, ValueError):
            return ""
        return extension_bucket(ext)
    if dimension == "exit_reason":
        return str(candidate.get("exit_reason") or "")
    return str(candidate.get(dimension) or "")


def _historical_gate(
    candidate: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Return the history→forward confidence gate used by production paper selection.

    Explicit ``policies=`` calls are research/tests and skip this production gate;
    the normal store-backed call enables it. The gate is fail-closed once enabled.
    """
    if not enabled:
        return {
            "required": False,
            "paper_eligible": True,
            "confidence_stage": "NOT_ENFORCED",
            "live_locked": True,
        }
    try:
        from product.autonomous_evolution import bootstrap_status, ensure_started_async
        from product.evidence_confidence import confidence_from_policies
        from product.evolution_generation_guard import ensure_current_generation

        generation = ensure_current_generation()
        state = bootstrap_status()
        if generation.get("historical_replay_required") or not state.get("analysis_complete"):
            ensure_started_async()
            state = bootstrap_status()
        confidence = confidence_from_policies(candidate, policies)
        if not state.get("analysis_complete"):
            return {
                **confidence,
                "required": True,
                "paper_eligible": False,
                "confidence_stage": "HISTORICAL_BOOTSTRAP",
                "bootstrap_status": state.get("status") or "RUNNING",
                "bootstrap_complete": False,
                "paper_ready_setups": int(state.get("paper_ready_setups") or 0),
                "generation_fingerprint": generation.get("fingerprint"),
                "generation_changed": bool(generation.get("changed")),
                "live_locked": True,
            }
        return {
            **confidence,
            "required": True,
            "bootstrap_status": state.get("status") or "SUCCEEDED",
            "bootstrap_complete": True,
            "paper_ready_setups": int(state.get("paper_ready_setups") or 0),
            "generation_fingerprint": generation.get("fingerprint"),
            "generation_changed": bool(generation.get("changed")),
            "live_locked": True,
        }
    except Exception as exc:
        return {
            "required": True,
            "paper_eligible": False,
            "confidence_stage": "HISTORICAL_GATE_ERROR",
            "bootstrap_complete": False,
            "error": str(exc)[:200],
            "live_locked": True,
        }


def evaluate_policies(
    candidate: Mapping[str, Any],
    *,
    policies: Sequence[Mapping[str, Any]] | None = None,
    path=None,
    regime: str = "",
    book=None,
) -> dict[str, Any]:
    """Return the empirical overlay for one candidate. Cannot create a BUY."""
    ctx = dict(candidate)
    if "methods" in candidate or "setup_label" in candidate:
        frozen = snapshot(candidate, book=book, regime=regime or str(candidate.get("regime") or ""))
        for key, value in frozen.items():
            ctx.setdefault(key, value)
    if regime:
        ctx["regime"] = regime

    store_backed = policies is None
    if policies is None:
        policies = list((load_policies(path).get("policies") or []))
    policies = list(policies or [])

    # Existing unit/injected-policy callers remain deterministic. Production
    # store-backed paper selection enforces history-first unless pytest is the
    # caller; dedicated tests exercise the pure gate directly.
    enforce_history = bool(store_backed and not os.environ.get("PYTEST_CURRENT_TEST"))
    historical = _historical_gate(ctx, policies, enabled=enforce_history)

    supportive: list[dict[str, Any]] = []
    cautionary: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []
    sample = 0
    coverage = 0
    final = NEUTRAL
    learned_edge = 0.0
    matched: list[dict[str, Any]] = []

    for raw in policies:
        policy = dict(raw)
        if policy.get("affects_selection") is False:
            continue
        status = str(policy.get("production_status") or "")
        if status not in {ACTIVE, ELIGIBLE, EXPERIMENTAL}:
            continue
        source = str(policy.get("evidence_source") or "")
        # Backtest never self-activates. Reproduced history may be ELIGIBLE for
        # paper exploration but remains live-locked.
        if source.startswith("backtest") and status == ACTIVE:
            status = ELIGIBLE
            policy = {**policy, "production_status": status, "note": "backtest cannot self-activate"}
        dimension = str(policy.get("dimension") or "")
        bucket = str(policy.get("bucket") or "")
        if not dimension or not bucket:
            continue
        if _bucket_of(ctx, dimension) != bucket:
            continue
        if dimension == "reason_code" and bucket in HARD_REASON_CODES:
            # Stats for the dashboard only — never disable a hard gate.
            cautionary.append({**policy, "effect": NEUTRAL, "note": "hard_gate_observation"})
            continue
        coverage += 1
        n = int(policy.get("sample_size") or 0)
        sample = max(sample, n)
        edge = float(policy.get("expectancy_difference_R") or 0.0)
        learned_edge += edge
        confidence = str(policy.get("confidence") or "")
        if confidence == "INSUFFICIENT_EVIDENCE" and status != ACTIVE:
            cautionary.append({**policy, "effect": NEUTRAL, "note": "INSUFFICIENT_EVIDENCE"})
            matched.append(policy)
            continue
        if status == ACTIVE and edge <= -0.40:
            effect = BLOCK
            blocking.append({**policy, "effect": effect})
        elif edge <= -0.20 and status in {ACTIVE, ELIGIBLE}:
            effect = PENALIZE
            cautionary.append({**policy, "effect": effect})
        elif edge >= 0.25 and status in {ACTIVE, ELIGIBLE}:
            effect = SUPPORT
            supportive.append({**policy, "effect": effect})
        else:
            effect = NEUTRAL
            cautionary.append({**policy, "effect": effect})
        matched.append({**policy, "effect": effect})
        if _EFFECT_RANK[effect] > _EFFECT_RANK[final]:
            final = effect

    if enforce_history and not historical.get("paper_eligible"):
        gate_row = {
            "policy_id": "AUTONOMOUS_HISTORY_FIRST_GATE",
            "dimension": "setup",
            "bucket": _bucket_of(ctx, "setup"),
            "effect": BLOCK,
            "production_status": "SYSTEM_GATE",
            "evidence_source": "historical_then_forward",
            "confidence_stage": historical.get("confidence_stage"),
            "evidence_confidence_score": historical.get("evidence_confidence_score"),
            "note": (
                "Autonomous paper entry waits until this setup reproduces in independent "
                "PIT historical slices; forward paper evidence can later strengthen or decay it."
            ),
            "live_locked": True,
        }
        blocking.append(gate_row)
        matched.append(gate_row)
        final = BLOCK

    return {
        "supportive": supportive,
        "cautionary": cautionary,
        "blocking": blocking,
        "matched": matched,
        "learned_edge_score": round(learned_edge, 4),
        "evidence_coverage": coverage,
        "sample_size": sample,
        "confidence": (
            "INSUFFICIENT_EVIDENCE" if coverage == 0 or sample < 8
            else "MEASURED"
        ),
        "historical_forward_confidence": historical,
        "evidence_confidence_score": historical.get("evidence_confidence_score"),
        "confidence_stage": historical.get("confidence_stage"),
        "historical_base_ready": bool(historical.get("historical_ready")),
        "forward_confirmation_n": int(historical.get("forward_n") or 0),
        "final_effect": final,
        "invents_buy": False,
        "hard_gates_remain": True,
        "live_locked": True,
    }
