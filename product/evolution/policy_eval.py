"""Pure, read-only policy evaluation over a frozen market snapshot.

PAPER-execution safety boundary: this module contains NO broker call, NO
book mutation, NO Telegram send, and NO import of execution.trade_executor
or book-mutating functions. It only transforms a canonical score/context
with one immutable bounded policy manifest. The real PAPER engine resolves
the domain's current CHAMPION before selection and applies these same bounded
weights inside its existing hard-gated decision seam; Challengers use this
module read-only against frozen snapshots. Execution remains owned by the
existing PAPER engine, never by Evolution itself.

Challengers are bounded, interpretable re-weightings of the SAME real
scoring function QuantTerm already runs (product.decision_context.
score_breakdown), never a parallel scoring engine and never an arbitrary
random strategy. A policy with every weight at its neutral default produces
byte-identical output to the unweighted production score_breakdown() --
that is what makes a freshly-registered CHAMPION policy a true baseline.
"""
from __future__ import annotations

from typing import Any, Mapping

from product import decision_context

ENTER_NOW = "ENTER_NOW"
REJECT = "REJECT"

# ── bounded, named weight vocabulary -- NOT arbitrary free parameters ───────
# Multipliers scale an EXISTING, already-interpretable score_breakdown() part.
# Clamped so one policy can never silently dominate via an extreme weight.
_MULT_KEYS = (
    "tier_mult",
    "evidence_families_mult",
    "desk_score_mult",
    "dd_mult",
    "relative_strength_mult",
    "entry_quality_mult",
    "empirical_policy_mult",
)
_MULT_MIN, _MULT_MAX = 0.0, 3.0
# Additive bonuses for dimensions score_breakdown doesn't itemize separately.
_BONUS_MIN, _BONUS_MAX = -5.0, 5.0
# Regimes where a stricter policy may stand down harder than baseline.
_RISK_OFF_REGIMES = ("DISTRIBUTION", "TRENDING_BEAR", "BEAR")

RECOGNIZED_WEIGHTS = _MULT_KEYS + (
    "volume_confirmation_bonus",
    "sector_confirmation_bonus",
    "regime_standdown_mult",
    "min_empirical_sample",
    "extension_penalty_mult",
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weight(policy: Mapping[str, Any], key: str, default: float) -> float:
    weights = policy.get("weights") or {}
    try:
        raw = float(weights.get(key, default))
    except (TypeError, ValueError):
        return default
    return raw


def apply_policy_weights(
    breakdown: Mapping[str, Any], context: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-weight the REAL score_breakdown() parts per the policy's bounded
    manifest. Returns a new breakdown dict; never mutates the input."""
    parts = [dict(p) for p in breakdown.get("parts") or []]
    for part in parts:
        mult_key = f"{part['key']}_mult"
        if mult_key in _MULT_KEYS:
            mult = _clamp(_weight(policy, mult_key, 1.0), _MULT_MIN, _MULT_MAX)
            part["points"] = round(float(part["points"]) * mult, 4)
            if mult != 1.0:
                part["source"] = f"{part['source']} (policy weight x{mult:g})"

    extra_parts: list[dict[str, Any]] = []
    vol_bonus = _clamp(_weight(policy, "volume_confirmation_bonus", 0.0), _BONUS_MIN, _BONUS_MAX)
    if vol_bonus:
        liquidity = str(context.get("liquidity") or "")
        points = vol_bonus if liquidity == "liq_high" else (-vol_bonus if liquidity == "liq_thin" else 0.0)
        extra_parts.append({
            "key": "policy_volume_confirmation", "points": round(points, 4),
            "source": f"liquidity={liquidity or 'unknown'} (policy bonus {vol_bonus:+g})",
        })
    sector_bonus = _clamp(_weight(policy, "sector_confirmation_bonus", 0.0), _BONUS_MIN, _BONUS_MAX)
    if sector_bonus:
        sector_status = str(context.get("method_sector") or "unknown")
        points = sector_bonus if sector_status == "pass" else (-sector_bonus if sector_status == "fail" else 0.0)
        extra_parts.append({
            "key": "policy_sector_confirmation", "points": round(points, 4),
            "source": f"method_sector={sector_status} (policy bonus {sector_bonus:+g})",
        })
    regime_mult = _clamp(_weight(policy, "regime_standdown_mult", 1.0), _MULT_MIN, _MULT_MAX)
    if regime_mult != 1.0 and str(context.get("regime") or "") in _RISK_OFF_REGIMES:
        extra_parts.append({
            "key": "policy_regime_standdown", "points": round(-4.0 * regime_mult, 4),
            "source": f"regime={context.get('regime')} (policy standdown x{regime_mult:g})",
        })

    extension_mult = _clamp(_weight(policy, "extension_penalty_mult", 1.0), _MULT_MIN, _MULT_MAX)
    extension = context.get("extension_pct")
    try:
        extension_value = float(extension) if extension is not None else None
    except (TypeError, ValueError):
        extension_value = None
    if extension_mult != 1.0 and extension_value is not None and extension_value > 0:
        extra_parts.append({
            "key": "policy_extension_penalty",
            "points": round(-min(5.0, extension_value / 2.0) * extension_mult, 4),
            "source": f"extension_pct={extension_value:g} (policy penalty x{extension_mult:g})",
        })

    all_parts = parts + extra_parts
    total = round(sum(float(p["points"]) for p in all_parts), 4)
    return {**breakdown, "parts": all_parts, "selection_rank": total}


def evaluate_snapshot(
    snapshot: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure function: snapshot + policy -> what this policy would decide.

    Never mutates any store, never places an order. Returns a complete,
    explainable, reproducible verdict: given the same (snapshot, policy)
    this always returns the same result (score_breakdown is deterministic
    over its inputs; the only external call here is read-only).
    """
    ctx = dict(snapshot.get("context") or {})
    card = dict(snapshot.get("card") or {})
    symbol = str(snapshot.get("symbol") or card.get("symbol") or "").upper()

    base_breakdown = decision_context.score_breakdown(card, policy=None, context=ctx)
    weighted = apply_policy_weights(base_breakdown, ctx, policy)
    adjusted_score = float(weighted["selection_rank"])

    min_sample = policy.get("weights", {}).get("min_empirical_sample")
    sample_size = int((ctx.get("empirical") or {}).get("sample_size") or 0)
    if min_sample is not None and sample_size < int(min_sample):
        return {
            "policy_id": policy.get("policy_id"),
            "market_snapshot_id": snapshot.get("market_snapshot_id"),
            "domain": snapshot.get("domain"),
            "symbol": symbol,
            "decision": REJECT,
            "reason_code": "POLICY_MIN_SAMPLE_NOT_MET",
            "adjusted_score": adjusted_score,
            "breakdown": weighted,
            "entry": ctx.get("entry"),
            "stop": ctx.get("stop"),
            "target": ctx.get("target"),
        }

    missing = ctx.get("missing_evidence") or []
    if "entry" in missing or "stop" in missing:
        decision, reason = REJECT, "NO_VALID_ENTRY"
    elif str(ctx.get("dd_status") or "") in {"FAIL", "FAILED", "BLOCK", "AVOID"}:
        decision, reason = REJECT, "DD_GATE_FAILED"
    else:
        decision, reason = ENTER_NOW, "ELIGIBLE"

    return {
        "policy_id": policy.get("policy_id"),
        "market_snapshot_id": snapshot.get("market_snapshot_id"),
        "domain": snapshot.get("domain"),
        "symbol": symbol,
        "decision": decision,
        "reason_code": reason,
        "adjusted_score": adjusted_score,
        "breakdown": weighted,
        "entry": ctx.get("entry"),
        "stop": ctx.get("stop"),
        "target": ctx.get("target"),
        "sector": ctx.get("sector"),
        "setup_label": ctx.get("setup_label"),
    }
