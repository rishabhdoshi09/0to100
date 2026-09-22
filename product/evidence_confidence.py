"""Evidence confidence ladder: reproduced history -> trustworthy forward paper.

The score exposed here is an evidence-strength composite, not a win
probability. Positive forward confirmation is accepted only from taken-paper
sources that the learning policy marked selection-eligible. Conservative
negative gross evidence may still decay confidence.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _setup(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("setup")
        or candidate.get("setup_label")
        or candidate.get("primary_thesis")
        or ""
    ).strip()


def confidence_from_policies(
    candidate: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    *,
    generation_fingerprint: str = "",
) -> dict[str, Any]:
    setup = _setup(candidate)
    hist = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"HIST_SETUP::{setup}"),
        {},
    )
    raw_forward = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"SETUP::{setup}"),
        {},
    )

    hist_generation = str(hist.get("generation_fingerprint") or "")
    generation_match = bool(
        not generation_fingerprint
        or (hist_generation and hist_generation == generation_fingerprint)
    )
    hist_ready = bool(hist.get("historical_reproduced_positive") and generation_match)
    hist_score = float(hist.get("historical_confidence_score") or 0.0)

    source = str(raw_forward.get("evidence_source") or "")
    is_taken_forward = source.startswith("paper_forward_taken")
    raw_n = int(raw_forward.get("sample_size") or 0)
    raw_edge = float(raw_forward.get("expectancy_difference_R") or 0.0)
    affects = raw_forward.get("affects_selection") is not False

    # Positive confidence is allowed only from actual taken-paper evidence whose
    # integrity contract permits it to affect selection. Gross-only positive
    # evidence stays observation-only. Negative gross evidence remains a
    # conservative upper bound and is allowed to reduce confidence.
    trusted_positive = bool(is_taken_forward and affects and raw_edge > 0)
    trusted_negative = bool(is_taken_forward and raw_edge < 0)
    forward_usable = bool(trusted_positive or trusted_negative)
    forward_n = raw_n if forward_usable else 0
    forward_edge = raw_edge if forward_usable else 0.0

    forward_sample = min(1.0, forward_n / 30.0)
    edge_component = math.tanh(forward_edge / 0.50) if forward_n else 0.0
    forward_score = max(
        0.0,
        min(100.0, 50.0 + 30.0 * edge_component + 20.0 * forward_sample),
    )

    if hist and generation_fingerprint and not generation_match:
        combined = 0.0
        stage = "HISTORICAL_GENERATION_MISMATCH"
    elif not hist_ready:
        combined = min(49.0, hist_score)
        stage = "HISTORICAL_UNPROVEN"
    elif forward_n <= 0:
        combined = min(79.0, hist_score)
        stage = (
            "FORWARD_EVIDENCE_UNTRUSTED"
            if raw_n and not forward_usable
            else "HISTORICAL_BASE"
        )
    else:
        forward_weight = min(0.70, 0.20 + 0.50 * forward_sample)
        combined = hist_score * (1.0 - forward_weight) + forward_score * forward_weight
        if forward_n < 8:
            stage = "FORWARD_EARLY"
        elif forward_edge <= -0.20:
            stage = "FORWARD_DECAYED"
        elif forward_n < 20:
            stage = "FORWARD_CALIBRATING"
        else:
            stage = "FORWARD_CONFIRMED" if forward_edge > 0 else "FORWARD_WEAK"

    combined = round(max(0.0, min(95.0, combined)), 1)
    paper_eligible = bool(hist_ready)
    if forward_n >= 5 and forward_edge <= -0.25:
        paper_eligible = False

    return {
        "setup": setup,
        "historical_ready": hist_ready,
        "historical_generation_fingerprint": hist_generation,
        "generation_fingerprint": generation_fingerprint,
        "historical_generation_match": generation_match,
        "historical_n": int(hist.get("sample_size") or 0),
        "historical_mean_R": hist.get("expectancy_R"),
        "historical_splits": int(hist.get("splits_tested") or 0),
        "historical_positive_splits": int(hist.get("positive_splits") or 0),
        "historical_confidence_score": round(hist_score, 1),
        "forward_n": forward_n,
        "forward_observed_n": raw_n,
        "forward_mean_R": raw_forward.get("expectancy_R"),
        "forward_source": source,
        "forward_policy_id": str(raw_forward.get("policy_id") or ""),
        "forward_policy_version": int(raw_forward.get("version") or 0),
        "forward_thesis_hash": str(raw_forward.get("thesis_hash") or ""),
        "forward_rules_hash": str(raw_forward.get("rules_hash") or ""),
        "forward_calibration_snapshot_id": str(raw_forward.get("calibration_snapshot_id") or ""),
        "forward_version_identity_pinned": bool(
            raw_forward.get("thesis_hash")
            or raw_forward.get("rules_hash")
            or raw_forward.get("calibration_snapshot_id")
        ),
        "forward_trusted_positive": trusted_positive,
        "forward_trusted_negative": trusted_negative,
        "forward_confidence_score": round(forward_score, 1) if forward_n else 0.0,
        "evidence_confidence_score": combined,
        "confidence_stage": stage,
        "paper_eligible": paper_eligible,
        "is_win_probability": False,
        "live_locked": True,
    }


def confidence_breakdown(
    candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    final_effect: str = "NEUTRAL",
    policy_sample_size: int = 0,
    policy_coverage: int = 0,
    matched_policies: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Explain the existing evidence-confidence score without inventing sub-scores.

    Regime, sector, setup quality and extension are decision context. Unless an
    independently measured conditional policy exists, they remain labels/facts —
    never fabricated percentages that appear to add up to the final confidence.
    """
    row = dict(candidate or {})
    ev = dict(evidence or {})
    methods = dict(row.get("methods") or {})
    if not methods:
        methods = {}

    def method_status(name: str) -> str:
        raw = methods.get(name) or {}
        return str(raw.get("status") or "unknown") if isinstance(raw, Mapping) else "unknown"

    extension_pct = row.get("extension_pct")
    if extension_pct is None:
        extension_pct = row.get("ema20_extension_pct")
    if extension_pct is None:
        extension_pct = row.get("from_ema20_pct")
    try:
        extension_pct = float(extension_pct) if extension_pct is not None else None
    except (TypeError, ValueError):
        extension_pct = None

    forward_identity = {
        "policy_id": str(ev.get("forward_policy_id") or ""),
        "policy_version": int(ev.get("forward_policy_version") or 0),
        "thesis_hash": str(ev.get("forward_thesis_hash") or ""),
        "rules_hash": str(ev.get("forward_rules_hash") or ""),
        "calibration_snapshot_id": str(ev.get("forward_calibration_snapshot_id") or ""),
        "pinned": bool(ev.get("forward_version_identity_pinned")),
    }

    return {
        "schema_version": 1,
        "final": {
            "evidence_confidence_score": ev.get("evidence_confidence_score"),
            "confidence_stage": str(ev.get("confidence_stage") or "UNKNOWN"),
            "paper_eligible": bool(ev.get("paper_eligible")),
            "is_win_probability": False,
            "meaning": "Evidence strength for this setup; not probability of winning.",
        },
        "setup": {
            "label": _setup(row),
            "reco_tier": str(row.get("reco_tier") or ""),
            "family_confirms": int(row.get("family_confirms") or row.get("method_confirms") or 0),
            "method_status": {
                key: method_status(key)
                for key in ("tape", "sepa", "funds", "trend", "rs", "ev", "case", "sector")
            },
            "desk_score": row.get("score"),
            "measurement": "DECISION_CONTEXT",
        },
        "historical": {
            "ready": bool(ev.get("historical_ready")),
            "n": int(ev.get("historical_n") or 0),
            "mean_R": ev.get("historical_mean_R"),
            "splits_tested": int(ev.get("historical_splits") or 0),
            "positive_splits": int(ev.get("historical_positive_splits") or 0),
            "confidence_score": ev.get("historical_confidence_score"),
            "generation_fingerprint": str(ev.get("historical_generation_fingerprint") or ""),
            "generation_match": bool(ev.get("historical_generation_match")),
            "measurement": "PIT_REPRODUCED_HISTORY",
        },
        "forward": {
            "trusted_n": int(ev.get("forward_n") or 0),
            "observed_n": int(ev.get("forward_observed_n") or 0),
            "mean_R": ev.get("forward_mean_R"),
            "confidence_score": ev.get("forward_confidence_score"),
            "source": str(ev.get("forward_source") or ""),
            "trusted_positive": bool(ev.get("forward_trusted_positive")),
            "trusted_negative": bool(ev.get("forward_trusted_negative")),
            "version_identity": forward_identity,
            "version_status": "PINNED" if forward_identity["pinned"] else "UNPINNED",
            "measurement": "REAL_FORWARD_PAPER" if ev.get("forward_n") else "NO_TRUSTED_FORWARD_SAMPLE",
        },
        "regime": {
            "value": str(row.get("regime") or row.get("market_state") or ""),
            "conditional_score": None,
            "measurement": "CONTEXT_ONLY",
        },
        "sector": {
            "value": str(row.get("sector") or ""),
            "state": str(row.get("sector_state") or ""),
            "method_status": method_status("sector"),
            "conditional_score": None,
            "measurement": "CONTEXT_ONLY",
        },
        "extension": {
            "entry_state": str(row.get("entry_state") or ""),
            "chase_risk": bool(row.get("chase_risk")),
            "extension_pct": extension_pct,
            "penalty_score": None,
            "measurement": "DECISION_GATE_CONTEXT",
        },
        "learning_policy": {
            "final_effect": str(final_effect or "NEUTRAL"),
            "max_sample_size": int(policy_sample_size or 0),
            "matched_policy_count": int(policy_coverage or 0),
            "matched_policy_ids": [
                str(item.get("policy_id") or "")
                for item in matched_policies
                if isinstance(item, Mapping) and item.get("policy_id")
            ],
        },
        "missing_evidence": list(row.get("missing_evidence") or []),
        "live_locked": True,
        "note": (
            "Only historical and trusted forward sections contribute to the evidence-confidence "
            "ladder. Context sections are shown separately and are not fabricated additive scores."
        ),
    }
