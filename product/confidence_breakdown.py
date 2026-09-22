"""Truth-preserving confidence decomposition for recommendation cards.

This module does not create a new prediction score. It explains the factual
inputs around QuantTerm's existing evidence-confidence ladder:

- setup quality from the saved scan/checklist
- regime/market context when explicitly measured
- sector leadership when explicitly measured
- extension/chase state from the same entry gate
- generation-scoped reproduced historical evidence
- trusted forward taken-paper evidence
- immutable calibration/signal identities
- the existing evidence-confidence composite

Missing inputs stay UNMEASURED. Historical research replay that is not the
production generation gate is shown separately and never blended into the final
confidence score. The final score is evidence strength, not win probability.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if out != out else out
    except (TypeError, ValueError):
        return None


def _setup(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("setup")
        or candidate.get("setup_label")
        or candidate.get("primary_thesis")
        or ""
    ).strip()


def load_context() -> dict[str, Any]:
    """Load current read-only evidence/version context once per workspace."""
    try:
        from product.learning_policy_store import load_policies
        policies = [
            dict(row)
            for row in (load_policies().get("policies") or [])
            if isinstance(row, Mapping)
        ]
    except Exception:
        policies = []

    try:
        from product.evolution_generation_guard import current_generation
        generation = dict(current_generation() or {})
    except Exception:
        generation = {}

    try:
        from product.trading_thesis import manifest
        thesis = dict(manifest() or {})
    except Exception:
        thesis = {}

    try:
        from scan.calibration_snapshot import load_current
        calibration = dict(load_current() or {})
    except Exception:
        calibration = {}

    try:
        from scan.signal_registry import load_registry
        registry = dict(load_registry() or {})
    except Exception:
        registry = {}

    return {
        "policies": policies,
        "generation": generation,
        "thesis": thesis,
        "calibration": calibration,
        "signal_registry": registry,
    }


def _setup_quality(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = _float(
        candidate.get("setup_quality")
        if candidate.get("setup_quality") is not None
        else candidate.get("score")
    )
    return {
        "status": "MEASURED" if value is not None else "UNMEASURED",
        "value": None if value is None else round(value, 1),
        "unit": "checklist_0_100",
        "label": "Setup Quality",
        "is_win_probability": False,
        "source": "saved_scan_checklist",
    }


def _regime_support(
    candidate: Mapping[str, Any],
    market_ctx: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ctx = dict(market_ctx or {})
    regime = str(candidate.get("regime") or candidate.get("market_regime") or "").strip()
    support = str(
        candidate.get("market_support")
        or ctx.get("market_support")
        or ""
    ).strip()
    detail = str(
        candidate.get("market_support_detail")
        or ctx.get("market_support_detail")
        or ""
    ).strip()
    # Market breadth/support is useful context but must not be renamed a regime.
    return {
        "status": "MEASURED" if regime else "UNMEASURED",
        "regime": regime,
        "market_support": support or "Unmeasured",
        "detail": detail,
        "source": "saved_candidate_regime" if regime else "market_breadth_context_only",
        "note": (
            ""
            if regime
            else "No explicit regime label is persisted on this card; market breadth is shown separately and is not substituted as a regime."
        ),
    }


def _sector_support(candidate: Mapping[str, Any]) -> dict[str, Any]:
    score = _float(candidate.get("sector_leadership_score"))
    label = str(candidate.get("sector_leadership_label") or "").strip()
    breadth = str(candidate.get("sector_breadth") or "").strip()
    momentum = str(candidate.get("sector_momentum") or "").strip()
    measured = score is not None or bool(label or breadth or momentum)
    return {
        "status": "MEASURED" if measured else "UNMEASURED",
        "sector": str(candidate.get("sector") or "").strip(),
        "leadership_score": None if score is None else round(score, 2),
        "leadership_label": label,
        "breadth": breadth,
        "momentum": momentum,
        "source": "saved_sector_leadership" if measured else "",
    }


def _extension_state(candidate: Mapping[str, Any]) -> dict[str, Any]:
    entry_state = str(candidate.get("entry_state") or "").strip()
    chase = bool(candidate.get("chase_risk"))
    extension = _float(candidate.get("extension_pct"))
    rsi = _float(candidate.get("rsi"))

    lowered = entry_state.lower()
    if chase or "extended" in lowered or "pullback" in lowered:
        impact = "BLOCKING_OR_PENALIZING"
    elif entry_state in {"ready", "Ready to trade", "READY"}:
        impact = "CLEAR"
    elif entry_state:
        impact = "WATCH"
    else:
        impact = "UNMEASURED"

    return {
        "status": "MEASURED" if (entry_state or extension is not None or chase) else "UNMEASURED",
        "entry_state": entry_state,
        "chase_risk": chase,
        "extension_pct": None if extension is None else round(extension, 3),
        "rsi": None if rsi is None else round(rsi, 2),
        "impact": impact,
        "numeric_penalty": None,
        "note": (
            "No synthetic penalty is invented; the production entry/chase gate remains authoritative."
        ),
    }


def _research_historical_policy(
    setup: str,
    thesis_hash: str,
    policies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not setup or not thesis_hash:
        return {"available": False, "reason": "setup_or_thesis_identity_missing"}
    policy_id = f"HIST_SETUP::{thesis_hash}::{setup}"
    row = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == policy_id),
        {},
    )
    if not row:
        return {
            "available": False,
            "policy_id": policy_id,
            "reason": "no_thesis_scoped_historical_paper_policy",
        }
    return {
        "available": True,
        "policy_id": policy_id,
        "sample_size": int(row.get("sample_size") or 0),
        "expectancy_R": row.get("expectancy_R"),
        "historical_confidence_score": row.get("historical_confidence_score"),
        "lower_95_R": row.get("historical_lower_95_R"),
        "upper_95_R": row.get("historical_upper_95_R"),
        "generation_fingerprint": str(row.get("generation_fingerprint") or ""),
        "not_promotion_evidence": row.get("not_promotion_evidence") is True,
        "not_real_pnl": row.get("not_real_pnl") is True,
        "affects_final_confidence": False,
        "evidence_source": str(row.get("evidence_source") or ""),
    }


def _forward_version_truth(
    setup: str,
    policies: Sequence[Mapping[str, Any]],
    *,
    thesis_hash: str,
    generation_fingerprint: str,
) -> dict[str, Any]:
    row = next(
        (
            dict(p)
            for p in policies
            if str(p.get("policy_id") or "") == f"SETUP::{setup}"
        ),
        {},
    )
    if not row:
        return {
            "available": False,
            "version_status": "NO_FORWARD_SETUP_POLICY",
            "exact_current_version_proven": False,
        }
    policy_thesis = str(row.get("thesis_hash") or "")
    policy_generation = str(row.get("generation_fingerprint") or "")
    exact = bool(
        (policy_thesis and thesis_hash and policy_thesis == thesis_hash)
        or (
            policy_generation
            and generation_fingerprint
            and policy_generation == generation_fingerprint
        )
    )
    return {
        "available": True,
        "policy_id": str(row.get("policy_id") or ""),
        "sample_size": int(row.get("sample_size") or 0),
        "evidence_source": str(row.get("evidence_source") or ""),
        "policy_thesis_hash": policy_thesis,
        "policy_generation_fingerprint": policy_generation,
        "exact_current_version_proven": exact,
        "version_status": (
            "EXACT_CURRENT_VERSION"
            if exact
            else "UNPINNED_AGGREGATE"
        ),
        "note": (
            ""
            if exact
            else "This aggregate does not persist enough thesis/generation identity to claim exact-version forward evidence."
        ),
    }


def build_confidence_breakdown(
    candidate: Mapping[str, Any],
    *,
    market_ctx: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose one explanation without altering selection or score semantics."""
    ctx = dict(context or load_context())
    policies = [
        dict(row)
        for row in (ctx.get("policies") or [])
        if isinstance(row, Mapping)
    ]
    generation = dict(ctx.get("generation") or {})
    thesis = dict(ctx.get("thesis") or {})
    calibration = dict(ctx.get("calibration") or {})
    registry = dict(ctx.get("signal_registry") or {})

    generation_fp = str(generation.get("fingerprint") or "")
    thesis_hash = str(thesis.get("thesis_hash") or "")
    setup = _setup(candidate)

    try:
        from product.evidence_confidence import confidence_from_policies
        evidence = confidence_from_policies(
            candidate,
            policies,
            generation_fingerprint=generation_fp,
        )
    except Exception as exc:
        evidence = {
            "setup": setup,
            "historical_ready": False,
            "historical_n": 0,
            "forward_n": 0,
            "forward_observed_n": 0,
            "evidence_confidence_score": 0.0,
            "confidence_stage": "CONFIDENCE_UNAVAILABLE",
            "paper_eligible": False,
            "is_win_probability": False,
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }

    identities = dict(calibration.get("identities") or {})
    signal_summary = dict(registry.get("summary") or {})

    return {
        "schema_version": 1,
        "available": True,
        "setup": setup,
        "components": {
            "setup_quality": _setup_quality(candidate),
            "regime_support": _regime_support(candidate, market_ctx),
            "sector_support": _sector_support(candidate),
            "extension": _extension_state(candidate),
            "production_history": {
                "status": (
                    "REPRODUCED"
                    if evidence.get("historical_ready")
                    else "UNPROVEN"
                ),
                "sample_size": int(evidence.get("historical_n") or 0),
                "mean_R": evidence.get("historical_mean_R"),
                "splits_tested": int(evidence.get("historical_splits") or 0),
                "positive_splits": int(evidence.get("historical_positive_splits") or 0),
                "confidence_score": evidence.get("historical_confidence_score"),
                "generation_fingerprint": str(
                    evidence.get("historical_generation_fingerprint") or ""
                ),
                "generation_match": bool(
                    evidence.get("historical_generation_match")
                ),
                "source": "generation_scoped_autonomous_evolution",
            },
            "forward_evidence": {
                "trusted_sample_size": int(evidence.get("forward_n") or 0),
                "observed_sample_size": int(evidence.get("forward_observed_n") or 0),
                "mean_R": evidence.get("forward_mean_R"),
                "confidence_score": evidence.get("forward_confidence_score"),
                "source": str(evidence.get("forward_source") or ""),
                "trusted_positive": bool(evidence.get("forward_trusted_positive")),
                "trusted_negative": bool(evidence.get("forward_trusted_negative")),
                **_forward_version_truth(
                    setup,
                    policies,
                    thesis_hash=thesis_hash,
                    generation_fingerprint=generation_fp,
                ),
            },
            "research_historical_replay": _research_historical_policy(
                setup, thesis_hash, policies
            ),
            "calibration": {
                "snapshot_id": str(calibration.get("snapshot_id") or ""),
                "immutable": calibration.get("immutable") is True,
                "data_identity": str(identities.get("data_identity") or ""),
                "thesis_hash": str(identities.get("thesis_hash") or thesis_hash),
                "feature_version": str(identities.get("feature_version") or ""),
                "model_version": str(identities.get("model_version") or ""),
                "signal_registry_version": str(
                    identities.get("signal_registry_version")
                    or registry.get("registry_version")
                    or ""
                ),
                "scanner_catalog": signal_summary.get("scanner_catalog"),
                "forward_calibrated": signal_summary.get("forward_calibrated"),
            },
        },
        "final": {
            "evidence_strength_score": evidence.get("evidence_confidence_score"),
            "stage": str(evidence.get("confidence_stage") or "UNMEASURED"),
            "paper_eligible": bool(evidence.get("paper_eligible")),
            "is_win_probability": False,
            "basis": "generation-scoped reproduced history + trusted forward taken-paper evidence",
        },
        "identities": {
            "generation_fingerprint": generation_fp,
            "thesis_hash": thesis_hash,
            "calibration_snapshot_id": str(calibration.get("snapshot_id") or ""),
        },
        "truth_note": (
            "Setup/regime/sector/extension context explains the decision but is not "
            "silently blended into the evidence-strength score. Research-only historical "
            "replay is shown separately. Missing measurements stay unmeasured."
        ),
        "live_locked": True,
    }


def unavailable_breakdown(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "reason": str(reason or "unavailable"),
        "final": {
            "evidence_strength_score": None,
            "stage": "UNAVAILABLE",
            "paper_eligible": False,
            "is_win_probability": False,
        },
        "live_locked": True,
    }
