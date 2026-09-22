from __future__ import annotations

from product.confidence_breakdown import build_confidence_breakdown


def _candidate():
    return {
        "symbol": "TCS",
        "setup_label": "VCP",
        "score": 82,
        "regime": "BULL_TREND",
        "market_support": "Strong",
        "market_support_detail": "Breadth constructive",
        "sector": "Technology",
        "sector_leadership_score": 78,
        "sector_leadership_label": "Leading",
        "sector_breadth": "broad",
        "sector_momentum": "positive",
        "entry_state": "ready",
        "chase_risk": False,
        "extension_pct": 1.2,
        "rsi": 64.0,
    }


def _context(*, research_history: bool = True, forward_version: bool = False):
    hist = {
        "policy_id": "HIST_SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 60,
        "expectancy_R": 0.34,
        "historical_reproduced_positive": True,
        "historical_confidence_score": 72.0,
        "generation_fingerprint": "gen-current",
        "splits_tested": 4,
        "positive_splits": 4,
        "evidence_source": "backtest_reproduced",
        "production_status": "ELIGIBLE",
        "affects_selection": True,
    }
    forward = {
        "policy_id": "SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 18,
        "expectancy_R": 0.26,
        "shrunk_expectancy_R": 0.20,
        "evidence_source": "paper_forward_taken_execution_adjusted",
        "production_status": "ELIGIBLE",
        "confidence": "MEASURED",
        "affects_selection": True,
    }
    if forward_version:
        forward["generation_fingerprint"] = "gen-current"
    rows = [hist, forward]
    if research_history:
        rows.append({
            "policy_id": "HIST_SETUP::thesis-current::VCP",
            "dimension": "historical_setup",
            "bucket": "VCP",
            "sample_size": 28,
            "expectancy_R": 0.31,
            "historical_confidence_score": 68.0,
            "historical_lower_95_R": 0.08,
            "historical_upper_95_R": 0.54,
            "generation_fingerprint": "research-trade-set",
            "thesis_hash": "thesis-current",
            "evidence_source": "backtest_historical_replay",
            "not_promotion_evidence": True,
            "not_real_pnl": True,
            "affects_selection": False,
        })
    return {
        "policies": rows,
        "generation": {"fingerprint": "gen-current"},
        "thesis": {"thesis_hash": "thesis-current"},
        "calibration": {
            "snapshot_id": "cal-1",
            "immutable": True,
            "identities": {
                "data_identity": "nse:2026-09-21",
                "thesis_hash": "thesis-current",
                "feature_version": "features-v1",
                "model_version": "cal-v1",
                "signal_registry_version": "sig-v1",
            },
        },
        "signal_registry": {
            "registry_version": "sig-v1",
            "summary": {"scanner_catalog": 17, "forward_calibrated": 16},
        },
    }


def test_breakdown_exposes_components_without_calling_score_win_probability():
    out = build_confidence_breakdown(_candidate(), context=_context())

    assert out["available"] is True
    assert out["final"]["is_win_probability"] is False
    assert out["final"]["stage"] == "FORWARD_CONFIRMED"
    assert out["final"]["paper_eligible"] is True
    assert out["components"]["setup_quality"]["value"] == 82.0
    assert out["components"]["regime_support"]["regime"] == "BULL_TREND"
    assert out["components"]["sector_support"]["leadership_score"] == 78.0
    assert out["components"]["extension"]["impact"] == "CLEAR"
    assert out["components"]["production_history"]["sample_size"] == 60
    assert out["components"]["forward_evidence"]["trusted_sample_size"] == 18
    assert out["components"]["calibration"]["snapshot_id"] == "cal-1"


def test_market_breadth_is_not_silently_substituted_for_missing_regime():
    candidate = _candidate()
    candidate.pop("regime")
    out = build_confidence_breakdown(
        candidate,
        market_ctx={
            "market_support": "Strong",
            "market_support_detail": "breadth healthy",
        },
        context=_context(),
    )
    regime = out["components"]["regime_support"]
    assert regime["status"] == "UNMEASURED"
    assert regime["regime"] == ""
    assert regime["market_support"] == "Strong"
    assert "not substituted as a regime" in regime["note"]


def test_unversioned_forward_aggregate_is_explicitly_not_exact_current_version():
    out = build_confidence_breakdown(_candidate(), context=_context(forward_version=False))
    forward = out["components"]["forward_evidence"]

    assert forward["trusted_sample_size"] == 18
    assert forward["exact_current_version_proven"] is False
    assert forward["version_status"] == "UNPINNED_AGGREGATE"
    assert "does not persist enough" in forward["note"]


def test_forward_policy_with_current_generation_proof_is_labeled_exact():
    out = build_confidence_breakdown(_candidate(), context=_context(forward_version=True))
    forward = out["components"]["forward_evidence"]

    assert forward["exact_current_version_proven"] is True
    assert forward["version_status"] == "EXACT_CURRENT_VERSION"
    assert forward["note"] == ""


def test_research_historical_policy_is_visible_but_does_not_change_final_score():
    with_research = build_confidence_breakdown(
        _candidate(),
        context=_context(research_history=True),
    )
    without_research = build_confidence_breakdown(
        _candidate(),
        context=_context(research_history=False),
    )

    research = with_research["components"]["research_historical_replay"]
    assert research["available"] is True
    assert research["not_promotion_evidence"] is True
    assert research["affects_final_confidence"] is False
    assert (
        with_research["final"]["evidence_strength_score"]
        == without_research["final"]["evidence_strength_score"]
    )


def test_extended_candidate_exposes_gate_state_without_inventing_numeric_penalty():
    candidate = _candidate()
    candidate["entry_state"] = "extended"
    candidate["chase_risk"] = True
    out = build_confidence_breakdown(candidate, context=_context())

    ext = out["components"]["extension"]
    assert ext["impact"] == "BLOCKING_OR_PENALIZING"
    assert ext["numeric_penalty"] is None
    assert "No synthetic penalty" in ext["note"]
