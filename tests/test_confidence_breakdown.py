from __future__ import annotations

from product.evidence_confidence import confidence_breakdown, confidence_from_policies
from product import evidence_policy_engine as EPE


def _candidate():
    return {
        "symbol": "TCS",
        "setup_label": "VCP",
        "reco_tier": "high_conviction",
        "family_confirms": 3,
        "score": 82,
        "sector": "Technology",
        "sector_state": "LEADING",
        "regime": "TRENDING_BULL",
        "entry_state": "ready",
        "chase_risk": False,
        "extension_pct": 2.5,
        "methods": {
            "tape": {"status": "pass"},
            "sepa": {"status": "pass"},
            "funds": {"status": "pass"},
            "trend": {"status": "pass"},
            "rs": {"status": "pass"},
            "ev": {"status": "unknown"},
            "case": {"status": "unknown"},
            "sector": {"status": "pass"},
        },
    }


def _policies(*, pinned=False):
    forward_extra = (
        {
            "thesis_hash": "thesis-a",
            "rules_hash": "rules-a",
            "calibration_snapshot_id": "cal-a",
        }
        if pinned
        else {}
    )
    return [
        {
            "policy_id": "HIST_SETUP::VCP",
            "historical_reproduced_positive": True,
            "generation_fingerprint": "gen-a",
            "historical_confidence_score": 72.0,
            "sample_size": 60,
            "expectancy_R": 0.34,
            "splits_tested": 6,
            "positive_splits": 5,
        },
        {
            "policy_id": "SETUP::VCP",
            "version": 4,
            "sample_size": 30,
            "expectancy_R": 0.30,
            "expectancy_difference_R": 0.30,
            "evidence_source": "paper_forward_taken_execution_adjusted",
            "affects_selection": True,
            "production_status": "ACTIVE",
            **forward_extra,
        },
    ]


def test_confidence_breakdown_keeps_context_out_of_fake_additive_scores():
    evidence = confidence_from_policies(
        _candidate(),
        _policies(),
        generation_fingerprint="gen-a",
    )
    out = confidence_breakdown(
        _candidate(),
        evidence,
        final_effect="SUPPORT",
        policy_sample_size=30,
        policy_coverage=1,
        matched_policies=_policies(),
    )

    assert out["final"]["evidence_confidence_score"] == evidence["evidence_confidence_score"]
    assert out["final"]["is_win_probability"] is False
    assert out["historical"]["n"] == 60
    assert out["forward"]["trusted_n"] == 30
    assert out["regime"]["value"] == "TRENDING_BULL"
    assert out["regime"]["conditional_score"] is None
    assert out["regime"]["measurement"] == "CONTEXT_ONLY"
    assert out["sector"]["conditional_score"] is None
    assert out["extension"]["extension_pct"] == 2.5
    assert out["extension"]["penalty_score"] is None
    assert "not fabricated additive scores" in out["note"]


def test_unversioned_forward_evidence_is_visible_as_unpinned():
    evidence = confidence_from_policies(
        _candidate(),
        _policies(pinned=False),
        generation_fingerprint="gen-a",
    )
    out = confidence_breakdown(_candidate(), evidence)

    assert evidence["forward_n"] == 30
    assert out["forward"]["version_status"] == "UNPINNED"
    assert out["forward"]["version_identity"]["pinned"] is False
    assert out["forward"]["version_identity"]["thesis_hash"] == ""


def test_pinned_forward_evidence_exposes_exact_identity_without_changing_score():
    loose = confidence_from_policies(
        _candidate(),
        _policies(pinned=False),
        generation_fingerprint="gen-a",
    )
    pinned = confidence_from_policies(
        _candidate(),
        _policies(pinned=True),
        generation_fingerprint="gen-a",
    )
    out = confidence_breakdown(_candidate(), pinned)

    assert pinned["evidence_confidence_score"] == loose["evidence_confidence_score"]
    assert out["forward"]["version_status"] == "PINNED"
    assert out["forward"]["version_identity"] == {
        "policy_id": "SETUP::VCP",
        "policy_version": 4,
        "thesis_hash": "thesis-a",
        "rules_hash": "rules-a",
        "calibration_snapshot_id": "cal-a",
        "pinned": True,
    }


def test_policy_engine_attaches_breakdown_without_changing_final_effect(monkeypatch):
    historical = confidence_from_policies(
        _candidate(),
        _policies(),
        generation_fingerprint="gen-a",
    )
    monkeypatch.setattr(
        EPE,
        "_historical_gate",
        lambda *_a, **_k: {
            **historical,
            "required": True,
            "paper_eligible": True,
            "bootstrap_complete": True,
            "live_locked": True,
        },
    )

    out = EPE.evaluate_policies(
        _candidate(),
        policies=_policies(),
        enforce_history=True,
    )

    assert out["final_effect"] == "SUPPORT"
    assert out["confidence_breakdown"]["final"]["evidence_confidence_score"] == out["evidence_confidence_score"]
    assert out["confidence_breakdown"]["learning_policy"]["final_effect"] == "SUPPORT"
    assert out["confidence_breakdown"]["learning_policy"]["matched_policy_count"] >= 1
    assert out["invents_buy"] is False
    assert out["live_locked"] is True
