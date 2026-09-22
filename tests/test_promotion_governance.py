from __future__ import annotations

from product.promotion_governance import promotion_dossier


def _rule_comparison(**over):
    row = {
        "oos_n": 35,
        "forward_n": 35,
        "sample_size": 35,
        "expectancy": 0.40,
        "drawdown": 2.0,
        "execution_adjusted_expectancy": 0.25,
        "execution_adjusted_n": 35,
        "execution_adjusted_coverage": 1.0,
        "regime_breakdown": {
            "TRENDING_BULL": {"n": 20, "expectancy": 0.5, "drawdown": 1.0},
            "SIDEWAYS": {"n": 15, "expectancy": 0.2, "drawdown": 1.0},
        },
        "sector_breakdown": {
            "Technology": {"n": 20, "expectancy": 0.3, "drawdown": 1.0},
            "Energy": {"n": 15, "expectancy": 0.5, "drawdown": 1.0},
        },
    }
    row.update(over)
    return row


def test_rule_challenger_dossier_can_be_eligible_without_live_authority():
    out = promotion_dossier(
        _rule_comparison(),
        component="rule_challenger:test",
        require_execution_adjusted_edge=True,
        require_calibration_edge=False,
        min_oos_n=30,
        min_forward_n=20,
    )

    assert out["decision"] == "ELIGIBLE"
    assert out["blockers"] == []
    assert out["live_locked"] is True
    assert out["explicit_promotion_required"] is True


def test_sufficiently_sampled_negative_regime_blocks_headline_positive_edge():
    cmp = _rule_comparison(
        regime_breakdown={
            "TRENDING_BULL": {"n": 25, "expectancy": 0.7},
            "SIDEWAYS": {"n": 10, "expectancy": -0.15},
        }
    )
    out = promotion_dossier(
        cmp,
        component="rule_challenger:test",
        require_execution_adjusted_edge=True,
        min_oos_n=30,
        min_forward_n=20,
    )

    assert out["decision"] == "KEEP_SHADOW"
    assert "REGIME_INSTABILITY:SIDEWAYS" in out["blockers"]


def test_small_negative_bucket_is_descriptive_not_fake_confidence():
    cmp = _rule_comparison(
        sector_breakdown={
            "Technology": {"n": 30, "expectancy": 0.5},
            "TinySample": {"n": 3, "expectancy": -1.0},
        }
    )
    out = promotion_dossier(
        cmp,
        component="rule_challenger:test",
        require_execution_adjusted_edge=True,
        min_oos_n=30,
        min_forward_n=20,
    )

    assert out["decision"] == "ELIGIBLE"
    assert not any(x.startswith("SECTOR_INSTABILITY:TINYSAMPLE") for x in out["blockers"])


def test_probabilistic_challenger_requires_exact_version_calibration_edge():
    cmp = {
        "oos_n": 40,
        "forward_n": 40,
        "expectancy": 0.20,
        "improvement": 0.03,
        "improvement_lower_95": 0.01,
        "exact_version_evidence": True,
        "drawdown": 3.0,
        "regime_breakdown": {"TRENDING_BULL": {"n": 40, "expectancy": 0.2}},
        "sector_breakdown": {"STRONG": {"n": 40, "expectancy": 0.2}},
    }
    good = promotion_dossier(
        cmp,
        component="paper_selection_classifier:v1",
        require_calibration_edge=True,
        require_execution_adjusted_edge=False,
        min_oos_n=30,
        min_forward_n=30,
    )
    bad = promotion_dossier(
        {**cmp, "exact_version_evidence": False},
        component="paper_selection_classifier:v1",
        require_calibration_edge=True,
        require_execution_adjusted_edge=False,
        min_oos_n=30,
        min_forward_n=30,
    )

    assert good["decision"] == "ELIGIBLE"
    assert bad["decision"] == "KEEP_SHADOW"
    assert "EXACT_VERSION_FORWARD_EVIDENCE_MISSING" in bad["blockers"]


def test_execution_and_drawdown_regressions_fail_closed():
    out = promotion_dossier(
        _rule_comparison(
            execution_adjusted_n=20,
            execution_adjusted_coverage=0.6,
            drawdown=4.0,
        ),
        component="rule_challenger:test",
        require_execution_adjusted_edge=True,
        min_oos_n=30,
        min_forward_n=20,
        champion_drawdown=2.0,
    )

    assert "EXECUTION_EVIDENCE_INCOMPLETE" in out["blockers"]
    assert "DRAWDOWN_REGRESSION" in out["blockers"]
    assert out["decision"] == "KEEP_SHADOW"
