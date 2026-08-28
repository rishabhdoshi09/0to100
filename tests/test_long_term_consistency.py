from scan.long_term_consistency import (
    evidence_adjusted_combined,
    hardened_sector_of,
    reconcile_record,
)


def test_sparse_fundamentals_cannot_manufacture_high_composite():
    row = reconcile_record({
        "symbol": "MACPOWER",
        "score": 86.0,
        "technical_score": 86.0,
        "fundamental_score": 100.0,
        "fundamental_coverage": 0.045,
        "combined_score": 93.7,
        "classification": "NEEDS_FUNDAMENTALS",
        "timing": "WAIT_FOR_BASE",
        "verdict": "LONG_TERM_BUY",
    })
    assert row["combined_score_unadjusted"] == 93.7
    assert row["combined_score"] == evidence_adjusted_combined(86.0, 100.0, 0.045)
    assert row["combined_score"] < 50
    assert row["verdict"] == "NEEDS_FUNDAMENTALS"
    assert row["technical_verdict"] == "LONG_TERM_BUY"
    assert row["verdict_reconciled"] is True


def test_avoid_review_overrides_technical_buy_label():
    row = reconcile_record({
        "symbol": "IDEAFORGE",
        "technical_score": 76.8,
        "fundamental_score": 19.4,
        "fundamental_coverage": 0.864,
        "classification": "AVOID_REVIEW",
        "timing": "WAIT_FOR_BASE",
        "verdict": "LONG_TERM_BUY",
    })
    assert row["verdict"] == "AVOID_REVIEW"


def test_chase_risk_cannot_remain_long_term_buy():
    row = reconcile_record({
        "symbol": "YATHARTH",
        "technical_score": 86.4,
        "fundamental_score": 74.4,
        "fundamental_coverage": 0.864,
        "classification": "QUALITY_BUT_EXPENSIVE",
        "timing": "ACCUMULATE_ON_PULLBACK",
        "chase_risk": True,
        "verdict": "LONG_TERM_BUY",
    })
    assert row["verdict"] == "WAIT_FOR_BASE"


def test_union_bank_is_not_scored_as_unknown_industrial_company():
    assert hardened_sector_of("UNIONBANK") == "Banking & Finance"
