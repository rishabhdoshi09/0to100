"""Deterministic acceptance tests for the F&O threshold-review evidence layer
(product/fno_threshold_review.py): recommendation-only, never an autonomous
gate change.
"""
from __future__ import annotations

from product.fno_threshold_review import (
    MATERIAL_GAP_PP,
    MIN_SAMPLE,
    review_min_score_threshold,
)


def test_never_autonomously_changes_the_gate():
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket={})
    assert result["autonomous_change_applied"] is False
    assert "recommendation only" in result["action_required"].lower()


def test_insufficient_sample_below_threshold_makes_no_recommendation_to_act():
    fixture = {
        "50_59": {"not_taken": {"MISSED_WINNER": 5, "AVOIDED_LOSER": 1}},
    }
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket=fixture)
    assert result["below_threshold_sample"] == 6
    assert result["recommendation"] == "INSUFFICIENT_SAMPLE"


def test_large_sample_with_material_winner_gap_considers_lowering():
    fixture = {
        "50_59": {"not_taken": {"MISSED_WINNER": 25, "AVOIDED_LOSER": 2, "CORRECT_REJECTION": 3}},
    }
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket=fixture)
    assert result["below_threshold_sample"] == 30
    assert result["below_threshold_would_be_winner_rate_pct"] - result["below_threshold_would_be_loser_rate_pct"] >= MATERIAL_GAP_PP
    assert result["recommendation"] == "CONSIDER_LOWERING_THRESHOLD"


def test_large_sample_without_material_gap_keeps_current_threshold():
    fixture = {
        "50_59": {"not_taken": {"MISSED_WINNER": 10, "AVOIDED_LOSER": 10, "CORRECT_REJECTION": 10}},
    }
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket=fixture)
    assert result["below_threshold_sample"] == 30
    assert result["recommendation"] == "KEEP_CURRENT_THRESHOLD"


def test_below_and_above_threshold_populations_never_mix():
    """A 'taken' row parked under a bucket that is actually below the
    threshold (or a 'not_taken' row above it) must never be read -- only the
    row matching this threshold's own below/above split counts."""
    fixture = {
        "50_59": {
            "not_taken": {"MISSED_WINNER": 25, "AVOIDED_LOSER": 2},
            "taken": {"MISSED_WINNER": 999},  # should never be counted: below threshold
        },
        "70_79": {
            "taken": {"CORRECT_REJECTION": 5},
            "not_taken": {"AVOIDED_LOSER": 999},  # should never be counted: above threshold
        },
    }
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket=fixture)
    assert result["below_threshold_sample"] == 27
    assert result["above_threshold_sample"] == 5


def test_empty_evidence_is_insufficient_not_an_exception():
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket={})
    assert result["recommendation"] == "INSUFFICIENT_SAMPLE"
    assert result["below_threshold_sample"] == 0


def test_min_sample_and_gap_constants_are_exposed_for_the_ui():
    result = review_min_score_threshold(min_score_to_take=60.0, by_bucket={})
    assert result["min_sample"] == MIN_SAMPLE
    assert result["material_gap_pp"] == MATERIAL_GAP_PP
