"""One crossing point from the pipeline's cards to the canonical Decision.

The defect this closes is not a crash — it is drift. Every consumer that reads
card keys for itself is a second opinion about what the desk decided, and two
opinions can disagree for months without anything failing.
"""
from __future__ import annotations

import pytest

from product.decision import AVOID, BUY, CONFLICTING, MISSING, SUPPORTING, WAIT, WATCH
from product.decision_adapter import decision_from_card, decisions_from_cards


def _card(**overrides):
    card = {
        "symbol": "infy",
        "reco_tier": "high_conviction",
        "primary_thesis": "VCP",
        "entry": 100.0,
        "stop": 95.0,
        "target": 115.0,
        "allows_recommend": True,
        "scan_scanned_at": "2026-09-12T03:00:00+00:00",
        "methods": [{"id": "tape", "label": "Tape", "status": "pass", "points": 10}],
        "families": [{"id": "price", "status": "SUPPORTIVE"}],
        "dd_status": "pass",
    }
    card.update(overrides)
    return card


def test_symbols_are_normalised_once_at_the_crossing():
    assert decision_from_card(_card()).symbol == "INFY"


@pytest.mark.parametrize("tier,expected", [
    ("high_conviction", BUY),
    ("good_setup", BUY),
    ("watch", WATCH),
    ("wait", WAIT),
    ("avoid", AVOID),
])
def test_card_tiers_map_onto_decision_states(tier, expected):
    assert decision_from_card(_card(reco_tier=tier)).state == expected


def test_a_hard_gate_beats_a_good_score():
    """allows_recommend is a refusal, not a hint."""
    decision = decision_from_card(_card(allows_recommend=False))
    assert decision.state == WATCH
    assert not decision.may_open_paper_position


def test_an_unrecognised_tier_never_becomes_a_buy():
    assert decision_from_card(_card(reco_tier="something_new")).state == WAIT


@pytest.mark.parametrize("status,direction", [
    ("pass", SUPPORTING),
    ("fail", CONFLICTING),
    ("unknown", MISSING),
])
def test_method_status_becomes_evidence_direction(status, direction):
    decision = decision_from_card(_card(
        methods=[{"id": "tape", "label": "Tape", "status": status}]
    ))
    bucket = {
        SUPPORTING: decision.supporting_evidence,
        CONFLICTING: decision.conflicting_evidence,
        MISSING: decision.missing_evidence,
    }[direction]
    assert any(item.id == "method:tape" for item in bucket)


def test_a_neutral_family_is_recorded_as_a_gap_not_as_support():
    """'We looked and it said nothing' is missing evidence, not a supporting fact."""
    decision = decision_from_card(_card(
        families=[{"id": "macro", "status": "NEUTRAL"}]
    ))
    assert any(i.id == "family:macro" for i in decision.missing_evidence)
    assert not any(i.id == "family:macro" for i in decision.supporting_evidence)


def test_an_unreadable_status_is_missing_not_neutral():
    decision = decision_from_card(_card(
        methods=[{"id": "tape", "label": "Tape", "status": "brand_new_word"}]
    ))
    assert any(i.id == "method:tape" for i in decision.missing_evidence)


def test_absent_fundamentals_are_stated_not_scored():
    decision = decision_from_card(_card(dd_status=""))
    gap = next(i for i in decision.missing_evidence if i.id == "fundamentals")
    assert "not available" in gap.label.lower()


def test_conflicts_reach_the_conflicting_bucket():
    decision = decision_from_card(_card(conflicts=["RSI extended", ""]))
    labels = [i.label for i in decision.conflicting_evidence]
    assert "RSI extended" in labels
    assert "" not in labels


def test_the_stop_is_always_an_invalidation_condition():
    decision = decision_from_card(_card())
    assert any("95" in condition for condition in decision.invalidation_conditions)


def test_two_reads_of_one_card_are_the_same_decision():
    card = _card()
    assert decision_from_card(card).decision_id == decision_from_card(card).decision_id


def test_cards_without_a_symbol_are_dropped_not_guessed():
    decisions = decisions_from_cards([_card(), {"symbol": ""}, {"no": "symbol"}])
    assert [d.symbol for d in decisions] == ["INFY"]


def test_nan_and_blank_prices_do_not_become_zero():
    decision = decision_from_card(_card(entry="", stop=float("nan"), target=None))
    assert decision.entry is None
    assert decision.stop is None
    assert decision.target is None
    assert not decision.may_open_paper_position
