"""Demote-only historical veto on the discovery board (Invariant 6).

EMAMILTD (2026-11-11) was labelled BUY while its own replay of 50 analogs said
mean -0.37R, 37% winners, P(edge>0)=0.02.
"""
from __future__ import annotations

from product.decision_service import historical_demotion

EMAMILTD = {"raw_n": 50, "effective_n": 48.53, "mean_R": -0.3689, "shrunk_mean_R": -0.3167,
            "upper_95_R": 0.0002, "p_edge_positive": 0.02}


def test_strong_negative_replay_demotes_with_reason():
    reason = historical_demotion(EMAMILTD)
    assert reason.startswith("Demoted to WATCH") and "50 historical analogs" in reason
    assert "-0.32R" in reason and "P(edge>0)=0.02" in reason


def test_thin_or_ambiguous_replay_never_demotes():
    assert historical_demotion(dict(EMAMILTD, effective_n=12)) == ""        # too few
    assert historical_demotion(dict(EMAMILTD, p_edge_positive=0.2)) == ""   # not clear
    assert historical_demotion(dict(EMAMILTD, upper_95_R=0.4)) == ""        # CI reaches a real edge
    assert historical_demotion({}) == "" and historical_demotion(None) == ""
    assert historical_demotion(dict(EMAMILTD, p_edge_positive="x")) == ""


def test_positive_replay_is_not_a_promotion_path():
    good = dict(EMAMILTD, mean_R=0.5, shrunk_mean_R=0.4, upper_95_R=0.9, p_edge_positive=0.97)
    assert historical_demotion(good) == ""
