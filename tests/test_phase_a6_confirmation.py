"""Phase A.6 confirmation freezes and verdict helpers (network-free)."""
from __future__ import annotations

from research.phase_a6.confirmation import _next_action, _verdict
from research.phase_a6.frozen_hypothesis import (
    CONFIRMATION_PROTOCOL,
    DISCOVERY,
    REJECTED_BRANCHES,
)


def test_four_branches_rejected_and_one_survivor():
    assert len(REJECTED_BRANCHES) == 4
    assert DISCOVERY["surviving_interaction"] == "signal_x_network_concentration"
    assert CONFIRMATION_PROTOCOL["experiment_id"] == "EXP-A6-CONF-01"
    assert CONFIRMATION_PROTOCOL["do_not_overwrite"] == "EXP-A5A6-01"
    assert CONFIRMATION_PROTOCOL["production_authority"] is False
    assert "no_post_hoc_threshold" in CONFIRMATION_PROTOCOL["network_concentration"]


def test_verdict_requires_discovery_direction():
    primary = {"delta_corr": -0.2, "p": 0.01, "n_low": 100, "n_high": 100}
    risk = {"ok": True, "economic_risk_meaning": True}
    incr = {"incremental": True}
    v, _ = _verdict(primary, risk, incr)
    assert v == "FAILED_CONFIRMATION"
    assert _next_action(v) == "REJECT"


def test_verdict_inconclusive_without_economic_risk():
    primary = {"delta_corr": 0.2, "p": 0.01, "n_low": 100, "n_high": 100}
    risk = {"ok": True, "economic_risk_meaning": False}
    incr = {"incremental": True}
    v, _ = _verdict(primary, risk, incr)
    assert v == "INCONCLUSIVE"
    assert _next_action(v) == "RETEST_ONLY_WITH_NEW_INDEPENDENT_DATA"


def test_verdict_confirmed_all_gates():
    primary = {"delta_corr": 0.2, "p": 0.01, "n_low": 100, "n_high": 100}
    risk = {"ok": True, "economic_risk_meaning": True}
    incr = {"incremental": True}
    v, _ = _verdict(primary, risk, incr)
    assert v == "CONFIRMED"
    assert _next_action(v) == "DESIGN_SEPARATE_POLICY_EXPERIMENT"
