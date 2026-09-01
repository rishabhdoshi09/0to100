"""Phase 8 — Adversarial research / red team."""

from __future__ import annotations

from product.adversarial_research import FAILED, FRAGILE, INSUFFICIENT_EVIDENCE, SURVIVED, attack


def test_insufficient_sample():
    out = attack([0.4, 0.2, -0.1])
    assert out["status"] == INSUFFICIENT_EVIDENCE
    assert out["can_promote"] is False


def test_survived_edge():
    pnls = [0.6] * 12 + [0.4] * 8 + [0.2] * 6 + [-0.1] * 4
    out = attack(pnls, cost_shock=0.05, slippage_shock=0.03, delay_miss_rate=0.05)
    assert out["status"] in {SURVIVED, FRAGILE}
    assert out["sample_size"] == len(pnls)
    assert out["multiple_testing_aware"] is True
    if out["status"] == SURVIVED:
        assert out["can_promote"] is True


def test_fragile_when_top_winners_removed_or_costs_rise():
    # Headline looks great because of two huge winners.
    pnls = [5.0, 4.5] + [0.05] * 18
    out = attack(pnls, cost_shock=0.2, slippage_shock=0.2)
    assert out["status"] in {FRAGILE, FAILED}
    assert out["can_promote"] is False


def test_failed_when_expectancy_not_positive():
    out = attack([-0.3] * 25)
    assert out["status"] == FAILED
    assert out["can_promote"] is False


def test_fragile_cannot_become_production():
    pnls = [1.2] * 5 + [-0.05] * 20
    out = attack(pnls)
    assert out["can_promote"] is False or out["status"] != SURVIVED
