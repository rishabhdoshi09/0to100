"""Learning lane: keep forward paper evidence flowing when production takes nothing.

Owner 2026-11-11: "don't stop taking paper trades" -- the first live cycle saw
107 names and entered 0 (94 LOW_QUALITY_SETUP).
"""
from __future__ import annotations

import product.paper_autopilot as PA


def _decision(symbol, *, reason=PA.LOW_QUALITY_SETUP, entry_state="ready", score=70.0, chase=False):
    card = {"symbol": symbol, "entry_state": entry_state, "score": score, "chase_risk": chase,
            "reco_tier": "watch"}
    return PA.AutopilotDecision(symbol, PA.WATCH, reason, card=card)


def _enter(card):
    return PA.AutopilotDecision(card["symbol"], PA.ENTER_NOW, "OK", card=card, selection_score=1.0)


def test_risk_override_can_only_lower_risk():
    assert PA._risk_pct_for({"exploration_risk_pct": 0.5}) == 0.5
    assert PA._risk_pct_for({"exploration_risk_pct": 3.0}) == PA.DEFAULT_RISK_PCT
    assert PA._risk_pct_for({"exploration_risk_pct": "x"}) == PA.DEFAULT_RISK_PCT
    assert PA._risk_pct_for({}) == PA.DEFAULT_RISK_PCT


def test_best_ready_research_name_is_reevaluated_at_half_risk(tmp_path):
    seen = []

    def evaluate(card):
        seen.append(card)
        return _enter(card)

    decisions = [_decision("LOW", score=50), _decision("BEST", score=80),
                 _decision("EXT", score=99, entry_state="extended"),
                 _decision("CHASE", score=98, chase=True),
                 _decision("DD", score=97, reason=PA.DD_GATE_FAILED)]
    pick = PA._exploration_pick(decisions, day="2026-11-11", evaluate=evaluate,
                                path=tmp_path / "x.json")
    assert pick.symbol == "BEST" and pick.policy_effect == PA.EXPLORATION
    probe = seen[0]
    assert probe["reco_tier"] == "good_setup" and probe["lane"] == PA.EXPLORATION
    assert probe["exploration_risk_pct"] == PA.EXPLORATION_RISK_PCT


def test_other_gates_still_apply_and_daily_budget_holds(tmp_path):
    path = tmp_path / "x.json"
    blocked = lambda card: PA.AutopilotDecision(card["symbol"], PA.BLOCK, "MAX_PORTFOLIO_RISK", card=card)
    assert PA._exploration_pick([_decision("A")], day="2026-11-11", evaluate=blocked, path=path) is None
    PA._record_exploration("2026-11-11", "A", path=path)
    assert PA._exploration_taken("2026-11-11", path=path) == ["A"]
    assert PA._exploration_pick([_decision("B")], day="2026-11-11", evaluate=_enter, path=path) is None
    assert PA._exploration_pick([_decision("B")], day="2026-11-12", evaluate=_enter, path=path).symbol == "B"


def test_owner_can_switch_the_lane_off(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_EXPLORATION", "0")
    assert PA._exploration_pick([_decision("A")], day="2026-11-11", evaluate=_enter,
                                path=tmp_path / "x.json") is None
