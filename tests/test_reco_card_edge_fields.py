"""Recommendation cards carry the scanner's evidence-gate fields so the paper
learning lane can explore a withheld BUY (2026-11-16: 94 cards, none carried
edge_gated, so the lane took nothing)."""
from __future__ import annotations


def _row(**over):
    row = {"symbol": "APOLLO", "company": "Apollo Micro", "verdict": "WATCH", "status": "Watch",
           "score": 68.8, "rsi": 61.0, "volume_ratio": 2.1, "entry": 475.6, "stop": 423.4,
           "target": 580.0, "price": 476.0, "signals": ["MOMENTUM", "BREAKOUT_52W"],
           "edge_gated": True, "edge_r": 0.003, "edge_lower_r": -0.08, "edge_source": "signals",
           "chase_risk": False}
    row.update(over)
    return row


def test_card_keeps_the_evidence_gate_fields():
    from product.recommendations_workspace import card_from_row

    card = card_from_row(_row(), category_id="breakouts", category_label="Breakouts")
    assert card["edge_gated"] is True and card["edge_r"] == 0.003
    assert card["edge_lower_r"] == -0.08 and card["edge_source"] == "signals"
    assert card["chase_risk"] is False
    plain = card_from_row(_row(edge_gated=False, edge_r=None, edge_lower_r=None, edge_source=""),
                          category_id="breakouts", category_label="Breakouts")
    assert plain["edge_gated"] is False and plain["edge_r"] is None


def test_learning_lane_can_explore_a_withheld_card(tmp_path):
    import product.paper_autopilot as PA
    from product.recommendations_workspace import card_from_row

    card = card_from_row(_row(), category_id="breakouts", category_label="Breakouts")
    seen = []

    def evaluate(c):
        seen.append(c)
        return PA.AutopilotDecision(c["symbol"], PA.ENTER_NOW, "OK", card=c, selection_score=1.0)

    decision = PA.AutopilotDecision("APOLLO", PA.WATCH, PA.LOW_QUALITY_SETUP, card=card)
    pick = PA._exploration_pick([decision], day="2026-11-16", evaluate=evaluate, path=tmp_path / "x.json")
    assert pick is not None and pick.symbol == "APOLLO"
    assert seen[0]["exploration_source"] == "EDGE_GATED"


def test_a_chasing_withheld_card_is_never_explored(tmp_path):
    import product.paper_autopilot as PA
    from product.recommendations_workspace import card_from_row

    card = card_from_row(_row(chase_risk=True), category_id="breakouts", category_label="Breakouts")
    decision = PA.AutopilotDecision("APOLLO", PA.WATCH, PA.LOW_QUALITY_SETUP, card=card)
    assert PA._exploration_pick([decision], day="2026-11-16",
                                evaluate=lambda c: PA.AutopilotDecision(c["symbol"], PA.ENTER_NOW, "OK", card=c),
                                path=tmp_path / "x.json") is None
