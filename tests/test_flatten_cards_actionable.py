"""One card per symbol, and it must be the actionable one (2026-11-11 regression:
Wealth Builders, listed first, hid high-conviction Momentum cards from paper)."""
from __future__ import annotations

from product.autopilot_journal import flatten_cards


def _ws(*cats):
    return {"categories": [{"id": cid, "cards": cards} for cid, cards in cats]}


def test_high_conviction_card_beats_an_earlier_watch_card():
    ws = _ws(
        ("wealth_builders", [{"symbol": "rpel", "reco_tier": "watch", "entry_state": "watch", "category_id": "wealth_builders"}]),
        ("momentum_breakouts", [{"symbol": "RPEL", "reco_tier": "high_conviction", "entry_state": "ready", "category_id": "momentum_breakouts"},
                                {"symbol": "PNB", "reco_tier": "good_setup", "entry_state": "ready"}]),
    )
    cards = flatten_cards(ws)
    assert [c["symbol"] for c in cards] == ["RPEL", "PNB"]          # first-seen order kept
    assert cards[0]["reco_tier"] == "high_conviction" and cards[0]["category_id"] == "momentum_breakouts"


def test_ties_keep_category_order_and_ready_breaks_ties():
    ws = _ws(("a", [{"symbol": "X", "reco_tier": "good_setup", "entry_state": "near_setup", "tag": 1}]),
             ("b", [{"symbol": "X", "reco_tier": "good_setup", "entry_state": "ready", "tag": 2}]),
             ("c", [{"symbol": "X", "reco_tier": "good_setup", "entry_state": "ready", "tag": 3}]))
    assert flatten_cards(ws)[0]["tag"] == 2


def test_empty_and_junk_are_valid():
    assert flatten_cards(None) == [] and flatten_cards({"categories": [{"cards": ["x", {"symbol": ""}]}]}) == []
