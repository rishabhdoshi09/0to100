"""The decision board and the WHY payload, served from the saved scan.

The dishonesty this guards against is an empty list. "Nothing qualified today"
and "we never scanned" look identical on a screen that renders both as zero
rows, and they are opposite facts: the first is the desk working, the second is
the desk broken.
"""
from __future__ import annotations

import pytest

from product.decision import BUY, WATCH
from product.decision_service import decision_board, decision_why, decisions_from_workspace


def _workspace(*cards, scanned_at: str = "2026-09-12T03:00:00+00:00") -> dict:
    return {
        "schema_version": 4,
        "scan_scanned_at": scanned_at,
        "categories": [{"id": "momentum_breakouts", "cards": list(cards)}],
    }


def _card(symbol: str, **overrides) -> dict:
    card = {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "primary_thesis": "VCP",
        "entry": 100.0,
        "stop": 95.0,
        "target": 115.0,
        "allows_recommend": True,
        "scan_scanned_at": "2026-09-12T03:00:00+00:00",
        "methods": [{"id": "tape", "label": "Tape", "status": "pass"}],
        "families": [{"id": "price", "status": "SUPPORTIVE"}],
        "dd_status": "pass",
    }
    card.update(overrides)
    return card


@pytest.fixture(autouse=True)
def _isolated_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))


def test_no_saved_scan_is_stated_not_rendered_as_nothing_qualified():
    board = decision_board(workspace={})
    assert board["available"] is False
    assert board["state"] == "NO_DECISIONS"
    assert "No saved whole-market scan" in board["reason"]
    assert board["decisions"] == []


def test_a_scan_that_qualified_nobody_says_so_distinctly():
    board = decision_board(workspace=_workspace())
    assert board["available"] is True
    assert board["state"] == "NO_CANDIDATES"
    assert "nothing qualified" in board["reason"]


def test_the_board_ranks_and_explains_every_row():
    board = decision_board(workspace=_workspace(_card("INFY"), _card("TCS", reco_tier="watch")))
    assert board["state"] == "DECIDED"
    assert board["counts"] == {BUY: 1, WATCH: 1}
    assert board["actionable"] == 1
    for row in board["decisions"]:
        assert row["decision_id"]
        assert row["why"]
        assert row["context_key"]
        assert set(row["evidence_counts"]) == {"SUPPORTING", "CONFLICTING", "MISSING"}


def test_the_board_surfaces_evidence_gaps_without_opening_each_name():
    board = decision_board(workspace=_workspace(
        _card("INFY", dd_status=""), _card("TCS", dd_status=""),
    ))
    assert board["evidence_gaps"]["fundamentals"] == 2


def test_expected_R_is_computed_from_the_levels_not_taken_on_trust():
    board = decision_board(workspace=_workspace(_card("INFY")))
    assert board["decisions"][0]["expected_R"] == pytest.approx(3.0)


def test_why_for_a_name_the_desk_never_decided():
    """Absent from the shortlist is still an answer, and it must be a specific one.

    This used to assert a single dead sentence for every unshortlisted name.
    The page now has to say WHICH kind of absence it is, so the assertion is
    tightened to the stance rather than relaxed.
    """
    payload = decision_why("ZZZZ", workspace=_workspace(_card("INFY")))
    assert payload["available"] is False
    assert payload["symbol"] == "ZZZZ"
    assert payload["stance"] in {
        "NOT_SHORTLISTED", "NOT_EVALUATED", "NOT_A_TRADABLE_SYMBOL",
    }
    assert payload["reason"]
    # the old behaviour claimed nothing was known; the new one must be explicit
    assert "in_latest_scan" in payload


def test_why_carries_the_ranking_and_the_unfilled_sections():
    payload = decision_why("infy", workspace=_workspace(_card("INFY")))
    assert payload["available"] is True
    assert payload["symbol"] == "INFY"
    assert payload["ranking"]["decision_id"] == payload["decision_id"]
    assert payload["ranking_explanation"]
    assert isinstance(payload["unfilled_sections"], list)


def test_why_requires_a_symbol():
    with pytest.raises(ValueError):
        decision_why("", workspace=_workspace(_card("INFY")))


def test_cards_keep_their_category_when_they_cross():
    decisions = decisions_from_workspace(_workspace(_card("INFY")))
    assert [d.symbol for d in decisions] == ["INFY"]
    assert decisions[0].provenance["built_from"] == "recommendation_card"


def test_the_board_is_a_pure_function_of_the_workspace():
    workspace = _workspace(_card("INFY"), _card("TCS"))
    assert decision_board(workspace=workspace) == decision_board(workspace=workspace)


def test_the_api_exposes_both_decision_routes():
    import terminal_product_api as api

    paths = {route.path for route in api.app.routes if hasattr(route, "path")}
    assert "/api/decisions" in paths
    assert "/api/decisions/{symbol}/why" in paths


# ---------------------------------------------------------------------------
# The regime label the canonical path conditions on.
#
# Audited defect, self-inflicted: the endpoint read market["regime"] and
# market["state"], neither of which the market payload has. Every decision
# therefore carried an empty market_state, every context key said
# regime=UNKNOWN, and the whole point of conditioning evidence on the tape
# quietly stopped happening while everything still looked fine.
# ---------------------------------------------------------------------------
def test_the_market_regime_reaches_the_context_key(monkeypatch):
    import terminal_api as core
    import terminal_product_api as api

    monkeypatch.setattr(core, "_market_payload",
                        lambda: {"available": True, "health": "Healthy"})
    assert api._market_regime() == "HEALTHY"


def test_an_unavailable_market_view_yields_no_regime_rather_than_a_label(monkeypatch):
    """"Unavailable" as a regime would split the evidence cells in two."""
    import terminal_api as core
    import terminal_product_api as api

    monkeypatch.setattr(core, "_market_payload",
                        lambda: {"available": False, "health": "Unavailable"})
    assert api._market_regime() == ""


def test_a_broken_market_view_never_breaks_the_decision_board(monkeypatch):
    import terminal_api as core
    import terminal_product_api as api

    def explode():
        raise RuntimeError("index store is down")

    monkeypatch.setattr(core, "_market_payload", explode)
    assert api._market_regime() == ""


def test_the_regime_actually_conditions_the_evidence_cell():
    from product.decision_ranking import decision_context_key

    healthy = decisions_from_workspace(_workspace(_card("INFY")), market_state="HEALTHY")
    narrow = decisions_from_workspace(_workspace(_card("INFY")), market_state="NARROW")
    assert decision_context_key(healthy[0]) != decision_context_key(narrow[0])
    assert "regime=HEALTHY" in decision_context_key(healthy[0])
    assert "regime=NARROW" in decision_context_key(narrow[0])


def test_best_trades_use_the_same_production_selection_seam(monkeypatch):
    import product.paper_autopilot as PA

    calls = []

    class Result:
        def __init__(self, symbol, decision, score):
            self.symbol = symbol
            self.decision = decision
            self.selection_score = score
            self.reason_code = "ELIGIBLE" if decision == PA.ENTER_NOW else "WAIT_FOR_ENTRY"
            self.policy_effect = "NEUTRAL"
            self.card = _card(symbol, score=score)
            self.context = {}
            self.portfolio = {}

    def select(card, **kwargs):
        calls.append((card["symbol"], dict(kwargs)))
        if card["symbol"] == "INFY":
            return Result("INFY", PA.ENTER_NOW, 97.0)
        return Result("TCS", PA.WAIT, 99.0)

    monkeypatch.setattr(PA, "evaluate_selection_candidate", select)
    board = decision_board(
        workspace=_workspace(
            _card("INFY", score=70),
            _card("TCS", score=95),
        )
    )

    assert [row["symbol"] for row in board["best_trades"]] == ["INFY"]
    assert board["best_trades"][0]["production_selection_score"] == 97.0
    assert board["best_trades"][0]["discovery_decision"] == PA.ENTER_NOW
    assert board["best_trades_mode"] == "PRODUCTION_THESIS_DISCOVERY"
    assert all(kwargs["enforce_history"] is False for _symbol, kwargs in calls)


def test_best_trade_search_is_not_limited_by_board_display_limit(monkeypatch):
    import product.paper_autopilot as PA

    cards = [
        _card(f"S{i:02d}", score=100 - i)
        for i in range(45)
    ]
    cards.append(_card("ZZBEST", score=1))

    class Result:
        def __init__(self, card):
            self.symbol = card["symbol"]
            self.decision = PA.ENTER_NOW if card["symbol"] == "ZZBEST" else PA.WAIT
            self.selection_score = 999.0 if card["symbol"] == "ZZBEST" else 0.0
            self.reason_code = "ELIGIBLE" if self.decision == PA.ENTER_NOW else "WAIT_FOR_ENTRY"
            self.policy_effect = "NEUTRAL"
            self.card = dict(card)
            self.context = {}
            self.portfolio = {}

    monkeypatch.setattr(
        PA,
        "evaluate_selection_candidate",
        lambda card, **_kwargs: Result(card),
    )

    board = decision_board(workspace=_workspace(*cards), limit=40)

    assert len(board["decisions"]) == 40
    assert [row["symbol"] for row in board["best_trades"]] == ["ZZBEST"]
    assert board["best_trades"][0]["production_selection_score"] == 999.0



def test_best_trade_discovery_respects_restored_open_positions(monkeypatch):
    from types import SimpleNamespace
    import product.decision_service as DS
    import product.paper_autopilot as PA

    class Book:
        capital = 100_000.0
        max_positions = 5
        open = {
            ("QT_RECO", "INFY"): SimpleNamespace(
                symbol="INFY",
                sector="IT",
                approved_risk_pct=1.0,
                risk_amount=1000.0,
            ),
        }

    class Result:
        def __init__(self, card):
            self.symbol = card["symbol"]
            self.decision = PA.ENTER_NOW
            self.selection_score = float(card.get("score") or 0.0)
            self.reason_code = "ELIGIBLE"
            self.policy_effect = "NEUTRAL"
            self.card = dict(card)
            self.context = {}
            self.portfolio = {}

    monkeypatch.setattr(DS, "_read_only_paper_book", lambda: Book())
    monkeypatch.setattr(
        PA,
        "evaluate_selection_candidate",
        lambda card, **_kwargs: Result(card),
    )

    board = decision_board(
        workspace=_workspace(
            _card("INFY", score=99, sector="IT"),
            _card("TCS", score=90, sector="IT"),
        )
    )

    # INFY scores higher but is already held. The read-only discovery pass must
    # not advertise a duplicate position as the next actionable best trade.
    assert [row["symbol"] for row in board["best_trades"]] == ["TCS"]
    assert board["best_trades"][0]["restored_paper_positions_considered"] is True


def test_best_trade_discovery_exposes_capacity_only_reserves_without_relaxing_three_entry_cap(monkeypatch):
    import product.paper_autopilot as PA
    import product.portfolio_selection_authority as portfolio

    class Result:
        def __init__(self, card):
            self.symbol = card["symbol"]
            self.decision = PA.ENTER_NOW
            self.selection_score = float(card.get("score") or 0.0)
            self.reason_code = "ELIGIBLE"
            self.policy_effect = "NEUTRAL"
            self.card = dict(card)
            self.context = {}
            self.portfolio = {}

    monkeypatch.setattr(
        PA,
        "evaluate_selection_candidate",
        lambda card, **_kwargs: Result(card),
    )

    def capacity_three(ranked, **kwargs):
        assert kwargs["max_new"] == 3
        kept = list(ranked[:3])
        diverted = []
        for _score, decision in ranked[3:]:
            decision.decision = PA.WAIT
            decision.reason_code = "NOT_TOP_OF_PORTFOLIO"
            decision.portfolio = {"decision": PA.WAIT, "reason_code": "NOT_TOP_OF_PORTFOLIO"}
            diverted.append(decision)
        return kept, diverted

    monkeypatch.setattr(portfolio, "apply_portfolio_authority", capacity_three)

    cards = [_card(f"S{i}", score=100 - i) for i in range(5)]
    board = decision_board(workspace=_workspace(*cards))

    assert [row["symbol"] for row in board["best_trades"]] == ["S0", "S1", "S2", "S3", "S4"]
    assert [row["production_candidate_status"] for row in board["best_trades"]] == [
        "ENTER_NOW", "ENTER_NOW", "ENTER_NOW", "RESERVE_CAPACITY", "RESERVE_CAPACITY"
    ]
    assert [row["production_execution_slot"] for row in board["best_trades"]] == [
        True, True, True, False, False
    ]
    assert all(
        row["production_reason_code"] == "NOT_TOP_OF_PORTFOLIO"
        for row in board["best_trades"][3:]
    )
