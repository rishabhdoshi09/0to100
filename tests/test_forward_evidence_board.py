"""The screen that says whether the desk has earned any evidence yet.

Its job is to be as willing to say "nothing" as to say "here is the edge". A
dashboard that renders an empty state as a tidy row of zeros invites the
reader to treat the zeros as measurements, and zero settled trades is not a
0% win rate — it is an absence of any claim at all.
"""
from __future__ import annotations

import pytest

from product import conditional_evidence as CE
from product.decision_chain import Outcome
from product.evidence_class import HISTORICAL_REPLAY, PAPER_FORWARD, TEST_FIXTURE
from product.forward_evidence_board import (
    ACCUMULATING,
    MEASURED,
    NO_MARKET_EVIDENCE,
    build_forward_evidence_board,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "pol.json"))


def _settle(n: int, *, realized_R: float, context: str,
            evidence_class: str = PAPER_FORWARD) -> None:
    for i in range(n):
        outcome = Outcome(
            position_id=f"pos_{context}_{i}", paper_order_id=f"ord_{i}",
            paper_intent_id=f"int_{i}", decision_id=f"dec_{i}", symbol="AAA",
            realized_R=realized_R, exit_reason="STOP",
            exit_session=f"2026-09-{(i % 28) + 1:02d}",
            resolved_at=f"2026-10-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
            evidence_class=evidence_class,
        )
        CE.record_outcome(outcome, context_key=context, evidence_class=evidence_class)


def test_nothing_settled_says_nothing_settled():
    board = build_forward_evidence_board()
    assert board["state"] == NO_MARKET_EVIDENCE
    assert board["settled_trades"] == 0
    assert "has settled" in board["headline"]
    assert "earned" in board["headline"]


def test_a_thin_sample_is_accumulating_not_measured():
    _settle(5, realized_R=1.0, context="setup=VCP|regime=HEALTHY")
    board = build_forward_evidence_board()
    assert board["state"] == ACCUMULATING
    assert board["settled_trades"] == 5
    assert board["cells_usable_for_ranking"] == 0
    assert f"{CE.MIN_SAMPLE} needed" in board["headline"]


def test_a_full_sample_is_measured():
    _settle(CE.MIN_SAMPLE, realized_R=0.5, context="setup=VCP|regime=HEALTHY")
    board = build_forward_evidence_board()
    assert board["state"] == MEASURED
    assert board["cells_usable_for_ranking"] == 1


def test_sample_size_is_reported_by_setup_and_by_regime():
    _settle(4, realized_R=1.0, context="setup=VCP|regime=HEALTHY")
    _settle(6, realized_R=-1.0, context="setup=VCP|regime=NARROW")
    _settle(3, realized_R=2.0, context="setup=CUP|regime=HEALTHY")
    board = build_forward_evidence_board()

    by_setup = {row["setup"]: row for row in board["by_setup"]}
    assert by_setup["VCP"]["count"] == 10
    assert by_setup["CUP"]["count"] == 3

    by_regime = {row["regime"]: row for row in board["by_regime"]}
    assert by_regime["HEALTHY"]["count"] == 7
    assert by_regime["NARROW"]["count"] == 6


def test_grouped_expectancy_is_weighted_by_sample_not_averaged_over_cells():
    """Nine trades at -1R and one at +1R is not "an average of zero"."""
    _settle(9, realized_R=-1.0, context="setup=VCP|regime=A")
    _settle(1, realized_R=1.0, context="setup=VCP|regime=B")
    row = next(r for r in build_forward_evidence_board()["by_setup"] if r["setup"] == "VCP")
    assert row["count"] == 10
    assert row["expectancy_R"] == pytest.approx(-0.8)


def test_the_distribution_shows_where_outcomes_landed():
    _settle(2, realized_R=-2.5, context="setup=VCP|regime=A")
    _settle(3, realized_R=0.5, context="setup=CUP|regime=A")
    _settle(1, realized_R=3.0, context="setup=FLAG|regime=A")
    distribution = build_forward_evidence_board()["r_distribution"]
    assert distribution["<= -2R"] == 2
    assert distribution["0..1R"] == 3
    assert distribution[">= 2R"] == 1


def test_open_trades_are_unresolved_not_missing():
    book = {"open": [
        {"symbol": "AAA", "entry_date": "2026-09-10", "decision_id": "dec_1",
         "context_key": "setup=VCP|regime=HEALTHY"},
        {"symbol": "BBB", "entry_date": "2026-09-09", "decision_id": ""},
    ]}
    board = build_forward_evidence_board(book=book)
    assert board["unresolved_count"] == 2
    assert [r["symbol"] for r in board["unresolved"]] == ["BBB", "AAA"]
    assert board["unattributable_open"] == 1


def test_a_live_book_object_works_as_well_as_the_persisted_json():
    from research.auto_research.paper_book import PaperBook

    book = PaperBook(capital=100_000.0)
    book.open_position("s1", "AAA", 100.0, 95.0, 115.0, "2026-09-12", 10,
                       decision_id="dec_x", context_key="setup=VCP")
    board = build_forward_evidence_board(book=book)
    assert board["unresolved_count"] == 1
    assert board["unresolved"][0]["attributable"] is True


def test_replay_and_fixture_rows_are_reported_but_never_counted():
    _settle(40, realized_R=3.0, context="setup=VCP|regime=A",
            evidence_class=HISTORICAL_REPLAY)
    _settle(40, realized_R=3.0, context="setup=VCP|regime=A",
            evidence_class=TEST_FIXTURE)
    board = build_forward_evidence_board()
    assert board["state"] == NO_MARKET_EVIDENCE, (
        "eighty profitable replay rows are still not market evidence"
    )
    assert board["settled_trades"] == 0
    assert board["non_market_evidence"]["historical_replay_cells"] == 1
    assert board["non_market_evidence"]["test_fixture_cells"] == 1
    assert "never counted" in board["non_market_evidence"]["note"]


def test_the_board_names_the_class_it_is_reporting():
    board = build_forward_evidence_board()
    assert board["evidence_class"] == PAPER_FORWARD
    assert board["min_sample"] == CE.MIN_SAMPLE


def test_the_api_exposes_the_board():
    import terminal_product_api as api

    paths = {route.path for route in api.app.routes if hasattr(route, "path")}
    assert "/api/forward-evidence" in paths


# ---------------------------------------------------------------------------
# Maturity grades DEPTH of evidence, never quality of it.
# ---------------------------------------------------------------------------
from product.forward_evidence_board import DEVELOPING, EARLY, INSUFFICIENT_SAMPLE, MATURE


@pytest.mark.parametrize("n,expected", [
    (0, NO_MARKET_EVIDENCE),
    (1, INSUFFICIENT_SAMPLE),
    (29, INSUFFICIENT_SAMPLE),
    (30, EARLY),
    (89, EARLY),
    (90, DEVELOPING),
    (299, DEVELOPING),
    (300, MATURE),
])
def test_maturity_follows_the_deepest_context(n, expected):
    if n:
        _settle(n, realized_R=0.1, context="setup=VCP|regime=HEALTHY")
    assert build_forward_evidence_board()["state"] == expected


def test_maturity_is_depth_not_edge():
    """A mature losing strategy is mature. The label makes no claim."""
    _settle(300, realized_R=-1.0, context="setup=VCP|regime=HEALTHY")
    board = build_forward_evidence_board()
    assert board["state"] == MATURE
    assert board["by_setup"][0]["expectancy_R"] == pytest.approx(-1.0)
    assert "not an edge" in board["headline"]
    assert "never a claim" in board["maturity_thresholds"]["note"]


def test_many_thin_contexts_do_not_add_up_to_maturity():
    """Depth is per context. Sixty one-trade cells is not a measured setup."""
    for i in range(60):
        _settle(1, realized_R=0.5, context=f"setup=S{i}|regime=HEALTHY")
    board = build_forward_evidence_board()
    assert board["settled_trades"] == 60
    assert board["state"] == INSUFFICIENT_SAMPLE


def test_drawdown_is_reported_over_the_settled_path():
    """Expectancy hides the path; a strategy is lived through, not averaged."""
    _settle(3, realized_R=-1.0, context="setup=VCP|regime=A")
    _settle(2, realized_R=1.0, context="setup=VCP|regime=B")
    shape = build_forward_evidence_board()["risk_shape"]
    assert shape["max_drawdown_R"] == pytest.approx(-3.0)
    assert shape["measured_from"] == 5


def test_risk_shape_reports_absence_rather_than_zero():
    board = build_forward_evidence_board()
    shape = board["risk_shape"]
    assert shape["worst_mae_R"] is None
    assert shape["best_mfe_R"] is None
    assert shape["max_drawdown_R"] is None
    assert shape["calibration_gap"] is None
    assert "no confidence was recorded" in shape["calibration_note"]


def test_calibration_gap_says_which_direction_the_system_erred():
    key = "setup=VCP|regime=HEALTHY"
    for i in range(10):
        outcome = Outcome(
            position_id=f"pos_{i}", paper_order_id=f"ord_{i}",
            paper_intent_id=f"int_{i}", decision_id=f"dec_{i}", symbol="AAA",
            realized_R=-1.0, exit_reason="STOP", exit_session=f"2026-09-{i + 1:02d}",
            resolved_at=f"2026-10-01T00:00:{i:02d}+00:00",
            evidence_class=PAPER_FORWARD,
        )
        CE.record_outcome(outcome, context_key=key, evidence_class=PAPER_FORWARD,
                          calibrated_confidence=0.8)
    shape = build_forward_evidence_board()["risk_shape"]
    assert shape["calibration_gap"] == pytest.approx(0.8)
    assert "claimed more confidence than it earned" in shape["calibration_note"]
