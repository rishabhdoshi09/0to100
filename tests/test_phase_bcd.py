"""Phases B–D — freeze, outcomes, portfolio overlay, sizing, experiments."""
from __future__ import annotations

from product.decision_attribution import (
    MISSED_REENTRY,
    RATIONAL_REJECTION_DESPITE_RALLY,
    WAIT_RATIONALLY_MAINTAINED,
    attribute_outcome,
)
from product.decision_calibration import display_confidence
from product.decision_freeze import freeze, get_freeze
from product.decision_outcomes import path_metrics
from product.exit_engine import ATR, HOLD, INITIAL_INVALIDATION, evaluate_exit
from product.experiment_queue import from_failures
from product.learning_ledger import LEVEL_OBSERVE, OBSERVED, promote, record
from product.portfolio_committee import WAIT_PORTFOLIO, apply_overlay, evaluate_portfolio
from product.portfolio_heat import measure
from product.portfolio_stress import run_scenarios
from product.risk_audit import audit_levels, audit_sizing
from product.scorecards import reason_scorecards
from product.shadow_execution import PAPER_ENTERED, SHADOW_NOT_EXECUTED, freeze_shadow, is_paper_fill


def test_freeze_is_immutable(tmp_path):
    db = tmp_path / "freeze.db"
    first = freeze({
        "decision_id": "INFY:2026-06-12:1",
        "symbol": "INFY",
        "as_of": "2026-06-12",
        "decision": "WAIT",
        "entry": 1500,
        "stop": 1400,
        "reason_code": "ENTRY_TOO_EXTENDED",
        "versions": {"committee_version": "committee_v2_families"},
    }, path=db)
    second = freeze({
        "decision_id": "INFY:2026-06-12:1",
        "symbol": "INFY",
        "as_of": "2026-06-12",
        "decision": "BUY",
        "entry": 1,
        "stop": 1,
    }, path=db)
    stored = get_freeze("INFY:2026-06-12:1", path=db)
    assert first["decision"] == "WAIT"
    assert second["decision"] == "WAIT"
    assert stored["decision"] == "WAIT"
    assert stored["rewritten_after_outcome"] is False


def test_path_metrics_use_frozen_stop():
    bars = [
        {"high": 102, "low": 99, "close": 101},
        {"high": 108, "low": 97, "close": 98},
        {"high": 99, "low": 89, "close": 90},
    ]
    m = path_metrics(entry=100, stop=90, target=120, bars=bars)
    assert m["mfe_pct"] == 8.0
    assert m["mae_pct"] == -11.0
    assert m["stop_status"] == "HIT"
    assert m["time_to_invalidation"] == 3
    assert m["retrospective_stop"] is False


def test_wait_rally_is_not_automatically_a_miss():
    row = attribute_outcome({
        "symbol": "INFY",
        "decision": "WAIT",
        "reason_code": "ENTRY_TOO_EXTENDED",
        "entry": 100,
        "stop": 90,
        "forward_return_pct": 15,
        "classification": "MISSED_WINNER",
        "later_valid_entry": False,
        "original_decision_rational": True,
    })
    # analyze_decision_quality decides rationality; a rally alone is not MISSED_REENTRY
    assert row["wait_attribution"] != MISSED_REENTRY


def test_avoid_rally_can_be_rational_rejection():
    row = attribute_outcome({
        "symbol": "INFY",
        "decision": "AVOID",
        "reason_code": "INSUFFICIENT_EVIDENCE",
        "vetoes": [{"code": "INSUFFICIENT_EVIDENCE"}],
        "forward_return_pct": 12,
        "classification": "MISSED_WINNER",
    })
    assert row["avoid_attribution"] == RATIONAL_REJECTION_DESPITE_RALLY


def test_display_confidence_is_not_a_fake_percentage():
    shown = display_confidence(tier="high_conviction", sample_size=3)
    assert shown["is_probability"] is False
    assert "%" not in shown["display"] or "not a win probability" in shown["display"]
    measured = display_confidence(tier="high_conviction", sample_size=40, hit_rate=0.55)
    assert measured["is_probability"] is True
    assert measured["kind"] == "MEASURED_HIT_RATE"


def test_learning_cannot_promote_silently(tmp_path):
    db = tmp_path / "learn.db"
    row = record(
        learning_id="q1",
        question="Does wake fail after contraction?",
        promotion_state=OBSERVED,
        path=db,
    )
    assert row["level"] == LEVEL_OBSERVE
    import pytest
    with pytest.raises(ValueError, match="refusing silent promotion"):
        promote("q1", path=db)
    stored = __import__("product.learning_ledger", fromlist=["get"]).get("q1", path=db)
    assert stored["promotion_state"] == OBSERVED


def test_portfolio_overlay_preserves_buy():
    overlay = evaluate_portfolio(
        {"symbol": "INFY", "decision": "BUY", "sector": "IT"},
        open_positions=[{"symbol": "TCS", "sector": "IT"}, {"symbol": "WIPRO", "sector": "IT"}],
    )
    assert overlay["stock_decision"] == "BUY"
    assert overlay["thesis_preserved"] is True
    out = apply_overlay({"symbol": "INFY", "decision": "BUY", "candidate_state": "READY"}, overlay)
    assert out["decision"] == "BUY"
    if overlay["portfolio_verdict"] == WAIT_PORTFOLIO:
        assert out["execution_state"] == "BLOCKED_PORTFOLIO"


def test_sizing_uses_actual_stop():
    wide = audit_sizing(entry=100, stop=80, capital=1_000_000)
    tight = audit_sizing(entry=100, stop=99, capital=1_000_000)
    assert wide["uses_actual_stop_distance"] is True
    assert wide["naive_shares"] < tight["naive_shares"]
    levels = audit_levels({"entry": 100, "stop": 90, "target": 120, "atr": 5})
    assert levels["target_artificial"] is True
    assert levels["risk_basis"] == "ATR_2X"


def test_exit_engine_labels_atr_fallback():
    hit = evaluate_exit({"entry": 100, "stop": 90, "target": 120, "atr": 5}, last_price=89)
    assert hit["action"] == INITIAL_INVALIDATION
    hold = evaluate_exit({"entry": 100, "stop": 90, "target": 120, "atr": 5}, last_price=101)
    assert hold["action"] == HOLD
    assert hold["stop_kind"] == ATR
    assert hold["target_artificial"] is True


def test_shadow_is_not_a_paper_fill(tmp_path):
    row = freeze_shadow({"symbol": "INFY", "decision": "BUY", "entry": 100, "stop": 90}, path=tmp_path / "s.jsonl")
    assert row["status"] == SHADOW_NOT_EXECUTED
    assert is_paper_fill(row) is False
    assert row["status"] != PAPER_ENTERED


def test_heat_and_stress():
    positions = [
        {"symbol": "INFY", "entry": 100, "stop": 90, "qty": 100, "sector": "IT"},
        {"symbol": "TCS", "entry": 100, "stop": 90, "qty": 100, "sector": "IT"},
    ]
    heat = measure(positions, capital=1_000_000)
    assert heat["gross_open_risk_pct"] == 0.2
    stress = run_scenarios(positions, capital=1_000_000)
    assert stress["not_a_forecast"] is True
    assert "market_gap_down_5pct" in stress


def test_reason_scorecard_shows_sample_size():
    cards = reason_scorecards([
        {"reason_code": "EXTENDED", "entry": 100, "stop": 90, "forward_return_pct": 2},
        {"reason_code": "EXTENDED", "entry": 100, "stop": 90, "forward_return_pct": -1},
    ])
    assert cards["reasons"][0]["n"] == 2
    assert cards["reasons"][0]["ranked"] is False
    assert cards["affects_production"] is False


def test_experiment_queue_needs_repeated_pattern(tmp_path):
    q = tmp_path / "q.jsonl"
    none = from_failures([{"wait_attribution": "MISSED_REENTRY"}] * 3, path=q)
    assert none == []
    some = from_failures(
        [{"wait_attribution": "MISSED_REENTRY", "reason_code": "ENTRY_TOO_EXTENDED"}] * 8,
        path=q,
    )
    assert some[0]["question_kind"] == "MISSED_REENTRY_WAKE"
    assert some[0]["production_implication"] == "OBSERVE_ONLY"
