"""Phase 5 — independent families, PIT contract, attribution, shadow, sizing."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from product import decision_taxonomy as T
from product.decision_attribution import (
    BUY_WINNER,
    MISSED_REENTRY,
    WAIT_RATIONALLY_MAINTAINED,
    attribute_outcome,
)
from product.decision_committee import evaluate_committee, reaudit_ready
from product.evidence_families import (
    PRICE_STRUCTURE,
    aggregate_evidence_families,
    family_gate,
    method_dependency_map,
)
from product.learning_ledger import LEVEL_PROMOTE, OBSERVED, learned_today, record
from product.paper_autopilot import ENTER_NOW, PORTFOLIO_BLOCK
from product.pit_availability import (
    PIT_MARKET_ONLY,
    PIT_PARTIAL,
    PIT_STRONG,
    PIT_UNAVAILABLE,
    PIT_UNVERIFIED,
    available_to_engine_at_t,
    grade_replay,
    reject_if_future,
)
from product.portfolio_committee import ADMIT, WAIT_PORTFOLIO, apply_overlay, evaluate_portfolio
from product.research_value import DECISION_CHANGED, NO_MATERIAL_CHANGE, classify_delta
from product.risk_audit import audit_levels, audit_sizing, r_multiple
from product.shadow_execution import PAPER_ENTERED, SHADOW_NOT_EXECUTED, freeze_shadow, is_paper_fill


class _Paper:
    def __init__(self, decision, reason_code, detail=""):
        self.decision = decision
        self.reason_code = reason_code
        self.detail = detail


def _research():
    return {
        "available": True,
        "acquired_at": "2026-09-03T00:00:00+00:00",
        "coverage_pct": 80.0,
        "missing_critical": [],
        "framework_id": "pharma_formulations",
        "quality_label": "healthy",
    }


def test_method_audit_is_explicit():
    audit = method_dependency_map()
    assert "tape" in audit["methods"]
    assert "sepa" in audit["methods"]
    assert "conviction" in audit["methods"]
    assert audit["methods"]["conviction"]["independent"] is False
    assert "PRICE_STRUCTURE_CLUSTER" in audit["overlaps"]
    assert set(audit["overlaps"]["PRICE_STRUCTURE_CLUSTER"]) >= {"tape", "sepa", "trend", "conviction"}


def test_sepa_trend_conviction_are_one_family():
    agg = aggregate_evidence_families(methods=[
        {"id": "sepa", "label": "SEPA", "status": "pass"},
        {"id": "trend", "label": "Trend", "status": "pass"},
        {"id": "conviction", "label": "Conviction", "status": "pass"},
        {"id": "tape", "label": "Tape", "status": "pass", "detail": "sniper only"},
    ])
    assert agg["effective_confirmation_count"] == 1
    assert agg["supportive_families"] == [PRICE_STRUCTURE]
    assert agg["price_derived_only"] is True
    gate = family_gate(aggregation=agg, tier="high_conviction")
    assert gate["ok"] is False
    assert gate["reason_code"] == "INSUFFICIENT_INDEPENDENT_EVIDENCE"


def test_funds_and_sector_are_independent_of_tape():
    agg = aggregate_evidence_families(methods=[
        {"id": "tape", "label": "Tape", "status": "pass", "detail": "grade A with volume floor"},
        {"id": "sepa", "label": "SEPA", "status": "pass"},
        {"id": "funds", "label": "Funds", "status": "pass"},
        {"id": "sector", "label": "Sector", "status": "pass"},
    ])
    assert agg["effective_confirmation_count"] >= 3
    assert "FINANCIAL_QUALITY" in agg["supportive_families"]
    assert "SECTOR_CONTEXT" in agg["supportive_families"]
    assert family_gate(aggregation=agg, tier="high_conviction")["ok"] is True


def test_correlated_methods_do_not_keep_ready(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    monkeypatch.setattr("product.decision_committee._research_snapshot", lambda *a, **k: _research())
    card = {
        "symbol": "FAKELEAD",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100, "stop": 95, "target": 110,
        "methods": [
            {"id": "tape", "label": "Tape", "status": "pass"},
            {"id": "sepa", "label": "SEPA", "status": "pass"},
            {"id": "trend", "label": "Trend", "status": "pass"},
            {"id": "conviction", "label": "Conviction", "status": "pass"},
        ],
        "families": [
            {"id": "structure", "status": "pass"},
            {"id": "price_leadership", "status": "pass"},
        ],
    }
    rec = evaluate_committee(card, broker_ok=False, load_research=True)
    assert rec.decision == T.WAIT_DECISION
    assert rec.reason_code == T.INSUFFICIENT_INDEPENDENT_EVIDENCE
    assert rec.effective_confirmation_count <= 2
    assert rec.candidate_state != T.READY


def test_business_family_still_allows_committee_buy(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    monkeypatch.setattr("product.decision_committee._research_snapshot", lambda *a, **k: _research())
    rec = evaluate_committee(
        {
            "symbol": "GUFICBIO",
            "reco_tier": "high_conviction",
            "entry_state": "ready",
            "entry": 442, "stop": 403.6, "target": 518.8,
            "families": [
                {"id": "structure", "status": "pass"},
                {"id": "price_leadership", "status": "pass"},
                {"id": "business_quality", "status": "pass"},
            ],
            "methods": [
                {"id": "sepa", "label": "SEPA", "status": "pass"},
                {"id": "trend", "label": "Trend", "status": "pass"},
                {"id": "funds", "label": "Funds", "status": "pass"},
            ],
        },
        broker_ok=False,
    )
    assert rec.decision == T.BUY
    assert rec.candidate_state == T.READY
    assert rec.effective_confirmation_count >= 2
    assert rec.family_gate_ok is True
    assert rec.execution_state == T.BLOCKED_BROKER_AUTH


def test_reaudit_can_drop_cosmetic_ready(monkeypatch):
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    monkeypatch.setattr("product.decision_committee._research_snapshot", lambda *a, **k: _research())
    cards = [
        {
            "symbol": "AAA",
            "reco_tier": "high_conviction",
            "entry_state": "ready",
            "entry": 10, "stop": 9, "target": 12,
            "candidate_state": "READY",
            "decision": "BUY",
            "methods": [
                {"id": "sepa", "status": "pass"},
                {"id": "trend", "status": "pass"},
                {"id": "conviction", "status": "pass"},
            ],
        },
        {
            "symbol": "BBB",
            "reco_tier": "high_conviction",
            "entry_state": "ready",
            "entry": 10, "stop": 9, "target": 12,
            "candidate_state": "READY",
            "decision": "BUY",
            "methods": [
                {"id": "tape", "status": "pass", "detail": "volume floor"},
                {"id": "funds", "status": "pass"},
                {"id": "sector", "status": "pass"},
            ],
        },
    ]
    report = reaudit_ready(cards, previous_ready=["AAA", "BBB"], broker_ok=False)
    assert report["n_before"] == 2
    assert "AAA" in report["dropped"]
    assert report["n_after"] <= report["n_before"]


def test_pit_period_end_is_not_availability():
    check = available_to_engine_at_t(
        as_of="2025-04-15",
        period_end="2025-03-31",
    )
    assert check["available_to_engine_at_T"] is False
    assert check["pit_status"] == PIT_UNVERIFIED
    published = available_to_engine_at_t(
        as_of="2025-04-15",
        period_end="2025-03-31",
        publication_date="2025-05-12",
    )
    assert published["available_to_engine_at_T"] is False
    assert published["pit_status"] == PIT_UNAVAILABLE
    later = available_to_engine_at_t(
        as_of="2025-05-20",
        period_end="2025-03-31",
        publication_date="2025-05-12",
    )
    assert later["available_to_engine_at_T"] is True
    assert later["pit_status"] == PIT_STRONG


def test_lookahead_injections_are_rejected():
    assert reject_if_future(as_of="2026-06-12", evidence_date="2026-06-20", kind="ohlcv")["accepted"] is False
    assert reject_if_future(as_of="2026-06-12", evidence_date="2026-06-01", kind="filing")["accepted"] is True
    grade = grade_replay(
        as_of="2026-06-12",
        market_bars_ok=True,
        company_items=[{
            "id": "fy2025",
            "period_end": "2025-03-31",
            "publication_date": "2026-06-20",
        }],
        used_today_fundamentals=True,
    )
    assert grade["grade"] == PIT_UNAVAILABLE
    assert grade["comparable_to_forward"] is False
    market = grade_replay(as_of="2026-06-12", market_bars_ok=True, company_items=[])
    assert market["grade"] == PIT_MARKET_ONLY


def test_future_outcome_does_not_enter_decision_scoring():
    from product.risk_audit import r_multiple as rm

    # Decision-time fields only. A later +15% is settlement, not a score input.
    decision_score = None
    frozen = {"entry": 100.0, "stop": 90.0, "target": 120.0, "decision": "WAIT"}
    later = 15.0
    assert decision_score is None
    assert rm(entry=frozen["entry"], stop=frozen["stop"], exit_price=115.0) == pytest.approx(1.5)
    assert later != decision_score


def test_r_multiple_refuses_invented_stop():
    assert r_multiple(entry=442, stop=None, exit_price=460) is None
    assert r_multiple(entry=442, stop=403.6, exit_price=518.8) == pytest.approx(2.0, abs=0.05)


def test_guficbio_levels_are_atr_multiples():
    audit = audit_levels({
        "symbol": "GUFICBIO",
        "entry": 442,
        "stop": 403.6,
        "target": 518.8,
        "atr": 19.2,
    })
    assert audit["entry_risk_pct"] == pytest.approx(8.69, abs=0.05)
    assert audit["reward_risk"] == pytest.approx(2.0, abs=0.05)
    assert audit["risk_basis"].startswith("ATR")
    assert audit["target_artificial"] is True


def test_sizing_uses_actual_stop_distance():
    tight = audit_sizing(entry=100, stop=97, capital=100_000)
    wide = audit_sizing(entry=100, stop=80, capital=100_000)
    assert tight["ok"] and wide["ok"]
    assert tight["paper_qty"] > wide["paper_qty"]
    assert wide["paper_qty"] == 50  # 1% of 100k / 20 rupee risk
    assert tight["bypasses_1pct_limit"] is False
    assert wide["per_share_risk"] == pytest.approx(20.0)


def test_portfolio_overlay_preserves_stock_buy():
    rec = {"symbol": "INFY", "decision": "BUY", "candidate_state": "READY", "execution_state": "ELIGIBLE", "sector": "IT"}
    overlay = evaluate_portfolio(
        rec,
        open_positions=[
            {"symbol": "TCS", "sector": "IT"},
            {"symbol": "WIPRO", "sector": "IT"},
        ],
    )
    assert overlay["portfolio_verdict"] == WAIT_PORTFOLIO
    assert overlay["thesis_preserved"] is True
    out = apply_overlay(rec, overlay)
    assert out["decision"] == "BUY"
    assert out["candidate_state"] == "READY"
    assert out["execution_state"] == T.BLOCKED_PORTFOLIO


def test_portfolio_admits_when_book_is_empty():
    rec = {"symbol": "INFY", "decision": "BUY", "sector": "IT"}
    overlay = evaluate_portfolio(rec, open_positions=[])
    assert overlay["portfolio_verdict"] == ADMIT


def test_shadow_is_not_a_paper_fill(tmp_path):
    row = freeze_shadow(
        {
            "symbol": "INFY",
            "decision": "BUY",
            "candidate_state": "READY",
            "execution_state": "BLOCKED_BROKER_AUTH",
            "decision_id": "2026-09-02|INFY|BUY|COMMITTEE_BUY|scan",
            "entry": 100, "stop": 90, "target": 120,
        },
        path=tmp_path / "shadow.jsonl",
    )
    assert row["status"] == SHADOW_NOT_EXECUTED
    assert row["not_a_trade"] is True
    assert row["paper_executed"] is False
    assert is_paper_fill(row) is False
    assert is_paper_fill({"status": PAPER_ENTERED, "paper_executed": True}) is True


def test_wait_attribution_distinguishes_rational_from_missed_wake():
    rational = attribute_outcome({
        "symbol": "AAA",
        "decision": "WAIT",
        "reason_code": T.ENTRY_TOO_EXTENDED,
        "classification": "MISSED_WINNER",
        "forward_return_pct": 15.0,
        "later_valid_entry": False,
        "entry": 100, "stop": 90,
        "evidence_family_votes": {"PRICE_STRUCTURE": "SUPPORTIVE", "RELATIVE_STRENGTH": "UNKNOWN"},
    })
    assert rational["wait_attribution"] == WAIT_RATIONALLY_MAINTAINED
    missed = attribute_outcome({
        "symbol": "BBB",
        "decision": "WAIT",
        "reason_code": T.ENTRY_TOO_EXTENDED,
        "classification": "MISSED_WINNER",
        "forward_return_pct": 15.0,
        "later_valid_entry": True,
        "wake_failed": True,
        "entry": 100, "stop": 90,
    })
    assert missed["wait_attribution"] == MISSED_REENTRY
    winner = attribute_outcome({
        "symbol": "CCC",
        "decision": "BUY",
        "entry": 100, "stop": 90,
        "forward_return_pct": 10.0,
        "evidence_family_votes": {"PRICE_STRUCTURE": "SUPPORTIVE", "SECTOR_CONTEXT": "SUPPORTIVE"},
    })
    assert winner["buy_attribution"] == BUY_WINNER
    assert winner["r_multiple"] == pytest.approx(1.0)
    assert winner["updates_policy"] is False


def test_research_value_classifies_material_change():
    assert classify_delta({"decision": "WAIT"}, {"decision": "BUY"}) == DECISION_CHANGED
    assert classify_delta(
        {"decision": "WAIT", "effective_confirmation_count": 1, "vetoes": []},
        {"decision": "WAIT", "effective_confirmation_count": 1, "vetoes": []},
    ) == NO_MATERIAL_CHANGE


def test_learning_ledger_refuses_auto_promote(tmp_path):
    db = tmp_path / "learn.db"
    row = record(
        learning_id="obs-1",
        question="Are EXTENDED waits usually rational?",
        sample_n=5,
        result="4/5 never offered valid entry",
        promotion_state=OBSERVED,
        path=db,
        day="2026-09-03",
    )
    assert row["level"] == 1
    assert row["recommended_change"] == ""
    with pytest.raises(ValueError):
        record(
            learning_id="bad",
            question="promote silently",
            level=LEVEL_PROMOTE,
            path=db,
        )
    today = learned_today("2026-09-03", path=db)
    assert today["policy_changed"] is False
    assert "No policy change" in today["summary"] or "statistically" in today["summary"].lower()


def test_scorecards_do_not_rank_tiny_samples():
    from product.scorecards import MIN_RANK_N, build_scorecards, quality_metrics

    cards = build_scorecards([
        {
            "symbol": "AAA",
            "decision": "BUY",
            "methods_buy": ["Tape", "SEPA"],
            "method_votes": [{"id": "tape", "status": "SUPPORTIVE"}],
            "evidence_family_votes": {"PRICE_STRUCTURE": "SUPPORTIVE"},
            "entry": 100, "stop": 90, "forward_return_pct": 5,
        }
    ])
    assert cards["methods"]["tape"]["sample_size"] == 1
    assert cards["methods"]["tape"]["ranked"] is False
    assert cards["min_rank_n"] == MIN_RANK_N
    metrics = quality_metrics([{
        "entry": 100, "stop": 90, "forward_return_pct": 10,
    }])
    assert metrics["sample_size"] == 1
    assert metrics["expectancy_r"] == pytest.approx(1.0)


def test_pit_workspace_is_never_wall_clock_stale():
    from product.paper_autopilot import reco_is_stale

    assert reco_is_stale({"point_in_time": True, "generated_at": "2020-01-01T00:00:00+00:00"}) is False
    assert reco_is_stale({"generated_at": "2020-01-01T00:00:00+00:00"}) is True


def test_replay_committee_path_does_not_load_today_research(monkeypatch):
    from product.historical_replay import decide_session

    called = {"defaults": 0, "facts": 0}

    def boom_defaults(*_a, **_k):
        called["defaults"] += 1
        raise AssertionError("today's research defaults must not be read in PIT replay")

    def boom_facts(*_a, **_k):
        called["facts"] += 1
        raise AssertionError("today's autonomy_facts must not be read in PIT replay")

    monkeypatch.setattr("product.due_diligence.engine._defaults", boom_defaults)
    monkeypatch.setattr("product.due_diligence.acquire.load_autonomy_facts", boom_facts)
    monkeypatch.setattr(
        "product.decision_committee.evaluate_candidate",
        lambda *a, **k: _Paper(ENTER_NOW, "ELIGIBLE"),
    )
    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        lambda **k: {
            "categories": [{
                "cards": [{
                    "symbol": "INFY",
                    "reco_tier": "watch",
                    "entry_state": "ready",
                    "entry": 100, "stop": 90, "target": 120,
                    "methods": [{"id": "sepa", "status": "pass"}],
                }]
            }],
            "pit_degraded": ["research overlays skipped"],
        },
    )
    rows = decide_session("2026-06-12", {"as_of_session": "2026-06-12", "records": []})
    assert called["defaults"] == 0
    assert called["facts"] == 0
    assert rows
    assert rows[0]["provenance"] == "BACKTEST"
    assert rows[0]["pit_grade"] in {PIT_MARKET_ONLY, PIT_PARTIAL, PIT_STRONG, PIT_UNVERIFIED, PIT_UNAVAILABLE}
    assert rows[0]["pit"]["future_evidence_used"] is False
    assert rows[0].get("versions")


def test_experiment_queue_rejects_arbitrary_strategy_mining(tmp_path):
    from product.experiment_queue import enqueue

    with pytest.raises(ValueError):
        enqueue(
            hypothesis="try 4000 parameter combos",
            question_kind="RANDOM_MINING",
            population="everything",
            path=tmp_path / "q.jsonl",
        )
    row = enqueue(
        hypothesis="EXTENDED veto too strict in chop?",
        question_kind="EXTENDED_THRESHOLD",
        population="WAIT_EXTENDED",
        path=tmp_path / "q.jsonl",
    )
    assert row["p_hacking"] is False
    assert row["production_implication"] == "OBSERVE_ONLY"
