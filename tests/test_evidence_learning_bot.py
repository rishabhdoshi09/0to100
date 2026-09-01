"""Evidence-learning bot: policies must affect the next paper selection cycle."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from product.counterfactual_learning import (
    CORRECT_REJECTION,
    GOOD_WAIT,
    MISSED_WINNER,
    classify_forward,
    freeze_decision,
    settle,
)
from product.evidence_policy_engine import BLOCK, evaluate_policies
from product.learning_policy_store import ACTIVE, upsert_policy
from product.live_readiness import evaluate_live_readiness
from product.paper_autopilot import EVIDENCE_POLICY_BLOCK, run_reco_paper_cycle
from research.auto_research.paper_book import PaperBook


def _eligible_card():
    return {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": 3,
        "score": 82,
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }


def _workspace(cards):
    stamp = datetime(2026, 9, 1, 10, tzinfo=timezone.utc).isoformat()
    return {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "cards": cards}],
    }


def test_active_block_policy_is_consumed_on_next_cycle(tmp_path):
    path = tmp_path / "policies.json"
    upsert_policy(
        policy_id="CHASE_EXTENSION_V3",
        dimension="setup",
        bucket="VCP",
        sample_size=40,
        expectancy_R=-0.48,
        baseline_R=0.20,
        path=path,
        floors={"experimental": 3, "eligible": 5, "active": 10},
    )
    stored = __import__("json").loads(path.read_text())
    assert stored["policies"][0]["production_status"] == ACTIVE

    book = PaperBook(capital=100_000)
    card = _eligible_card()
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace=_workspace([card]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        as_of="2026-09-01",
        persist_journal=False,
        policy_path=path,
    )
    assert not book.open
    assert out["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def test_policy_cannot_invent_a_buy():
    effect = evaluate_policies(
        {"setup_label": "unknown-setup", "reco_tier": "watch"},
        policies=[{
            "policy_id": "X",
            "dimension": "setup",
            "bucket": "unknown-setup",
            "production_status": ACTIVE,
            "expectancy_difference_R": 2.0,
            "sample_size": 80,
            "confidence": "MEASURED",
        }],
    )
    assert effect["invents_buy"] is False
    assert effect["final_effect"] in { "SUPPORT", "NEUTRAL", "PENALIZE", BLOCK }


def test_counterfactual_classifications():
    assert classify_forward(entry=100, stop=94, target=115, forward_return_pct=-7.2) == CORRECT_REJECTION
    assert classify_forward(entry=100, stop=94, target=115, forward_return_pct=14.8) == MISSED_WINNER
    assert classify_forward(entry=100, stop=94, target=115, forward_return_pct=2.0, later_entered=True) == GOOD_WAIT


def test_counterfactual_freeze_and_settle(tmp_path):
    path = tmp_path / "cf.jsonl"
    row = freeze_decision(
        symbol="XYZ",
        reason_code="ENTRY_TOO_EXTENDED",
        decision="WAIT",
        entry=100,
        stop=94,
        target=115,
        as_of="2026-09-01",
        path=path,
    )
    settled = settle(row, forward_return_pct=12.0)
    assert settled["classification"] == MISSED_WINNER
    assert settled["outcome"]["not_pnl"] is True


def test_live_readiness_is_fail_closed():
    verdict = evaluate_live_readiness(
        settled_trades=500,
        trading_days=80,
        expectancy_R=0.4,
        max_drawdown_pct=8.0,
        distinct_regimes=3,
        stops_proven=True,
        critical_lanes_broken=False,
        rules_hash_stable=True,
    )
    assert verdict["live_enabled"] is False
    assert verdict["live_locked"] is True
    assert verdict["contract_ready"] is True


def test_taken_trade_updates_policy_consumed_next_cycle(tmp_path):
    """Winning taken trade → policy statistics → next similar candidate sees it."""
    path = tmp_path / "policies.json"
    # After a measured positive edge, SUPPORT does not invent BUY; eligible still enters.
    upsert_policy(
        policy_id="VCP_STRONG_SECTOR",
        dimension="setup",
        bucket="VCP",
        sample_size=32,
        expectancy_R=0.61,
        baseline_R=0.10,
        path=path,
        floors={"experimental": 3, "eligible": 5, "active": 10},
    )
    book = PaperBook(capital=100_000)
    card = _eligible_card()
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace=_workspace([card]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        as_of="2026-09-01",
        persist_journal=False,
        policy_path=path,
    )
    assert out["taken"]
    assert out["taken"][0]["policy_effect"] == "SUPPORT"


def test_settled_losing_trades_update_policy_consumed_on_next_cycle(tmp_path):
    """Taken → stop-out → policy statistics → next similar candidate is BLOCKED."""
    from product.paper_learning_loop import ingest_closed_trade, record_taken_evidence

    policy_path = tmp_path / "policies.json"
    floors = {"experimental": 2, "eligible": 3, "active": 4}
    book = PaperBook(capital=100_000)
    card = _eligible_card()
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace=_workspace([card]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        as_of="2026-09-01",
        persist_journal=True,
        policy_path=policy_path,
    )
    assert out["taken"]
    record_taken_evidence(out["taken"], as_of="2026-09-01")
    closed = book.mark("2026-09-02", {"TCS": (93.0, 95.0, 92.0, 93.5)})
    assert closed and closed[0].exit_reason in {"STOP", "GAP_STOP"}
    assert closed[0].realized_R < 0
    for _ in range(5):
        ingest_closed_trade(closed[0], path=policy_path, floors=floors)

    book2 = PaperBook(capital=100_000)
    blocked = run_reco_paper_cycle(
        book=book2,
        cards=[_eligible_card("INFY")],
        workspace=_workspace([_eligible_card("INFY")]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        as_of="2026-09-03",
        persist_journal=False,
        policy_path=policy_path,
    )
    assert not book2.open
    assert blocked["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def test_counterfactual_missed_winner_is_not_booked_as_pnl(tmp_path):
    from product.counterfactual_learning import freeze_decision, MISSED_WINNER
    from product.paper_learning_loop import settle_pending_counterfactuals

    freeze_decision(
        symbol="XYZ",
        reason_code="ENTRY_TOO_EXTENDED",
        decision="WAIT",
        entry=100,
        stop=94,
        target=115,
        as_of="2026-09-01",
    )
    out = settle_pending_counterfactuals(
        forward_return_by_symbol={"XYZ": 14.8},
        path=tmp_path / "policies.json",
        floors={"experimental": 1, "eligible": 2, "active": 3},
    )
    assert out["updated"] == 1
    assert out["classifications"][MISSED_WINNER] == 1
    book = PaperBook(capital=100_000)
    # Hard chase gate still holds — missed-winner stats must not disable it.
    refused = run_reco_paper_cycle(
        book=book,
        cards=[_eligible_card(chase_risk=True, entry_state="extended")],
        workspace=_workspace([_eligible_card(chase_risk=True, entry_state="extended")]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        as_of="2026-09-02",
        persist_journal=False,
        policy_path=tmp_path / "policies.json",
    )
    assert not book.open
    assert refused["waits"][0]["reason_code"] == "ENTRY_TOO_EXTENDED"


def test_learning_dashboard_has_no_ai_is_learning_copy():
    from product.paper_learning_loop import learning_dashboard
    payload = learning_dashboard()
    blob = json.dumps(payload).lower()
    assert "ai is learning" not in blob
    assert payload["live_locked"] is True
    assert payload["live_readiness"]["live_enabled"] is False
