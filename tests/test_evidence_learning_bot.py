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
    closed = book.mark({"TCS": (93.0, 95.0, 92.0, 93.5)}, "2026-09-02")
    assert closed and closed[0].exit_reason in {"STOP", "GAP_STOP"}
    assert closed[0].realized_R < 0
    for _ in range(5):
        ingest_closed_trade(closed[0], path=policy_path, floors=floors)

    book2 = PaperBook(capital=100_000)
    next_card = dict(_eligible_card())
    next_card["symbol"] = "INFY"
    blocked = run_reco_paper_cycle(
        book=book2,
        cards=[next_card],
        workspace=_workspace([next_card]),
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
    chased = dict(_eligible_card())
    chased["chase_risk"] = True
    chased["entry_state"] = "extended"
    refused = run_reco_paper_cycle(
        book=book,
        cards=[chased],
        workspace=_workspace([chased]),
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


def test_winning_taken_trade_updates_policy_consumed_next_cycle(tmp_path):
    from product.paper_learning_loop import ingest_closed_trade, record_taken_evidence

    policy_path = tmp_path / "policies.json"
    floors = {"experimental": 2, "eligible": 3, "active": 4, "conditional_mult": 1}
    book = PaperBook(capital=100_000)
    card = _eligible_card()
    out = run_reco_paper_cycle(
        book=book, cards=[card], workspace=_workspace([card]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-01",
        persist_journal=True, policy_path=policy_path,
    )
    assert out["taken"]
    assert out["taken"][0]["why"]["title"] == "WHY BOT TOOK THIS"
    record_taken_evidence(out["taken"], as_of="2026-09-01")
    closed = book.mark({"TCS": (100.0, 116.0, 99.0, 115.0)}, "2026-09-02")
    assert closed and closed[0].exit_reason in {"TARGET", "GAP_TARGET"}
    assert closed[0].realized_R > 0
    for _ in range(5):
        ingest_closed_trade(closed[0], path=policy_path, floors=floors)

    book2 = PaperBook(capital=100_000)
    nxt = dict(_eligible_card())
    nxt["symbol"] = "INFY"
    followed = run_reco_paper_cycle(
        book=book2, cards=[nxt], workspace=_workspace([nxt]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-03",
        persist_journal=False, policy_path=policy_path,
    )
    assert followed["taken"]
    assert followed["taken"][0]["policy_effect"] == "SUPPORT"


def test_correct_rejection_is_not_pnl_and_does_not_disable_dd_gate(tmp_path):
    from product.counterfactual_learning import CORRECT_REJECTION
    from product.paper_learning_loop import settle_pending_counterfactuals

    book = PaperBook(capital=100_000)
    failed = dict(_eligible_card())
    failed["dd_verdict"] = "FAIL"
    out = run_reco_paper_cycle(
        book=book, cards=[failed], workspace=_workspace([failed]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-01",
        persist_journal=True, policy_path=tmp_path / "policies.json",
    )
    assert out["rejections"][0]["reason_code"] == "DD_GATE_FAILED"
    settled = settle_pending_counterfactuals(
        forward_return_by_symbol={"TCS": -7.2},
        path=tmp_path / "policies.json",
        floors={"experimental": 1, "eligible": 2, "active": 3},
    )
    assert settled["classifications"][CORRECT_REJECTION] == 1
    again = run_reco_paper_cycle(
        book=PaperBook(capital=100_000), cards=[failed], workspace=_workspace([failed]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-02",
        persist_journal=False, policy_path=tmp_path / "policies.json",
    )
    assert again["rejections"][0]["reason_code"] == "DD_GATE_FAILED"


def test_better_entry_wait_classifies_good_wait(tmp_path):
    from product.counterfactual_learning import GOOD_WAIT, ledger_path
    import json as _json

    policy_path = tmp_path / "policies.json"
    chased = dict(_eligible_card())
    chased["chase_risk"] = True
    chased["entry_state"] = "extended"
    first = run_reco_paper_cycle(
        book=PaperBook(capital=100_000), cards=[chased], workspace=_workspace([chased]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-01",
        persist_journal=True, policy_path=policy_path,
    )
    assert first["waits"][0]["reason_code"] == "ENTRY_TOO_EXTENDED"
    ready = dict(_eligible_card())
    ready["entry"] = 102.0
    ready["cmp"] = 102.0
    second = run_reco_paper_cycle(
        book=PaperBook(capital=100_000), cards=[ready], workspace=_workspace([ready]),
        now=datetime(2026, 9, 1, 11, tzinfo=timezone.utc), as_of="2026-09-02",
        persist_journal=True, policy_path=policy_path,
    )
    assert second["taken"]
    rows = [ _json.loads(line) for line in ledger_path().read_text().splitlines() if line.strip() ]
    classified = [r for r in rows if r.get("classification") == GOOD_WAIT]
    assert classified


def test_missed_winner_can_weaken_learned_setup_block_not_hard_gates(tmp_path):
    from product.paper_learning_loop import settle_pending_counterfactuals
    from product.counterfactual_learning import freeze_decision

    policy_path = tmp_path / "policies.json"
    floors = {"experimental": 2, "eligible": 3, "active": 4, "conditional_mult": 1}
    upsert_policy(
        policy_id="SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=10,
        expectancy_R=-0.45,
        path=policy_path,
        floors=floors,
        extra={"shrinkage_k": 0.01},
    )
    freeze_decision(
        symbol="WIPRO",
        reason_code=EVIDENCE_POLICY_BLOCK,
        decision="BLOCK",
        entry=100, stop=94, target=115, as_of="2026-09-01",
        evidence={"setup_label": "VCP", "group": "REJECTED"},
    )
    settle_pending_counterfactuals(
        forward_return_by_symbol={"WIPRO": 14.8},
        path=policy_path,
        floors=floors,
    )
    nxt = dict(_eligible_card())
    nxt["symbol"] = "INFY"
    out = run_reco_paper_cycle(
        book=PaperBook(capital=100_000), cards=[nxt], workspace=_workspace([nxt]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-03",
        persist_journal=False, policy_path=policy_path,
    )
    # Weakened overlay is no longer a hard BLOCK; chase gate is unchanged separately.
    assert out["taken"] or (out["rejections"] and out["rejections"][0]["reason_code"] != EVIDENCE_POLICY_BLOCK)
    chased = dict(_eligible_card())
    chased["chase_risk"] = True
    chased["entry_state"] = "extended"
    waited = run_reco_paper_cycle(
        book=PaperBook(capital=100_000), cards=[chased], workspace=_workspace([chased]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-03",
        persist_journal=False, policy_path=policy_path,
    )
    assert waited["waits"][0]["reason_code"] == "ENTRY_TOO_EXTENDED"


def test_conditional_setup_sector_is_not_a_global_vcp_slogan(tmp_path):
    from product.paper_learning_loop import ingest_closed_trade

    policy_path = tmp_path / "policies.json"
    floors = {"experimental": 2, "eligible": 3, "active": 4, "conditional_mult": 1}
    for i in range(6):
        ingest_closed_trade(
            {"symbol": "TCS", "setup_label": "VCP", "sector": "Technology",
             "regime": "RISK_ON", "realized_R": 0.80, "exit_reason": "TARGET",
             "entry_date": "2026-08-01", "exit_date": f"2026-08-{i+10:02d}"},
            path=policy_path, floors=floors,
        )
        ingest_closed_trade(
            {"symbol": "ONGC", "setup_label": "VCP", "sector": "Energy",
             "regime": "RISK_ON", "realized_R": -0.90, "exit_reason": "STOP",
             "entry_date": "2026-08-01", "exit_date": f"2026-08-{i+10:02d}"},
            path=policy_path, floors=floors,
        )
    tech = dict(_eligible_card())
    tech["symbol"] = "INFY"
    tech["sector"] = "Technology"
    energy = dict(_eligible_card())
    energy["symbol"] = "RELIANCE"
    energy["sector"] = "Energy"
    book = PaperBook(capital=100_000)
    out = run_reco_paper_cycle(
        book=book, cards=[tech, energy], workspace=_workspace([tech, energy]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-01",
        persist_journal=False, policy_path=policy_path, max_new=3,
    )
    taken_symbols = {t["symbol"] for t in out["taken"]}
    rejected = {r["symbol"]: r["reason_code"] for r in out["rejections"]}
    assert "INFY" in taken_symbols
    assert rejected.get("RELIANCE") == EVIDENCE_POLICY_BLOCK


def test_not_surfaced_scan_names_are_frozen_not_entered():
    book = PaperBook(capital=100_000)
    card = _eligible_card()
    out = run_reco_paper_cycle(
        book=book, cards=[card], workspace=_workspace([card]),
        now=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), as_of="2026-09-01",
        persist_journal=False,
        scan_records=[{"symbol": "BBB", "score": 91, "verdict": "WATCH", "price": 50}],
    )
    assert out["not_surfaced"][0]["symbol"] == "BBB"
    assert all(p.symbol != "BBB" for p in book.open.values())


def test_live_adapter_is_fail_closed():
    from product.execution_adapter import LiveExecutionAdapter, LiveMoneyLocked
    try:
        LiveExecutionAdapter().submit(None)
    except LiveMoneyLocked:
        return
    raise AssertionError("live adapter must refuse")


def test_selection_rank_is_not_an_ai_buy_score():
    from product.decision_context import score_breakdown
    payload = score_breakdown(_eligible_card())
    assert payload["invents_buy"] is False
    assert payload["parts"]
    assert "not an ai buy score" in payload["note"].lower()


def test_one_observation_is_insufficient_evidence(tmp_path):
    path = tmp_path / "policies.json"
    row = upsert_policy(
        policy_id="TINY",
        dimension="setup",
        bucket="NR7",
        sample_size=1,
        expectancy_R=4.0,
        path=path,
    )
    assert row["production_status"] == "OBSERVING"
    assert row["confidence"] == "INSUFFICIENT_EVIDENCE"
    effect = evaluate_policies({"setup_label": "NR7"}, path=path)
    assert effect["final_effect"] != BLOCK
    assert effect["invents_buy"] is False
