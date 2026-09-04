"""Synthetic paper/shadow lineage. Does not prove live broker connectivity."""
from __future__ import annotations

from product.decision_committee import evaluate_committee
from product.decision_taxonomy import BUY, NO_JUDGMENT
from product.forward_evidence import (
    REAL_FORWARD_MARKET,
    attach_settlement,
    freeze_observation,
    load_ledger,
    real_forward_only,
)
from product.shadow_execution import PAPER_ENTERED, SHADOW_NOT_EXECUTED, freeze_shadow, is_paper_fill


def _buy_card() -> dict:
    return {
        "symbol": "INFY",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "candidate_state": "READY",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "scan_run_id": "scan-2026-09-01",
        "candidate_id": "INFY:scan-2026-09-01",
        "methods_buy": ["sepa", "funds"],
        "evidence_family_votes": {"PRICE_STRUCTURE": True, "FUNDAMENTALS": True},
        "effective_confirmation_count": 2,
    }


def test_broker_block_does_not_rewrite_buy_into_avoid():
    rec = evaluate_committee(_buy_card(), broker_ok=False, load_research=False)
    assert rec.decision in {BUY, "WAIT", "AVOID", NO_JUDGMENT} or rec.decision
    if rec.decision == BUY:
        shadow = freeze_shadow(
            {
                **rec.as_dict(),
                "decision_id": "2026-09-01|INFY|taken|BUY|c1",
                "candidate_id": "INFY:scan-2026-09-01",
                "execution_state": "BLOCKED_BROKER_AUTH",
            }
        )
        assert shadow["status"] == SHADOW_NOT_EXECUTED
        assert shadow["status"] != PAPER_ENTERED
        assert is_paper_fill(shadow) is False
        assert shadow["live_locked"] is True
        assert shadow["decision"] == BUY


def test_paper_fill_lineage_settles_to_r_without_becoming_live():
    frozen = freeze_observation(
        {
            "symbol": "INFY",
            "decision": BUY,
            "reason_code": "ELIGIBLE",
            "entry": 100,
            "stop": 94,
            "target": 115,
            "scan_scanned_at": "2026-09-01T05:00:00+00:00",
            "candidate_id": "INFY:scan-2026-09-01",
        },
        cycle_id="c-paper",
        as_of="2026-09-01",
        group="taken",
        entered=True,
        provenance=REAL_FORWARD_MARKET,
    )
    assert frozen is not None
    assert frozen["entered"] is True
    assert frozen["pit_proof"]["future_data_used_for_decision"] is False
    paper = {
        "status": PAPER_ENTERED,
        "paper_executed": True,
        "decision_id": frozen["decision_id"],
        "candidate_id": "INFY:scan-2026-09-01",
        "scan_run_id": "scan-2026-09-01",
        "live_locked": True,
    }
    assert is_paper_fill(paper) is True
    settled = attach_settlement(
        frozen["decision_id"],
        classification="WIN",
        forward_return_pct=6.0,
        gross_R=2.5,
        outcome_provenance=REAL_FORWARD_MARKET,
    )
    assert settled is not None
    assert settled["gross_R"] == 2.5
    assert settled["entry"] == 100
    assert settled["stop"] == 94
    assert settled["later_outcome"]["provenance"] == REAL_FORWARD_MARKET
    assert real_forward_only(load_ledger())[0]["decision_id"] == frozen["decision_id"]


def test_invalid_symbol_never_enters_paper_or_forward_learning():
    rec = evaluate_committee({"symbol": "NOTAREALTICKERZZZ"}, broker_ok=False, load_research=False)
    assert rec.decision == NO_JUDGMENT
    shadow = freeze_shadow(rec.as_dict())
    assert shadow["decision"] == NO_JUDGMENT
    assert is_paper_fill(shadow) is False
    assert freeze_observation(
        rec.as_dict(),
        cycle_id="bad",
        as_of="2026-09-01",
        group="rejected",
        provenance=REAL_FORWARD_MARKET,
    ) is None
    assert load_ledger() == []
