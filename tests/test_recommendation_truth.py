"""Recommendation UI must follow frozen committee truth, not ensemble enthusiasm."""
from __future__ import annotations

from product.recommendation_truth import project_candidate_truth
from product.strategy_catalog import decorate_card


SCAN = "2026-09-03T09:45:00+00:00"
SYMBOL = "GUFICBIO"


def _candidate(**overrides):
    row = {
        "candidate_id": f"2026-09-03:{SYMBOL}",
        "symbol": SYMBOL,
        "scan_run_id": SCAN,
        "recommendation_id": f"{SCAN}:{SYMBOL}:high_conviction",
        "decision_id": f"2026-09-03|{SYMBOL}|BUY|COMMITTEE_BUY|{SCAN}",
        "paper_intent_id": f"2026-09-03|{SYMBOL}|BUY|COMMITTEE_BUY|{SCAN}:intent",
        "outcome_id": "",
        "decision": "BUY",
        "state": "READY",
        "entry_state": "ENTER_NOW",
        "execution_state": "BLOCKED_BROKER_AUTH",
        "reason": "COMMITTEE_BUY",
        "opportunity_id": SYMBOL,
        "wait_trigger_json": "{}",
    }
    row.update(overrides)
    return row


def test_exact_scan_buy_is_canonical_even_when_broker_auth_blocks_execution():
    out = project_candidate_truth(
        {"symbol": SYMBOL, "action_badge": "Watch"},
        scan_run_id=SCAN,
        candidate=_candidate(),
    )
    assert out["canonical_decision"] == "BUY"
    assert out["action_badge"] == "Buy"
    assert out["canonical_candidate_state"] == "READY"
    assert out["canonical_entry_state"] == "ENTER_NOW"
    assert out["canonical_execution_state"] == "BLOCKED_BROKER_AUTH"
    assert out["decision_truth_status"] == "CANONICAL_CURRENT_SCAN"
    assert out["decision_match_scope"] == "EXACT_SCAN_RUN"
    assert out["decision_id"].endswith(f"|{SCAN}")


def test_committee_wait_overrides_raw_ensemble_buy():
    out = project_candidate_truth(
        {"symbol": SYMBOL, "action_badge": "Buy"},
        scan_run_id=SCAN,
        candidate=_candidate(
            decision="WAIT",
            decision_id=f"2026-09-03|{SYMBOL}|WAIT|WAIT_EVIDENCE|{SCAN}",
            state="WAIT_EVIDENCE",
            entry_state="WAIT_EVIDENCE",
            execution_state="NOT_ELIGIBLE",
            reason="WAIT_EVIDENCE",
            wait_trigger_json='{"type":"EVIDENCE_COMPLETE"}',
        ),
    )
    assert out["raw_action_badge"] == "Buy"
    assert out["canonical_decision"] == "WAIT"
    assert out["action_badge"] == "Wait"
    assert out["wait_trigger"] == {"type": "EVIDENCE_COMPLETE"}


def test_same_session_rescan_cannot_reuse_previous_scan_decision():
    """A retained BUY after a new scan is pending, not a current BUY."""
    previous_scan = "2026-09-03T09:30:00+00:00"
    stale = _candidate(
        # Candidate row has already been touched by the new scan...
        scan_run_id=SCAN,
        recommendation_id=f"{SCAN}:{SYMBOL}:high_conviction",
        # ...but its durable decision still belongs to the previous scan.
        decision_id=f"2026-09-03|{SYMBOL}|BUY|COMMITTEE_BUY|{previous_scan}",
    )
    out = project_candidate_truth(
        {"symbol": SYMBOL, "action_badge": "Buy"},
        scan_run_id=SCAN,
        candidate=stale,
    )
    assert out["canonical_decision"] == "NO_JUDGMENT"
    assert out["action_badge"] == "No judgment"
    assert out["canonical_candidate_state"] == "UNJUDGED"
    assert out["decision_truth_status"] == "COMMITTEE_PENDING_FOR_SCAN"
    assert out["decision_match_scope"] == "EXACT_SCAN_CANDIDATE_ONLY"
    assert out["decision_id"] is None


def test_different_scan_candidate_is_never_projected():
    old_scan = "2026-09-02T09:45:00+00:00"
    out = project_candidate_truth(
        {"symbol": SYMBOL, "action_badge": "Buy"},
        scan_run_id=SCAN,
        candidate=_candidate(
            scan_run_id=old_scan,
            recommendation_id=f"{old_scan}:{SYMBOL}:high_conviction",
            decision_id=f"2026-09-02|{SYMBOL}|BUY|COMMITTEE_BUY|{old_scan}",
        ),
    )
    assert out["canonical_decision"] == "NO_JUDGMENT"
    assert out["action_badge"] == "No judgment"
    assert out["decision_truth_status"] == "CANDIDATE_SCAN_MISMATCH"


def test_no_candidate_never_leaks_raw_buy():
    out = project_candidate_truth(
        {"symbol": "NEWNAME", "action_badge": "Buy"},
        scan_run_id=SCAN,
        candidate=None,
    )
    assert out["raw_action_badge"] == "Buy"
    assert out["canonical_decision"] == "NO_JUDGMENT"
    assert out["action_badge"] == "No judgment"


def test_strategy_catalog_applies_canonical_overlay(monkeypatch):
    import product.recommendation_truth as truth

    monkeypatch.setattr(
        truth,
        "decorate_current_recommendation",
        lambda card: {
            "raw_action_badge": card.get("action_badge"),
            "canonical_decision": "WAIT",
            "decision_truth_status": "CANONICAL_CURRENT_SCAN",
            "decision_match_scope": "EXACT_SCAN_RUN",
            "action_badge": "Wait",
        },
    )
    out = decorate_card({"symbol": SYMBOL, "action_badge": "Buy", "methods": []})
    assert out["raw_action_badge"] == "Buy"
    assert out["canonical_decision"] == "WAIT"
    assert out["action_badge"] == "Wait"


def test_strategy_catalog_fails_closed_when_truth_store_is_unavailable(monkeypatch):
    import product.recommendation_truth as truth

    def boom(_card):
        raise RuntimeError("candidate store unavailable")

    monkeypatch.setattr(truth, "decorate_current_recommendation", boom)
    out = decorate_card({"symbol": SYMBOL, "action_badge": "Buy", "methods": []})
    assert out["raw_action_badge"] == "Buy"
    assert out["canonical_decision"] == "NO_JUDGMENT"
    assert out["decision_truth_status"] == "TRUTH_UNAVAILABLE"
    assert out["action_badge"] == "No judgment"
