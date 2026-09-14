from __future__ import annotations

from research.autonomy.paper_cycle_truth import merge_cycle_truth


def test_management_only_safety_block_does_not_override_reco_no_trade():
    merged = merge_cycle_truth(
        {
            "eligibility": "BLOCKED_SAFETY",
            "entry_block_reason": "RECO_SELECTION_AUTHORITY",
            "positions_opened": [],
        },
        {
            "eligibility": "NO_ELIGIBLE_TRADE",
            "entries_allowed": True,
            "entry_block_reason": "",
            "positions_opened": [],
            "cycle_reasons": ["NOT_SURFACED"],
            "source": "recommendation_selection_authority",
        },
        entries_allowed=True,
        entry_block_reason="",
        session_phase="intraday",
    )

    assert merged["eligibility"] == "NO_ELIGIBLE_TRADE"
    assert merged["management_eligibility"] == "BLOCKED_SAFETY"
    assert merged["management_entry_block_reason"] == "RECO_SELECTION_AUTHORITY"
    assert merged["entry_block_reason"] == ""
    assert merged["new_entries_allowed"] is True
    assert merged["reco_autopilot"]["eligibility"] == "NO_ELIGIBLE_TRADE"


def test_real_reco_safety_block_remains_blocked():
    merged = merge_cycle_truth(
        {"eligibility": "BLOCKED_SAFETY", "positions_opened": []},
        {
            "eligibility": "BLOCKED_SAFETY",
            "entries_allowed": False,
            "entry_block_reason": "ENTRY_WINDOW_CLOSED",
            "positions_opened": [],
        },
        entries_allowed=False,
        entry_block_reason="ENTRY_WINDOW_CLOSED",
        session_phase="premarket",
    )

    assert merged["eligibility"] == "BLOCKED_SAFETY"
    assert merged["new_entries_allowed"] is False
    assert merged["entry_block_reason"] == "ENTRY_WINDOW_CLOSED"


def test_traded_claim_without_reco_paperbook_fill_fails_closed():
    merged = merge_cycle_truth(
        {"eligibility": "BLOCKED_SAFETY", "positions_opened": []},
        {"eligibility": "TRADED", "entries_allowed": True, "positions_opened": []},
        entries_allowed=True,
        entry_block_reason="",
        session_phase="intraday",
    )

    assert merged["eligibility"] == "EXECUTION_INCONSISTENT"
    assert merged["execution_truth_error"] == "TRADED_WITHOUT_PERSISTED_POSITION"


def test_traded_requires_and_preserves_reco_paperbook_fill():
    opened = [("ensemble", "TCS")]
    merged = merge_cycle_truth(
        {"eligibility": "BLOCKED_SAFETY", "positions_opened": []},
        {"eligibility": "TRADED", "entries_allowed": True, "positions_opened": opened},
        entries_allowed=True,
        entry_block_reason="",
        session_phase="intraday",
    )

    assert merged["eligibility"] == "TRADED"
    assert merged["positions_opened"] == opened
