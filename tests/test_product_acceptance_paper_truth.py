"""FINAL acceptance must not confuse paper unavailability or intent with execution."""
from __future__ import annotations

import pytest

from scripts.run_product_acceptance import grade_paper_cycle_execution


@pytest.mark.parametrize(
    "summary,eligibility",
    [
        ("paper cycle BLOCKED_BROKER", "BLOCKED_BROKER"),
        ("paper cycle BLOCKED_BROKER_AUTH", "BLOCKED_BROKER_AUTH"),
        ("paper cycle NO_DATA", "NO_DATA"),
        ("paper cycle BLOCKED_SAFETY", "BLOCKED_SAFETY"),
        ("paper cycle PAPER_TRADING_DISABLED", "PAPER_TRADING_DISABLED"),
    ],
)
def test_pseudo_success_states_fail_even_when_job_succeeded(summary: str, eligibility: str):
    graded = grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": summary},
        last_cycle={"cycle_id": "c1", "eligibility": eligibility},
        observed=True,
    )
    assert graded["status"] == "FAIL"


def test_off_session_entry_window_close_is_valid_completed_behavior():
    graded = grade_paper_cycle_execution(
        job={
            "status": "SUCCEEDED",
            "result_summary": "paper cycle: no-op · premarket · no open positions · ENTRY_WINDOW_CLOSED",
        },
        last_cycle={"cycle_id": "c1", "eligibility": "ENTRY_WINDOW_CLOSED"},
        observed=True,
    )
    assert graded["status"] == "PASS"


def test_no_eligible_trade_can_pass_only_after_observed_succeeded_cycle():
    assert grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "NO_ELIGIBLE_TRADE"},
        last_cycle={"cycle_id": "c1", "eligibility": "NO_ELIGIBLE_TRADE"},
        observed=True,
    )["status"] == "PASS"
    assert grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "NO_ELIGIBLE_TRADE"},
        last_cycle={"cycle_id": "c1", "eligibility": "NO_ELIGIBLE_TRADE"},
        observed=False,
    )["status"] != "PASS"


def test_traded_without_persisted_opened_position_is_rejected():
    graded = grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "TRADED"},
        last_cycle={"cycle_id": "c1", "eligibility": "TRADED", "positions_opened": []},
        observed=True,
    )
    assert graded["status"] == "FAIL"
    assert "positions_opened" in graded["blocker_reason"]


def test_traded_with_persisted_opened_position_can_pass():
    graded = grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "TRADED"},
        last_cycle={
            "cycle_id": "c1",
            "eligibility": "TRADED",
            "positions_opened": [{"symbol": "TCS", "position_id": "paper:TCS:c1"}],
        },
        observed=True,
    )
    assert graded["status"] == "PASS"


def test_succeeded_without_recognized_terminal_outcome_fails_closed():
    graded = grade_paper_cycle_execution(
        job={"status": "SUCCEEDED", "result_summary": "cycle completed"},
        last_cycle={"cycle_id": "c1", "eligibility": ""},
        observed=True,
    )
    assert graded["status"] == "FAIL"
