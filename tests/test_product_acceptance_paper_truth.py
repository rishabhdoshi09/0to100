"""FINAL acceptance must not confuse paper unavailability or intent with execution."""
from __future__ import annotations

import pytest

import scripts.run_product_acceptance as acceptance
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


def test_strict_acceptance_waits_for_terminal_autonomy_job():
    """A persisted last_cycle may appear just before jobs_recent becomes SUCCEEDED."""
    old_tokens = set(acceptance._core.PAPER_CYCLE_DONE)
    old_learning = acceptance._core.grade_learning_dashboard
    old_soak = acceptance._core.grade_forward_soak
    old_paper = acceptance._core.grade_paper_status
    old_cycle = acceptance._core.grade_paper_cycle_execution
    try:
        acceptance._core.PAPER_CYCLE_DONE = {"TRADED", "NO_ELIGIBLE_TRADE"}
        acceptance._install_strict_core_contract()
        assert acceptance._core.PAPER_CYCLE_DONE == set()
        assert acceptance._core.grade_paper_cycle_execution is acceptance.grade_paper_cycle_execution
    finally:
        acceptance._core.PAPER_CYCLE_DONE = old_tokens
        acceptance._core.grade_learning_dashboard = old_learning
        acceptance._core.grade_forward_soak = old_soak
        acceptance._core.grade_paper_status = old_paper
        acceptance._core.grade_paper_cycle_execution = old_cycle


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
