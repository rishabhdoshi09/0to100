"""A component's status word and its detail must describe the same thing.

Found by operating the desk, not by a test. A running stack reported:

    autonomy | READY | DATA_BLOCKED

Status measured whether the supervisor process was alive; detail described
what the supervisor could actually do. Liveness is not capability — a
supervisor can run perfectly while unable to act — and the operator reads the
status word.

This is the same defect class already closed for the system-health lanes,
living on a second surface that never got the invariant. Fixing one instance
and assuming the class was closed is exactly the mistake this file exists to
stop repeating.
"""
from __future__ import annotations

import pytest

from product.runtime_lifecycle import (
    DEGRADED,
    READY,
    STARTING,
    _autonomy_component_status,
    _component,
)


@pytest.mark.parametrize("state,expected", [
    ("OBSERVING", READY),
    ("PAPER_ACTIVE", READY),
    ("RESEARCHING", READY),
    ("DATA_READY", READY),
    ("DATA_REFRESHING", READY),
    ("STARTING", STARTING),
    ("AUTH_REQUIRED", DEGRADED),
    ("DATA_BLOCKED", DEGRADED),
    ("DEGRADED", DEGRADED),
    ("HALTED", DEGRADED),
])
def test_autonomy_status_comes_from_the_supervisor_state(state, expected):
    assert _autonomy_component_status({"state": state}, alive=True) == expected


def test_a_live_supervisor_that_cannot_act_is_not_ready():
    """The exact observed defect."""
    assert _autonomy_component_status({"state": "DATA_BLOCKED"}, alive=True) != READY


def test_a_dead_supervisor_is_starting_whatever_its_last_state_said():
    assert _autonomy_component_status({"state": "OBSERVING"}, alive=False) == STARTING


def test_an_unknown_state_is_not_assumed_healthy():
    assert _autonomy_component_status({"state": "SOMETHING_NEW"}, alive=True) == DEGRADED


@pytest.mark.parametrize("detail", [
    "Market data is not ready — new paper trades are paused.",
    "Running with reduced capability — see details.",
    "Zerodha login is unavailable; non-broker autonomy can continue.",
    "Stopped. No new activity.",
])
def test_a_degraded_detail_can_never_carry_a_ready_status(detail):
    row = _component("autonomy", READY, detail=detail)
    assert row["status"] == DEGRADED
    assert row["status_demoted_from"] == READY
    assert "detail" in row["status_demoted_because"]


@pytest.mark.parametrize("detail", [
    "QuantTerm is observing the market normally.",
    "Terminal API is serving",
    "heartbeat 0s ago",
    "Research-report API on :8766",
    "",
])
def test_a_clean_detail_keeps_its_ready_status(detail):
    row = _component("api", READY, detail=detail)
    assert row["status"] == READY
    assert "status_demoted_from" not in row


def test_the_invariant_does_not_touch_statuses_that_are_already_honest():
    row = _component("official_history", "FAILED", detail="HISTORY_NOT_READY")
    assert row["status"] == "FAILED"
    assert "status_demoted_from" not in row


def test_no_component_in_a_live_payload_contradicts_itself():
    """Audit every component, not only the one that was observed wrong."""
    from product.runtime_lifecycle import inspect_runtime
    from product.system_health_contract import detail_contradicts_healthy

    payload = inspect_runtime(api_serving=False)
    offenders = [
        (c["name"], c["detail"]) for c in payload.get("components", [])
        if c["status"] == READY and detail_contradicts_healthy(c.get("detail", ""))
    ]
    assert not offenders, f"components claiming READY over a degraded detail: {offenders}"
