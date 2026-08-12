"""Smoke tests for fund cycle frozen protocols (network-free)."""
from __future__ import annotations

from research.fund_cycle import data as D


def test_frozen_protocols_complete():
    f = D.load_frozen()
    assert f["frozen_before_outcome_inspection"] is True
    assert f["foundation_package_id"] == "46ff79f58ee21c9e"
    assert f["parent_ohlcv_snapshot_id"] == "2f683be0c73eaa33"
    assert set(f["experiments"]) == {
        "EXP-FUND-01", "EXP-FUND-02", "EXP-FUND-03", "EXP-FUND-04",
    }
    assert f["no_ml"] is True
    assert f["production_authority"] is False
    assert f["experiments"]["EXP-FUND-01"]["holding_horizon_sessions"] == 21
    assert f["experiments"]["EXP-FUND-04"]["outlier_rule"].startswith("exclude pe > 200")


def test_next_action_mapping():
    assert D.next_action_for("CONFIRMED") == "ELIGIBLE_FOR_FOLLOWUP_RESEARCH"
    assert D.next_action_for("FAIL") == "REJECT_CLOSE_BRANCH"
    assert D.next_action_for("INCONCLUSIVE") == "HOLD_NO_TUNING"
