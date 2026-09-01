"""Gross-only paper outcomes must never become production selection policy."""
from __future__ import annotations

from product.learning_policy_store import record_measured_outcome


def test_gross_only_paper_outcome_is_audit_only(tmp_path):
    path = tmp_path / "policies.json"
    row = None
    for _ in range(35):
        row = record_measured_outcome(
            policy_id="SETUP::VCP",
            dimension="setup",
            bucket="VCP",
            realized_R=0.8,
            source="paper_forward_taken_gross_only",
            path=path,
            floors={"experimental": 2, "eligible": 3, "active": 4},
        )
    assert row is not None
    assert row["sample_size"] == 35
    assert row["expectancy_R"] > 0
    assert row["affects_selection"] is False
    assert row["evidence_only_reason"] == "EXECUTION_ADJUSTED_UNAVAILABLE"
    assert row["live_locked"] is True
