from __future__ import annotations

from product import opportunity_memory as om


def test_decision_change_is_durable_even_when_lifecycle_state_is_unchanged(tmp_path):
    db = tmp_path / "opportunity_memory.db"
    om.remember(
        symbol="TCS",
        session_date="2026-09-09",
        state="WAIT",
        decision="WAIT",
        entry_state="WAIT_FOR_ENTRY",
        execution_state="NOT_APPLICABLE",
        reason="entry not ready",
        path=db,
    )
    om.remember(
        symbol="TCS",
        session_date="2026-09-09",
        state="WAIT",
        decision="BUY",
        entry_state="ENTER_NOW",
        execution_state="BLOCKED_WINDOW",
        reason="entry triggered after committee confirmation",
        path=db,
    )

    events = om.events_for("TCS", path=db)
    changes = [row for row in events if row["event"] == "DECISION_CHANGED"]

    assert len(changes) == 1
    change = changes[0]
    assert change["old_decision"] == "WAIT"
    assert change["new_decision"] == "BUY"
    assert change["old_entry_state"] == "WAIT_FOR_ENTRY"
    assert change["new_entry_state"] == "ENTER_NOW"
    assert change["old_execution_state"] == "NOT_APPLICABLE"
    assert change["new_execution_state"] == "BLOCKED_WINDOW"

    current = om.get("TCS", path=db)
    assert current is not None
    assert current["last_decision"] == "BUY"
    assert current["last_entry_state"] == "ENTER_NOW"


def test_same_decision_does_not_emit_duplicate_decision_change(tmp_path):
    db = tmp_path / "opportunity_memory.db"
    for reason in ("first", "updated evidence"):
        om.remember(
            symbol="INFY",
            session_date="2026-09-09",
            state="WAIT",
            decision="WAIT",
            entry_state="WAIT_FOR_ENTRY",
            execution_state="NOT_APPLICABLE",
            reason=reason,
            path=db,
        )

    changes = [row for row in om.events_for("INFY", path=db) if row["event"] == "DECISION_CHANGED"]
    assert changes == []
