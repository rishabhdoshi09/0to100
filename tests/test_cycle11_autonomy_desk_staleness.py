"""Cycle 11: /api/autonomy-desk never presents a persisted loop summary as current.

On the owner's Mac (26 Nov) the endpoint served the product loop's summary from
2026-09-18: its ``readiness`` block said the Zerodha session was valid and execution was
ready (BROKER_LIVE_DATA_READY / EXECUTION_READY true) while the live broker check, in the
same payload, said LOGIN_REQUIRED. Nothing carried the summary's age. The request also
scanned the dormant 13.8 MB candidates.db in full (no session index): 4.9 s warm, 11.8 s cold.

Now the payload states the summary's age; a stale summary is marked so, its readiness
never claims a broker capability the live check denies, and the dormant candidate tables
are not scanned for it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from product import autonomous_loop as AL


def _summary(age: timedelta) -> dict:
    return {
        "generated_at": (datetime.now(timezone.utc) - age).isoformat(),
        "session_date": "2026-09-17",
        "trigger": "MARKET_SCAN",
        "readiness": {
            "schema_version": 1,
            "broker": {"state": "READY", "ready": True, "live_data_ready": True, "execution_ready": True,
                       "auth_status": "SESSION_VALID"},
            "capabilities": {"OFFICIAL_MARKET_DATA_READY": True, "BROKER_LIVE_DATA_READY": True,
                             "RESEARCH_DATA_READY": True, "OUTCOME_DATA_READY": True, "EXECUTION_READY": True},
        },
        "census": {"seen": 5},
    }


@pytest.fixture
def desk(monkeypatch):
    live = {"state": "LOGIN_REQUIRED", "ready": False, "live_data_ready": False, "execution_ready": False,
            "auth_status": "SESSION_EXPIRED"}
    monkeypatch.setattr(AL.RDY, "broker_status", lambda: dict(live))
    monkeypatch.setattr(AL, "project_events", lambda limit=24: [])
    monkeypatch.setattr(AL, "_what_it_learned", lambda summary=None: {})
    monkeypatch.setattr(AL, "replay_compatible_rows", lambda limit=8: [])
    monkeypatch.setattr(AL, "_operator_metrics", lambda session="": {})
    calls = []
    monkeypatch.setattr(AL.CL, "state_counts", lambda *a, **k: calls.append("state_counts") or {"WAIT": 1})
    monkeypatch.setattr(AL.CL, "list_candidates", lambda *a, **k: calls.append("list_candidates") or [])

    def run(summary):
        monkeypatch.setattr(AL, "load_summary", lambda: summary)
        return AL.desk_projection(), calls
    return run


def test_a_months_old_summary_is_marked_stale_and_claims_no_broker_readiness(desk):
    out, calls = desk(_summary(timedelta(days=69)))
    loop = out["loop_summary"]
    assert loop["stale"] is True and loop["generated_at"].startswith("20") and loop["age_hours"] > 24 * 60
    rd = out["readiness"]
    assert rd["stale"] is True and rd["as_of"] == loop["generated_at"]
    assert rd["broker"]["state"] == "LOGIN_REQUIRED"                       # the live check, not the persisted one
    assert rd["capabilities"]["BROKER_LIVE_DATA_READY"] is False
    assert rd["capabilities"]["EXECUTION_READY"] is False
    assert out["what_needs_the_operator"]["state"] == "LOGIN_REQUIRED"


def test_a_stale_summary_does_not_scan_the_dormant_candidate_tables(desk):
    out, calls = desk(_summary(timedelta(days=69)))
    assert calls == []
    assert out["candidate_lifecycle"] == {"withheld": "LOOP_SUMMARY_STALE"}
    assert out["lineage"]["candidates"] == [] and out["lineage"]["withheld"] == "LOOP_SUMMARY_STALE"


def test_a_summary_without_a_time_is_stale(desk):
    summary = _summary(timedelta(0))
    del summary["generated_at"]
    out, _ = desk(summary)
    assert out["loop_summary"]["stale"] is True and out["loop_summary"]["generated_at"] is None


def test_a_fresh_summary_is_served_as_before_with_the_live_broker(desk):
    out, calls = desk(_summary(timedelta(hours=2)))
    assert out["loop_summary"]["stale"] is False and "stale" not in out["readiness"]
    assert out["readiness"]["broker"]["state"] == "LOGIN_REQUIRED"
    assert out["readiness"]["capabilities"]["BROKER_LIVE_DATA_READY"] is False
    assert out["readiness"]["capabilities"]["OFFICIAL_MARKET_DATA_READY"] is True
    assert sorted(calls) == ["list_candidates", "state_counts"]
    assert out["candidate_lifecycle"] == {"WAIT": 1}
