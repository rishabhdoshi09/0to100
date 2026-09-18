"""A recently executed scan does not make its inputs current.

QuantTerm previously showed "MARKET DATA READY · 2026-09-11" beside
"AUTONOMOUS SCAN WORKING · last scan 13/9/2026". Both numbers were real, but
the second was a job execution instant rendered as a data date, so the desk
implied it held Saturday prices for a market that last traded on Friday.

These tests pin the two facts apart at the persistence layer, which is the only
place that can stop every downstream surface from re-conflating them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from product.scan_store import (
    UNKNOWN_FRESHNESS,
    build_scan_payload,
    load_scan,
    save_scan,
    scan_provenance,
)

FRIDAY_SESSION = {
    "available_session": "2026-09-11",
    "expected_latest_completed_session": "2026-09-11",
    "reason_code": "HISTORY_CURRENT",
    "current": True,
    "stale_sessions": 0,
}


def test_execution_instant_is_never_reported_as_the_data_date():
    ran_on_sunday = datetime(2026, 9, 13, 18, 20, tzinfo=timezone.utc)
    prov = scan_provenance(completed_at=ran_on_sunday, freshness=FRIDAY_SESSION)

    assert prov["scan_completed_at"].startswith("2026-09-13")
    assert prov["market_session_date"] == "2026-09-11"
    assert prov["price_data_as_of"] == "2026-09-11"
    # the whole point: these must not be the same value
    assert prov["market_session_date"] not in prov["scan_completed_at"]


def test_stale_history_is_reported_as_stale_even_when_the_scan_just_ran():
    stale = {
        "available_session": "2026-09-04",
        "expected_latest_completed_session": "2026-09-11",
        "reason_code": "HISTORY_STALE",
        "current": False,
        "stale_sessions": 5,
    }
    prov = scan_provenance(completed_at=datetime.now(timezone.utc), freshness=stale)
    assert prov["data_current"] is False
    assert prov["data_freshness"] == "HISTORY_STALE"
    assert prov["sessions_behind"] == 5


def test_unavailable_session_is_declared_not_guessed():
    prov = scan_provenance(freshness={})
    assert prov["market_session_date"] == ""
    assert prov["provenance_available"] is False
    assert prov["data_current"] is False
    assert prov["provenance_reason"], "an unavailable session must carry a reason"
    assert prov["price_source"] == ""


def test_duration_and_scan_id_come_from_real_timing():
    start = datetime(2026, 9, 13, 18, 20, tzinfo=timezone.utc)
    end = start + timedelta(seconds=42.5)
    prov = scan_provenance(started_at=start, completed_at=end, freshness=FRIDAY_SESSION)
    assert prov["scan_duration_s"] == 42.5
    assert prov["scan_id"]
    # deterministic for the same inputs
    assert prov["scan_id"] == scan_provenance(
        started_at=start, completed_at=end, freshness=FRIDAY_SESSION
    )["scan_id"]


def test_build_scan_payload_carries_provenance():
    payload = build_scan_payload({"AAA": "Alpha"}, [], freshness=FRIDAY_SESSION,
                                 universe_failed=7)
    prov = payload["provenance"]
    assert payload["schema_version"] == 2
    assert prov["market_session_date"] == "2026-09-11"
    assert prov["universe_failed"] == 7


def test_pre_provenance_artifacts_still_load_but_declare_unknown_session(tmp_path):
    legacy = tmp_path / "legacy_scan.json"
    save_scan({"schema_version": 1, "scanned_at": "2026-09-13T18:20:00+00:00",
               "records": []}, legacy)
    loaded = load_scan(legacy)
    assert loaded is not None, "a v1 artifact is still real data"
    prov = loaded["provenance"]
    assert prov["market_session_date"] == ""
    assert prov["data_freshness"] == UNKNOWN_FRESHNESS
    assert prov["provenance_available"] is False


def test_incoherent_timing_reports_unknown_duration_not_a_fake_zero():
    """A start after the finish is bad input, not a zero-second scan."""
    finish = datetime(2026, 9, 13, 18, 20, tzinfo=timezone.utc)
    prov = scan_provenance(started_at=finish + timedelta(minutes=5),
                           completed_at=finish, freshness=FRIDAY_SESSION)
    assert prov["scan_duration_s"] is None
