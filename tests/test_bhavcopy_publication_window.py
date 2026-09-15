"""Publication lag must not paralyse the desk, and must not become a lie.

Product Acceptance passed at 17:36 IST and failed at 18:13 IST on the identical
SHA. `latest_required_session` moved to today the moment the clock crossed the
18:00 publication cutoff, so the gate demanded an archive NSE had not published
yet and every scan blocked with HISTORY_STALE.

The fix is a publication window with four distinct truths -- completed session,
minimum required session, publication pending, usable for scan -- and one
authority that both freshness functions derive from. The grace is bounded by a
wall clock on the next trading day, never by "one more session".

Calendar used throughout (no holidays unless stated):
    Fri 2026-09-11 · Mon 2026-09-14 · Tue 2026-09-15 · Wed 2026-09-16
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

import research.intelligence.data.nse_calendar as CAL
from data.bhavcopy_runtime import official_history_freshness

FRI = "2026-09-11"
MON = "2026-09-14"
TUE = "2026-09-15"
WED = "2026-09-16"
NO_HOLIDAYS: set = set()


def _history(latest: str, *, sessions: int = 500, ready: bool = True) -> dict:
    return {"ready": ready, "sessions": sessions, "latest_date": latest}


def _freshness(latest: str, now: datetime, *, sessions: int = 500, ready: bool = True) -> dict:
    return official_history_freshness(
        _history(latest, sessions=sessions, ready=ready),
        now=now, holidays=NO_HOLIDAYS, load_cache=False,
    )


# ── A: pre-cutoff, previous session in hand ────────────────────────────────
def test_A_session_day_1759_with_previous_session_is_usable():
    f = _freshness(MON, datetime(2026, 9, 15, 17, 59))
    assert f["usable_for_scan"] is True
    assert f["reason_code"] == "HISTORY_PUBLICATION_PENDING"
    assert f["completed_session"] == TUE          # Tue closed at 15:30
    assert f["minimum_required_official_session"] == MON
    assert f["current"] is False                  # we do not hold Tue


# ── B: post-cutoff, today's archive absent ─────────────────────────────────
def test_B_1801_without_todays_archive_is_pending_usable_and_not_current():
    f = _freshness(MON, datetime(2026, 9, 15, 18, 1))
    assert f["current"] is False, "never call history CURRENT when Tue is absent"
    assert f["usable_for_scan"] is True
    assert f["publication_pending"] is True
    assert f["available_session"] == MON
    assert f["completed_session"] == TUE
    assert f["reason_code"] == "HISTORY_PUBLICATION_PENDING"


def test_B_the_1800_cutoff_no_longer_flips_usability():
    before = _freshness(MON, datetime(2026, 9, 15, 17, 59))
    after = _freshness(MON, datetime(2026, 9, 15, 18, 1))
    assert before["usable_for_scan"] == after["usable_for_scan"] is True
    assert before["current"] == after["current"] is False


# ── C: the archive lands ───────────────────────────────────────────────────
def test_C_archive_appearing_makes_history_current_immediately():
    f = _freshness(TUE, datetime(2026, 9, 15, 19, 30))
    assert f["current"] is True
    assert f["usable_for_scan"] is True
    assert f["publication_pending"] is False
    assert f["reason_code"] == "HISTORY_CURRENT"
    assert f["stale_sessions"] == 0


# ── D: grace persists overnight, up to the next session's deadline ─────────
@pytest.mark.parametrize("when", [
    datetime(2026, 9, 15, 23, 59),      # same evening
    datetime(2026, 9, 16, 6, 0),        # next morning, pre-deadline
    datetime(2026, 9, 16, 8, 59),       # one minute before the deadline
])
def test_D_grace_holds_until_the_next_session_deadline(when):
    f = _freshness(MON, when)
    assert f["usable_for_scan"] is True
    assert f["publication_pending"] is True
    assert f["current"] is False
    assert f["in_publication_grace"] is True


def test_D_deadline_is_the_next_trading_sessions_preopen():
    f = _freshness(MON, datetime(2026, 9, 15, 18, 30))
    assert f["publication_deadline"].startswith("2026-09-16T09:00")


# ── E: at the deadline, yesterday's archive is mandatory ───────────────────
@pytest.mark.parametrize("when", [
    datetime(2026, 9, 16, 9, 0),        # exactly the deadline
    datetime(2026, 9, 16, 9, 1),
    datetime(2026, 9, 16, 14, 0),
])
def test_E_missing_previous_session_blocks_from_the_deadline(when):
    f = _freshness(MON, when)
    assert f["usable_for_scan"] is False, "Tue's archive is mandatory once Wed opens"
    assert f["reason_code"] == "HISTORY_STALE"
    assert f["minimum_required_official_session"] == TUE
    assert f["in_publication_grace"] is False


def test_E_grace_cannot_span_a_whole_trading_session():
    # Tue's file still absent late on Wed -- a full session later. Never usable.
    f = _freshness(MON, datetime(2026, 9, 16, 18, 30))
    assert f["usable_for_scan"] is False
    assert f["reason_code"] == "HISTORY_STALE"


# ── F: more than one genuinely missing session ─────────────────────────────
def test_F_two_missing_completed_sessions_block_even_inside_grace():
    # Fri in hand, Tue evening: Mon is mandatory and absent.
    f = _freshness(FRI, datetime(2026, 9, 15, 18, 30))
    assert f["usable_for_scan"] is False
    assert f["reason_code"] == "HISTORY_STALE"
    assert f["stale_sessions"] == 2


# ── G: weekends and holidays ───────────────────────────────────────────────
def test_G_weekend_holds_fridays_session_as_current():
    f = _freshness("2026-09-18", datetime(2026, 9, 19, 11, 0))      # Fri data, Sat
    assert f["current"] is True
    assert f["usable_for_scan"] is True
    assert f["completed_session"] == "2026-09-18"


def test_G_sunday_does_not_invent_a_session():
    f = _freshness("2026-09-18", datetime(2026, 9, 20, 20, 0))      # Sunday night
    assert f["current"] is True
    assert f["completed_session"] == "2026-09-18"


def test_G_holiday_walks_back_to_the_previous_real_session():
    f = official_history_freshness(
        _history(MON), now=datetime(2026, 9, 15, 18, 30),
        holidays={TUE}, load_cache=False,
    )
    # Tue is a holiday, so Mon is the completed session and we hold it.
    assert f["completed_session"] == MON
    assert f["current"] is True
    assert f["usable_for_scan"] is True


def test_G_deadline_skips_a_holiday_to_the_next_real_session():
    window = CAL.publication_window(datetime(2026, 9, 15, 18, 30), {WED})
    assert window["completed_session"] == date(2026, 9, 15)
    # Wed is a holiday, so Tue's archive is mandatory from Thursday's pre-open.
    assert window["publication_deadline"].date() == date(2026, 9, 17)


# ── H: the acceptance verdict must not turn on the clock alone ─────────────
def test_H_readiness_does_not_flip_merely_because_1800_passed():
    import data.bhavcopy_runtime as BR
    from product.startup_check import _history_readiness

    verdicts, details = {}, {}
    saved = BR.official_history_freshness
    try:
        for label, when in (("17:59", datetime(2026, 9, 15, 17, 59)),
                            ("18:01", datetime(2026, 9, 15, 18, 1))):
            frozen = _freshness(MON, when)
            BR.official_history_freshness = lambda *a, **k: frozen
            verdicts[label], details[label] = _history_readiness()
    finally:
        BR.official_history_freshness = saved

    assert verdicts["17:59"] == verdicts["18:01"] == "READY"
    # Ready, and still honest about which archive is outstanding.
    assert "2026-09-14" in details["18:01"]
    assert "2026-09-15" in details["18:01"]


def test_H_data_lane_ready_status_is_accepted_as_evidence_ready():
    from product.startup_check import _EVIDENCE_READY_STATUSES
    assert "READY" in _EVIDENCE_READY_STATUSES


# ── I: the fail-closed cases stay fail-closed ──────────────────────────────
@pytest.mark.parametrize("kwargs,expected", [
    ({"ready": False}, "HISTORY_NOT_READY"),
    ({"sessions": 10}, "HISTORY_TOO_SHALLOW"),
])
def test_I_unready_or_shallow_history_still_blocks(kwargs, expected):
    f = _freshness(MON, datetime(2026, 9, 15, 18, 30), **kwargs)
    assert f["usable_for_scan"] is False
    assert f["current"] is False
    assert f["reason_code"] == expected


def test_I_future_dated_history_still_blocks():
    f = _freshness("2026-09-30", datetime(2026, 9, 15, 18, 30))
    assert f["usable_for_scan"] is False
    assert f["reason_code"] == "HISTORY_FUTURE_DATED"


def test_I_unparseable_history_date_still_blocks():
    f = _freshness("not-a-date", datetime(2026, 9, 15, 18, 30))
    assert f["usable_for_scan"] is False
    assert f["reason_code"] == "HISTORY_DATE_MISSING"


def test_I_market_ops_gates_scans_on_usable_not_current():
    import inspect

    import operations.market_ops as MO
    src = inspect.getsource(MO.MarketOperationsWorker._history_ready)
    body = src.split('"""', 2)[-1]
    assert 'freshness.get("usable_for_scan")' in body
    assert 'freshness.get("current")' not in body


# ── J: provenance exposes every truth separately ───────────────────────────
def test_J_provenance_fields_are_all_distinct_and_present():
    f = _freshness(MON, datetime(2026, 9, 15, 18, 30))
    assert f["available_session"] == MON
    assert f["completed_session"] == TUE
    assert f["minimum_required_official_session"] == MON
    assert f["expected_latest_completed_session"] == TUE
    assert f["publication_pending"] is True
    assert f["usable_for_scan"] is True
    assert f["current"] is False
    assert f["reason_code"] == "HISTORY_PUBLICATION_PENDING"
    assert f["stale_sessions"] == 1


# ── 7: one authority, no contradictions ────────────────────────────────────
@pytest.mark.parametrize("when", [
    datetime(2026, 9, 15, 11, 0),
    datetime(2026, 9, 15, 16, 0),
    datetime(2026, 9, 15, 17, 59),
    datetime(2026, 9, 15, 18, 1),
    datetime(2026, 9, 15, 23, 0),
    datetime(2026, 9, 16, 8, 59),
    datetime(2026, 9, 16, 9, 1),
    datetime(2026, 9, 16, 18, 0),
])
@pytest.mark.parametrize("latest", [FRI, MON, TUE])
def test_the_two_freshness_functions_never_disagree(latest, when):
    snapshot = CAL.snapshot_freshness(latest, now=when, holidays=NO_HOLIDAYS)
    official = _freshness(latest, when)
    assert snapshot["fresh"] == official["usable_for_scan"], (latest, when)
    assert snapshot["completed_session"] == official["completed_session"]
    assert (snapshot["minimum_required_official_session"]
            == official["minimum_required_official_session"])
    assert snapshot["publication_pending"] == official["publication_pending"]
