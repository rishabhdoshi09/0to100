"""Bounded shutdown and structural retry scheduling.

Audited defects:

  * ``build_store`` submitted every missing session to a thread pool and then
    drained it, so a shutdown signal was deferred until hundreds of downloads
    had finished. The worker printed "stopped" and kept working; both
    supervisors then blocked in ``wait`` with no timeout.
  * The autonomy supervisor requeued official-data blockers on every 15-second
    tick. That condition cannot change until the exchange publishes the next
    session, so it produced thousands of identical BLOCKED rows a day.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

import pytest

import data.bhavcopy_store as BS
from data.bhavcopy_store import HistoryAcquisitionCancelled
from research.autonomy import schedules as SCH


# --------------------------------------------------------------------------
# Cancellable history acquisition
# --------------------------------------------------------------------------

@pytest.fixture
def fake_sessions(monkeypatch, tmp_path):
    """A pool of 'missing' sessions whose downloads are instant and offline."""
    days = [datetime(2026, 1, 1).date() + timedelta(days=i) for i in range(60)]
    monkeypatch.setattr(BS, "_trading_days_back", lambda _n: days)
    monkeypatch.setattr(BS, "_day_path", lambda d: tmp_path / f"{d}.csv")
    return days


def test_acquisition_stops_promptly_when_asked(monkeypatch, fake_sessions):
    downloaded: list = []

    def fake_download(day, retries=2):
        downloaded.append(day)
        return True

    monkeypatch.setattr(BS, "_download_day", fake_download)

    with pytest.raises(HistoryAcquisitionCancelled):
        BS.build_store(days=60, should_stop=lambda: True)

    # It must not have worked through the whole backlog before noticing.
    assert len(downloaded) < len(fake_sessions)


def test_acquisition_reports_progress_before_cancelling(monkeypatch, fake_sessions):
    monkeypatch.setattr(BS, "_download_day", lambda day, retries=2: True)
    seen: list[tuple[int, int]] = []

    with pytest.raises(HistoryAcquisitionCancelled):
        BS.build_store(
            days=60,
            progress=lambda done, total: seen.append((done, total)),
            should_stop=lambda: True,
        )

    assert seen, "cancellation must not hide progress already made"
    assert seen[0][1] == len(fake_sessions)


def test_stop_event_is_a_valid_predicate(monkeypatch, fake_sessions):
    """The worker passes threading.Event.is_set directly."""
    monkeypatch.setattr(BS, "_download_day", lambda day, retries=2: True)
    stop = threading.Event()
    stop.set()

    with pytest.raises(HistoryAcquisitionCancelled):
        BS.build_store(days=60, should_stop=stop.is_set)


def test_no_stop_predicate_keeps_the_previous_behaviour(monkeypatch, fake_sessions):
    """Callers that never cancel must be unaffected."""
    monkeypatch.setattr(BS, "_download_day", lambda day, retries=2: True)
    monkeypatch.setattr(BS, "_MIN_DAYS", 10_000)  # bail out before parsing CSVs

    assert BS.build_store(days=60) == 0


def test_a_failing_stop_predicate_does_not_abort_acquisition(monkeypatch, fake_sessions):
    monkeypatch.setattr(BS, "_download_day", lambda day, retries=2: True)
    monkeypatch.setattr(BS, "_MIN_DAYS", 10_000)

    def broken():
        raise RuntimeError("predicate blew up")

    assert BS.build_store(days=60, should_stop=broken) == 0


# --------------------------------------------------------------------------
# Structural retry scheduling
# --------------------------------------------------------------------------

def test_structural_blocker_waits_for_the_publication_window():
    """Not 15 seconds: the next moment the data could actually exist."""
    morning = datetime(2026, 9, 11, 10, 0)  # Friday, before the EOD window
    wait = SCH.seconds_until_official_data_boundary(morning)
    assert wait > 3600


def test_inside_the_publication_window_polling_is_gentle_but_alive():
    inside = datetime(2026, 9, 11, 18, 30)
    wait = SCH.seconds_until_official_data_boundary(inside)
    assert 60 <= wait <= 900


def test_weekend_does_not_retry_until_a_session_day():
    saturday = datetime(2026, 9, 12, 12, 0)
    assert SCH.seconds_until_official_data_boundary(saturday) > 3600


def test_retry_delay_is_floored_and_capped():
    for moment in (
        datetime(2026, 9, 11, 0, 1),
        datetime(2026, 9, 11, 18, 6),
        datetime(2026, 9, 13, 23, 59),
    ):
        wait = SCH.seconds_until_official_data_boundary(moment)
        assert 60 <= wait <= 6 * 3600


def test_retry_delay_never_degenerates_to_the_tick_interval():
    """The regression: a 15s supervisor tick must never be the retry cadence."""
    for hour in range(0, 24):
        moment = datetime(2026, 9, 11, hour, 0)
        assert SCH.seconds_until_official_data_boundary(moment) >= 300
