"""A session whose EOD window the host slept through is recovered.

Owner's Mac 2026-11-12 20:05 -> 2026-11-15 20:22 IST: lid-closed on battery,
then a low-battery shutdown. Friday 2026-11-13's EOD window never ran, and the
closed-market tick only fetched official data inside the same-day window, so
Friday's archive, outcome resolution and reference upkeep all waited for
Monday while the desk showed Thursday's scan.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor

IST = ZoneInfo("Asia/Kolkata")
SUNDAY_EVENING = datetime(2026, 11, 15, 20, 30, tzinfo=IST)
NO_HOLIDAYS: set = set()


def _stub(tmp_path, *, clock=1000.0):
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: clock)
    stub = SimpleNamespace(jobs=store)
    stub._enqueue_reference_upkeep = lambda day: Supervisor._enqueue_reference_upkeep(stub, day)
    return stub, store


def _history(monkeypatch, available: str):
    import product.readiness as readiness

    monkeypatch.setattr(readiness, "official_history", lambda: {"available_session": available})


def test_missed_friday_is_fetched_on_sunday(tmp_path, monkeypatch):
    _history(monkeypatch, "2026-11-12")
    stub, store = _stub(tmp_path)
    key = Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING, NO_HOLIDAYS)
    assert key == "bhavcopy_update:2026-11-13:catchup:2026-11-15T20"
    rows = {j.idempotency_key: j.job_type for j in store.list()}
    assert rows == {
        key: SCH.BHAVCOPY_UPDATE,
        "corporate_actions:2026-11-13": SCH.CORPORATE_ACTIONS,
        "index_warmup:2026-11-13": SCH.INDEX_WARMUP,
    }


def test_same_hour_is_idempotent_and_next_hour_retries(tmp_path, monkeypatch):
    _history(monkeypatch, "2026-11-12")
    stub, store = _stub(tmp_path)
    Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING, NO_HOLIDAYS)
    Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING.replace(minute=59), NO_HOLIDAYS)
    assert len([j for j in store.list() if j.job_type == SCH.BHAVCOPY_UPDATE]) == 1
    Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING.replace(hour=21), NO_HOLIDAYS)
    assert len([j for j in store.list() if j.job_type == SCH.BHAVCOPY_UPDATE]) == 2
    # Reference upkeep stays once per session.
    assert len([j for j in store.list() if j.job_type == SCH.INDEX_WARMUP]) == 1


def test_present_archive_needs_no_recovery(tmp_path, monkeypatch):
    _history(monkeypatch, "2026-11-13")
    stub, store = _stub(tmp_path)
    assert Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING, NO_HOLIDAYS) is None
    assert store.list() == []


@pytest.mark.parametrize("now", [
    datetime(2026, 11, 13, 16, 0, tzinfo=IST),   # after close, before the EOD window
    datetime(2026, 11, 13, 19, 0, tzinfo=IST),   # inside the EOD window (normal path owns it)
])
def test_todays_archive_is_left_to_the_eod_window(tmp_path, monkeypatch, now):
    _history(monkeypatch, "2026-11-12")
    stub, store = _stub(tmp_path)
    assert Supervisor._enqueue_missed_eod_catchup(stub, now, NO_HOLIDAYS) is None
    assert store.list() == []


def test_late_publication_after_the_window_is_recovered_the_same_night(tmp_path, monkeypatch):
    _history(monkeypatch, "2026-11-12")
    stub, _store = _stub(tmp_path)
    key = Supervisor._enqueue_missed_eod_catchup(
        stub, datetime(2026, 11, 13, 21, 10, tzinfo=IST), NO_HOLIDAYS
    )
    assert key == "bhavcopy_update:2026-11-13:catchup:2026-11-13T21"


def test_unreadable_history_never_enqueues(tmp_path, monkeypatch):
    import product.readiness as readiness

    def boom():
        raise RuntimeError("store locked")

    monkeypatch.setattr(readiness, "official_history", boom)
    stub, store = _stub(tmp_path)
    assert Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING, NO_HOLIDAYS) is None
    assert store.list() == []


def test_catchup_rows_survive_legacy_retirement(tmp_path, monkeypatch):
    _history(monkeypatch, "2026-11-12")
    stub, store = _stub(tmp_path)
    key = Supervisor._enqueue_missed_eod_catchup(stub, SUNDAY_EVENING, NO_HOLIDAYS)
    store.enqueue(SCH.BHAVCOPY_UPDATE, idempotency_key="bhavcopy_update:2026-11-01")  # legacy row
    stub.deps = SimpleNamespace(
        active_snapshot_id=lambda: "",
        now_ist=lambda: SUNDAY_EVENING,
        holidays=lambda: NO_HOLIDAYS,
    )
    Supervisor._retire_legacy_recurring_work(stub)
    status = {j.idempotency_key: j.status for j in store.list() if j.job_type == SCH.BHAVCOPY_UPDATE}
    assert status[key] == JS.PENDING
    assert status["bhavcopy_update:2026-11-01"] == JS.CANCELLED
