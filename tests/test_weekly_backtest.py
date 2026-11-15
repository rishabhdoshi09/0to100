"""Weekly walk-forward calibration (frozen since 2026-08-21 on the owner's Mac)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

from research.autonomy.supervisor import Supervisor

SAT = datetime(2026, 11, 14, 10, 0)


def _patch(monkeypatch, tmp_path, *, symbols=900, age_days=10, running=False):
    import data.bhavcopy_store as store
    import scan.signal_backtest as SB

    out = tmp_path / "signal_backtest.json"
    out.write_text("{}")
    stamp = time.time() - age_days * 86400
    os.utime(out, (stamp, stamp))
    started = []
    monkeypatch.setattr(SB, "_OUT", out)
    monkeypatch.setattr(SB, "get_state", lambda: {"running": running})
    monkeypatch.setattr(SB, "run_in_background", lambda **k: started.append(1))
    monkeypatch.setattr(store, "store_symbols", lambda: [f"S{i}" for i in range(symbols)])
    return started


def _stub():
    return SimpleNamespace(BACKTEST_MAX_AGE_S=Supervisor.BACKTEST_MAX_AGE_S)


def test_saturday_with_stale_calibration_runs_once(monkeypatch, tmp_path):
    started = _patch(monkeypatch, tmp_path)
    assert Supervisor._maybe_weekly_backtest(_stub(), SAT) is True and started == [1]


def test_weekdays_fresh_results_and_running_jobs_do_not_run(monkeypatch, tmp_path):
    started = _patch(monkeypatch, tmp_path)
    assert Supervisor._maybe_weekly_backtest(_stub(), datetime(2026, 11, 13, 10)) is False   # Friday
    assert Supervisor._maybe_weekly_backtest(_stub(), datetime(2026, 11, 14, 7)) is False    # too early
    started2 = _patch(monkeypatch, tmp_path, age_days=2)
    assert Supervisor._maybe_weekly_backtest(_stub(), SAT) is False
    _patch(monkeypatch, tmp_path, running=True)
    assert Supervisor._maybe_weekly_backtest(_stub(), SAT) is False
    assert started == [] and started2 == []


def test_unloaded_store_never_runs(monkeypatch, tmp_path):
    import data.bhavcopy_runtime as runtime

    started = _patch(monkeypatch, tmp_path, symbols=0)
    monkeypatch.setattr(runtime, "ensure_loaded", lambda **k: {})
    assert Supervisor._maybe_weekly_backtest(_stub(), SAT) is False and started == []


def test_thin_backtest_never_overwrites_calibration(monkeypatch, tmp_path):
    import data.bhavcopy_store as store
    import scan.signal_backtest as SB

    out = tmp_path / "signal_backtest.json"
    out.write_text(json.dumps({"signals": {"KEEP": {}}}))
    monkeypatch.setattr(SB, "_OUT", out)
    monkeypatch.setattr(store, "store_symbols", lambda: [])
    SB.run_backtest()
    assert json.loads(out.read_text()) == {"signals": {"KEEP": {}}}


def test_sunday_catches_up_a_saturday_the_host_slept_through(monkeypatch, tmp_path):
    # Owner's Mac slept 2026-11-12 20:05 -> 2026-11-15 20:22: Saturday never ticked.
    started = _patch(monkeypatch, tmp_path, age_days=86)
    assert Supervisor._maybe_weekly_backtest(_stub(), datetime(2026, 11, 15, 20, 30), set()) is True
    assert started == [1]


def test_exchange_holiday_counts_as_a_closed_day(monkeypatch, tmp_path):
    started = _patch(monkeypatch, tmp_path)
    holidays = {"2026-11-24"}
    assert Supervisor._maybe_weekly_backtest(_stub(), datetime(2026, 11, 24, 11), holidays) is True
    assert started == [1]


def test_weeknight_only_when_two_cadences_overdue(monkeypatch, tmp_path):
    tuesday_night = datetime(2026, 11, 17, 21, 0)
    started = _patch(monkeypatch, tmp_path, age_days=10)
    assert Supervisor._maybe_weekly_backtest(_stub(), tuesday_night, set()) is False
    started = _patch(monkeypatch, tmp_path, age_days=14)
    assert Supervisor._maybe_weekly_backtest(_stub(), tuesday_night, set()) is True
    assert started == [1]
    # Never during the session or the morning run-up, however overdue.
    for hour in (5, 9, 12, 15, 19):
        started = _patch(monkeypatch, tmp_path, age_days=90)
        assert Supervisor._maybe_weekly_backtest(_stub(), datetime(2026, 11, 17, hour), set()) is False
        assert started == []


def test_a_run_that_did_not_save_is_not_relaunched_every_tick(monkeypatch, tmp_path):
    started = _patch(monkeypatch, tmp_path, age_days=86)
    stub = _stub()
    sunday = datetime(2026, 11, 15, 20, 30)
    assert Supervisor._maybe_weekly_backtest(stub, sunday, set()) is True
    # The run ended without writing (file still 86 days old): the next ticks wait.
    assert Supervisor._maybe_weekly_backtest(stub, sunday, set()) is False
    assert started == [1]
    stub._backtest_launched_at -= Supervisor.BACKTEST_RETRY_S + 1
    assert Supervisor._maybe_weekly_backtest(stub, sunday, set()) is True
    assert started == [1, 1]
