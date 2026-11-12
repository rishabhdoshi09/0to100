"""Reference data that silently froze on the owner's Mac (found 2026-11-12).

1. The supervisor's per-tick legacy cleanup cancelled the EOD rescan
   (``eod:scan:<snap>``) and the nightly ``index_warmup:<day>`` seconds after
   they were enqueued, so the desk had no verified scan overnight and the Kite
   index cache stopped on 2026-10-01.
2. Readers loaded that cache once per process and never rebuilt a deep-but-old
   cache, so Home showed the 1 Oct VIX/Nifty as current for six weeks.
3. The regime engine classified the market from those bars and labelled every
   logged BUY with it.
4. A CA refresh that completed after its job returned left
   ``corporate_actions_incomplete`` up until the next EOD.
"""
from __future__ import annotations

import os
import pickle
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor

_AFTER_EOD = datetime(2026, 11, 11, 18, 21)


class _Deps:
    def now_ist(self):
        return _AFTER_EOD

    def holidays(self):
        return {"2026-11-10"}

    def session_valid(self):
        return True

    def active_snapshot_id(self):
        return "snap-eod"

    def news_health(self):
        return {"running": True, "error": ""}


def _status(sup, key):
    for job in sup.jobs.list(limit=200):
        if job.idempotency_key == key:
            return job.status
    raise AssertionError(f"no job {key}")


# --- 1. per-tick cleanup keeps the current scheduler's own rows -------------

def test_tick_cleanup_keeps_current_eod_rescan_and_index_upkeep(tmp_path):
    sup = Supervisor(tmp_path / "auto", deps=_Deps())
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="eod:scan:snap-eod", input_snapshot_id="snap-eod")
    sup.jobs.enqueue(SCH.INDEX_WARMUP, idempotency_key="index_warmup:2026-11-11")

    sup._retire_legacy_recurring_work()
    sup._retire_legacy_recurring_work()  # every tick, not just once

    assert _status(sup, "eod:scan:snap-eod") == JS.PENDING
    assert _status(sup, "index_warmup:2026-11-11") == JS.PENDING


def test_tick_cleanup_still_retires_superseded_and_legacy_rows(tmp_path):
    sup = Supervisor(tmp_path / "auto", deps=_Deps())
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="eod:scan:snap-old", input_snapshot_id="snap-old")
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="market_scan:snap-eod:0930")
    sup.jobs.enqueue(SCH.INDEX_WARMUP, idempotency_key="index_warmup:2026-10-04")
    sup.jobs.enqueue(SCH.LONG_TERM_REFRESH, idempotency_key="long_term_refresh:2026-11-11")

    sup._retire_legacy_recurring_work()

    for key in (
        "eod:scan:snap-old",
        "market_scan:snap-eod:0930",
        "index_warmup:2026-10-04",
        "long_term_refresh:2026-11-11",
    ):
        assert _status(sup, key) == JS.CANCELLED, key


def test_reference_upkeep_survives_the_tick_that_follows_it(tmp_path):
    sup = Supervisor(tmp_path / "auto", deps=_Deps())
    sup._enqueue_reference_upkeep("2026-11-11")
    sup._retire_legacy_recurring_work()
    assert _status(sup, "corporate_actions:2026-11-11") == JS.PENDING
    assert _status(sup, "index_warmup:2026-11-11") == JS.PENDING


# --- 2. index cache readers reload and refresh ------------------------------

def _frame(last: str, n: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0 + i for i in range(n)]},
        index=pd.bdate_range(end=last, periods=n),
    )


def _write_cache(path, last: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump({"store": {"Nifty 50": _frame(last)}, "last_day": date.fromisoformat(last)}, fh)


def _isolate_index_store(monkeypatch, tmp_path, required="2026-11-11"):
    from data import index_store as idx

    clock = [1000.0]
    monkeypatch.setattr(idx.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(idx, "_KITE_PKL", tmp_path / "index_store_kite.pkl")
    monkeypatch.setattr(idx, "_kite_store", {})
    monkeypatch.setattr(idx, "_kite_last_day", None)
    monkeypatch.setattr(idx, "_kite_loaded_mtime", 0.0)
    monkeypatch.setattr(idx, "_kite_pkl_checked_at", 0.0)
    monkeypatch.setattr(idx, "_kite_stale_check_after", 0.0)
    monkeypatch.setattr(idx, "_required_index_session", lambda: date.fromisoformat(required))
    return idx, clock


def test_reader_reloads_cache_rewritten_by_another_process(monkeypatch, tmp_path):
    idx, clock = _isolate_index_store(monkeypatch, tmp_path)
    monkeypatch.setattr(idx, "_build_kite_index_store", lambda *_a, **_k: 0)
    _write_cache(idx._KITE_PKL, "2026-10-01")
    os.utime(idx._KITE_PKL, (1_000_000, 1_000_000))
    assert idx.kite_index_as_of() == date(2026, 10, 1)

    _write_cache(idx._KITE_PKL, "2026-11-11")  # the nightly warm-up in autonomy
    os.utime(idx._KITE_PKL, (2_000_000, 2_000_000))
    assert idx.kite_index_as_of() == date(2026, 10, 1)  # stat is throttled
    clock[0] += idx._KITE_PKL_CHECK_S + 1
    assert idx.kite_index_as_of() == date(2026, 11, 11)
    assert str(idx._kite_index_frame("^NSEI", build=False).index[-1].date()) == "2026-11-11"


def test_deep_but_old_cache_triggers_one_throttled_rebuild(monkeypatch, tmp_path):
    idx, clock = _isolate_index_store(monkeypatch, tmp_path)
    _write_cache(idx._KITE_PKL, "2026-10-01")
    builds = []
    monkeypatch.setattr(idx, "_build_kite_index_store", lambda *a, **k: builds.append(a) or 0)

    for _ in range(50):  # the scanner reads Nifty once per symbol
        assert idx._kite_index_frame("^NSEI") is not None
    assert len(builds) == 1

    clock[0] += idx._KITE_STALE_RETRY_S + 1
    idx._kite_index_frame("^NSEI")
    assert len(builds) == 2


def test_current_cache_is_never_rebuilt_by_readers(monkeypatch, tmp_path):
    idx, _clock = _isolate_index_store(monkeypatch, tmp_path)
    _write_cache(idx._KITE_PKL, "2026-11-11")
    monkeypatch.setattr(
        idx, "_build_kite_index_store", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("rebuilt"))
    )
    for _ in range(5):
        assert idx._kite_index_frame("^NSEI") is not None


# --- 3. the regime is withheld on weeks-old bars ----------------------------

def test_regime_stale_reason_flags_six_week_old_nifty():
    from core.regime_engine import _index_history_stale_reason

    reason = _index_history_stale_reason(
        _frame("2026-10-01"), now=datetime(2026, 11, 12, 10, 0), holidays={"2026-11-10"}
    )
    assert "2026-10-01" in reason and "stale" in reason.lower()


def test_regime_stale_reason_tolerates_evening_publication_lag():
    from core.regime_engine import _index_history_stale_reason

    hol = {"2026-11-10"}
    # 19:00 on the 11th: the 11th is required, the cache still ends on the 9th.
    assert _index_history_stale_reason(_frame("2026-11-09"), now=datetime(2026, 11, 11, 19, 0), holidays=hol) == ""
    assert _index_history_stale_reason(_frame("2026-11-11"), now=datetime(2026, 11, 12, 10, 0), holidays=hol) == ""


def test_compute_regime_withholds_market_read_from_stale_index(monkeypatch):
    import core.regime_engine as re

    re._CACHE.clear()
    monkeypatch.setattr(re, "_fetch_ohlcv", lambda *_a, **_k: _frame("2026-10-01"))
    monkeypatch.setattr(
        "research.intelligence.data.nse_calendar._now_ist", lambda: datetime(2026, 11, 12, 10, 0)
    )
    state = re.compute_regime()
    re._CACHE.clear()
    assert state.data_available is False
    assert state.market_regime == "UNAVAILABLE"
    assert state.vix == 0.0 and state.nifty_change_1d == 0.0
    assert "2026-10-01" in state.unavailable_reason


# --- 4. a completed CA refresh clears its own failure -----------------------

def test_completed_ca_refresh_clears_stale_failure(monkeypatch):
    from research.autonomy import parallel_runtime as PR

    saved = []
    sup = SimpleNamespace(failures={H.CA_INCOMPLETE}, _save_failures=lambda: saved.append(1))
    monkeypatch.setattr(PR, "_ca_future", None)
    monkeypatch.setattr(PR, "_ca_clear_last", 0.0)
    monkeypatch.setattr(PR, "_ca_status", lambda: {"coverage_complete": False})
    assert PR._clear_ca_failure_when_complete(sup, now=10.0) is False
    assert H.CA_INCOMPLETE in sup.failures

    monkeypatch.setattr(PR, "_ca_status", lambda: {"coverage_complete": True})
    assert PR._clear_ca_failure_when_complete(sup, now=20.0) is False  # throttled
    assert PR._clear_ca_failure_when_complete(sup, now=10.0 + PR._CA_CLEAR_INTERVAL_S) is True
    assert H.CA_INCOMPLETE not in sup.failures
    assert saved == [1]


def test_ca_failure_is_not_cleared_while_refresh_still_runs(monkeypatch):
    from research.autonomy import parallel_runtime as PR

    sup = SimpleNamespace(failures={H.CA_INCOMPLETE}, _save_failures=lambda: None)
    monkeypatch.setattr(PR, "_ca_future", SimpleNamespace(done=lambda: False))
    monkeypatch.setattr(PR, "_ca_clear_last", 0.0)
    monkeypatch.setattr(PR, "_ca_status", lambda: {"coverage_complete": True})
    assert PR._clear_ca_failure_when_complete(sup, now=500.0) is False
    assert H.CA_INCOMPLETE in sup.failures


def test_fresh_login_retries_a_rejected_index_refresh_without_waiting(monkeypatch, tmp_path):
    idx, clock = _isolate_index_store(monkeypatch, tmp_path)
    _write_cache(idx._KITE_PKL, "2026-10-01")
    identity = ["expired-token"]
    monkeypatch.setattr("data.kite_client.kite_session_identity", lambda: identity[0])
    monkeypatch.setattr(idx, "_kite_login_checked_at", 0.0)
    builds = []

    def build(*_a, **_k):
        builds.append(identity[0])
        idx._kite_failed_session = identity[0]   # rejected, as before the login
        return 0

    monkeypatch.setattr(idx, "_kite_failed_session", None)
    monkeypatch.setattr(idx, "_build_kite_index_store", build)
    idx._kite_index_frame("^NSEI")
    assert builds == ["expired-token"]

    clock[0] += idx._KITE_LOGIN_RECHECK_S + 1
    idx._kite_index_frame("^NSEI")               # same token: still throttled
    assert builds == ["expired-token"]

    identity[0] = "fresh-login-token"
    idx._kite_index_frame("^NSEI")               # recheck window not reached
    assert builds == ["expired-token"]
    clock[0] += idx._KITE_LOGIN_RECHECK_S + 1
    idx._kite_index_frame("^NSEI")
    assert builds == ["expired-token", "fresh-login-token"]


def test_regime_calibration_ignores_withheld_regime(monkeypatch):
    from scan import live_edge

    monkeypatch.setattr(
        live_edge, "_closed_rows",
        lambda: (_ for _ in ()).throw(AssertionError("must not learn from a withheld regime")),
    )
    assert live_edge.regime_calibration("UNAVAILABLE") == {}
    assert live_edge.regime_calibration("unknown") == {}


# --- 5. CA coverage of the open window does not expire at midnight ----------

def _ca_coverage(tmp_path, covered_to: str, *, success: bool = True):
    import json as _json
    from data import corporate_actions_resilient as CAR

    windows = CAR.build_windows(today=date(2026, 11, 12), years=1)
    records = {
        w.window_id: {
            "window_id": w.window_id, "covered_from": w.start.isoformat(),
            "covered_to": w.end.isoformat(), "success": True,
        }
        for w in windows
    }
    records["2026-H2"].update({"covered_to": covered_to, "success": success})
    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    events.write_text("[]")
    coverage.write_text(_json.dumps({"version": 1, "years": 1, "windows": records,
                                     "last_refresh_at": "2026-11-11T18:06:49+05:30"}))
    return CAR, events, coverage


def test_open_window_refreshed_through_last_session_is_complete(tmp_path, monkeypatch):
    CAR, events, coverage = _ca_coverage(tmp_path, "2026-11-11")
    monkeypatch.setattr(CAR, "_previous_session", lambda today: date(2026, 11, 11))
    status = CAR.coverage_status(years=1, today=date(2026, 11, 12), events_path=events, coverage_path=coverage)
    assert status["coverage_complete"] is True, status["missing_windows"]


def test_open_window_missing_the_last_session_is_still_incomplete(tmp_path, monkeypatch):
    CAR, events, coverage = _ca_coverage(tmp_path, "2026-11-06")
    monkeypatch.setattr(CAR, "_previous_session", lambda today: date(2026, 11, 11))
    status = CAR.coverage_status(years=1, today=date(2026, 11, 12), events_path=events, coverage_path=coverage)
    assert status["coverage_complete"] is False
    assert status["missing_windows"] == ["2026-H2"]

    CAR, events, coverage = _ca_coverage(tmp_path, "2026-11-11", success=False)
    monkeypatch.setattr(CAR, "_previous_session", lambda today: date(2026, 11, 11))
    status = CAR.coverage_status(years=1, today=date(2026, 11, 12), events_path=events, coverage_path=coverage)
    assert status["coverage_complete"] is False
