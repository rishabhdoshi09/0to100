from datetime import date, timedelta

import pandas as pd

from data import index_store as idx
from logger import _QuietHealthAccess, quiet_uvicorn_health_access


def test_days_to_download_skips_holes_behind_a_current_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "_DIR", tmp_path)
    last = date(2026, 8, 27)
    candidates = [date(2026, 8, 25), date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)]
    got = idx._days_to_download(candidates, last_day=last, have_store=True)
    assert got == [date(2026, 8, 28)]


def test_days_to_download_fetches_holes_when_store_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, "_DIR", tmp_path)
    candidates = [date(2026, 8, 26), date(2026, 8, 27)]
    got = idx._days_to_download(candidates, last_day=date(2026, 8, 27), have_store=False)
    assert got == candidates


def test_candidate_weekdays_uses_calendar_horizon_not_weekday_multiplier():
    start = date(2026, 9, 19)
    got = idx._candidate_weekdays(400, today=start)
    floor = start - timedelta(days=int(400 * 1.55) - 1)

    assert got[0] == date(2026, 9, 18)
    assert got[-1] >= floor
    assert 400 < len(got) < 500
    assert len(got) < int(400 * 1.55)


def test_get_index_ohlcv_continues_a_shallow_current_bootstrap(monkeypatch):
    def frame(n):
        return pd.DataFrame({"Close": [float(i + 1) for i in range(n)]})

    monkeypatch.setattr(idx, "_store", {"Nifty 50": frame(199)})
    monkeypatch.setattr(idx, "_last_day", date(2026, 9, 18))

    calls = []

    def fake_build(days=400):
        calls.append(days)
        depth = 205 if len(calls) == 1 else idx._REGIME_BOOTSTRAP_SESSIONS
        idx._store["Nifty 50"] = frame(depth)
        return 1

    monkeypatch.setattr(idx, "build_index_store", fake_build)

    out = idx.get_index_ohlcv("^NSEI")

    assert calls == [idx._REGIME_BOOTSTRAP_SESSIONS, idx._REGIME_BOOTSTRAP_SESSIONS]
    assert out is not None
    assert len(out) == idx._REGIME_BOOTSTRAP_SESSIONS


def test_health_access_log_filter_drops_watchdog_pings():
    quiet_uvicorn_health_access()
    filt = _QuietHealthAccess()

    class _Rec:
        def __init__(self, msg):
            self.msg = msg

        def getMessage(self):
            return self.msg

    assert filt.filter(_Rec('127.0.0.1:49657 - "GET /health HTTP/1.1" 200 OK')) is False
    assert filt.filter(_Rec('127.0.0.1:1 - "GET /api/health HTTP/1.1" 200 OK')) is False
    assert filt.filter(_Rec('127.0.0.1:1 - "GET /api/dashboard HTTP/1.1" 200 OK')) is True
