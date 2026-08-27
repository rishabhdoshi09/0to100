from datetime import date

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
