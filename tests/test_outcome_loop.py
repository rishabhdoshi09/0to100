"""Invariant 7 in the canonical stack: every published BUY is outcome-tracked.

Production audit 2026-11-11: signal_log's last row was 2026-07-30. Only the
legacy auto_scan loop called log_signal/update_outcomes, and the canonical
stack never starts it, so live edge / EV / drift learned from nothing new.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest


@pytest.fixture()
def tracker(tmp_path, monkeypatch):
    import core.signal_outcome_tracker as tk

    monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
    return tk


def _rows(tk):
    conn = tk._get_conn()
    try:
        return [tuple(r) for r in conn.execute(
            "SELECT symbol, signal_type, entry_price, stop_price, archetype FROM signal_log ORDER BY symbol")]
    finally:
        conn.close()


PAYLOAD = {"records": [
    {"symbol": "RPEL", "verdict": "BUY", "entry": 307.0, "stop": 290.0, "target": 360.0,
     "score": 81, "signals": ["SEPA", "PRE_BREAKOUT"]},
    {"symbol": "PNB", "verdict": "WATCH", "entry": 110.0, "stop": 104.0, "target": 120.0},
    {"symbol": "BAD", "verdict": "BUY", "entry": 100.0, "stop": 101.0, "target": 120.0},
    {"symbol": "nostop", "verdict": "BUY", "entry": 100.0, "stop": None},
]}


def test_published_buys_are_logged_on_a_session(tracker):
    from scan.market_scan_service import track_published_buys

    assert track_published_buys(PAYLOAD, today=date(2026, 11, 11)) == 1
    assert _rows(tracker) == [("RPEL", "UNIFIED_BUY", 307.0, 290.0, "SEPA|PRE_BREAKOUT")]


def test_two_schedulers_same_day_log_once(tracker):
    from scan.market_scan_service import track_published_buys

    track_published_buys(PAYLOAD, today=date(2026, 11, 11))
    track_published_buys(PAYLOAD, today=date(2026, 11, 11))
    assert len(_rows(tracker)) == 1


def test_holiday_rescan_logs_nothing(tracker):
    from scan.market_scan_service import track_published_buys

    # 2026-11-10 (Diwali Balipratipada) is in data/nse_holidays.json.
    assert track_published_buys(PAYLOAD, today=date(2026, 11, 10)) == 0
    assert track_published_buys(PAYLOAD, today=date(2026, 11, 14)) == 0   # Saturday
    assert _rows(tracker) == []


def test_tracking_failure_never_raises(monkeypatch):
    import core.signal_outcome_tracker as tk
    from scan.market_scan_service import track_published_buys

    monkeypatch.setattr(tk, "log_signal", lambda **_: (_ for _ in ()).throw(RuntimeError("disk")))
    assert track_published_buys(PAYLOAD, today=date(2026, 11, 11)) == 0


class _Deps:
    def now_ist(self): return datetime(2026, 7, 31, 18, 10)
    def holidays(self): return set()
    def active_snapshot_id(self): return None
    def official_history(self):
        return {"current": True, "available_session": "2026-07-31", "latest_date": "2026-07-31"}
    def resolve_outcomes(self, session_date, failures=()):
        return {"positions_closed": [], "outcomes_recorded": [], "as_of": session_date}


def test_outcome_job_settles_signal_ledger(monkeypatch):
    import core.signal_outcome_tracker as tk
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    calls = []
    monkeypatch.setattr(tk, "update_outcomes", lambda *a, **k: calls.append(1))
    result = JOBS.run_outcome_resolution(JOBS._Ctx(_Deps()))
    assert result.status == JS.SUCCEEDED and calls == [1]


def test_signal_ledger_failure_does_not_fail_the_outcome_job(monkeypatch):
    import core.signal_outcome_tracker as tk
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    def boom(*a, **k):
        raise RuntimeError("bhav store locked")

    monkeypatch.setattr(tk, "update_outcomes", boom)
    result = JOBS.run_outcome_resolution(JOBS._Ctx(_Deps()))
    assert result.status == JS.SUCCEEDED
    assert "bhav store locked" in result.metadata.get("signal_outcomes_error", "")
