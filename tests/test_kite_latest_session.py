"""Kite live quotes are the latest session during market hours.

A cash scan must not wait for options history or a full historical rewrite
when official bhavcopy + today's Kite OHLCV are already on the store.
"""
from datetime import datetime

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy import supervisor_state as ST


class _FailingSnapshot:
    blocker = "options_history_incomplete"
    active_pointer = None
    snapshot_id = None
    quality = {}

    def status(self, name):
        return "FAIL"


class _LiveDeps:
    def __init__(self, *, live=None, snap=None, scan=None):
        self._live = live if live is not None else {
            "ready": True, "symbols": 2633, "source": "kite_quotes",
            "session_date": "2026-08-24",
        }
        self._snap = snap
        self._scan = scan if scan is not None else {
            "summary": {"with_any_setup": 4, "momentum": 2, "near_breakout": 1},
        }

    def session_valid(self):
        return True

    def auth_health(self):
        from research.autonomy import auth as AUTH
        return AUTH.AuthHealth(AUTH.SESSION_VALID, "test")

    def activate(self):
        return _FailingSnapshot()

    def active_snapshot_id(self):
        return self._snap

    def live_market_ready(self):
        return dict(self._live)

    def run_scan(self):
        return self._scan

    def now_ist(self):
        return datetime(2026, 8, 24, 10, 15)

    def holidays(self):
        return set()


def test_data_refresh_uses_kite_live_when_snapshot_blocked():
    result = JOBS.run_data_refresh(JOBS._Ctx(_LiveDeps()))
    assert result.status == JS.SUCCEEDED
    assert result.state_hint == ST.DATA_READY
    assert H.SNAPSHOT_STALE in result.clears
    assert H.OPTIONS_HISTORY_INCOMPLETE in result.clears
    assert result.metadata["source"] == "kite_quotes"
    assert result.metadata["symbols"] == 2633


def test_data_refresh_uses_kite_live_when_activate_raises():
    class Boom(_LiveDeps):
        def activate(self):
            raise OSError(24, "Too many open files")

    result = JOBS.run_data_refresh(JOBS._Ctx(Boom()))
    assert result.status == JS.SUCCEEDED
    assert result.metadata["source"] == "kite_quotes"


def test_data_refresh_still_blocks_without_snapshot_or_live():
    result = JOBS.run_data_refresh(JOBS._Ctx(_LiveDeps(live={"ready": False})))
    assert result.status == JS.BLOCKED
    assert H.SNAPSHOT_STALE in result.failures
    assert result.state_hint == ST.DATA_BLOCKED


def test_eod_refresh_accepts_today_kite_session():
    class Report:
        active_pointer = "snap-old"
        snapshot_id = "snap-old"
        blocker = ""
        quality = {"date_range": ("2026-08-21", "2026-08-22")}

        def status(self, name):
            return "PASS"

    class Deps(_LiveDeps):
        def activate(self):
            return Report()

        def active_snapshot_info(self):
            return {"latest_date": "2026-08-22"}

    ctx = JOBS._Ctx(Deps())
    ctx.required_session_date = "2026-08-24"
    result = JOBS.run_data_refresh(ctx)
    assert result.status == JS.SUCCEEDED
    assert result.metadata["session_date"] == "2026-08-24"


def test_market_scan_runs_on_kite_live_without_snapshot():
    result = JOBS.run_market_scan(JOBS._Ctx(_LiveDeps(snap=None)))
    assert result.status == JS.SUCCEEDED
    assert "scan complete" in result.summary
    assert result.metadata["momentum"] == 2


def test_market_scan_still_blocks_without_snapshot_or_live():
    class NoOfficial(_LiveDeps):
        def official_history(self):
            return {"current": False, "available_session": "", "latest_date": ""}

    result = JOBS.run_market_scan(JOBS._Ctx(NoOfficial(snap=None, live={"ready": False})))
    assert result.status == JS.BLOCKED
    assert result.blocked_on in {JOBS.DEP_OFFICIAL, "OFFICIAL_MARKET_DATA_READY", JOBS.DEP_DATA}
    assert H.SNAPSHOT_STALE in result.failures


def test_live_session_ready_reports_kite_overlay(monkeypatch):
    from data import nse_live as NL

    monkeypatch.setattr(NL, "_is_trading_now", lambda: True)
    monkeypatch.setattr(NL, "apply_live_to_store", lambda: 2633)
    monkeypatch.setattr("data.bhavcopy_store.is_ready", lambda: True)
    monkeypatch.setattr("data.bhavcopy_store._store_sessions", 200, raising=False)
    import data.bhavcopy_store as bs
    monkeypatch.setattr(bs, "_store_sessions", 200)
    monkeypatch.setattr(bs, "_store_last_day", None)
    monkeypatch.setattr(bs, "_store", {"AAA": object()})
    monkeypatch.setattr(bs, "is_ready", lambda: True)
    info = NL.live_session_ready(apply=True)
    assert info["ready"] is True
    assert info["symbols"] == 2633
    assert info["source"] == "kite_quotes"


def test_operation_store_reuses_thread_connection(tmp_path):
    from operations.store import OperationStore
    store = OperationStore(tmp_path / "ops.db")
    first, _ = store.enqueue("MARKET_SCAN", lane="market_scan")
    leased = store.lease_next("market_scan", worker_pid=7)
    assert leased is not None
    assert leased["operation_id"] == first["operation_id"]
    cached = store._local.con
    store.progress(leased["operation_id"], stage="SCANNING", message="kite")
    assert store._local.con is cached
    again = store.get(leased["operation_id"])
    assert again["stage"] == "SCANNING"


def test_parse_request_token_accepts_full_redirect_url():
    from data.kite_client import parse_request_token

    assert parse_request_token("abc123") == "abc123"
    assert parse_request_token(
        "http://127.0.0.1/?request_token=abc123&action=login&status=success"
    ) == "abc123"
    assert parse_request_token("") == ""
