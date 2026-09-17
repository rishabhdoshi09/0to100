"""Production-truth regressions: readiness, paper, replay, learning, launcher."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from product.data_readiness import project_official_data_readiness
from product.home_os import PREPARING, build_home_os
from research.autonomy import jobs as JOBS
from research.autonomy import supervisor_state as ST


IST_OPEN = datetime(2026, 9, 17, 10, 45)
IST_CLOSED = datetime(2026, 9, 17, 22, 10)


def test_history_current_makes_data_ready_even_when_store_unloaded():
    projected = project_official_data_readiness(
        freshness={
            "current": True,
            "reason_code": "HISTORY_CURRENT",
            "available_session": "2026-09-17",
            "expected_latest_completed_session": "2026-09-17",
            "stale_sessions": 0,
        },
        data={"ready": False},
        bhav={"ready": False, "latest_date": "2026-09-17"},
        operations_running=False,
    )
    assert projected["history_current"] is True
    assert projected["data_ready"] is True
    assert projected["data_ready"] == projected["history_current"]
    assert projected["lane_status"] == "Ready"
    assert projected["store_loaded"] is False
    assert projected["operations_running"] is False


def test_stale_history_is_waiting_not_ready():
    projected = project_official_data_readiness(
        freshness={
            "current": False,
            "reason_code": "HISTORY_STALE",
            "available_session": "2026-09-15",
            "expected_latest_completed_session": "2026-09-17",
            "stale_sessions": 2,
        },
        data={"ready": True},
        bhav={"ready": True, "latest_date": "2026-09-15"},
    )
    assert projected["data_ready"] is False
    assert projected["history_current"] is False
    assert projected["lane_status"] == "Waiting"


def test_home_os_does_not_wait_when_official_history_is_current():
    os = build_home_os(
        dashboard={
            "autonomy": {"state": "DATA_READY", "running": True},
            "data": {
                "ready": False,
                "history_current": True,
                "bhavcopy": {
                    "ready": False,
                    "latest_date": "2026-09-17",
                    "current": True,
                    "reason_code": "HISTORY_CURRENT",
                    "expected_latest_completed_session": "2026-09-17",
                    "available_session": "2026-09-17",
                    "stale_sessions": 0,
                },
            },
        },
        paper={"enabled": True, "open_positions": [], "closed_trades": []},
        why={"available": False},
        soak={"real_forward_observations": 0, "insufficient_evidence": True},
        scan={"scanned_at": "2026-09-17T05:00:00+00:00", "records": [{"symbol": "TCS"}]},
        reco={"schema_version": 4, "categories": []},
        now=IST_OPEN,
    )
    assert os["history_freshness"]["reason_code"] == "HISTORY_CURRENT"
    assert os["today"]["data_fresh"] is True
    assert os["system"]["data"]["status"] == "Ready"
    assert os["system"]["data"]["status_code"] == "READY"
    assert "getting the latest market data" not in os["subtext"].lower()
    assert os["state"] != PREPARING


def test_disk_session_date_can_be_current_without_in_memory_store():
    from data.bhavcopy_runtime import official_history_freshness

    freshness = official_history_freshness(
        {
            "ready": False,
            "sessions": 0,
            "latest_date": "",
            "csv_latest_date": "2026-09-15",
            "csv_files": 400,
        },
        now=datetime(2026, 9, 15, 19, 30),
        holidays=set(),
        load_cache=False,
        require_store=False,
    )
    assert freshness["reason_code"] == "HISTORY_CURRENT"
    assert freshness["current"] is True
    assert freshness["store_loaded"] is False


def test_scan_gate_still_requires_loaded_store():
    from data.bhavcopy_runtime import official_history_freshness

    freshness = official_history_freshness(
        {
            "ready": False,
            "sessions": 0,
            "latest_date": "",
            "csv_latest_date": "2026-09-15",
            "csv_files": 400,
        },
        now=datetime(2026, 9, 15, 19, 30),
        holidays=set(),
        load_cache=False,
        require_store=True,
    )
    assert freshness["reason_code"] == "HISTORY_NOT_READY"
    assert freshness["usable_for_scan"] is False


def test_api_data_ready_matches_history_current_when_pickle_unloaded(monkeypatch):
    import terminal_api

    monkeypatch.setattr(terminal_api, "_warm_bhavcopy_cache", lambda: None)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        lambda *, load_cache=False: {
            "ready": False,
            "cache_exists": True,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_latest_date": "2026-09-17",
            "csv_files": 400,
            "minimum_sessions": 60,
        },
    )
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda history, load_cache=False, **_kwargs: {
            "current": True,
            "reason_code": "HISTORY_CURRENT",
            "available_session": "2026-09-17",
            "expected_latest_completed_session": "2026-09-17",
            "stale_sessions": 0,
            "store_loaded": False,
        },
    )
    monkeypatch.setattr(terminal_api, "_snapshot_payload", lambda: {"ready": False})
    payload = terminal_api._data_payload(
        {"available": True, "records": [1]},
        {"available": False, "records": []},
        {"running": False},
        {"available": True},
        {"available": True},
    )
    assert payload["history_current"] is True
    assert payload["ready"] is True
    assert payload["ready"] == payload["history_current"]
    assert payload["store_loaded"] is False
    assert payload["operations_running"] is False
    assert payload["lane_status"] == "Ready"


class _PaperDeps:
    def __init__(self, *, now, snapshot=None, official=False):
        self._now = now
        self.snapshot = snapshot
        self.official = official
        self.seen = {}

    def now_ist(self):
        return self._now

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return self.snapshot

    def official_history(self):
        return {"current": self.official, "source": "official_nse"}

    def live_market_ready(self):
        return {}

    def run_paper_cycle(self, entries_allowed, reason="", phase="", failures=()):
        self.seen = {
            "allowed": bool(entries_allowed),
            "reason": str(reason or ""),
            "phase": str(phase or ""),
        }
        return {"eligibility": "NO_ELIGIBLE_TRADE"}


def test_missing_market_data_is_data_unavailable_not_no_trade():
    deps = _PaperDeps(now=IST_OPEN, snapshot=None, official=False)
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))
    assert result.status == JOBS.JS.SUCCEEDED
    assert result.metadata["eligibility"] == "DATA_UNAVAILABLE"
    assert result.metadata["failure_class"] == "DATA_OR_PROVIDER"
    assert deps.seen["reason"] == "NO_DATA_SNAPSHOT"
    assert "NO_ELIGIBLE_TRADE" not in result.summary


def test_closed_session_without_data_is_still_data_unavailable():
    deps = _PaperDeps(now=IST_CLOSED, snapshot=None, official=False)
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))
    assert result.metadata["eligibility"] == "DATA_UNAVAILABLE"
    assert deps.seen["reason"] == "NO_DATA_SNAPSHOT"
    assert deps.seen["reason"] != "ENTRY_WINDOW_CLOSED"


def test_auth_health_hint_is_observing_not_data_refreshing():
    class _Auth:
        valid = True
        status = "SESSION_VALID"
        error_code = ""
        reason = "ok"

        def as_dict(self):
            return {"status": self.status}

    class _Deps:
        def auth_health(self):
            return _Auth()

    result = JOBS.run_auth_health(JOBS._Ctx(_Deps()))
    assert result.state_hint == ST.OBSERVING
    assert result.state_hint != ST.DATA_REFRESHING


def test_canonical_launchers_source_setsid_compat_and_wait_for_http():
    root = Path(__file__).resolve().parents[1]
    shim = (root / "scripts" / "_setsid_compat.sh").read_text(encoding="utf-8")
    complete = (root / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    stack = (root / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "_setsid_compat.sh" in complete
    assert "_setsid_compat.sh" in stack
    assert "os.setsid()" in shim
    assert "wait_for_frontend" in stack
    assert "url_ok" in stack
    assert "Desk UI is answering" in stack
    assert "answered HTTP" in complete


def test_setsid_shim_is_a_function_when_setsid_is_missing(tmp_path):
    import os
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[1]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "python3", "python"):
        found = shutil.which(name)
        if found:
            target = bindir / name
            if not target.exists():
                target.symlink_to(found)
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"source {root / 'scripts' / '_setsid_compat.sh'}\n"
        "type setsid\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = str(bindir)
    result = subprocess.run(
        [str(bindir / "bash"), str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "setsid is a function" in combined
