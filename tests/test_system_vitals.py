"""System vitals: silent failures surface within minutes (2026-11-11 audit)."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime

from product.system_vitals import FAIL, OK, WARN, VitalsAlerter, collect, worst

NOW = datetime(2026, 11, 11, 12, 0)


def _logs(tmp_path, *, buys=1, cycles=1, scan_age_h=0.1, fno_age_h=0.2, ca_age_d=1):
    root = tmp_path / "logs"
    (root / "autonomy").mkdir(parents=True)
    (root / "market_ops").mkdir()
    (root / "product").mkdir()
    con = sqlite3.connect(root / "signal_outcomes.db")
    con.execute("CREATE TABLE signal_log (logged_at TEXT)")
    for _ in range(buys):
        con.execute("INSERT INTO signal_log VALUES ('2026-11-11T09:00:00')")
    con.commit(); con.close()
    con = sqlite3.connect(root / "autonomy" / "jobs.db")
    con.execute("CREATE TABLE jobs (job_type TEXT, status TEXT, created_at REAL)")
    for _ in range(cycles):
        con.execute("INSERT INTO jobs VALUES ('paper_cycle','SUCCEEDED',?)", (NOW.timestamp() - 600,))
    con.commit(); con.close()
    con = sqlite3.connect(root / "market_ops" / "jobs.db")
    con.execute("CREATE TABLE operations (kind TEXT, status TEXT, finished_at REAL)")
    con.execute("INSERT INTO operations VALUES ('FNO_REFRESH','SUCCEEDED',?)", (NOW.timestamp() - fno_age_h * 3600,))
    con.commit(); con.close()
    for rel, age_s in (("product/latest_momentum_scan.json", scan_age_h * 3600),
                       ("ca_events.json", ca_age_d * 86400), ("nse_industry_map.json", 3600),
                       ("product/pit_warehouse.db", 3600), ("signal_backtest.json", 3600)):
        path = root / rel
        path.write_text("{}")
        stamp = NOW.timestamp() - age_s
        os.utime(path, (stamp, stamp))
    (root / "indices").mkdir()
    os.utime(root / "indices", (NOW.timestamp() - 3600,) * 2)
    return root


def _by_key(vitals):
    return {v["key"]: v["status"] for v in vitals}


def test_healthy_session_is_all_ok(tmp_path):
    vitals = collect(now=NOW, logs=_logs(tmp_path), session_day=True)
    assert worst(vitals) == OK, vitals


def test_dead_outcome_loop_and_paper_bot_fail_loudly(tmp_path):
    root = _logs(tmp_path, buys=0, cycles=0)
    scan = root / "product" / "latest_momentum_scan.json"
    stamp = scan.stat().st_mtime
    scan.write_text(json.dumps({"records": [{"symbol": "X", "verdict": "BUY"}]}))  # published, not logged
    os.utime(scan, (stamp, stamp))
    vitals = _by_key(collect(now=NOW, logs=root, session_day=True))
    assert vitals["outcome_logging"] == FAIL and vitals["paper_cycles"] == FAIL


def test_frozen_scan_fno_and_reference_data_are_caught(tmp_path):
    vitals = _by_key(collect(now=NOW, logs=_logs(tmp_path, scan_age_h=3, fno_age_h=5, ca_age_d=54),
                             session_day=True))
    assert vitals["scan_fresh"] == FAIL and vitals["fno_refresh"] == FAIL
    assert vitals["corporate_actions"] == WARN


def test_holidays_do_not_raise_session_failures(tmp_path):
    vitals = _by_key(collect(now=NOW, logs=_logs(tmp_path, buys=0, cycles=0, scan_age_h=20),
                             session_day=False))
    assert vitals["outcome_logging"] == OK and vitals["paper_cycles"] == OK and vitals["scan_fresh"] == OK


def test_alerts_once_per_vital_per_day():
    sent = []
    alerter = VitalsAlerter(sent.append)
    bad = [{"key": "fno_refresh", "label": "F&O", "status": FAIL, "detail": "x"}]
    assert alerter.observe(bad, "2026-11-11") and len(sent) == 1
    assert alerter.observe(bad, "2026-11-11") == [] and len(sent) == 1
    assert alerter.observe(bad, "2026-11-12") and len(sent) == 2
    assert alerter.observe([{"key": "a", "label": "A", "status": OK, "detail": ""}], "2026-11-12") == []


def test_stale_signal_calibration_warns(tmp_path):
    root = _logs(tmp_path)
    stamp = NOW.timestamp() - 82 * 86400
    os.utime(root / "signal_backtest.json", (stamp, stamp))
    assert _by_key(collect(now=NOW, logs=root, session_day=True))["signal_calibration"] == WARN


def _keys(vitals):
    return {v["key"]: v for v in vitals}


def test_off_session_scan_older_than_last_close_is_stale_not_ok(tmp_path):
    # Sunday 2026-11-15 20:30: the newest scan is Thursday's (host slept through Friday).
    sunday = datetime(2026, 11, 15, 20, 30)
    root = _logs(tmp_path)
    scan = root / "product" / "latest_momentum_scan.json"
    thursday = datetime(2026, 11, 12, 18, 23).timestamp()
    os.utime(scan, (thursday, thursday))
    vital = _keys(collect(now=sunday, logs=root, session_day=False))["scan_fresh"]
    assert vital["status"] == WARN and "2026-11-13" in vital["detail"]


def test_off_session_scan_after_last_close_is_ok(tmp_path):
    sunday = datetime(2026, 11, 15, 20, 30)
    root = _logs(tmp_path)
    scan = root / "product" / "latest_momentum_scan.json"
    friday_eod = datetime(2026, 11, 13, 18, 20).timestamp()
    os.utime(scan, (friday_eod, friday_eod))
    assert _keys(collect(now=sunday, logs=root, session_day=False))["scan_fresh"]["status"] == OK


def test_scan_before_tonights_eod_rescan_is_not_flagged_early(tmp_path):
    # 18:00 on a session day: today's EOD rescan is still due.
    evening = datetime(2026, 11, 13, 18, 0)
    root = _logs(tmp_path)
    scan = root / "product" / "latest_momentum_scan.json"
    midday = datetime(2026, 11, 13, 15, 0).timestamp()
    os.utime(scan, (midday, midday))
    assert _keys(collect(now=evening, logs=root, session_day=True))["scan_fresh"]["status"] == OK


def test_official_history_vital_states(tmp_path):
    root = _logs(tmp_path)
    sunday = datetime(2026, 11, 15, 20, 30)
    friday_evening = datetime(2026, 11, 13, 18, 30)
    pending = {"reason_code": "HISTORY_PUBLICATION_PENDING", "available_session": "2026-11-12",
               "completed_session": "2026-11-13"}
    missed = _keys(collect(now=sunday, logs=root, session_day=False, history=pending))["official_history"]
    assert missed["status"] == WARN and "2026-11-13" in missed["detail"]
    publishing = _keys(collect(now=friday_evening, logs=root, session_day=True, history=pending))
    assert publishing["official_history"]["status"] == OK
    current = {"reason_code": "HISTORY_CURRENT", "available_session": "2026-11-13",
               "completed_session": "2026-11-13"}
    assert _keys(collect(now=sunday, logs=root, session_day=False, history=current))["official_history"]["status"] == OK
    stale = {"reason_code": "HISTORY_STALE", "available_session": "2026-11-05",
             "completed_session": "2026-11-13"}
    assert _keys(collect(now=sunday, logs=root, session_day=False, history=stale))["official_history"]["status"] == FAIL


def test_foreign_logs_root_without_history_skips_the_canonical_probe(tmp_path):
    assert "official_history" not in _keys(collect(now=NOW, logs=_logs(tmp_path), session_day=True))


def test_watch_runs_in_session_and_after_the_eod_window():
    from product.system_vitals import watch_due

    assert watch_due(datetime(2026, 11, 13, 10, 0))
    assert watch_due(datetime(2026, 11, 15, 20, 30))
    assert not watch_due(datetime(2026, 11, 13, 18, 30))
    assert not watch_due(datetime(2026, 11, 14, 3, 0))


def test_canonical_history_probe_accepts_the_tz_aware_desk_clock(monkeypatch):
    # Production passes now_ist() (tz-aware); nse_calendar compares naive IST
    # datetimes, so the probe raised TypeError and the vital read "unreadable".
    from zoneinfo import ZoneInfo

    import data.bhavcopy_runtime as runtime
    import product.system_vitals as vitals_mod

    seen = {}
    real = runtime.official_history_freshness

    def fake_freshness(history=None, *, now=None, **kwargs):
        seen["now"] = now
        return real(
            {"latest_date": "2026-11-12", "sessions": 500, "ready": True},
            now=now, holidays=set(), load_cache=False, require_store=False,
        )

    monkeypatch.setattr(runtime, "official_history_freshness", fake_freshness)
    monkeypatch.setattr(vitals_mod, "_logs", lambda: __import__("pathlib").Path("/nonexistent-qt-logs"))
    aware = datetime(2026, 11, 15, 21, 20, tzinfo=ZoneInfo("Asia/Kolkata"))
    vital = _keys(collect(now=aware, session_day=False))["official_history"]
    assert seen["now"].tzinfo is None and seen["now"].hour == 21
    assert vital["status"] == WARN and "2026-11-13" in vital["detail"], vital


def test_honest_zero_buy_day_is_not_a_dead_outcome_loop(tmp_path):
    root = _logs(tmp_path, buys=0)
    scan = root / "product" / "latest_momentum_scan.json"
    stamp = scan.stat().st_mtime
    scan.write_text(json.dumps({"records": [{"symbol": "X", "verdict": "WATCH"}]}))
    os.utime(scan, (stamp, stamp))
    vital = _keys(collect(now=NOW, logs=root, session_day=True))["outcome_logging"]
    assert vital["status"] == OK and "evidence gate" in vital["detail"]
    scan.write_text(json.dumps({"records": [{"symbol": "X", "verdict": "BUY"}]}))
    os.utime(scan, (stamp, stamp))
    vital = _keys(collect(now=NOW, logs=root, session_day=True))["outcome_logging"]
    assert vital["status"] == FAIL and "published 1 BUY" in vital["detail"]
