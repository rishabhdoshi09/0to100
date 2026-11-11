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
                       ("product/pit_warehouse.db", 3600)):
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
    vitals = _by_key(collect(now=NOW, logs=_logs(tmp_path, buys=0, cycles=0), session_day=True))
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
