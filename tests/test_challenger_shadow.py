"""Prospective shadow ledger: frozen before entry, exact horizon, paired decision."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from contextlib import closing

from product import challenger_shadow as CS
from product import learning_scorecard as LS

IST = timezone(timedelta(hours=5, minutes=30))
SESSION = "2026-11-18"                       # Wed; next session Thu 19 Nov
HOLIDAYS = set(json.loads((Path(__file__).resolve().parents[1] / "data" / "nse_holidays.json").read_text()))
# the ten NSE sessions after 18 Nov (24 Nov is a holiday)
WINDOW = ["2026-11-19", "2026-11-20", "2026-11-23", "2026-11-25", "2026-11-26",
          "2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03"]


def _sessions(start: str, end: str) -> list[pd.Timestamp]:
    from research.intelligence.data.nse_calendar import is_session

    return [d for d in pd.date_range(start, end) if is_session(d.date(), HOLIDAYS)]


def _frame(seed: int, end: str = SESSION, n: int = 320) -> pd.DataFrame:
    idx = [d for d in _sessions("2025-01-01", end)][-n:]
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6}, index=pd.DatetimeIndex(idx))


def _scan(symbols, *, session=SESSION, completed="2026-11-18T12:30:00+00:00", **over):
    records = [{"symbol": s, "score": 90 - i, "verdict": "WATCH"} for i, s in enumerate(symbols)]
    prov = {"scan_id": "scan-abc", "market_session_date": session, "expected_session_date": session,
            "data_current": True, "scan_completed_at": completed}
    prov.update(over.pop("provenance", {}))
    return {"schema_version": 2, "as_of_session": session, "records": records, "provenance": prov, **over}


def _clock(text: str):
    stamp = datetime.fromisoformat(text).replace(tzinfo=IST)
    return lambda: stamp


EVENING = _clock("2026-11-18T20:00:00")


def _rows(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return ([dict(r) for r in conn.execute("SELECT * FROM sessions")],
                [dict(r) for r in conn.execute("SELECT * FROM predictions ORDER BY champion_rank")])


@pytest.fixture
def frames():
    return {f"S{i}": _frame(i) for i in range(8)}


def test_freezes_actual_predictions_and_model_version_before_entry(tmp_path, frames, monkeypatch):
    monkeypatch.setenv("QT_BUILD_SHA", "abc123def")
    path = tmp_path / "shadow.db"
    out = CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    assert out["recorded"] == 8 and out["reason"] == "PROSPECTIVE"
    (meta,), preds = _rows(path)
    assert meta["status"] == CS.FROZEN and meta["scan_id"] == "scan-abc" and meta["candidates"] == 8
    assert meta["entry_deadline"].startswith("2026-11-19T09:15:00")
    assert datetime.fromisoformat(meta["frozen_at"]) < datetime.fromisoformat(meta["entry_deadline"])
    model = json.loads(meta["model_version"])
    assert model["code_sha"] == "abc123def" and model["hypothesis"] == CS.HYPOTHESIS
    assert model["challenger_spec_sha"] == CS.challenger_spec_sha() and model["horizon_sessions"] == 10
    universe = json.loads(meta["universe"])
    assert universe["candidates"] == 8 and len(universe["candidates_sha256"]) == 64
    assert [p["champion_rank"] for p in preds] == list(range(1, 9))          # the scan's own order
    assert [p["champion_score"] for p in preds] == [90 - i for i in range(8)]
    assert sorted(p["challenger_rank"] for p in preds) == list(range(1, 9))


def test_writes_are_idempotent(tmp_path, frames):
    path = tmp_path / "shadow.db"
    CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    again = CS.record_session(_scan(frames), frames.get, clock=_clock("2026-11-18T23:00:00"),
                              holidays=HOLIDAYS, path=path)
    assert again["reason"] == "already_recorded"
    metas, preds = _rows(path)
    assert len(metas) == 1 and len(preds) == 8


def test_a_zero_candidate_session_still_gets_a_durable_marker(tmp_path):
    path = tmp_path / "shadow.db"
    out = CS.record_session(_scan([]), {}.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    assert out["recorded"] == 0 and out["reason"] == "PROSPECTIVE"
    (meta,), preds = _rows(path)
    assert meta["status"] == CS.FROZEN and meta["candidates"] == 0 and meta["frozen"] == 0 and preds == []
    assert CS.record_session(_scan([]), {}.get, clock=EVENING, holidays=HOLIDAYS,
                             path=path)["reason"] == "already_recorded"


def test_after_the_first_eligible_entry_nothing_can_be_frozen(tmp_path, frames):
    path = tmp_path / "shadow.db"
    late = CS.record_session(_scan(frames), frames.get, clock=_clock("2026-11-19T09:15:00"),
                             holidays=HOLIDAYS, path=path)
    assert late["reason"] == "ENTRY_DEADLINE_PASSED" and late["retry"] is False
    skew = CS.record_session(_scan(frames), frames.get, clock=_clock("2026-11-19T08:50:00"),
                             holidays=HOLIDAYS, path=path)          # inside the clock-skew margin
    assert skew["reason"] == "ENTRY_DEADLINE_PASSED"
    ok = CS.record_session(_scan(frames), frames.get, clock=_clock("2026-11-19T08:40:00"),
                           holidays=HOLIDAYS, path=path)
    assert ok["reason"] == "PROSPECTIVE"
    path.unlink()
    stale = CS.record_session(_scan(frames), frames.get, clock=_clock("2026-11-20T21:00:00"),
                              holidays=HOLIDAYS, path=path)
    assert stale["reason"] == "SCAN_NOT_THE_LATEST_COMPLETED_SESSION" and stale["retry"] is False
    assert not path.exists() or _rows(path) == ([], [])


@pytest.mark.parametrize("scan,reason,retry", [
    (_scan(["A"], pit=True), "HISTORICAL_REPLAY_IS_NOT_PROSPECTIVE", False),
    (_scan(["A"], provenance={"data_current": False}), "SCAN_INPUT_STALE", True),
    (_scan(["A"], provenance={"expected_session_date": "2026-11-19"}), "SCAN_INPUT_STALE", True),
    (_scan(["A"], provenance={"scan_id": ""}), "SCAN_IDENTITY_MISSING", True),
    (_scan(["A"], completed="2026-11-18T08:00:00+00:00"), "SCAN_BEFORE_SESSION_CLOSE", True),
    (_scan(["A"], completed="2026-11-19T08:00:00+00:00"), "SCAN_TIME_INCOHERENT", True),
    (_scan(["A"], session="2026-11-10"), "SCAN_SESSION_NOT_AN_NSE_SESSION", False),
    (None, "NO_SCAN", True),
])
def test_only_a_fresh_identified_live_session_close_scan_is_prospective(tmp_path, scan, reason, retry):
    path = tmp_path / "shadow.db"
    out = CS.record_session(scan, {}.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    assert (out["reason"], out["retry"], out["recorded"]) == (reason, retry, 0)
    assert not path.exists()


def test_a_naive_clock_is_refused(tmp_path, frames):
    naive = lambda: datetime(2026, 11, 18, 20, 0)
    out = CS.record_session(_scan(frames), frames.get, clock=naive, holidays=HOLIDAYS, path=tmp_path / "s.db")
    assert out["reason"] == "CLOCK_NOT_TIMEZONE_AWARE" and not (tmp_path / "s.db").exists()


def test_a_scan_id_cannot_be_frozen_for_a_second_session(tmp_path):
    path = tmp_path / "shadow.db"
    CS.record_session(_scan([], session="2026-11-17", completed="2026-11-17T12:30:00+00:00"), {}.get,
                      clock=_clock("2026-11-17T20:00:00"), holidays=HOLIDAYS, path=path)
    again = CS.record_session(_scan([]), {}.get, clock=EVENING, holidays=HOLIDAYS, path=path)   # same scan_id
    assert again["reason"] == "SCAN_ID_ALREADY_FROZEN_FOR_ANOTHER_SESSION"


def test_an_unpublished_official_bar_waits_instead_of_freezing_an_empty_session(tmp_path, frames):
    path = tmp_path / "shadow.db"
    out = CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path,
                            session_published=False)
    assert out["reason"] == "OFFICIAL_SESSION_BAR_NOT_PUBLISHED" and out["retry"] is True
    assert not path.exists()


def test_an_uncovered_calendar_freezes_nothing(tmp_path, frames):
    out = CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays={"2026-11-10"},
                            path=tmp_path / "shadow.db")
    assert out["reason"].startswith("EXCHANGE_CALENDAR_NOT_COVERED") and out["recorded"] == 0


def test_freezing_needs_the_calendar_only_through_the_entry_deadline(tmp_path):
    late_dec = _scan([], session="2026-12-22", completed="2026-12-22T12:30:00+00:00")
    out = CS.record_session(late_dec, {}.get, clock=_clock("2026-12-22T20:00:00"), holidays=HOLIDAYS,
                            path=tmp_path / "shadow.db")
    assert out["reason"] == "PROSPECTIVE"                     # window reaches 2027; freeze is still valid
    last = _scan([], session="2026-12-31", completed="2026-12-31T12:30:00+00:00")
    out = CS.record_session(last, {}.get, clock=_clock("2026-12-31T20:00:00"), holidays=HOLIDAYS,
                            path=tmp_path / "shadow2.db")
    assert out["reason"].startswith("EXCHANGE_CALENDAR_NOT_COVERED")   # next session is in 2027


def test_a_crash_mid_write_leaves_no_partial_session(tmp_path, frames, monkeypatch):
    path = tmp_path / "shadow.db"
    real = CS.build_predictions

    def poisoned(*a, **k):
        rows, excl, n = real(*a, **k)
        rows[-1]["champion_verdict"] = object()            # sqlite cannot bind it: fails mid-insert
        return rows, excl, n

    monkeypatch.setattr(CS, "build_predictions", poisoned)
    with pytest.raises(sqlite3.Error):
        CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    assert _rows(path) == ([], [])                          # marker and predictions rolled back together
    monkeypatch.setattr(CS, "build_predictions", real)
    assert CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS,
                             path=path)["recorded"] == 8


def test_future_bars_do_not_change_what_was_frozen(tmp_path, frames):
    later = {k: pd.concat([v, v.iloc[-10:].set_axis(pd.DatetimeIndex(pd.to_datetime(WINDOW))) * 3])
             for k, v in frames.items()}
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=a)
    CS.record_session(_scan(later), later.get, clock=EVENING, holidays=HOLIDAYS, path=b)
    assert _rows(a)[1] == _rows(b)[1]
    assert _rows(a)[0][0]["payload_sha256"] == _rows(b)[0][0]["payload_sha256"]


def _full_frames(n=8, missing=None, seed0=0):
    out = {}
    for i in range(n):
        f = _frame(seed0 + i, end="2026-12-31")
        if missing and f"S{i}" in missing:
            f = f.drop(pd.Timestamp(missing[f"S{i}"]))
        out[f"S{i}"] = f
    return out


def test_outcomes_are_measured_over_exactly_the_nse_window():
    full = _full_frames()
    window = [date.fromisoformat(d) for d in WINDOW]
    rows = [{"symbol": s} for s in full]
    got = CS.session_outcomes(SESSION, rows, full.get, window)
    assert got["missing_bars"] == 0
    for r, ret in got["complete"]:
        f = full[r["symbol"]]
        assert ret == pytest.approx(f.loc["2026-12-03", "close"] / f.loc["2026-11-19", "open"] - 1)


def test_a_missing_symbol_bar_inside_the_window_is_detected_and_excluded():
    full = _full_frames(missing={"S2": "2026-11-26", "S5": "2026-12-03"})
    got = CS.session_outcomes(SESSION, [{"symbol": s} for s in full], full.get,
                              [date.fromisoformat(d) for d in WINDOW])
    assert got["missing_bars"] == 2
    assert {r["symbol"] for r, _ in got["complete"]} == {f"S{i}" for i in range(8)} - {"S2", "S5"}


def test_only_admissible_matured_sessions_are_evaluated(tmp_path):
    path = tmp_path / "shadow.db"
    cut = {k: v[v.index <= SESSION] for k, v in _full_frames().items()}
    CS.record_session(_scan(cut), cut.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    full = _full_frames()
    rep = CS.evaluate(full.get, maturity=lambda s: LS.IMMATURE, holidays=HOLIDAYS, path=path)
    assert rep["sessions_evaluated"] == 0 and rep["session_outcomes"] == {"WINDOW_IMMATURE": 1}
    rep = CS.evaluate(full.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["sessions_evaluated"] == 1 and rep["status"] == "SHADOW_ONLY"
    assert rep["decision"] == "INSUFFICIENT EVIDENCE"                      # one session decides nothing
    many_missing = _full_frames(missing={f"S{i}": "2026-11-25" for i in range(3)})
    rep = CS.evaluate(many_missing.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["session_outcomes"] == {"TOO_MANY_MISSING_SYMBOL_BARS": 1} and rep["missing_symbol_bars"] == 3


def test_backfilled_late_or_edited_rows_are_never_evidence(tmp_path):
    path = tmp_path / "shadow.db"
    cut = {k: v[v.index <= SESSION] for k, v in _full_frames().items()}
    CS.record_session(_scan(cut), cut.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    # (a) an EARLIER session written after it with plausible timestamps: a backfill
    earlier = "2026-11-17"
    cut17 = {k: v[v.index <= earlier] for k, v in cut.items()}
    rows, excl, n = CS.build_predictions(earlier, _scan(cut17, session=earlier)["records"], cut17.get)
    model = CS.model_version({}, 10)
    CS._write(path, {"session": earlier, "status": CS.FROZEN, "frozen_at": "2026-11-17T14:00:00+00:00",
                     "entry_deadline": "2026-11-18T09:15:00+05:30", "scan_id": "x", "model_version": model,
                     "candidates": n, "payload_sha256": CS._payload_sha(earlier, model, "x", rows)}, rows)
    # (b) a row frozen after its own deadline
    late = "2026-11-19"
    cut19 = {k: v[v.index <= late] for k, v in _full_frames().items()}
    rows19, _, n19 = CS.build_predictions(late, _scan(cut19, session=late)["records"], cut19.get)
    CS._write(path, {"session": late, "status": CS.FROZEN, "frozen_at": "2026-11-20T03:30:00+00:00",
                     "entry_deadline": "2026-11-20T09:15:00+05:30", "scan_id": "y", "model_version": model,
                     "candidates": n19, "payload_sha256": CS._payload_sha(late, model, "y", rows19)}, rows19)
    full = _full_frames()
    rep = CS.evaluate(full.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["session_outcomes"]["INSERTED_AFTER_A_LATER_SESSION"] == 1
    assert rep["session_outcomes"]["FROZEN_AFTER_ENTRY_DEADLINE"] == 1
    # (c) an edited prediction no longer matches its frozen hash
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE predictions SET champion_score = champion_score + 1 WHERE session=?", (SESSION,))
    rep = CS.evaluate(full.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["session_outcomes"].get("PAYLOAD_HASH_MISMATCH") == 1 and rep["sessions_evaluated"] == 0


def test_unfrozen_sessions_become_visible_missed_gaps(tmp_path):
    path = tmp_path / "shadow.db"
    CS.record_session(_scan([]), {}.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    missed = CS.mark_missed(now=datetime(2026, 11, 23, 21, 0, tzinfo=IST), holidays=HOLIDAYS, path=path)
    assert missed == ["2026-11-19", "2026-11-20"]          # 23 Nov is still prospective tonight
    metas, _ = _rows(path)
    assert {m["session"]: m["status"] for m in metas} == {SESSION: CS.FROZEN, "2026-11-19": CS.MISSED,
                                                         "2026-11-20": CS.MISSED}
    rep = CS.evaluate({}.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["sessions_missed"] == 2 and rep["session_outcomes"].get("MISSED") == 2


def _daily(n, edge, noise, seed):
    from research.intelligence.data.nse_calendar import next_session
    rng = np.random.default_rng(seed)
    out, day = {}, date(2025, 12, 31)
    for _ in range(n):
        day = next_session(day, HOLIDAYS)
        base = 0.05 + rng.normal(0, 0.05)
        ch = base + rng.normal(0, noise)
        out[day.isoformat()] = {"challenger": ch + edge, "champion": base + rng.normal(0, noise),
                                "paired_ic_improvement": 0.0}
        out[day.isoformat()]["paired_ic_improvement"] = (out[day.isoformat()]["challenger"]
                                                         - out[day.isoformat()]["champion"])
    return out


def test_two_positive_correlations_are_not_evidence_the_challenger_is_better():
    res = CS.shadow_inference(_daily(220, edge=0.0, noise=0.02, seed=1), holidays=HOLIDAYS, horizon=10)
    assert res["challenger"]["verdict"] == "POSITIVE_IC" and res["champion"]["verdict"] == "POSITIVE_IC"
    assert res["paired_ic_improvement"]["verdict"] == "NOT_DISTINGUISHABLE"


def test_a_consistent_paired_improvement_is_detected():
    res = CS.shadow_inference(_daily(220, edge=0.03, noise=0.02, seed=2), holidays=HOLIDAYS, horizon=10)
    assert res["paired_ic_improvement"]["verdict"] == "CHALLENGER_BETTER"
    assert res["paired_ic_improvement"]["independent_sessions"] >= 20
    assert res["paired_ic_improvement"]["holm_adjusted_p"] < 0.05


def test_a_consistently_worse_challenger_is_called_worse_not_better():
    res = CS.shadow_inference(_daily(220, edge=-0.03, noise=0.02, seed=4), holidays=HOLIDAYS, horizon=10)
    assert res["paired_ic_improvement"]["verdict"] == "CHALLENGER_WORSE"


def test_a_marginal_paired_win_must_survive_the_multiple_testing_correction(monkeypatch):
    from research import overlap_inference as OI
    real = OI.overlap_robust_test

    def marginal(cells, horizon, **kw):
        out = real(cells, horizon, **kw)
        if out["status"] == OI.OK:                # force t so one-sided p ~0.04: passes alone, not after Holm
            out["t"], out["df"] = 1.80, 21
            out["lower_95_one_sided"] = 1e-6
        return out

    monkeypatch.setattr(OI, "overlap_robust_test", marginal)
    res = CS.shadow_inference(_daily(220, edge=0.03, noise=0.02, seed=2), holidays=HOLIDAYS, horizon=10)
    p = res["paired_ic_improvement"]
    assert p["p_one_sided_better"] < 0.05 <= p["holm_adjusted_p"] and p["verdict"] == "NOT_DISTINGUISHABLE"


def test_thin_paired_evidence_or_an_uncovered_calendar_decides_nothing():
    res = CS.shadow_inference(_daily(150, edge=0.05, noise=0.01, seed=3), holidays=HOLIDAYS, horizon=10)
    assert res["paired_ic_improvement"]["verdict"] == "INSUFFICIENT EVIDENCE"
    res = CS.shadow_inference(_daily(220, edge=0.05, noise=0.01, seed=3), holidays={"2026-01-26"}, horizon=10)
    assert res["paired_ic_improvement"]["reason"] == "EXCHANGE_CALENDAR_NOT_COVERED"


def test_shadow_writes_nothing_but_its_own_ledger(tmp_path, frames):
    from core.runtime_paths import logs_dir
    before = {p for p in logs_dir().rglob("*") if p.is_file()} if logs_dir().exists() else set()
    path = tmp_path / "shadow.db"
    CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    CS.evaluate(frames.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    after = {p for p in logs_dir().rglob("*") if p.is_file()} if logs_dir().exists() else set()
    assert after == before


# ── TOCTOU, crash/restart, timing (2026-11-20 review) ────────────────────────

def _header(session, scan_id="race-scan", status=None, frozen_at="2026-11-18T14:30:00+00:00"):
    model = CS.model_version({}, 10)
    return {"session": session, "status": status or CS.FROZEN, "frozen_at": frozen_at,
            "entry_deadline": "2026-11-19T09:15:00+05:30", "scan_id": scan_id,
            "scan_completed_at": "2026-11-18T12:30:00+00:00", "model_version": model, "candidates": 0,
            "payload_sha256": CS._payload_sha(session, model, scan_id, [])}


def test_concurrent_threads_cannot_freeze_one_scan_id_twice(tmp_path):
    import threading
    path = tmp_path / "shadow.db"
    CS._write(path, _header("2026-11-01", scan_id="seed"), [])          # schema exists
    barrier, results = threading.Barrier(8), []

    def worker(i):
        barrier.wait()
        results.append(CS._write(path, _header(f"2026-11-{10 + i:02d}"), []))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1
    assert results.count("SCAN_ID_ALREADY_FROZEN_FOR_ANOTHER_SESSION") == 7
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE scan_id='race-scan'").fetchone()[0] == 1


def test_concurrent_processes_respect_the_database_boundary(tmp_path):
    import subprocess, sys
    path = tmp_path / "shadow.db"
    CS._write(path, _header("2026-11-01", scan_id="seed"), [])
    code = ("import sys, json; sys.path.insert(0, %r); from product import challenger_shadow as CS; "
            "from tests.test_challenger_shadow import _header; "
            "print(json.dumps(CS._write(__import__('pathlib').Path(%r), _header(sys.argv[1]), [])))")
    root = str(Path(__file__).resolve().parents[1])
    procs = [subprocess.Popen([sys.executable, "-c", code % (root, str(path)), f"2026-11-{12 + i:02d}"],
                              stdout=subprocess.PIPE, text=True, cwd=root) for i in range(4)]
    outs = [json.loads(p.communicate(timeout=60)[0].strip().splitlines()[-1]) for p in procs]
    assert outs.count(True) == 1 and outs.count("SCAN_ID_ALREADY_FROZEN_FOR_ANOTHER_SESSION") == 3


def test_same_session_writes_stay_idempotent_under_a_race(tmp_path):
    import threading
    path = tmp_path / "shadow.db"
    CS._write(path, _header("2026-11-01", scan_id="seed"), [])
    barrier, results = threading.Barrier(6), []

    def worker():
        barrier.wait()
        results.append(CS._write(path, _header("2026-11-18", scan_id="same"), []))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1 and results.count("already_recorded") == 5


def test_a_restart_after_a_crash_resumes_without_duplicates(tmp_path, frames, monkeypatch):
    path = tmp_path / "shadow.db"
    real = CS._connect
    calls = {"n": 0}

    def crash_once(p):
        conn = real(p)
        calls["n"] += 1
        if calls["n"] == 1:
            class Dying:
                def __getattr__(self, name):
                    return getattr(conn, name)

                def executemany(self, *a, **k):
                    raise sqlite3.OperationalError("disk I/O error (simulated power loss)")
            return Dying()
        return conn

    monkeypatch.setattr(CS, "_connect", crash_once)
    with pytest.raises(sqlite3.OperationalError):
        CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    monkeypatch.setattr(CS, "_connect", real)
    assert _rows(path) == ([], [])
    assert CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS, path=path)["recorded"] == 8
    assert CS.record_session(_scan(frames), frames.get, clock=EVENING, holidays=HOLIDAYS,
                             path=path)["reason"] == "already_recorded"


def test_the_margin_cannot_certify_a_scan_completed_after_the_cutoff(tmp_path):
    path = tmp_path / "shadow.db"
    h = _header("2026-11-18", scan_id="late-scan")
    h["scan_completed_at"] = "2026-11-19T03:50:00+00:00"           # 09:20 IST, after the open
    h["frozen_at"] = "2026-11-18T14:30:00+00:00"                   # a backdated write timestamp
    assert CS._write(path, h, []) is True
    rep = CS.evaluate({}.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    assert rep["session_outcomes"] == {"SCAN_COMPLETED_OUTSIDE_THE_FREEZE_WINDOW": 1}


def test_the_deadline_never_depends_on_the_holiday_calendar():
    friday = date(2026, 11, 20)
    assert CS.entry_deadline(friday).isoformat() == "2026-11-21T09:15:00+05:30"      # Saturday, not Monday
    assert CS.entry_deadline(date(2026, 11, 23), set()) == CS.entry_deadline(date(2026, 11, 23), HOLIDAYS)


# ── migration, hard crash, corruption, exceptional sessions (2026-11-20 review) ──

def test_a_pre_index_ledger_with_duplicate_scan_ids_is_migrated_not_broken(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(CS._SCHEMA)                                       # the old schema: no unique index
    for i, session in enumerate(("2026-11-16", "2026-11-17", "2026-11-18")):
        h = _header(session, scan_id="dup")
        conn.execute("INSERT INTO sessions(session,status,reason,hypothesis,schema_version,frozen_at,entry_deadline,"
                     "scan_id,scan_completed_at,model_version,candidates,frozen,exclusions,universe,payload_sha256) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (session, "FROZEN", "", CS.HYPOTHESIS, CS.SCHEMA_VERSION, h["frozen_at"], h["entry_deadline"],
                      "dup", h["scan_completed_at"], "{}", 0, 0, "{}", "{}", "x"))
    conn.commit()
    conn.close()
    with closing(CS._connect(path)) as c:                                 # migrates instead of raising
        rows = c.execute("SELECT session, status, reason FROM sessions ORDER BY rowid").fetchall()
        assert c.execute("SELECT name FROM sqlite_master WHERE name='one_session_per_scan_id'").fetchone()
    assert rows[0] == ("2026-11-16", "FROZEN", "")
    assert [r[1:] for r in rows[1:]] == [("MISSED", CS.QUARANTINED)] * 2


def test_a_writer_killed_mid_transaction_leaves_nothing_and_releases_the_lock(tmp_path):
    import signal, subprocess, sys, time as _time
    path = tmp_path / "shadow.db"
    CS._write(path, _header("2026-11-01", scan_id="seed"), [])
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys, time, sqlite3; sys.path.insert(0, %r); from product import challenger_shadow as CS; "
            "c = CS._connect(__import__('pathlib').Path(%r)); c.execute('BEGIN IMMEDIATE'); "
            "c.execute(\"INSERT INTO sessions(session,status,reason,hypothesis,schema_version,frozen_at,entry_deadline,"
            "scan_id,scan_completed_at,model_version,candidates,frozen,exclusions,universe,payload_sha256) VALUES"
            "('2026-11-18','FROZEN','','h',2,'t','d','killed','t','{}',0,0,'{}','{}','x')\"); "
            "print('inside', flush=True); time.sleep(60)") % (root, str(path))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=root)
    assert proc.stdout.readline().strip() == "inside"
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    assert CS._write(path, _header("2026-11-18", scan_id="after-crash"), []) is True     # lock released
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE scan_id='killed'").fetchone()[0] == 0


def test_a_corrupt_ledger_is_reported_never_recreated(tmp_path, monkeypatch):
    path = tmp_path / "shadow.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        CS.evaluate({}.get, maturity=lambda s: LS.MATURED, holidays=HOLIDAYS, path=path)
    with pytest.raises(sqlite3.DatabaseError):
        CS.record_session(_scan([]), {}.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    assert path.read_bytes().startswith(b"this is not a sqlite database")            # untouched


def test_an_unrecorded_exceptional_session_never_silently_changes_the_horizon(tmp_path):
    path = tmp_path / "shadow.db"
    cut = {k: v[v.index <= SESSION] for k, v in _full_frames().items()}
    CS.record_session(_scan(cut), cut.get, clock=EVENING, holidays=HOLIDAYS, path=path)
    full = _full_frames()
    verified = [d.date() for d in _sessions("2026-01-01", "2026-12-31")]
    normal = LS.maturity_checker(verified_sessions=verified, holidays=HOLIDAYS, last_completed=date(2026, 12, 31))
    assert CS.evaluate(full.get, maturity=normal, holidays=HOLIDAYS, path=path)["sessions_evaluated"] == 1
    saturday = verified + [date(2026, 11, 28)]                    # an official bar on a Saturday in the window
    odd = LS.maturity_checker(verified_sessions=saturday, holidays=HOLIDAYS, last_completed=date(2026, 12, 31))
    rep = CS.evaluate(full.get, maturity=odd, holidays=HOLIDAYS, path=path)
    assert rep["sessions_evaluated"] == 0
    assert rep["session_outcomes"] == {"WINDOW_NOT_YET_VERIFIED": 1}
