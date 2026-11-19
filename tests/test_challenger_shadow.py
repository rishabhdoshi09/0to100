"""Prospective shadow ledger: frozen before entry, exact horizon, paired decision."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
