from datetime import date, timedelta
from types import SimpleNamespace
import sqlite3

import pandas as pd
import pytest

from gauntlet.geometry import GRID, MAX_HORIZON, GeometryRun


def _days(n=180):
    first = date(2025, 1, 1)
    return [(first + timedelta(days=i)).isoformat() for i in range(n)]


def test_variants_share_purged_windows_and_retain_censored_setups():
    days = _days()
    sessions = dict(zip(days, range(len(days))))
    run = GeometryRun(sqlite3.connect(":memory:"), cutoff=days[80], end=days[160], binding={})
    bars = pd.DataFrame({"open": [100] * MAX_HORIZON, "high": [104] * MAX_HORIZON,
                         "low": [100] * MAX_HORIZON, "close": [103] * MAX_HORIZON},
                        index=pd.to_datetime(days[81:81 + MAX_HORIZON]))
    setup = SimpleNamespace(entry=100, stop=98, target=106)
    assert list(run.measure(days[80 - MAX_HORIZON], setup, bars, sessions)) == []
    outcomes = {row[2]: row for row in run.measure(days[80], setup, bars, sessions)}
    assert set(outcomes) == set(GRID)
    assert outcomes["target_3pct"][0] == "evaluation"
    assert outcomes["target_3pct"][5] == pytest.approx(2.68)
    assert outcomes["target_3pct"][6] == 1
    assert {row[3] for row in run.measure(days[80], setup, bars.iloc[:0], sessions)} == {"CENSORED"}


def test_symbol_resume_is_atomic_and_selection_does_not_use_evaluation(tmp_path):
    days = _days()
    sessions = dict(zip(days, range(len(days))))
    db = sqlite3.connect(tmp_path / "research.sqlite")
    run = GeometryRun(db, cutoff=days[80], end=days[160], binding={"data": "version 1"})
    dev = [("development", day, variant, "WIN", r, r, 1, day)
           for day in days[:25] for variant, r in (("scanner", 1), ("short", 2)) for _ in range(4)]
    holdout = [("evaluation", days[90], variant, "LOSS" if r < 0 else "WIN", r, r, 1, days[91])
               for variant, r in (("scanner", 5), ("short", -5))]
    with pytest.raises(sqlite3.ProgrammingError):
        run.save("BROKEN", [("incomplete",)])
    assert not run.completed("BROKEN")
    run.save("TCS", dev + holdout)
    run.save("TCS", dev + holdout)
    resumed = GeometryRun(db, cutoff=days[80], end=days[160], binding={"data": "version 1"})
    assert resumed.completed("TCS")
    report = resumed.report(["TCS"], sessions)
    assert report["selected"] == "short"
    assert report["development"]["short"]["filled"] == 100
    assert report["evaluation"]["short"]["mean_net_return_pct"] == -5
    assert report["evaluation"]["short"]["settled_daily_fixed_risk_max_drawdown_R"] == 5
    assert report["evaluation"]["short"]["development_probability_positive_net"] > 0.99
    with pytest.raises(ValueError, match="different snapshot"):
        GeometryRun(db, cutoff=days[80], end=days[160], binding={"data": "version 2"})


def test_observations_are_censored_and_committed_with_outcomes(tmp_path):
    days = _days()
    db = sqlite3.connect(tmp_path / "observations.sqlite")
    run = GeometryRun(db, cutoff=days[80], end=days[160], binding={"observations": True})
    history = pd.DataFrame({"close": [100.] * 60, "volume": [200_000.] * 60})
    forward = pd.DataFrame({"open": [102.] * 3, "close": [104.] * 3})
    row = run.observe(days[60], history, forward, None)
    with pytest.raises(sqlite3.IntegrityError):
        run.save("TCS", [], [row, row])
    assert not run.completed("TCS")
    assert db.execute("SELECT count(*) FROM geometry_observations").fetchone() == (0,)
    run.save("TCS", [], [row])
    run.save("TCS", [], [row])
    assert db.execute("SELECT features, outcome FROM geometry_observations").fetchall() == [row[1:]]
    import json
    outcome = json.loads(row[2])
    assert outcome["next_open_gap"] == pytest.approx(.02)
    assert outcome["close_return"] is None


def test_incremental_observations_match_clean_after_backfill_and_extension():
    from scan.signal_backtest import GRID_SESSION_ANCHORED, _run_backtest_inner

    dates = pd.bdate_range("2024-01-01", periods=230)
    days = list(dates.strftime("%Y-%m-%d"))
    archives = {day: day for day in days[:-1]}
    scanned = []

    class Scanner:
        def _analyze(self, symbol, frame):
            scanned.append((symbol, frame.index[-1].strftime("%Y-%m-%d")))
            return SimpleNamespace(entry=100, stop=98, target=106,
                                   signals=["setup"], score=20, verdict="WATCH")

    def replay(db, end, symbols, gap=None):
        prefix = days[:end + 1]
        digests = {day: None if day == gap else archives[day] for day in prefix[:-1]}
        binding = dict(exploratory=True, observations=True, symbols=symbols,
                       sessions=prefix, archive_sha256=digests)
        run = GeometryRun(db, cutoff=days[140], end=days[end], binding=binding)
        run.scoped_sessions = {day: i for i, day in enumerate(prefix)}
        index = dates[:end]
        if gap:
            index = index[index != pd.Timestamp(gap)]
        frame = pd.DataFrame({"open": 101., "high": 106., "low": 98., "close": 103.,
                              "volume": 500_000.}, index=index)
        run.scoped_frames = dict.fromkeys(symbols, frame)
        _run_backtest_inner(Scanner(), {}, symbols, 5, len(days), MAX_HORIZON, 0,
                            decision_grid=GRID_SESSION_ANCHORED, geometry=run)
        return run

    incremental = sqlite3.connect(":memory:")
    clean = sqlite3.connect(":memory:")
    old = replay(incremental, 195, ["TCS"], gap=days[125])
    scanned.clear()
    replay(incremental, 195, ["INFY", "TCS"], gap=days[125])
    assert scanned and {symbol for symbol, _ in scanned} == {"INFY"}
    scanned.clear()
    replay(incremental, 229, ["INFY", "TCS"])
    assert scanned and min(day for _, day in scanned) >= days[125 - MAX_HORIZON]
    with pytest.raises(ValueError, match="scope changed"):
        old.report(["TCS"], old.scoped_sessions)
    replay(clean, 229, ["INFY", "TCS"])
    for table in ("geometry_observations", "geometry_outcomes"):
        query = f"SELECT * FROM {table} ORDER BY 1,2,3,4"
        assert incremental.execute(query).fetchall() == clean.execute(query).fetchall()



def test_frozen_snapshot_cache_survives_a_failed_read(tmp_path, monkeypatch):
    from data import bhavcopy_store

    cache = tmp_path / "store_cache.pkl"
    cache.write_bytes(b"archive from an incompatible environment")
    monkeypatch.setattr(bhavcopy_store, "_PKL", cache)

    assert not bhavcopy_store._load_pkl(remove_invalid=False)
    assert cache.read_bytes() == b"archive from an incompatible environment"



def test_report_scores_all_variants_with_development_only_exit_probabilities():
    days = _days()
    sessions = dict(zip(days, range(len(days))))
    run = GeometryRun(sqlite3.connect(":memory:"), cutoff=days[80], end=days[160], binding={})
    run.save("TCS", [
        (split, days[day], variant, exit_kind, -1.0, -1.0, 1, days[day + 1])
        for split, day, exit_kind in (
            ("development", 2, "WIN"), ("development", 3, "TARGET_GAP_SCRATCH"),
            ("development", 4, "SCRATCH"), ("evaluation", 90, "FLAT"),
        )
        for variant in GRID
    ])
    report = run.report(["TCS"], sessions)
    assert set(report["evaluation"]) == set(GRID)
    assert report["selected"] is None
    exits = report["evaluation"]["scanner"]["exit_calibration"]
    assert exits["target_first"]["development_probability"] == pytest.approx(2.5 / 4.5)
    assert exits["stop_first"]["development_probability"] == pytest.approx(1.5 / 4.5)
    assert exits["time_expiry"]["development_probability"] == pytest.approx(0.5 / 4.5)
    assert exits["time_expiry"]["evaluation_rate"] == 1.0
    assert exits["time_expiry"]["brier"] == pytest.approx((1 - 0.5 / 4.5) ** 2)


def test_reuses_successful_preflight_and_retries_failed_preflight(tmp_path):
    checkpoint = tmp_path / "study.db"
    with sqlite3.connect(checkpoint) as db:
        run = GeometryRun(db, cutoff="2025-05-01", end="2025-08-01", binding={})
        assert run.validate_once(lambda: {"ok": False}) == {"ok": False}
        assert run.validate_once(lambda: {"ok": True}) == {"ok": True}
    with sqlite3.connect(checkpoint) as db:
        resumed = GeometryRun(db, cutoff="2025-05-01", end="2025-08-01", binding={})
        assert resumed.validate_once(lambda: pytest.fail("preflight repeated")) == {"ok": True}


def test_scope_quarantines_be_corruption_action_and_missing_session():
    from gauntlet.validator import scoped_equity_frames

    days = pd.bdate_range("2026-06-01", periods=7)
    header = "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY,PREV_CLOSE\n"

    def payload(i, series="EQ", previous=100, embedded=None):
        return (header + f"TCS,{series},{(embedded or days[i]).date()},100,101,99,100,1000,{previous}\n").encode()

    frames, report = scoped_equity_frames(
        zip(days.date, [payload(0), payload(1, "BE"), payload(2), payload(3, previous=55),
                        payload(4), None, payload(6, embedded=days[5])]),
        ["TCS"], {"TCS": [{"ex_date": days[4]}]},
    )
    assert list(frames["TCS"].index) == [days[0], days[2]]
    assert report["quarantined_symbol_sessions_by_reason"] == {
        "trade_to_trade_BE": 1, "unreconciled_previous_close": 1,
        "documented_corporate_action": 1,
    }
    assert set(report["market_wide_excluded_sessions"]) == {str(days[5].date()), str(days[6].date())}


def test_scoped_scanner_does_not_join_feature_or_exit_windows_across_a_gap():
    from scan.signal_backtest import GRID_SESSION_ANCHORED, _run_backtest_inner

    days = pd.bdate_range("2026-01-01", periods=195)
    gap = days[80]
    scoped = days.delete(80)
    scanner_days = {}

    class Scanner:
        def _analyze(self, symbol, frame):
            scanner_days[frame.index[-1]] = frame.index[0]
            return SimpleNamespace(entry=100, stop=98, target=106, signals=["setup"])

    run = GeometryRun(sqlite3.connect(":memory:"), cutoff=days[110].strftime("%Y-%m-%d"),
                      end=days[-1].strftime("%Y-%m-%d"), binding={"exploratory": True})
    run.scoped_sessions = {day.strftime("%Y-%m-%d"): i for i, day in enumerate(days)}
    run.scoped_frames = {"TCS": pd.DataFrame(
        {"open": 99., "high": 99., "low": 98., "close": 99., "volume": 500_000}, index=scoped)}
    report = _run_backtest_inner(Scanner(), {}, ["TCS"], 1, len(days), MAX_HORIZON, 0,
                                 decision_grid=GRID_SESSION_ANCHORED, geometry=run)
    assert report["evidence"] == "exploratory_historical_research"
    assert all(day < gap or day >= days[140] for day in scanner_days)
    assert scanner_days[days[140]] == days[81]
    assert report["development"]["scanner"]["censored"] > 0
