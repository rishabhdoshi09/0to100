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
                         "low": [100] * MAX_HORIZON, "close": [103] * MAX_HORIZON})
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
    dev = [("development", day, variant, "WIN", r, r, 1)
           for day in days[:25] for variant, r in (("scanner", 1), ("short", 2)) for _ in range(4)]
    holdout = [("evaluation", days[90], variant, "LOSS" if r < 0 else "WIN", r, r, 1)
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
    assert report["evaluation"]["short"]["development_probability_positive_net"] > 0.99
    with pytest.raises(ValueError, match="different snapshot"):
        GeometryRun(db, cutoff=days[80], end=days[160], binding={"data": "version 2"})



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
        (split, days[day], variant, exit_kind, -1.0, -1.0, 1)
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
