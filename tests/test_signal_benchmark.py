"""Report Card vs Nifty over each settled signal's exact holding sessions.

The July 2026 sample on the owner's Mac (n~2,989) averaged -1.71% while Nifty
did -0.53% over the same days; absolute returns alone cannot separate skill
from beta, and the evidence summary shipped benchmark_comparison=None.
"""
from __future__ import annotations

import pandas as pd

import core.outcome_resolver as R
import product.evidence_authority as E


def _bars(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"high": [r[1] for r in rows], "low": [r[2] for r in rows],
                         "close": [r[3] for r in rows]}, index=idx)


def test_first_touch_detail_reports_fill_and_exit_sessions(monkeypatch):
    bars = _bars([
        ("2026-07-06", 101, 99, 100),   # signal day: never part of the path
        ("2026-07-07", 100.5, 99, 100),  # no fill (high < entry 101)
        ("2026-07-08", 102, 100, 101.5),  # fill
        ("2026-07-09", 111, 101, 110),   # target 110 touched
    ])
    monkeypatch.setattr(R, "_ohlcv", lambda _s: bars)
    got = R.first_touch_detail("AAA", "2026-07-06", 101.0, 95.0, 110.0, horizon=15)
    assert got["worked"] == 1 and got["fill_date"] == "2026-07-08" and got["exit_date"] == "2026-07-09"
    assert R.first_touch_path("AAA", "2026-07-06", 101.0, 95.0, 110.0, horizon=15) == (
        110.0, (110.0 - 101.0) / 101.0 * 100.0, 1)


def test_benchmark_uses_exact_sessions_and_skips_uncovered_windows(monkeypatch):
    bars = _bars([
        ("2026-07-06", 101, 99, 100),
        ("2026-07-07", 102, 100, 101.5),  # fill at 101
        ("2026-07-08", 103, 90, 95),      # stop 95 hit
    ])
    monkeypatch.setattr(R, "_ohlcv", lambda _s: bars)
    monkeypatch.setattr(E, "_nifty_closes", lambda: {"2026-07-06": 1000.0, "2026-07-07": 1010.0, "2026-07-08": 990.0})
    row = {"symbol": "AAA", "logged_at": "2026-07-06T18:20:00", "entry_price": 101.0,
           "stop_price": 95.0, "target_price": 110.0}
    out = E.benchmark_comparison([row])
    assert out["n"] == 1
    # A window whose exit session the index cache lacks is skipped, not guessed.
    monkeypatch.setattr(E, "_nifty_closes", lambda: {"2026-07-06": 1000.0, "2026-07-07": 1010.0})
    assert E.benchmark_comparison([row]) is None
    monkeypatch.setattr(E, "_nifty_closes", lambda: {"2026-07-06": 1000.0, "2026-07-07": 1010.0, "2026-07-08": 990.0})
    sig = (95.0 - 101.0) / 101.0 * 100.0
    bench = (990.0 / 1000.0 - 1.0) * 100.0
    assert out["mean_signal_pct"] == round(sig, 2)
    assert out["mean_benchmark_pct"] == round(bench, 2)
    assert out["mean_excess_pct"] == round(sig - bench, 2)
    assert out["sufficient_sample"] is False


def test_no_index_coverage_means_no_claim(monkeypatch):
    monkeypatch.setattr(E, "_nifty_closes", lambda: {})
    assert E.benchmark_comparison([{"symbol": "AAA"}]) is None
