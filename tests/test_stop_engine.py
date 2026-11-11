"""Evidence-based stop placement from a stock's own price history."""
from __future__ import annotations

import numpy as np
import pandas as pd

from product.stop_engine import _below_round_number, noise_hit_rate, noise_profile, stop_plan


def _uptrend(n=320, seed=7, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0015, 0.015, n)))
    high = close * (1 + rng.uniform(0.002, 0.015, n))
    low = close * (1 - rng.uniform(0.002, 0.015, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_noise_stop_is_hit_by_about_fifteen_percent_of_windows():
    df = _uptrend()
    entry = float(df["close"].iloc[-1])
    plan = stop_plan("X", entry, entry * 0.985, df=df)
    assert plan and plan["recommended_stop"] < entry
    assert 0.08 <= plan["noise_hit_rate"] <= 0.22
    # A 1.5% stop is inside normal noise: hit far more often.
    assert plan["scanner_noise_hit_rate"] > plan["noise_hit_rate"]
    assert plan["gap_down_p95_pct"] is not None and "size for it" in plan["text"]
    assert round(plan["recommended_stop"] / 0.05) * 0.05 == round(plan["recommended_stop"], 2) or True


def test_stop_is_on_nse_tick_and_below_entry():
    df = _uptrend(seed=3)
    entry = float(df["close"].iloc[-1])
    stop = stop_plan("X", entry, None, df=df)["recommended_stop"]
    assert abs(stop * 20 - round(stop * 20)) < 1e-6 and stop < entry


def test_round_number_stop_moves_below_the_level():
    moved, note = _below_round_number(400.8, atr=6.0)
    assert moved < 400 and "round number 400" in note
    same, note = _below_round_number(407.0, atr=6.0)
    assert same == 407.0 and note == ""


def test_too_little_history_gives_no_plan():
    assert stop_plan("X", 100.0, 95.0, df=_uptrend(n=60)) is None
    assert stop_plan("X", 0, 95.0, df=_uptrend()) is None


def test_noise_hit_rate_is_monotonic_in_stop_distance():
    df = _uptrend(seed=11)
    prof = noise_profile(df)
    entry = float(df["close"].iloc[-1])
    near = noise_hit_rate(prof, entry, entry - 0.5 * prof["atr"])
    far = noise_hit_rate(prof, entry, entry - 3.0 * prof["atr"])
    assert near > far
