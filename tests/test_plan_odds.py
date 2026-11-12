"""No-edge probabilistic view of a recommendation plan (owner request 2026-11-12)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from product.plan_odds import plan_odds


def _walk(n=260, vol=0.015, drift=0.0, seed=7):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    low = close * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range(end="2026-11-11", periods=n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def test_probabilities_are_a_partition_and_respect_reward_risk():
    bars = _walk()
    sym = plan_odds("AAA", 100.0, 95.0, 105.0, bars=bars)
    assert abs(sym["p_target_first"] + sym["p_stop_first"] + sym["p_neither"] - 1.0) < 0.002
    # Symmetric % plan, drift removed: roughly a coin flip between the two
    # levels (+5% is a slightly shorter log move than -5%).
    assert 0.25 < sym["p_target_first"] < 0.5 and abs(sym["p_target_first"] - sym["p_stop_first"]) < 0.1
    assert sym["breakeven_win_rate"] == 0.5 and sym["reward_risk"] == 1.0

    far = plan_odds("AAA", 100.0, 95.0, 115.0, bars=bars)
    assert far["p_target_first"] < sym["p_target_first"]          # 3R target is harder
    assert far["breakeven_win_rate"] == 0.25


def test_drift_is_removed_so_a_trending_history_is_not_an_edge():
    up = plan_odds("UP", 100.0, 95.0, 105.0, bars=_walk(drift=0.004))
    assert abs(up["p_target_first"] - up["p_stop_first"]) < 0.08
    assert abs(up["no_edge_expected_r"]) < 0.15


def test_cone_widens_with_horizon_and_is_deterministic():
    bars = _walk()
    a = plan_odds("AAA", 100.0, 95.0, 110.0, bars=bars)
    b = plan_odds("AAA", 100.0, 95.0, 110.0, bars=bars)
    assert a == b
    widths = [c["p90_pct"] - c["p10_pct"] for c in a["cone"]]
    assert [c["sessions"] for c in a["cone"]] == [5, 10, 15]
    assert widths[0] < widths[1] < widths[2]


def test_invalid_plans_or_thin_history_make_no_claim():
    bars = _walk()
    assert plan_odds("AAA", 100.0, 101.0, 110.0, bars=bars) is None   # stop above entry
    assert plan_odds("AAA", 100.0, 95.0, 99.0, bars=bars) is None     # target below entry
    assert plan_odds("AAA", 100.0, 95.0, 110.0, bars=_walk(n=60)) is None
