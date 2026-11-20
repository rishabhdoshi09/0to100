"""PR-2026-11-21-MOM12-1: the frozen spec, point-in-time scoring, honest missing data (network-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import momentum_portfolio as MP


def _frames(n_sym=120, n_days=330, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    out = {}
    for k in range(n_sym):
        drift = 0.0006 * (k % 7 - 3)
        c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n_days)))
        out[f"S{k:03d}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                                         "volume": np.full(n_days, 2e6)}, index=idx)
    return out


def _member(frames):
    return {s: ("2000-01-01", None) for s in frames}


def test_the_spec_is_frozen():
    assert MP.spec_sha256() == MP.FROZEN_SHA256
    spec = MP.load_spec()
    assert spec["can_create_buy"] is False and spec["can_place_orders"] is False


def test_scores_use_only_bars_up_to_the_cutoff():
    frames = _frames()
    spec = MP.load_spec()
    panel = MP.build_panel(frames)
    i = MP.rebalance_indices(panel["dates"])[-2]
    before = MP.eligible_scores(panel, i, spec, _member(frames))
    poisoned = {s: df.copy() for s, df in frames.items()}
    cut = pd.Timestamp(panel["dates"][i - 1])
    for df in poisoned.values():
        df.loc[df.index > cut, ["open", "high", "low", "close"]] *= 3.0      # the future, multiplied
    after = MP.eligible_scores(MP.build_panel(poisoned), i, spec, _member(frames))
    assert before and before == after


def test_the_held_out_year_needs_an_explicit_single_look():
    frames = _frames()
    with pytest.raises(PermissionError):
        MP.evaluate(MP.build_panel(frames), MP.load_spec(), member=_member(frames),
                    coverage=("2000-01-01", None), period="held_out")


def test_a_name_that_stops_trading_is_never_priced_at_a_missing_bar():
    close = np.array([[100.0], [90.0], [np.nan], [np.nan]])
    r, stopped = MP.holding_return(close, 0, 0, 3)
    assert stopped and r == pytest.approx(-0.10)                          # last traded close, then cash
    r_half, _ = MP.holding_return(close, 0, 0, 3, remainder_loss=0.5)
    assert r_half == pytest.approx(0.9 * 0.5 - 1.0)
    assert np.isnan(MP.holding_return(np.array([[np.nan], [90.0]]), 0, 0, 1)[0])   # not executable


def test_rebalances_outside_verified_coverage_are_not_evaluated():
    frames = _frames()
    panel = MP.build_panel(frames)
    spec = MP.load_spec()
    full = MP.evaluate(panel, spec, member=_member(frames), coverage=("2000-01-01", None))
    none = MP.evaluate(panel, spec, member=_member(frames), coverage=(None, None))
    late = MP.evaluate(panel, spec, member=_member(frames), coverage=(panel["dates"][200], None))
    assert full["months"] > late["months"] >= 0 and none["months"] == 0


def test_costs_only_lower_the_net_and_the_traded_price_floor_uses_the_traded_price():
    frames = _frames()
    panel = MP.build_panel(frames)
    spec = MP.load_spec()
    base = MP.evaluate(panel, spec, member=_member(frames), coverage=("2000-01-01", None))
    dear = dict(spec, costs={**spec["costs"], "per_side_pct": 1.0})
    worse = MP.evaluate(panel, dear, member=_member(frames), coverage=("2000-01-01", None))
    assert worse["secondary"]["mean_monthly_net"] < base["secondary"]["mean_monthly_net"]
    m = MP._multipliers(["2023-01-02", "2023-02-01", "2023-03-01"], [("2023-02-15", 10.0)])
    assert list(m) == [10.0, 10.0, 1.0]                                    # a later 10:1 split
