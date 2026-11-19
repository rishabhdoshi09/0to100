"""Pre-declared exit-construction experiment in the weekly walk-forward (research only)."""

from __future__ import annotations

import pandas as pd

import scan.signal_backtest as SB


def _fwd(highs, lows, closes, opens=None):
    n = len(highs)
    return pd.DataFrame({"open": opens or [closes[0]] * n, "high": highs, "low": lows, "close": closes},
                        index=pd.bdate_range("2026-01-05", periods=n))


def test_trail_turns_a_drifting_winner_into_a_scratch_only_in_trail_variants():
    # Fills at 100, touches +2.5% (arms the trail), dips back to 99.9, then drifts to 104.
    highs = [100.5, 102.5, 101.0] + [104.0] * 17
    lows = [99.5, 100.5, 99.8] + [103.0] * 17
    closes = [100.2, 102.0, 100.4] + [103.8] * 17
    opens = [99.9, 100.3, 101.5] + [103.5] * 17           # no gap through the trailed stop
    got = {n: (o, r) for n, o, r in SB.exit_variant_results(100.0, 94.0, 112.0, _fwd(highs, lows, closes, opens), 10)}
    assert got["V0_prod_be_h"][0] == "SCRATCH"            # stopped at entry after arming
    assert got["V1_no_trail_h"][0] == "FLAT" and got["V1_no_trail_h"][1] > 0.5
    assert got["V2_be_2h"][0] == "SCRATCH"
    assert got["V3_no_trail_2h"][1] > 0.5


def test_no_fill_is_not_a_trade_in_any_variant():
    highs = [99.0] * 20
    out = SB.exit_variant_results(100.0, 94.0, 112.0, _fwd(highs, [98.0] * 20, [98.5] * 20), 10)
    assert out == []


def _hist(n=300, drift=0.0, vol=1.0, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01 * vol, n))
    return pd.DataFrame({"open": close, "high": close * (1 + 0.01 * vol), "low": close * (1 - 0.01 * vol), "close": close},
                        index=pd.bdate_range("2025-01-01", periods=n))


def test_confirmation_features_are_pit_and_oriented_higher_is_better():
    calm, wild = SB.confirmation_features(_hist(vol=0.5)), SB.confirmation_features(_hist(vol=3.0))
    assert calm["low_atr"] > wild["low_atr"]                 # lower ATR scores higher
    assert SB.confirmation_features(_hist(n=40)) is None      # too little history: no value invented


def _day(i: int) -> str:
    from datetime import date, timedelta
    return (date(2024, 1, 1) + timedelta(days=i)).isoformat()


def _rows(n_days, ic_sign=1.0, noise=0.0, per_day=8, seed=0, start=0, feats=("low_atr",)):
    import random
    rng = random.Random(seed)
    rows, index = [], {}
    for d in range(n_days):
        day = _day(start + d)
        index[day] = start + d
        for k in range(per_day):
            x = float(k)
            r = ic_sign * x + rng.gauss(0, noise)
            row = {"day": day, "r": r, "low_atr": x, "near_high": rng.random(), "not_extended": rng.random()}
            rows.append(row)
    return rows, index


def test_zero_dispersion_is_never_decisive():
    """Perfect, identical daily ICs: a degenerate sample, not maximal significance."""
    rows, index = _rows(400, ic_sign=1.0, noise=0.0)               # 40 blocks, IC = 1 every day
    rep = SB.confirmation_report(rows, horizon=10, session_index=index)
    f = rep["features"]["low_atr"]
    assert f["verdict"] == "INSUFFICIENT EVIDENCE"
    assert f["reason"] == "DEGENERATE_ZERO_DISPERSION"
    assert "p_two_sided" not in f


def test_fewer_than_twenty_independent_blocks_cannot_confirm():
    rows, index = _rows(190, ic_sign=1.0, noise=2.0)               # 19 blocks of 10 sessions
    f = SB.confirmation_report(rows, horizon=10, session_index=index)["features"]["low_atr"]
    assert f["independent_sessions"] == 19
    assert f["verdict"] == "INSUFFICIENT EVIDENCE"


def test_many_overlapping_days_do_not_count_as_independent():
    rows, index = _rows(300, ic_sign=1.0, noise=2.0)
    packed = {day: i // 6 for i, day in enumerate(sorted(index))}  # 300 days squeezed into 50 sessions
    f = SB.confirmation_report(rows, horizon=10, session_index=packed)["features"]["low_atr"]
    assert f["independent_sessions"] == 5
    assert f["verdict"] == "INSUFFICIENT EVIDENCE"


def test_real_relationship_over_enough_blocks_is_confirmed_and_reversed_is_contradicted():
    rows, index = _rows(400, ic_sign=1.0, noise=6.0, seed=1)
    assert SB.confirmation_report(rows, horizon=10, session_index=index)["features"]["low_atr"]["verdict"] == "CONFIRMED"
    rows, index = _rows(400, ic_sign=-1.0, noise=6.0, seed=2)
    assert SB.confirmation_report(rows, horizon=10, session_index=index)["features"]["low_atr"]["verdict"] == "CONTRADICTED"


def test_pure_noise_is_not_confirmed():
    rows, index = _rows(400, ic_sign=0.0, noise=1.0, seed=3)
    for feat in ("low_atr", "near_high", "not_extended", "composite"):
        v = SB.confirmation_report(rows, horizon=10, session_index=index)["features"][feat]["verdict"]
        assert v in {"NOT_CONFIRMED", "INSUFFICIENT EVIDENCE"}, (feat, v)


def test_tie_corrected_spearman_matches_scipy():
    from scipy.stats import spearmanr
    xs, ys = [1, 1, 2, 3, 3, 3, 7], [2, 1, 4, 3, 5, 5, 9]
    assert abs(SB.spearman_ic(xs, ys) - spearmanr(xs, ys).correlation) < 1e-12


def test_constant_feature_or_outcome_carries_no_ic():
    assert SB.spearman_ic([5, 5, 5, 5], [1, 2, 3, 4]) is None
    assert SB.spearman_ic([1, 2, 3, 4], [0, 0, 0, 0]) is None
    rows, index = _rows(400, ic_sign=1.0, noise=1.0)
    for r in rows:
        r["low_atr"] = 1.0                                         # tied within every day
    f = SB.confirmation_report(rows, horizon=10, session_index=index)["features"]["low_atr"]
    assert f["days_with_ic"] == 0 and f["verdict"] == "INSUFFICIENT EVIDENCE"


def test_unavailable_calendar_reaches_no_verdict(monkeypatch):
    monkeypatch.setattr(SB, "_session_ordinals", lambda days: None)
    rows, _ = _rows(400, ic_sign=1.0, noise=6.0, seed=1)
    f = SB.confirmation_report(rows, horizon=10)["features"]["low_atr"]
    assert f["verdict"] == "INSUFFICIENT EVIDENCE" and f["reason"] == "EXCHANGE_CALENDAR_UNAVAILABLE"


def test_discovery_window_rows_never_count():
    rows, index = _rows(60, ic_sign=1.0, noise=6.0, start=(905))   # 2026-06-23 onward: inside the window
    inside = [r for r in rows if "2026-05-26" <= r["day"] <= "2026-07-30"]
    assert len(inside) >= 200                                      # a real discovery-window sample
    rep = SB.confirmation_report(inside, horizon=10, session_index=index)
    assert rep["days_used"] == 0 and rep["trades_used"] == 0


def test_real_calendar_skips_holidays_and_weekends():
    idx = SB._session_ordinals(["2026-10-30", "2026-11-02"])        # Fri, then Mon
    assert idx is not None and idx["2026-11-02"] - idx["2026-10-30"] == 1


def _exit_stats(n_days, mean, noise, seed=0, index_from=0):
    import random
    rng = random.Random(seed)
    st, index = {"trades": 0, "wins": 0, "scratches": 0, "r_sum": 0.0, "clusters": {}}, {}
    for d in range(n_days):
        day = _day(index_from + d)
        index[day] = index_from + d
        for _ in range(5):
            r = mean + rng.gauss(0, noise)
            st["trades"] += 1
            st["wins"] += r > 0
            st["r_sum"] += r
            SB._add_cluster(st["clusters"], day, r)
    return st, index


def test_exit_report_uses_blocks_and_refuses_thin_or_degenerate_samples():
    st, index = _exit_stats(400, 0.3, 0.5, seed=4)                     # 40 blocks
    rep = SB.exit_variant_report({"V3_no_trail_2h": st}, horizon=10, session_index=index)
    v = rep["variants"]["V3_no_trail_2h"]
    assert v["independent_sessions"] == 40 and v["verdict"] == "POSITIVE"
    st, index = _exit_stats(150, 0.3, 0.5, seed=5)                     # 15 blocks
    assert SB.exit_variant_report({"V0_prod_be_h": st}, horizon=10, session_index=index)[
        "variants"]["V0_prod_be_h"]["verdict"] == "INSUFFICIENT EVIDENCE"
    st, index = _exit_stats(400, 0.3, 0.0, seed=6)                     # identical R everywhere
    v = SB.exit_variant_report({"V1_no_trail_h": st}, horizon=10, session_index=index)["variants"]["V1_no_trail_h"]
    assert v["verdict"] == "INSUFFICIENT EVIDENCE" and v["reason"] == "DEGENERATE_ZERO_DISPERSION"
    assert rep["status"] == "RESEARCH_ONLY"


def test_missing_holiday_calendar_is_treated_as_unavailable(monkeypatch):
    import research.intelligence.data.nse_calendar as CAL
    monkeypatch.setattr(CAL, "load_holidays", lambda: set())
    assert SB._session_ordinals(["2026-10-30", "2026-11-02"]) is None
