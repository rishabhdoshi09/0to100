"""The negative-result diagnosis is read-only, point-in-time and honest about power."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from research import edge_diagnosis as ED


def _frames(n_sym=4, n=320, seed=7, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    out = {}
    for k in range(n_sym):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.018, n)))
        opn = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
        high = np.maximum.reduce([close * (1 + np.abs(rng.normal(0, 0.008, n))), opn, close])
        low = np.minimum.reduce([close * (1 - np.abs(rng.normal(0, 0.008, n))), opn, close])
        vol = rng.lognormal(14, 0.4, n)
        vol[rng.random(n) < 0.05] *= 3
        out[f"SYM{k}"] = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": vol},
                                      index=idx)
    return out


def _inf(lo, hi, *, n=200, blocks=40, status="OK"):
    return {"status": status, "n": n, "independent_sessions": blocks, "ci95": [lo, hi], "mean": (lo + hi) / 2}


@pytest.mark.parametrize("lo,hi,expected", [
    (-0.30, -0.02, ED.NEGATIVE_EDGE),          # measurably harmful: not a power problem
    (-0.04, 0.03, ED.NO_MATERIAL_EDGE),        # straddles zero but even the top is below the +0.05R floor
    (-0.20, 0.25, ED.UNDERPOWERED),            # cannot tell an edge worth trading from nothing
    (0.01, 0.30, ED.POSITIVE_IN_SAMPLE),       # a hypothesis only
])
def test_classification_separates_no_edge_from_low_power(lo, hi, expected):
    assert ED.classify(_inf(lo, hi), floor=0.05) == expected


@pytest.mark.parametrize("inf", [_inf(0.1, 0.3, n=29), _inf(0.1, 0.3, blocks=19), _inf(0.1, 0.3, status="INSUFFICIENT"),
                                 {"status": "OK", "n": 99, "independent_sessions": 30, "ci95": None}, None])
def test_thin_or_degenerate_evidence_makes_no_statement(inf):
    assert ED.classify(inf, floor=0.05) == ED.INSUFFICIENT


def test_minimum_detectable_effect_is_the_one_sided_power_formula():
    from scipy.stats import t
    assert ED.mde(0.1, 30) == pytest.approx((t.ppf(0.95, 30) + t.ppf(0.80, 30)) * 0.1)
    assert ED.mde(0.05, 30) < ED.mde(0.1, 30)
    assert ED.mde(None, 30) is None and ED.mde(0.1, 0) is None


def test_coverage_starts_after_the_last_failed_or_missing_window():
    cov = {"windows": {
        "a": {"covered_from": "2020-01-01", "covered_to": "2020-06-30", "success": True},
        "b": {"covered_from": "2020-07-01", "covered_to": "2020-12-31", "success": False},
        "c": {"covered_from": "2021-01-01", "covered_to": "2021-06-30", "success": True},
        "d": {"covered_from": "2021-09-01", "covered_to": "2021-12-31", "success": True},     # a gap before it
        "e": {"covered_from": "2022-01-01", "covered_to": "2022-06-30", "success": True}}}
    assert ED.coverage_start(cov) == "2021-09-01"
    assert ED.coverage_start({}) is None


def test_the_strict_tier_needs_the_whole_52_week_lookback_adjusted():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-01", periods=600)]
    ordinal = {d: i for i, d in enumerate(days)}
    tiers = ED.tier_map(ordinal, days[100])
    assert tiers[days[99]] == "UNCOVERED" and tiers[days[100]] == "ENTRY_ONLY"
    assert tiers[days[100 + 251]] == "ENTRY_ONLY" and tiers[days[100 + 252]] == "STRICT"
    assert set(ED.tier_map(ordinal, None).values()) == {"UNCOVERED"}


def test_membership_is_as_of_the_decision_date():
    m = ED.membership([{"symbol": "a", "listed": "2022-01-01", "delisted": "2024-01-01"},
                       {"symbol": "B", "listed": "2023-06-01"}, {"bad": 1}])
    assert ED.is_member(m, "A", "2021-12-31") is False and ED.is_member(m, "A", "2023-12-31") is True
    assert ED.is_member(m, "A", "2024-01-01") is False and ED.is_member(m, "B", "2026-01-01") is True
    assert ED.is_member(m, "ZZZ", "2023-01-01") is None


def test_the_equal_weight_benchmark_needs_breadth_and_keeps_delisted_names():
    idx = pd.bdate_range("2024-01-01", periods=30)
    up = {f"U{k}": pd.DataFrame({"close": 100 * 1.01 ** np.arange(30)}, index=idx) for k in range(60)}
    gone = {"DELISTED": pd.DataFrame({"close": 100 * 0.5 ** np.arange(10)}, index=idx[:10])}   # -50%/day, clipped
    level = ED.equal_weight_index({**up, **gone}, min_names=50)
    first = level.iloc[0] - 1                                       # the day DELISTED still traded
    assert first == pytest.approx((60 * 0.01 + 1 * -ED.EW_DAILY_CLIP) / 61)
    assert level.pct_change().iloc[-1] == pytest.approx(0.01)       # after it stopped trading
    assert ED.equal_weight_index(dict(list(up.items())[:10]), min_names=50).empty


def _rows(values, days, cost=0.05, risk=0.04):
    return [{"symbol": "X", "decision": d, "signals": ["S"], "net_r": v, "gross_r": v + cost, "cost_r": cost,
             "risk_frac": risk, "gross_ret": (v + cost) * risk, "bench_ret": 0.0, "excess_ret": v * risk,
             "regime": "BULL", "exit_reason": "FLAT", "holding_period": 10} for v, d in zip(values, days)]


def test_duplicated_or_same_session_evidence_cannot_manufacture_a_verdict():
    rng = np.random.default_rng(3)
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=400)]
    ordinal = {d: i for i, d in enumerate(days)}
    picked = days[::4]
    vals = list(rng.normal(0.02, 0.6, len(picked)))
    once = ED._signal_report(_rows(vals, picked), [], ordinal, 0.05, len(days))
    twenty = ED._signal_report(_rows(vals * 20, picked * 20), [], ordinal, 0.05, len(days))
    assert twenty["trades"] == 20 * once["trades"]
    assert twenty["net_r"]["ci95"] == pytest.approx(once["net_r"]["ci95"], abs=1e-3)    # no free precision
    assert twenty["verdict"] == once["verdict"] != ED.POSITIVE_IN_SAMPLE


def test_cost_and_slippage_sensitivity_only_ever_lower_the_mean():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=200)]
    ordinal = {d: i for i, d in enumerate(days)}
    rep = ED._signal_report(_rows([0.1, -0.2, 0.3] * 20, days[:60]), [], ordinal, 0.05, 200)
    s = rep["cost_sensitivity_mean_r"]
    assert s["cost_x0"] > s["cost_x0.5"] > s["cost_x1"] > s["cost_x2"]
    assert s["cost_x1_plus_0.10pct_slippage_each_side"] < s["cost_x1"]
    assert rep["breakeven_cost_multiple"] == pytest.approx(rep["gross_r"]["mean"] / rep["mean_cost_r"], abs=0.01)


def test_a_positive_gross_edge_eaten_by_costs_is_named():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=300)]
    ordinal = {d: i for i, d in enumerate(days)}
    rep = ED._signal_report(_rows([-0.03] * 60, days[::5][:60], cost=0.08), [], ordinal, 0.05, 300)
    assert "COSTS_FLIP_SIGN" in rep["drivers"]


def test_the_diagnosis_runs_the_production_walk_forward_and_writes_nothing(tmp_path):
    import data.bhavcopy_store as BS
    import scan.signal_backtest as SB
    from scan.unified_scanner import UnifiedScanner

    before = (BS.get_ohlcv, BS.store_symbols, SB.simulate_trade, SB.MIN_SYMBOLS_TO_SAVE, UnifiedScanner._analyze)
    report_existed = SB._OUT.exists()
    ledger_env = os.environ["QT_ALPHA_LEDGER"]
    frames = _frames()
    out = ED.collect(frames, symbols=4, sample_step=5)
    assert out["events"] and out["trades"]
    assert all(t["decision"] and t["decision"] < t["exit_datetime"] for t in out["trades"])
    assert {e["fate"] for e in out["events"]} <= {"FILLED", "NO_FILL", "RISK_SKIP"}
    assert sum(e["fate"] == "FILLED" for e in out["events"]) == len(out["trades"])
    assert out["provenance"]["eval_spec"] == SB._eval_spec_tag(10)
    # nothing leaked: no report, no production alpha look, every hook restored
    assert SB._OUT.exists() == report_existed
    assert os.environ["QT_ALPHA_LEDGER"] == ledger_env and not os.path.exists(ledger_env)
    assert (BS.get_ohlcv, BS.store_symbols, SB.simulate_trade, SB.MIN_SYMBOLS_TO_SAVE,
            UnifiedScanner._analyze) == before

    cov = {"windows": {"a": {"covered_from": "2023-06-01", "covered_to": "2024-12-31", "success": True}}}
    uni = [{"symbol": "SYM0", "listed": "2023-01-02"}, {"symbol": "SYM1", "listed": "2023-01-02"},
           {"symbol": "SYM2", "listed": "2023-09-01"},                          # listed later
           {"symbol": "SYM3", "listed": "2023-01-02", "delisted": "2023-10-02"}]  # delisted
    rep = ED.analyse(out, frames, coverage=cov, universe_rows=uni, ew_min_names=3)
    early_sym2 = sum(1 for t in out["trades"] if t["symbol"] == "SYM2" and t["decision"] < "2023-09-01")
    late_sym3 = sum(1 for t in out["trades"] if t["symbol"] == "SYM3" and t["decision"] >= "2023-10-02")
    assert rep["exclusions"]["not_a_member_on_decision_date"] == early_sym2 + late_sym3
    assert rep["trades_kept"] == rep["trades_captured"] - early_sym2 - late_sym3
    tiers = rep["tiers"]
    assert tiers["COVERED_ALL"]["trades"] == tiers["STRICT"]["trades"] + tiers["ENTRY_ONLY"]["trades"]
    assert sum(tiers[t]["trades"] for t in ED.TIERS) == rep["trades_kept"]
    any_row = tiers["COVERED_ALL"]["signals"][ED.ANY]
    assert any_row["trades"] == tiers["COVERED_ALL"]["trades"] and any_row["verdict"] in {
        ED.INSUFFICIENT, ED.NEGATIVE_EDGE, ED.NO_MATERIAL_EDGE, ED.UNDERPOWERED, ED.POSITIVE_IN_SAMPLE}
    assert "| ANY_SIGNAL |" in ED.summary_table(rep, "COVERED_ALL")


def test_future_bars_cannot_change_an_earlier_decision_or_its_outcome():
    frames = _frames()
    base = ED.collect(frames, symbols=4, sample_step=5)
    days = list(next(iter(frames.values())).index)
    cut = days[250]
    poisoned = {}
    for sym, df in frames.items():
        p = df.copy()
        p.loc[p.index >= cut, ["open", "high", "low", "close"]] *= 3.0          # a fake future
        p.loc[p.index >= cut, "volume"] *= 10
        poisoned[sym] = p
    after = ED.collect(poisoned, symbols=4, sample_step=5)
    safe = days[250 - ED.HORIZON - 1].strftime("%Y-%m-%d")                      # window ends before the cut

    def early(c):
        ev = sorted((e["symbol"], e["decision"], e["signals"], e["fate"]) for e in c["events"] if e["decision"] < safe)
        tr = sorted((t["symbol"], t["decision"], t["net_R"]) for t in c["trades"] if t["decision"] < safe)
        return ev, tr

    assert early(base)[0] and early(base) == early(after)
    assert [e for e in after["events"] if e["decision"] >= safe] != [e for e in base["events"] if e["decision"] >= safe]


def test_the_copy_loader_refuses_the_live_external_volume():
    with pytest.raises(SystemExit, match="internal SSD"):
        ED.load_copy("/Volumes/QuantTermStorage/QuantTerm/runtime")
