"""PR-2026-11-20-52WH: frozen, shadow-only, point-in-time, one pre-registered analysis."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research import prereg_52wh as P

SPEC = P.load_spec()


def test_the_spec_is_frozen_and_cannot_be_edited_silently():
    assert P.spec_sha256() == P.FROZEN_SHA256, "the pre-registration was edited after freezing: register a new one"
    assert SPEC["lane"] == "SHADOW_ONLY" and SPEC["can_create_buy"] is False and SPEC["can_place_orders"] is False
    assert SPEC["primary_outcome"]["multiplicity"].startswith("a single pre-registered hypothesis")


def test_the_frozen_sample_size_is_the_stated_power_calculation():
    pw = SPEC["power"]
    assert P.required_blocks(pw["target_effect_ic"], pw["assumed_sd_ic_per_block"], power=pw["power"]) \
        == pw["required_blocks"] == SPEC["stopping"]["final_analysis_at_blocks"] == 69
    assert P.required_blocks(0.03, 0.12) > 69 > P.required_blocks(0.03, 0.08)


def test_the_module_has_no_path_to_orders_the_gate_or_the_paper_book():
    src = inspect.getsource(P)
    for forbidden in ("execution", "kite", "place_trade", "paper_book", "PaperBook", "unified_scanner"):
        assert forbidden not in src


def _frame(n=300, start="2025-01-01", seed=0, price=100.0, volume=1e6):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": volume}, index=idx)


def _member(*syms, listed="2020-01-01", delisted=None):
    return {s: (listed, delisted) for s in syms}


def test_scores_use_bars_up_to_the_decision_session_only():
    f = {"A": _frame(seed=1), "B": _frame(seed=2), "C": _frame(seed=3)}
    day = f["A"].index[280].strftime("%Y-%m-%d")
    base = P.score_session(f, day, member=_member("A", "B", "C"), spec=SPEC)
    poisoned = {k: v.copy() for k, v in f.items()}
    for v in poisoned.values():
        v.loc[v.index > day, ["open", "high", "low", "close"]] *= 5.0
    assert base and P.score_session(poisoned, day, member=_member("A", "B", "C"), spec=SPEC) == base
    hist = f["A"][f["A"].index <= day]
    assert base["A"] == pytest.approx(hist["close"].iloc[-1] / hist["high"].iloc[-252:].max())


def test_the_universe_filters_are_point_in_time():
    day_i = 280
    f = {"OK": _frame(seed=1), "PENNY": _frame(seed=2, price=5.0), "ILLIQUID": _frame(seed=3, volume=100.0),
         "YOUNG": _frame(seed=4).iloc[-100:], "GONE": _frame(seed=5), "LATER": _frame(seed=6), "NOROW": _frame(seed=7)}
    day = f["OK"].index[day_i].strftime("%Y-%m-%d")
    member = {**_member("OK", "PENNY", "ILLIQUID", "YOUNG"), **_member("GONE", delisted=day),
              **_member("LATER", listed="2099-01-01")}
    assert set(P.score_session(f, day, member=member, spec=SPEC)) == {"OK"}


def test_outcomes_use_the_exact_window_and_exclude_missing_bars():
    f = {"A": _frame(seed=1), "B": _frame(seed=2)}
    days = [d.strftime("%Y-%m-%d") for d in f["A"].index]
    window = days[281:291]
    f["B"] = f["B"].drop(f["B"].index[285])
    rets, missing = P.block_returns(f, days[280], window, ["A", "B", "Z"])
    assert set(rets) == {"A"} and missing == 2
    assert rets["A"] == pytest.approx(f["A"].loc[window[-1], "close"] / f["A"].loc[window[0], "open"] - 1)


def test_decision_sessions_start_after_the_freeze_and_never_overlap():
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-11-02", periods=80)]
    got = P.decision_sessions(sessions, SPEC)
    assert got[0] == "2026-11-23" and all(d > SPEC["frozen_on"] for d in got)
    idx = {d: i for i, d in enumerate(sessions)}
    assert all(idx[b] - idx[a] == 10 for a, b in zip(got, got[1:]))


def test_the_final_analysis_runs_once_at_the_frozen_size():
    rng = np.random.default_rng(0)
    assert P.final_test(list(rng.normal(0.05, 0.1, 40)), SPEC)["status"] == "NOT_DUE"
    strong = P.final_test(list(rng.normal(0.05, 0.1, 100)), SPEC)
    assert strong["blocks"] == 69 and strong["status"] == "PASS"         # extra blocks are never used
    null = [P.final_test(list(np.random.default_rng(s).normal(0.0, 0.1, 69)), SPEC)["status"] for s in range(400)]
    assert null.count("PASS") / len(null) < 0.08                          # ~5% false positives under H0
    assert P.futility([-0.01] * 35, SPEC) and not P.futility([0.01] * 35, SPEC) and not P.futility([-1] * 10, SPEC)
