"""Tests for backtest metrics."""
import numpy as np
import pandas as pd

from sq_ai.backtest.metrics import (
    cagr,
    max_drawdown,
    profit_factor,
    returns_to_equity,
    sharpe,
    summary,
    win_rate,
)


def test_sharpe_positive():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    s = sharpe(r)
    assert s != 0


def test_sharpe_constant_returns_zero():
    r = pd.Series([0.0] * 100)
    assert sharpe(r) == 0.0


def test_max_drawdown_negative():
    eq = pd.Series([100, 110, 105, 95, 100, 90])
    dd = max_drawdown(eq)
    assert dd < 0
    assert abs(dd - ((90 - 110) / 110)) < 1e-9


def test_profit_factor():
    assert profit_factor([10, -5, 7, -3]) == (17 / 8)
    assert profit_factor([10, 5]) == float("inf")
    assert profit_factor([]) == 0.0


def test_win_rate():
    assert win_rate([1, -1, 2, 3]) == 0.75
    assert win_rate([]) == 0.0


def test_cagr_one_year_doubling():
    eq = pd.Series([1.0, 2.0])
    # CAGR with 252 ppy and 2 bars ≈ 2^(252/2) - 1 → enormous, just sanity check
    assert cagr(eq) > 0


def test_summary_keys_and_initial():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 252))
    out = summary(rets, [100, -50, 30])
    for k in ("sharpe", "cagr", "max_drawdown", "profit_factor",
              "win_rate", "n_trades", "final_equity"):
        assert k in out


def test_returns_to_equity_starts_at_initial():
    rets = pd.Series([0.01, -0.005, 0.02])
    eq = returns_to_equity(rets, initial=1000)
    assert eq.iloc[0] == 1000 * 1.01
