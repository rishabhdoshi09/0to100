"""Backtest performance metrics."""
from __future__ import annotations

import math

import pandas as pd


def returns_to_equity(returns: pd.Series, initial: float = 1_000_000.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def sharpe(returns: pd.Series, periods_per_year: int = 252,
           rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    sd = excess.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def profit_factor(trade_pnls: list[float] | pd.Series) -> float:
    s = pd.Series(trade_pnls).dropna()
    if s.empty:
        return 0.0
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(trade_pnls: list[float] | pd.Series) -> float:
    s = pd.Series(trade_pnls).dropna()
    if s.empty:
        return 0.0
    return float((s > 0).sum() / len(s))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def summary(returns: pd.Series, trade_pnls: list[float],
            initial: float = 1_000_000.0) -> dict:
    eq = returns_to_equity(returns, initial)
    return {
        "sharpe": round(sharpe(returns), 3),
        "cagr": round(cagr(eq), 4),
        "max_drawdown": round(max_drawdown(eq), 4),
        "profit_factor": round(profit_factor(trade_pnls), 3),
        "win_rate": round(win_rate(trade_pnls), 3),
        "n_trades": len(trade_pnls),
        "final_equity": float(eq.iloc[-1]) if not eq.empty else initial,
    }
