"""Plain-language projection over a real Backtester result."""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping


@dataclass(frozen=True)
class BacktestSummary:
    trades: int
    closed_trades: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    expectancy_inr: float | None
    profit_factor: float | None
    verdict: str
    warning: str


def summarize_backtest(result: Mapping[str, Any]) -> BacktestSummary:
    """Summarize only values contained in the real engine result.

    The event-driven engine does not itself run Reality Check/DSR/FDR evidence
    tests, so this projection never upgrades an operational backtest to PASS.
    """
    initial = float(result.get("initial_capital") or 0.0)
    final = float(result.get("final_equity") or initial)
    journal = list(result.get("trade_journal") or ())
    equity = list(result.get("equity_curve") or ())
    sells = [row for row in journal if str(row.get("action", "")).upper() == "SELL"]
    pnls = [float(row.get("realized_pnl") or 0.0) for row in sells]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]

    values = [float(row.get("equity") or 0.0) for row in equity if row.get("equity") is not None]
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = inf
    else:
        profit_factor = None

    closed = len(sells)
    expectancy = sum(pnls) / closed if closed else None
    total_return = ((final / initial) - 1) * 100 if initial > 0 else 0.0
    win_rate = len(wins) / closed * 100 if closed else 0.0

    if closed == 0:
        warning = "No completed trades were produced; no performance conclusion is possible."
    elif closed < 30:
        warning = "The completed-trade sample is small and highly uncertain."
    else:
        warning = (
            "This is an operational backtest, not a scientific PASS. Use the Research evidence "
            "suite for confidence intervals, multiple-testing checks and out-of-sample validation."
        )

    return BacktestSummary(
        trades=len(journal),
        closed_trades=closed,
        initial_capital=initial,
        final_equity=final,
        total_return_pct=total_return,
        max_drawdown_pct=max_drawdown * 100,
        win_rate_pct=win_rate,
        expectancy_inr=expectancy,
        profit_factor=profit_factor,
        verdict="INCONCLUSIVE",
        warning=warning,
    )
