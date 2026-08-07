"""Beginner backtest service over QuantTerm's existing realistic Backtester.

This module selects data and translates results. It does not implement another
execution simulator or another portfolio/risk engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class BacktestRequest:
    strategy: str = "Core technical strategy"
    universe: str = "Selected stock"
    symbol: str = "RELIANCE"
    days: int = 756
    capital: float = 100_000.0


@dataclass(frozen=True)
class BacktestSummary:
    request: dict
    symbols_requested: int
    symbols_loaded: int
    starting_amount: float
    ending_amount: float
    profit_loss: float
    return_pct: float
    round_trips: int
    wins: int
    losses: int
    win_rate_pct: float
    largest_fall_pct: float
    trading_costs: float
    nifty_return_pct: float | None
    comparison: str
    trustworthiness: str
    conclusion: str
    equity_curve: tuple[dict, ...] = field(default_factory=tuple)
    trades: tuple[dict, ...] = field(default_factory=tuple)
    scientific_details: dict = field(default_factory=dict)
    walk_forward: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_symbols(universe: str, symbol: str) -> list[str]:
    label = str(universe or "Selected stock")
    if label == "Nifty 50":
        from data.nse_universe import NIFTY50
        return list(dict.fromkeys(str(s).upper() for s in NIFTY50))
    if label == "Current F&O stocks":
        from data.fno_universe import current_fno_universe
        return list(dict.fromkeys(current_fno_universe().symbols))
    clean = str(symbol or "RELIANCE").strip().upper()
    return [clean] if clean else ["RELIANCE"]


def _normalise_history(df: pd.DataFrame | None, days: int) -> pd.DataFrame | None:
    if df is None or len(df) < 30:
        return None
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if any(col not in out.columns for col in needed):
        return None
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna(subset=["open", "high", "low", "close"])
    return out.tail(max(30, int(days)))


def load_historical_data(
    symbols: list[str],
    days: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    from scan.bulk_fetcher import get_cached, prefetch
    prefetch(symbols, period=f"{max(120, int(days))}d", progress=progress)
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _normalise_history(get_cached(symbol), days)
        if frame is not None:
            data[symbol] = frame
    return data


def _benchmark_return(data: dict[str, pd.DataFrame]) -> float | None:
    if not data:
        return None
    start = min(df.index.min() for df in data.values())
    end = max(df.index.max() for df in data.values())
    try:
        nifty = None
        try:
            from data.index_store import get_index_ohlcv
            nifty = get_index_ohlcv("^NSEI")
        except Exception:
            nifty = None
        if nifty is None or nifty.empty:
            from core.regime_engine import _fetch_ohlcv
            nifty = _fetch_ohlcv("^NSEI", period="1y")
        if nifty is None or nifty.empty:
            return None
        close_col = "Close" if "Close" in nifty.columns else "close"
        series = nifty.loc[(nifty.index >= start) & (nifty.index <= end), close_col].dropna()
        if len(series) < 2:
            return None
        return float((series.iloc[-1] / series.iloc[0] - 1) * 100)
    except Exception:
        return None


def interpret_result(
    request: BacktestRequest,
    result: dict[str, Any],
    metrics: dict[str, Any],
    *,
    requested: int,
    loaded: int,
    nifty_return: float | None,
    walk_forward: dict | None = None,
) -> BacktestSummary:
    journal = list(result.get("trade_journal", []))
    sells = [row for row in journal if str(row.get("action", "")).upper() == "SELL"]
    pnls = [float(row.get("realized_pnl", 0.0) or 0.0) for row in sells]
    wins = sum(1 for pnl in pnls if pnl > 0)
    losses = sum(1 for pnl in pnls if pnl <= 0)
    ending = float(result.get("final_equity", request.capital) or request.capital)
    pnl = ending - float(request.capital)
    ret = (pnl / request.capital * 100) if request.capital else 0.0
    fills = list(result.get("fills", []))
    costs = sum(float(row.get("transaction_cost", 0.0) or 0.0) +
                float(row.get("slippage_cost", 0.0) or 0.0) for row in fills)

    if nifty_return is None:
        comparison = "Nifty comparison unavailable"
    elif ret > nifty_return:
        comparison = f"Beat Nifty by {ret - nifty_return:.1f} percentage points"
    else:
        comparison = f"Underperformed Nifty by {nifty_return - ret:.1f} percentage points"

    coverage = loaded / max(requested, 1)
    if loaded == 0:
        trust = "Not usable — no validated history loaded"
    elif len(sells) < 10:
        trust = "Weak evidence — fewer than 10 completed trades"
    elif coverage < 0.8:
        trust = "Incomplete — too many requested stocks lacked data"
    elif walk_forward and float(walk_forward.get("pct_profitable_oos_windows", 0) or 0) < 50:
        trust = "Weak out-of-sample evidence"
    elif walk_forward:
        trust = "Stronger check completed — inspect out-of-sample details"
    else:
        trust = "Indicative only — enable the walk-forward check for stronger evidence"

    max_dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    if ret <= 0:
        conclusion = "This strategy lost money in the selected test. Do not rely on it yet."
    elif nifty_return is not None and ret <= nifty_return:
        conclusion = "It made money but did not beat a simple Nifty comparison."
    elif max_dd >= 25:
        conclusion = "Return was positive, but the fall was too large for a comfortable retail experience."
    elif len(sells) < 10:
        conclusion = "The result is positive but based on too few completed trades."
    else:
        conclusion = "The historical result is promising, but PAPER_AUTO still needs unseen-market evidence."

    scientific = dict(metrics)
    scientific["symbols_requested"] = requested
    scientific["symbols_loaded"] = loaded
    scientific["costs_modelled_inr"] = round(costs, 2)
    return BacktestSummary(
        request=asdict(request),
        symbols_requested=requested,
        symbols_loaded=loaded,
        starting_amount=float(request.capital),
        ending_amount=ending,
        profit_loss=pnl,
        return_pct=ret,
        round_trips=len(sells),
        wins=wins,
        losses=losses,
        win_rate_pct=(wins / len(sells) * 100) if sells else 0.0,
        largest_fall_pct=max_dd,
        trading_costs=costs,
        nifty_return_pct=nifty_return,
        comparison=comparison,
        trustworthiness=trust,
        conclusion=conclusion,
        equity_curve=tuple(result.get("equity_curve", [])),
        trades=tuple(journal),
        scientific_details=scientific,
        walk_forward=walk_forward,
    )


def run_beginner_backtest(
    request: BacktestRequest,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestSummary:
    symbols = resolve_symbols(request.universe, request.symbol)
    data = load_historical_data(symbols, request.days, progress=progress)
    if not data:
        empty = {"final_equity": request.capital, "trade_journal": [], "equity_curve": [], "fills": []}
        return interpret_result(request, empty, {}, requested=len(symbols), loaded=0, nifty_return=None)

    from analytics.reporter import PerformanceReporter
    from backtest.backtester import Backtester
    engine = Backtester(historical_data=data, initial_capital=request.capital, use_llm=False)
    result = engine.run()
    curve = PerformanceReporter._build_equity_df(result.get("equity_curve", []))
    metrics = (PerformanceReporter()._compute_metrics(
        curve, result.get("trade_journal", []), request.capital
    ) if not curve.empty else {})

    wf = None
    if request.strategy == "Core technical + walk-forward reliability check":
        try:
            from backtest.walk_forward import WalkForwardValidator
            wf = WalkForwardValidator(initial_capital=request.capital).run(data) or None
        except Exception as exc:
            wf = {"error": str(exc)}

    return interpret_result(
        request,
        result,
        metrics,
        requested=len(symbols),
        loaded=len(data),
        nifty_return=_benchmark_return(data),
        walk_forward=wf,
    )
