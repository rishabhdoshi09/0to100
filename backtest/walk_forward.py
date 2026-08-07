"""
Walk-Forward Validator.

Runs rolling in-sample parameter optimisation followed by out-of-sample
validation to measure overfitting and robustness of the technical strategy.

Invariants:
  - OOS slice is NEVER passed to the IS Backtester under any circumstances.
  - All slicing is by integer position (iloc) on a shared sorted date index.
  - No LLM calls — use_llm=False throughout (fast, free, deterministic).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics.reporter import PerformanceReporter
from backtest.backtester import Backtester
from config import settings
from logger import get_logger

log = get_logger(__name__)


@dataclass
class _WindowResult:
    window_idx: int
    is_start: Any
    is_end: Any
    oos_start: Any
    oos_end: Any
    best_params: Dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_total_return: float
    oos_max_drawdown_pct: float
    oos_win_rate: float
    oos_equity_curve: List[Dict[str, Any]] = field(default_factory=list)


class WalkForwardValidator:
    """
    Rolling walk-forward optimisation over a historical dataset.

    Parameters
    ----------
    in_sample_days : int
        Number of trading days in each IS window (default 252 ≈ 1 year).
    out_of_sample_days : int
        Number of trading days in each OOS window (default 63 ≈ 3 months).
    initial_capital : float
        Starting capital for every sub-backtest.
    """

    _PARAM_GRID: Dict[str, List[Any]] = {
        "zscore_entry":   [-2.0, -1.5, -1.0],
        "zscore_exit":    [1.0,  1.5,  2.0],
        "rsi_entry_max":  [45,   55,   65],
        "rsi_exit_min":   [60,   65,   70],
    }

    def __init__(
        self,
        in_sample_days: int = settings.walkforward_is_days,
        out_of_sample_days: int = settings.walkforward_oos_days,
        initial_capital: float = settings.backtest_initial_capital,
    ) -> None:
        self._is_days = in_sample_days
        self._oos_days = out_of_sample_days
        self._capital = initial_capital
        self._reporter = PerformanceReporter()

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute the full walk-forward validation.

        Parameters
        ----------
        historical_data : dict[str, pd.DataFrame]
            Symbol → OHLCV DataFrame (same format used by Backtester).

        Returns
        -------
        Summary dict with OOS metrics, overfitting ratio, and full equity curve.
        """
        if not historical_data:
            log.warning("walk_forward_empty_data")
            return {}

        # Build a shared sorted date index from all symbols
        all_dates = sorted(set.union(*[set(df.index) for df in historical_data.values()]))
        n = len(all_dates)

        log.info(
            "walk_forward_start",
            total_bars=n,
            is_days=self._is_days,
            oos_days=self._oos_days,
            symbols=list(historical_data.keys()),
        )

        if n < self._is_days + self._oos_days:
            log.warning(
                "walk_forward_insufficient_data",
                required=self._is_days + self._oos_days,
                available=n,
            )
            return {}

        windows: List[_WindowResult] = []
        window_idx = 0
        is_start_idx = 0

        while True:
            is_end_idx = is_start_idx + self._is_days
            oos_end_idx = is_end_idx + self._oos_days

            # Stop when we no longer have a full IS window
            if is_end_idx > n:
                break
            # If OOS window goes past the end, clamp it
            oos_end_idx = min(oos_end_idx, n)
            if is_end_idx >= oos_end_idx:
                break

            is_dates = all_dates[is_start_idx:is_end_idx]
            oos_dates = all_dates[is_end_idx:oos_end_idx]

            # Strict IS slice — iloc ensures no OOS data leaks in
            is_data = {
                sym: df.loc[df.index.isin(is_dates)]
                for sym, df in historical_data.items()
                if not df.loc[df.index.isin(is_dates)].empty
            }
            # Strict OOS slice — never touches is_data
            oos_data = {
                sym: df.loc[df.index.isin(oos_dates)]
                for sym, df in historical_data.items()
                if not df.loc[df.index.isin(oos_dates)].empty
            }

            if not is_data or not oos_data:
                log.warning("walk_forward_window_empty_data", window=window_idx)
                is_start_idx += self._oos_days
                window_idx += 1
                continue

            log.info(
                "walk_forward_window",
                window=window_idx,
                is_start=str(is_dates[0]),
                is_end=str(is_dates[-1]),
                oos_start=str(oos_dates[0]),
                oos_end=str(oos_dates[-1]),
            )

            # ── IS grid search ────────────────────────────────────────────
            best_params, best_is_sharpe = self._grid_search(is_data)
            log.info(
                "walk_forward_best_is_params",
                window=window_idx,
                params=best_params,
                is_sharpe=round(best_is_sharpe, 3),
            )

            # ── OOS evaluation with best IS params ────────────────────────
            oos_result = self._run_with_params(oos_data, best_params)
            oos_metrics = oos_result["metrics"]
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            oos_total_return = oos_metrics.get("total_return_pct", 0.0)
            oos_max_dd = oos_metrics.get("max_drawdown_pct", 0.0)
            oos_win_rate = oos_metrics.get("win_rate_pct", 0.0)

            log.info(
                "walk_forward_oos_result",
                window=window_idx,
                oos_sharpe=round(oos_sharpe, 3),
                oos_return=round(oos_total_return, 2),
                oos_max_dd=round(oos_max_dd, 2),
            )

            windows.append(_WindowResult(
                window_idx=window_idx,
                is_start=is_dates[0],
                is_end=is_dates[-1],
                oos_start=oos_dates[0],
                oos_end=oos_dates[-1],
                best_params=best_params,
                is_sharpe=best_is_sharpe,
                oos_sharpe=oos_sharpe,
                oos_total_return=oos_total_return,
                oos_max_drawdown_pct=oos_max_dd,
                oos_win_rate=oos_win_rate,
                oos_equity_curve=oos_result["equity_curve"],
            ))

            # Walk forward by one OOS window
            is_start_idx += self._oos_days
            window_idx += 1

        if not windows:
            log.warning("walk_forward_no_windows_completed")
            return {}

        summary = self._aggregate(windows)
        log.info(
            "walk_forward_complete",
            windows=len(windows),
            mean_oos_sharpe=round(summary["mean_oos_sharpe"], 3),
            is_oos_ratio=round(summary["is_oos_sharpe_ratio"], 3),
        )
        return summary

    # ── Grid search ────────────────────────────────────────────────────────

    def _grid_search(
        self, is_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Any], float]:
        """Exhaustive grid search over _PARAM_GRID on the IS window."""
        keys = list(self._PARAM_GRID.keys())
        values = list(self._PARAM_GRID.values())
        combos = list(itertools.product(*values))

        best_params: Dict[str, Any] = {}
        best_sharpe: float = float("-inf")

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                result = self._run_with_params(is_data, params)
                sharpe = result["metrics"].get("sharpe_ratio", float("-inf"))
            except Exception as exc:
                log.warning("grid_search_combo_failed", params=params, error=str(exc))
                sharpe = float("-inf")

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        if not best_params:
            # Fallback to default params if all combos failed
            best_params = {k: v[0] for k, v in self._PARAM_GRID.items()}

        return best_params, best_sharpe

    # ── Run a backtester with overridden technical signal params ──────────

    def _run_with_params(
        self,
        data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Instantiate a Backtester, monkey-patch _technical_signal with the
        given param set, run it, and return metrics + equity curve.
        """
        zscore_entry = params.get("zscore_entry", -1.5)
        zscore_exit = params.get("zscore_exit", 1.5)
        rsi_entry_max = params.get("rsi_entry_max", 55)
        rsi_exit_min = params.get("rsi_exit_min", 65)

        bt = Backtester(
            historical_data=data,
            initial_capital=self._capital,
            use_llm=False,
        )

        # Override the technical signal logic via subclass-style patch
        from llm.signal_validator import TradingSignal

        def _patched_technical_signal(
            _self,
            symbol: str,
            indicators: Dict[str, Any],
            bar_close: float,
        ) -> TradingSignal:
            zscore = indicators.get("zscore_20")
            rsi = indicators.get("rsi_14")
            has_position = _self._portfolio.has_position(symbol)

            action = "HOLD"
            confidence = 0.5
            reasoning = "no_clear_signal"

            if zscore is not None and rsi is not None:
                if zscore < zscore_entry and rsi < rsi_entry_max and not has_position:
                    action = "BUY"
                    confidence = min(0.92, 0.60 + abs(zscore) * 0.08)
                    reasoning = f"wf_buy: zscore={zscore:.2f}, rsi={rsi:.1f}"
                elif has_position and (zscore > zscore_exit or rsi > rsi_exit_min):
                    action = "SELL"
                    confidence = min(0.92, 0.60 + abs(zscore) * 0.08)
                    reasoning = f"wf_exit: zscore={zscore:.2f}, rsi={rsi:.1f}"

            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                time_horizon="swing",
                position_size=settings.max_position_size_pct,
                reasoning=reasoning,
                risk_level="medium",
            )

        # Bind the patched method to the backtester instance
        import types
        bt._technical_signal = types.MethodType(_patched_technical_signal, bt)

        result = bt.run()

        metrics = self._compute_metrics_only(
            equity_curve=result["equity_curve"],
            trade_journal=result["trade_journal"],
            initial_capital=result["initial_capital"],
        )

        return {
            "metrics": metrics,
            "equity_curve": result["equity_curve"],
            "trade_journal": result["trade_journal"],
            "final_equity": result["final_equity"],
        }

    def _compute_metrics_only(
        self,
        equity_curve: List[Dict[str, Any]],
        trade_journal: List[Dict[str, Any]],
        initial_capital: float,
    ) -> Dict[str, Any]:
        """
        Compute performance metrics without writing any output files.
        Reuses PerformanceReporter's private computation methods.
        """
        if not equity_curve:
            return {
                "sharpe_ratio": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "cagr_pct": 0.0,
            }
        # Access private helpers — acceptable because we own both modules
        equity_df = PerformanceReporter._build_equity_df(equity_curve)
        return self._reporter._compute_metrics(equity_df, trade_journal, initial_capital)

    # ── Aggregation ────────────────────────────────────────────────────────

    def _aggregate(self, windows: List[_WindowResult]) -> Dict[str, Any]:
        all_oos_sharpes = [w.oos_sharpe for w in windows]
        all_is_sharpes = [w.is_sharpe for w in windows]
        mean_oos = float(sum(all_oos_sharpes) / len(all_oos_sharpes))
        mean_is = float(sum(all_is_sharpes) / len(all_is_sharpes))

        profitable = sum(1 for s in all_oos_sharpes if s > 0)
        pct_profitable = profitable / len(all_oos_sharpes) * 100

        # IS/OOS ratio: > 2.0 suggests overfitting
        if mean_oos != 0:
            is_oos_ratio = abs(mean_is / mean_oos)
        elif mean_is > 0:
            is_oos_ratio = float("inf")
        else:
            is_oos_ratio = 1.0

        # Which parameter set won IS optimisation most often
        from collections import Counter
        param_freq: Counter = Counter()
        for w in windows:
            key = tuple(sorted(w.best_params.items()))
            param_freq[key] += 1
        best_params_frequency = {
            str(dict(k)): v for k, v in param_freq.most_common()
        }

        # Concatenate OOS equity curves in chronological order
        full_curve: List[Dict[str, Any]] = []
        for w in sorted(windows, key=lambda x: x.window_idx):
            full_curve.extend(w.oos_equity_curve)

        window_details = [
            {
                "window": w.window_idx,
                "is_start": str(w.is_start),
                "is_end": str(w.is_end),
                "oos_start": str(w.oos_start),
                "oos_end": str(w.oos_end),
                "best_params": w.best_params,
                "is_sharpe": round(w.is_sharpe, 3),
                "oos_sharpe": round(w.oos_sharpe, 3),
                "oos_total_return_pct": round(w.oos_total_return, 3),
                "oos_max_drawdown_pct": round(w.oos_max_drawdown_pct, 3),
                "oos_win_rate_pct": round(w.oos_win_rate, 2),
            }
            for w in windows
        ]

        return {
            "all_oos_sharpes": all_oos_sharpes,
            "mean_oos_sharpe": round(mean_oos, 3),
            "mean_is_sharpe": round(mean_is, 3),
            "pct_profitable_oos_windows": round(pct_profitable, 1),
            "is_oos_sharpe_ratio": round(is_oos_ratio, 3),
            "best_params_frequency": best_params_frequency,
            "full_oos_equity_curve": full_curve,
            "total_windows": len(windows),
            "window_details": window_details,
        }
