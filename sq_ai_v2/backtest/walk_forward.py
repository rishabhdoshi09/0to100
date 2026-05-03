"""
Walk-forward validation and automated retraining.

Algorithm:
  For each fold:
    1. Train on [fold_start, fold_end - test_months].
    2. Test on [fold_end - test_months, fold_end].
    3. Record performance metrics.
  Report distribution of out-of-sample Sharpe ratios.

The walk_forward.py also handles the continuous retraining loop:
  • Every N days, trigger TrainingPipeline.run() on the latest 3-year window.
  • Alert if OOS Sharpe < threshold or win rate < 45%.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from config.settings import settings
from backtest.engine import BacktestEngine
from models.train_pipeline import TrainingPipeline

warnings.filterwarnings("ignore")

# dateutil may not be installed; fall back to timedelta approximation
try:
    from dateutil.relativedelta import relativedelta
    _HAS_DATEUTIL = True
except ImportError:
    _HAS_DATEUTIL = False


def _add_months(d: date, months: int) -> date:
    if _HAS_DATEUTIL:
        return d + relativedelta(months=months)
    return d + timedelta(days=months * 30)


class WalkForwardValidator:
    """
    Runs rolling walk-forward validation across all historical data.
    """

    def __init__(
        self,
        train_years: int = settings.walk_forward_train_years,
        test_months: int = settings.walk_forward_test_months,
    ) -> None:
        self.train_years = train_years
        self.test_months = test_months

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = settings.backtest_initial_capital,
    ) -> Dict[str, Any]:
        """
        Execute walk-forward across the full dataset.
        Returns dict with fold results and aggregate statistics.
        """
        # Find date range
        all_dates = sorted(
            set.union(*[set(df.index) for df in data.values()])
        )
        if not all_dates:
            return {"error": "no data"}

        start = all_dates[0].date() if hasattr(all_dates[0], "date") else all_dates[0]
        end = all_dates[-1].date() if hasattr(all_dates[-1], "date") else all_dates[-1]

        # Build fold boundaries
        folds = self._build_folds(start, end)
        logger.info(f"Walk-forward: {len(folds)} folds ({start} → {end})")

        fold_results: List[Dict] = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(
                f"Fold {fold_idx+1}/{len(folds)}: "
                f"train={train_start}→{train_end} test={test_start}→{test_end}"
            )

            # 1. Slice data for this fold
            train_data = self._slice_data(data, train_start, train_end)
            test_data = self._slice_data(data, test_start, test_end)

            if not train_data or not test_data:
                logger.warning(f"Fold {fold_idx+1}: insufficient data")
                continue

            # 2. Train on train window
            pipeline = TrainingPipeline()
            try:
                train_metrics = pipeline.run(train_data)
            except Exception as exc:
                logger.error(f"Fold {fold_idx+1} training failed: {exc}")
                train_metrics = {}

            # 3. Backtest on test window
            engine = BacktestEngine(test_data, initial_capital=initial_capital)
            try:
                bt_results = engine.run()
                test_stats = bt_results.get("stats", {})
            except Exception as exc:
                logger.error(f"Fold {fold_idx+1} backtest failed: {exc}")
                test_stats = {}

            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_metrics": train_metrics,
                    "test_stats": test_stats,
                }
            )

            # 4. Alert on poor performance
            if test_stats:
                self._check_performance_alerts(fold_idx + 1, test_stats)

        return self._aggregate_results(fold_results)

    # ── Fold building ─────────────────────────────────────────────────────

    def _build_folds(
        self, start: date, end: date
    ) -> List[Tuple[date, date, date, date]]:
        folds = []
        train_end = start + timedelta(days=self.train_years * 365)

        while train_end < end:
            train_start = train_end - timedelta(days=self.train_years * 365)
            test_start = train_end
            test_end = _add_months(test_start, self.test_months)

            if test_end > end:
                test_end = end

            folds.append((train_start, train_end, test_start, test_end))
            train_end = _add_months(train_end, self.test_months)

        return folds

    # ── Data slicing ──────────────────────────────────────────────────────

    @staticmethod
    def _slice_data(
        data: Dict[str, pd.DataFrame],
        start: date,
        end: date,
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for sym, df in data.items():
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            sliced = df[mask]
            if len(sliced) >= 60:
                result[sym] = sliced
        return result

    # ── Performance alerts ─────────────────────────────────────────────────

    @staticmethod
    def _check_performance_alerts(fold: int, stats: Dict) -> None:
        sharpe = stats.get("sharpe", 0)
        win_rate = stats.get("win_rate", 0)

        if sharpe < -0.5:
            logger.warning(f"Fold {fold}: ALERT — Sharpe={sharpe:.2f} < -0.5")
        if win_rate < 0.45 and stats.get("n_trades", 0) >= 10:
            logger.warning(f"Fold {fold}: ALERT — Win rate={win_rate:.1%} < 45%")

    # ── Aggregate metrics ─────────────────────────────────────────────────

    @staticmethod
    def _aggregate_results(fold_results: List[Dict]) -> Dict:
        sharpes = [f["test_stats"].get("sharpe", 0) for f in fold_results if f.get("test_stats")]
        win_rates = [f["test_stats"].get("win_rate", 0) for f in fold_results if f.get("test_stats")]
        max_dds = [f["test_stats"].get("max_drawdown", 0) for f in fold_results if f.get("test_stats")]
        returns = [f["test_stats"].get("total_return", 0) for f in fold_results if f.get("test_stats")]

        aggregate = {
            "n_folds": len(fold_results),
            "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0,
            "median_sharpe": float(np.median(sharpes)) if sharpes else 0,
            "pct_positive_sharpe": float(np.mean([s > 0 for s in sharpes])) if sharpes else 0,
            "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0,
            "mean_max_drawdown": float(np.mean(max_dds)) if max_dds else 0,
            "mean_return": float(np.mean(returns)) if returns else 0,
            "folds": fold_results,
        }

        logger.info(
            f"Walk-forward aggregate: "
            f"Sharpe={aggregate['mean_sharpe']:.2f} "
            f"WinRate={aggregate['mean_win_rate']:.1%} "
            f"MaxDD={aggregate['mean_max_drawdown']:.1%}"
        )
        return aggregate
