"""
Performance reporter.

Computes from equity curve + trade journal:
  - CAGR
  - Sharpe ratio (annualized)
  - Max drawdown
  - Win rate
  - Average trade duration
  - Profit factor
  - Full trade journal export (CSV + JSON)
  - Equity curve plot (PNG)
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from logger import get_logger

log = get_logger(__name__)

_TRADING_DAYS_PER_YEAR = 252
_RISK_FREE_RATE = 0.065  # 6.5% — approx Indian 10yr gilt yield


class PerformanceReporter:
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._out = output_dir or settings.log_dir

    def generate_report(
        self,
        equity_curve: List[Dict[str, Any]],
        trade_journal: List[Dict[str, Any]],
        initial_capital: float,
        label: str = "simplequant",
    ) -> Dict[str, Any]:
        """
        Compute all metrics and write output files.
        Returns the metrics dict.
        """
        if not equity_curve:
            log.warning("empty_equity_curve_no_report")
            return {}

        equity_df = self._build_equity_df(equity_curve)
        metrics = self._compute_metrics(equity_df, trade_journal, initial_capital)

        self._write_trade_journal(trade_journal, label)
        self._write_equity_csv(equity_df, label)
        self._write_metrics_json(metrics, label)
        self._plot_equity_curve(equity_df, metrics, label)

        self._print_summary(metrics)
        return metrics

    # ── Metric Computation ─────────────────────────────────────────────────

    def _compute_metrics(
        self,
        equity_df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
    ) -> Dict[str, Any]:
        final_equity = float(equity_df["equity"].iloc[-1])
        total_return = (final_equity - initial_capital) / initial_capital

        cagr = self._cagr(equity_df, initial_capital)
        sharpe = self._sharpe(equity_df)
        max_dd, max_dd_pct = self._max_drawdown(equity_df)
        win_rate, avg_win, avg_loss, profit_factor = self._trade_stats(trades)
        avg_duration = self._avg_trade_duration(trades)
        total_trades = len([t for t in trades if t["action"] in ("BUY", "SELL")])
        realized_pnl = sum(t.get("realized_pnl", 0.0) for t in trades)

        return {
            "initial_capital": round(initial_capital, 2),
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 3),
            "cagr_pct": round(cagr * 100, 3),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_inr": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct * 100, 3),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_trade_duration_days": round(avg_duration, 1),
            "total_realized_pnl": round(realized_pnl, 2),
        }

    def _cagr(self, equity_df: pd.DataFrame, initial_capital: float) -> float:
        if len(equity_df) < 2:
            return 0.0
        start = equity_df.index[0]
        end = equity_df.index[-1]
        years = (end - start).days / 365.25
        if years <= 0 or initial_capital <= 0:
            return 0.0
        final = float(equity_df["equity"].iloc[-1])
        try:
            return (final / initial_capital) ** (1 / years) - 1
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _sharpe(self, equity_df: pd.DataFrame) -> float:
        returns = equity_df["equity"].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        mean_daily = float(returns.mean())
        std_daily = float(returns.std())
        if std_daily == 0:
            return 0.0
        daily_rf = _RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
        excess = mean_daily - daily_rf
        return float((excess / std_daily) * math.sqrt(_TRADING_DAYS_PER_YEAR))

    def _max_drawdown(self, equity_df: pd.DataFrame) -> Tuple[float, float]:
        equity = equity_df["equity"]
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        max_dd = float(drawdown.min())   # most negative value
        peak = float(roll_max[drawdown.idxmin()])
        max_dd_pct = max_dd / peak if peak > 0 else 0.0
        return abs(max_dd), abs(max_dd_pct)

    def _trade_stats(
        self, trades: List[Dict[str, Any]]
    ) -> Tuple[float, float, float, float]:
        sell_trades = [t for t in trades if t.get("action") == "SELL"]
        if not sell_trades:
            return 0.0, 0.0, 0.0, 0.0

        pnls = [t.get("realized_pnl", 0.0) for t in sell_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return win_rate, avg_win, avg_loss, profit_factor

    def _avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """
        Estimate average holding duration by pairing BUY/SELL records
        per symbol in order of timestamp.
        """
        from collections import defaultdict
        entries: Dict[str, List[datetime]] = defaultdict(list)
        durations: List[float] = []

        for trade in sorted(trades, key=lambda x: x.get("timestamp", "")):
            sym = trade.get("symbol", "")
            ts_str = trade.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue

            if trade.get("action") == "BUY":
                entries[sym].append(ts)
            elif trade.get("action") == "SELL" and entries[sym]:
                entry_ts = entries[sym].pop(0)
                dur = (ts - entry_ts).total_seconds() / 86_400  # days
                durations.append(dur)

        return float(np.mean(durations)) if durations else 0.0

    # ── Output Writers ─────────────────────────────────────────────────────

    def _write_trade_journal(
        self, trades: List[Dict[str, Any]], label: str
    ) -> None:
        if not trades:
            return
        path = self._out / f"{label}_trade_journal.csv"
        keys = list(trades[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trades)
        log.info("trade_journal_written", path=str(path))

    def _write_equity_csv(self, equity_df: pd.DataFrame, label: str) -> None:
        path = self._out / f"{label}_equity_curve.csv"
        equity_df.to_csv(path)
        log.info("equity_curve_written", path=str(path))

    def _write_metrics_json(self, metrics: Dict[str, Any], label: str) -> None:
        path = self._out / f"{label}_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log.info("metrics_written", path=str(path))

    def _plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        metrics: Dict[str, Any],
        label: str,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.suptitle(
                f"SimpleQuant AI — {label}\n"
                f"CAGR: {metrics.get('cagr_pct', 0):.1f}%  |  "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
                f"Max DD: {metrics.get('max_drawdown_pct', 0):.1f}%  |  "
                f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%",
                fontsize=11,
            )

            # Equity curve
            ax1 = axes[0]
            ax1.plot(equity_df.index, equity_df["equity"], color="#0d6efd", linewidth=1.5)
            ax1.fill_between(
                equity_df.index,
                equity_df["equity"].cummax(),
                equity_df["equity"],
                alpha=0.25,
                color="red",
                label="Drawdown",
            )
            ax1.set_ylabel("Equity (INR)")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[1]
            roll_max = equity_df["equity"].cummax()
            dd_pct = (equity_df["equity"] - roll_max) / roll_max * 100
            ax2.fill_between(equity_df.index, dd_pct, 0, color="red", alpha=0.5)
            ax2.set_ylabel("Drawdown %")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            path = self._out / f"{label}_equity_curve.png"
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.info("equity_plot_saved", path=str(path))
        except Exception as exc:
            log.warning("plot_failed", error=str(exc))

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="SimpleQuant AI — Performance Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        display_metrics = [
            ("Initial Capital (INR)", f"₹{metrics.get('initial_capital', 0):,.0f}"),
            ("Final Equity (INR)", f"₹{metrics.get('final_equity', 0):,.0f}"),
            ("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"),
            ("CAGR", f"{metrics.get('cagr_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ("Total Trades", str(metrics.get('total_trades', 0))),
            ("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%"),
            ("Avg Win (INR)", f"₹{metrics.get('avg_win_inr', 0):,.0f}"),
            ("Avg Loss (INR)", f"₹{metrics.get('avg_loss_inr', 0):,.0f}"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
            ("Avg Trade Duration", f"{metrics.get('avg_trade_duration_days', 0):.1f} days"),
        ]
        for name, val in display_metrics:
            table.add_row(name, val)

        console.print(table)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_equity_df(equity_curve: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
