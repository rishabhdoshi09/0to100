"""
Model decay monitor — tracks live vs backtest performance degradation.

Records completed live trades to SQLite (live_trades table).
Computes rolling 30-day Sharpe per strategy and compares to expected
backtest Sharpe. Sends Telegram alert if ratio drops below threshold.

Usage
-----
from monitoring.decay_monitor import ModelDecayMonitor

mon = ModelDecayMonitor()

# Log a completed trade
mon.log_trade(
    symbol="RELIANCE",
    strategy="lgbm",
    entry_time=datetime(...),
    exit_time=datetime(...),
    entry_price=2450.0,
    exit_price=2490.0,
    quantity=10,
)

# Check for decay and alert
report = mon.compute_decay_metrics()
mon.check_and_alert(report)
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/screener_cache.db")
_DECAY_THRESHOLD = 0.7       # alert if live_sharpe / backtest_sharpe < 0.7
_ROLLING_WINDOW_DAYS = 30
_RISK_FREE_RATE_DAILY = 0.06 / 252   # ~6% annualised


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT    NOT NULL,
            strategy      TEXT    NOT NULL DEFAULT 'ensemble',
            entry_time    REAL    NOT NULL,
            exit_time     REAL    NOT NULL,
            entry_price   REAL    NOT NULL,
            exit_price    REAL    NOT NULL,
            quantity      REAL    NOT NULL DEFAULT 1,
            pnl           REAL    NOT NULL,
            return_pct    REAL    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_benchmarks (
            strategy      TEXT    PRIMARY KEY,
            sharpe        REAL    NOT NULL,
            updated_at    REAL    NOT NULL
        )
    """)
    conn.commit()
    return conn


class ModelDecayMonitor:
    """
    Monitor for live vs backtest performance decay.

    Log trades as they complete, then call compute_decay_metrics() to
    compare rolling live Sharpe against stored backtest benchmarks.
    """

    # ── Trade logging ──────────────────────────────────────────────────────

    def log_trade(
        self,
        symbol: str,
        strategy: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        quantity: float = 1.0,
    ) -> None:
        pnl = (exit_price - entry_price) * quantity
        return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

        with _connect() as conn:
            conn.execute(
                """INSERT INTO live_trades
                   (symbol, strategy, entry_time, exit_time,
                    entry_price, exit_price, quantity, pnl, return_pct)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    symbol.upper(),
                    strategy,
                    entry_time.timestamp(),
                    exit_time.timestamp(),
                    entry_price,
                    exit_price,
                    quantity,
                    round(pnl, 4),
                    round(return_pct, 6),
                ),
            )
            conn.commit()
        log.info("trade_logged", symbol=symbol, strategy=strategy,
                 pnl=round(pnl, 2))

    # ── Benchmarks ─────────────────────────────────────────────────────────

    def set_backtest_benchmark(self, strategy: str, sharpe: float) -> None:
        """Store expected Sharpe from a backtest run for later comparison."""
        with _connect() as conn:
            conn.execute(
                """INSERT INTO backtest_benchmarks (strategy, sharpe, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(strategy) DO UPDATE SET
                       sharpe=excluded.sharpe, updated_at=excluded.updated_at""",
                (strategy, round(sharpe, 4), time.time()),
            )
            conn.commit()
        log.info("backtest_benchmark_set", strategy=strategy, sharpe=round(sharpe, 4))

    # ── Metrics ────────────────────────────────────────────────────────────

    def compute_decay_metrics(self) -> pd.DataFrame:
        """
        Compute rolling 30-day live Sharpe per strategy.
        Returns DataFrame with columns:
          strategy, trade_count, live_sharpe, backtest_sharpe, decay_ratio, alert
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_ROLLING_WINDOW_DAYS)).timestamp()

        with _connect() as conn:
            trades = pd.read_sql_query(
                "SELECT strategy, return_pct, exit_time FROM live_trades WHERE exit_time > ?",
                conn,
                params=(cutoff,),
            )
            benchmarks = pd.read_sql_query(
                "SELECT strategy, sharpe FROM backtest_benchmarks",
                conn,
            )

        if trades.empty:
            log.info("decay_monitor_no_recent_trades")
            return pd.DataFrame(
                columns=["strategy", "trade_count", "live_sharpe",
                         "backtest_sharpe", "decay_ratio", "alert"]
            )

        bench_map: Dict[str, float] = (
            dict(zip(benchmarks["strategy"], benchmarks["sharpe"]))
            if not benchmarks.empty else {}
        )

        rows = []
        for strategy, grp in trades.groupby("strategy"):
            returns = grp["return_pct"].values
            live_sharpe = self._sharpe(returns)
            bt_sharpe = bench_map.get(strategy, 1.0)
            decay_ratio = live_sharpe / bt_sharpe if bt_sharpe != 0 else None
            alert = decay_ratio is not None and decay_ratio < _DECAY_THRESHOLD

            rows.append({
                "strategy":       strategy,
                "trade_count":    len(returns),
                "live_sharpe":    round(live_sharpe, 3),
                "backtest_sharpe": round(bt_sharpe, 3),
                "decay_ratio":    round(decay_ratio, 3) if decay_ratio is not None else None,
                "alert":          alert,
            })

        df = pd.DataFrame(rows)
        log.info("decay_metrics_computed", strategies=len(df),
                 alerts=int(df["alert"].sum()))
        return df

    def check_and_alert(
        self,
        metrics: Optional[pd.DataFrame] = None,
        threshold: float = _DECAY_THRESHOLD,
    ) -> List[str]:
        """
        Send Telegram alerts for strategies whose decay_ratio < threshold.
        Returns list of alerted strategy names.
        """
        if metrics is None:
            metrics = self.compute_decay_metrics()

        if metrics.empty:
            return []

        alerted = []
        for _, row in metrics[metrics["alert"]].iterrows():
            strategy = row["strategy"]
            msg = (
                f"⚠️ Model Decay Alert — {strategy}\n"
                f"Live Sharpe: {row['live_sharpe']:.3f} | "
                f"Backtest Sharpe: {row['backtest_sharpe']:.3f}\n"
                f"Decay ratio: {row['decay_ratio']:.2f} (threshold: {threshold})\n"
                f"Trades in window: {row['trade_count']}\n"
                f"Consider disabling: python main.py strategy disable --name {strategy}"
            )
            log.warning("model_decay_alert", strategy=strategy,
                        decay_ratio=row["decay_ratio"])
            self._send_telegram(msg)
            alerted.append(strategy)

        return alerted

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - _RISK_FREE_RATE_DAILY
        std = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @staticmethod
    def _send_telegram(message: str) -> None:
        try:
            from config import settings
            import requests
            token = settings.telegram_bot_token
            chat_id = settings.telegram_chat_id
            if not token or not chat_id:
                log.debug("telegram_not_configured_skipping_alert")
                return
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=10,
            )
        except Exception as exc:
            log.warning("telegram_alert_failed", error=str(exc))
