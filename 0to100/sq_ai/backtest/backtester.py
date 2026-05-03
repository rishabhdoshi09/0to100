"""Backtester – uses the SAME CompositeSignal + RiskManager as live trading.

Single-symbol or list of symbols, daily bars.  Outputs equity curve and
trade ledger; metrics are computed by ``sq_ai.backtest.metrics``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sq_ai.backend.risk_manager import RiskConfig, RiskManager
from sq_ai.backtest.metrics import summary
from sq_ai.signals.composite_signal import (
    CompositeSignal,
    atr,
    regime_from_smas,
    rsi,
    zscore,
)


@dataclass
class Trade:
    symbol: str
    entry_idx: int
    entry_price: float
    qty: int
    stop: float
    target: float
    exit_idx: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[Trade] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        composite: CompositeSignal | None = None,
        risk: RiskManager | None = None,
        initial_equity: float = 1_000_000.0,
        commission_bps: float = 3.0,
        slippage_bps: float = 5.0,
        warmup: int = 50,
    ) -> None:
        self.composite = composite or CompositeSignal()
        self.risk = risk or RiskManager(RiskConfig())
        self.initial_equity = initial_equity
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.warmup = warmup

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _features_at(df: pd.DataFrame, i: int) -> dict:
        window = df.iloc[: i + 1].tail(200)
        close = window["close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        ret = close.pct_change()
        return {
            "sma_20": sma20,
            "sma_50": sma50,
            "volatility_20": float(ret.rolling(20).std(ddof=0).iloc[-1] * math.sqrt(252)),
            "momentum_5d": float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0,
            "volume_trend": float(window["volume"].iloc[-1] / window["volume"].rolling(20).mean().iloc[-1])
                if window["volume"].rolling(20).mean().iloc[-1] else 1.0,
            "rsi": float(rsi(close, 14).iloc[-1]),
            "atr": float(atr(window, 14).iloc[-1]),
            "regime": regime_from_smas(sma20, sma50),    # ← FIX
            "zscore_20": float(zscore(close, 20).iloc[-1]),
            "close": float(close.iloc[-1]),
        }

    def _round_trip_cost(self, price: float, qty: int) -> float:
        bps = (self.commission_bps + self.slippage_bps) / 10_000.0
        return abs(price * qty) * bps

    # ---------------------------------------------------------------- main
    def run_single(self, df: pd.DataFrame, symbol: str = "SYM") -> BacktestResult:
        if len(df) < self.warmup + 5:
            raise ValueError("not enough bars")

        df = df.copy()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        cash = self.initial_equity
        equity_open_today = self.initial_equity
        position: Trade | None = None
        trades: list[Trade] = []
        equity_track: list[float] = []
        last_date_marker = None

        for i in range(self.warmup, len(df)):
            price = float(df["close"].iloc[i])
            today = df.index[i].date() if hasattr(df.index[i], "date") else i
            if today != last_date_marker:
                equity_open_today = cash + (
                    position.qty * price if position else 0
                )
                last_date_marker = today

            features = self._features_at(df, i)
            signal = self.composite.compute(features)

            # ----- EXITS first ---------------------------------------------
            if position is not None:
                hit_stop = price <= position.stop
                hit_tgt = price >= position.target
                age = i - position.entry_idx
                time_stop = age >= self.risk.cfg.time_stop_bars
                if hit_stop or hit_tgt or time_stop:
                    exit_price = position.stop if hit_stop else (
                        position.target if hit_tgt else price
                    )
                    cost = self._round_trip_cost(exit_price, position.qty)
                    pnl = (exit_price - position.entry_price) * position.qty - cost
                    cash += position.qty * exit_price - cost
                    position.exit_idx = i
                    position.exit_price = exit_price
                    position.pnl = pnl
                    position.reason = "stop" if hit_stop else ("target" if hit_tgt else "time")
                    trades.append(position)
                    position = None

            # ----- ENTRIES --------------------------------------------------
            if position is None and signal["direction"] == 1:
                equity_now = cash
                killed, _ = self.risk.kill_switch_triggered(
                    equity_now, equity_open_today, gross_exposure=0.0,
                    equity=equity_now,
                )
                if not killed:
                    atr_v = features["atr"]
                    qty = self.risk.position_size(
                        equity=equity_now, price=price, atr_value=atr_v,
                        confidence=signal["confidence"] / 100.0,
                    )
                    if qty > 0 and qty * price <= cash:
                        stop, tgt = self.risk.stop_and_target(price, atr_v, side=1)
                        cost = self._round_trip_cost(price, qty)
                        cash -= qty * price + cost
                        position = Trade(
                            symbol=symbol, entry_idx=i, entry_price=price,
                            qty=qty, stop=stop, target=tgt,
                        )

            mark = cash + (position.qty * price if position else 0)
            equity_track.append(mark)

        equity = pd.Series(equity_track, index=df.index[self.warmup:])
        returns = equity.pct_change().fillna(0.0)
        stats = summary(returns, [t.pnl for t in trades], self.initial_equity)
        return BacktestResult(equity=equity, returns=returns, trades=trades, stats=stats)
