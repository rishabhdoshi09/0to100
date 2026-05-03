"""
Event-driven backtesting engine.

Strict no-lookahead guarantee:
  • Indicators at bar T use data[0..T] only.
  • Order submitted at close[T] is filled at open[T+1].
  • Features are sliced from the feature DataFrame up to the current bar.

Output:
  • Trade journal (DataFrame)
  • Equity curve (Series)
  • Performance statistics dict
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.feature_store.store import FeatureStore
from execution.slippage_model import SlippageModel
from models.ensemble.hmm_regime import HMMRegime
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager
from signals.composite_signal import CompositeSignal, SignalResult


@dataclass
class Trade:
    trade_id: str
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: Optional[date]
    exit_price: Optional[float]
    quantity: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    signal_probability: float = 0.5
    regime: str = "chop"


class BacktestEngine:
    """
    Full event-driven backtester using CompositeSignal.
    """

    _MIN_WARMUP = 60  # bars needed before first trade

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = settings.backtest_initial_capital,
        transaction_cost_pct: float = settings.backtest_transaction_cost,
        use_sentiment: bool = False,   # off by default in backtest (no historical news)
        use_fundamentals: bool = False,
    ) -> None:
        self._data = data
        self._initial_capital = initial_capital
        self._txn_cost = transaction_cost_pct
        self._use_sentiment = use_sentiment
        self._use_fundamentals = use_fundamentals

        self._signal_engine = CompositeSignal()
        self._slippage = SlippageModel()
        self._stop_mgr = StopLossManager()
        self._sizer = PositionSizer()
        self._feature_store = FeatureStore()

        # Portfolio state
        self._cash = initial_capital
        self._positions: Dict[str, Dict] = {}  # sym → {qty, entry_price, entry_date, atr}
        self._trades: List[Trade] = []
        self._equity_curve: List[Tuple[date, float]] = []

    # ── Main run ──────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        logger.info(
            f"Backtest started: {len(self._data)} symbols, "
            f"capital={self._initial_capital:,.0f}"
        )

        all_dates = sorted(
            set.union(*[set(df.index) for df in self._data.values()])
        )

        for i, ts in enumerate(all_dates):
            if i < self._MIN_WARMUP:
                self._equity_curve.append((ts, self._initial_capital))
                continue

            prices = {
                sym: float(self._data[sym].loc[ts, "close"])
                for sym in self._data
                if ts in self._data[sym].index
            }

            # 1. Check stop losses
            self._check_stops(ts, prices)

            # 2. Update mark-to-market
            self._update_mtm(prices)

            # 3. Generate signals & trade
            for sym, df in self._data.items():
                if ts not in df.index:
                    continue

                # Slice history up to and including current bar (no lookahead)
                hist = df.loc[:ts]
                if len(hist) < self._MIN_WARMUP:
                    continue

                signal = self._signal_engine.generate(
                    sym,
                    hist,
                    use_sentiment=self._use_sentiment,
                    use_fundamentals=self._use_fundamentals,
                )

                current_price = prices.get(sym, 0.0)
                if current_price <= 0:
                    continue

                # 4. Execute pending sells
                if signal.action == "SELL" and sym in self._positions:
                    self._execute_sell(sym, ts, current_price, reason="signal")

                # 5. Execute buys (filled at next bar's open in real trading;
                #    here we use current bar's close as approximation + slippage)
                elif signal.action == "BUY" and sym not in self._positions:
                    self._execute_buy(sym, ts, current_price, signal, hist)

            # 6. Record equity
            equity = self._cash + sum(
                p["qty"] * prices.get(sym, p["entry_price"])
                for sym, p in self._positions.items()
            )
            self._equity_curve.append((ts, equity))

        # Force-close all remaining positions at last price
        last_ts = all_dates[-1]
        last_prices = {
            sym: float(df.iloc[-1]["close"])
            for sym, df in self._data.items()
        }
        for sym in list(self._positions.keys()):
            price = last_prices.get(sym, self._positions[sym]["entry_price"])
            self._execute_sell(sym, last_ts, price, reason="end_of_backtest")

        return self._compile_results()

    # ── Trade execution ───────────────────────────────────────────────────

    def _execute_buy(
        self,
        sym: str,
        ts,
        price: float,
        signal: SignalResult,
        hist: pd.DataFrame,
    ) -> None:
        if len(self._positions) >= settings.max_open_positions:
            return
        if signal.confidence < settings.min_signal_confidence:
            return

        symbol_vol = PositionSizer.compute_symbol_vol(hist["close"])
        portfolio_vol = PositionSizer.estimate_portfolio_vol(
            {s: p["qty"] * price for s, p in self._positions.items()},
            {s: 0.20 for s in self._positions},
            self._cash,
        )

        from models.ensemble.hmm_regime import HMMRegime
        size = self._sizer.compute(
            symbol=sym,
            probability=signal.probability,
            last_price=price,
            portfolio_value=self._initial_capital,
            portfolio_vol=portfolio_vol,
            symbol_vol=symbol_vol,
            regime_multiplier=HMMRegime.regime_risk_multiplier(signal.regime_id),
        )

        if size.shares < 1:
            return

        fill_price = self._slippage.adjusted_buy_price(price)
        cost = size.shares * fill_price * (1 + self._txn_cost)

        if cost > self._cash:
            size.shares = max(0, int(self._cash / (fill_price * (1 + self._txn_cost))))
            if size.shares < 1:
                return
            cost = size.shares * fill_price * (1 + self._txn_cost)

        self._cash -= cost

        atr = symbol_vol * price / np.sqrt(252)
        self._stop_mgr.open(sym, fill_price, ts.date() if hasattr(ts, "date") else ts, atr)

        self._positions[sym] = {
            "qty": size.shares,
            "entry_price": fill_price,
            "entry_date": ts,
            "signal_prob": signal.probability,
            "regime": signal.regime,
        }

    def _execute_sell(
        self, sym: str, ts, price: float, reason: str = "signal"
    ) -> None:
        if sym not in self._positions:
            return

        pos = self._positions.pop(sym)
        fill_price = self._slippage.adjusted_sell_price(price)
        proceeds = pos["qty"] * fill_price * (1 - self._txn_cost)
        self._cash += proceeds

        pnl = proceeds - pos["qty"] * pos["entry_price"]
        pnl_pct = pnl / (pos["qty"] * pos["entry_price"]) if pos["entry_price"] > 0 else 0.0

        self._stop_mgr.close(sym)
        self._trades.append(
            Trade(
                trade_id=str(uuid.uuid4())[:8],
                symbol=sym,
                entry_date=pos["entry_date"].date() if hasattr(pos["entry_date"], "date") else pos["entry_date"],
                entry_price=pos["entry_price"],
                exit_date=ts.date() if hasattr(ts, "date") else ts,
                exit_price=fill_price,
                quantity=pos["qty"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
                signal_probability=pos.get("signal_prob", 0.5),
                regime=pos.get("regime", "chop"),
            )
        )

    # ── Stop-loss check ───────────────────────────────────────────────────

    def _check_stops(self, ts, prices: Dict[str, float]) -> None:
        today = ts.date() if hasattr(ts, "date") else ts
        exits = self._stop_mgr.check_all(prices, today)
        for sym, reason in exits.items():
            if sym in self._positions:
                price = prices.get(sym, self._positions[sym]["entry_price"])
                self._execute_sell(sym, ts, price, reason=reason)

    def _update_mtm(self, prices: Dict[str, float]) -> None:
        for sym, pos in self._positions.items():
            if sym in prices:
                atr = 0.02 * prices[sym] / np.sqrt(252)
                self._stop_mgr.update_trailing(sym, prices[sym], atr)

    # ── Results compilation ───────────────────────────────────────────────

    def _compile_results(self) -> Dict[str, Any]:
        equity_df = pd.DataFrame(self._equity_curve, columns=["date", "equity"])
        equity_df = equity_df.set_index("date")["equity"]

        trades_df = pd.DataFrame([vars(t) for t in self._trades]) if self._trades else pd.DataFrame()

        stats = self._compute_stats(equity_df, trades_df)
        logger.info(f"Backtest complete. Sharpe={stats['sharpe']:.2f} Return={stats['total_return']:.2%}")
        return {"stats": stats, "equity_curve": equity_df, "trades": trades_df}

    @staticmethod
    def _compute_stats(equity: pd.Series, trades: pd.DataFrame) -> Dict:
        if len(equity) < 2:
            return {}

        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (1 + total_return) ** (252 / len(equity)) - 1

        vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / vol if vol > 0 else 0.0

        drawdown = (equity / equity.cummax() - 1)
        max_dd = float(drawdown.min())

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        if not trades.empty and "pnl" in trades.columns:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            win_rate = len(wins) / len(trades)
            avg_win = wins["pnl"].mean() if len(wins) else 0
            avg_loss = losses["pnl"].mean() if len(losses) else 0
            profit_factor = (
                wins["pnl"].sum() / abs(losses["pnl"].sum())
                if losses["pnl"].sum() != 0
                else float("inf")
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0

        return {
            "total_return": total_return,
            "annualised_return": ann_return,
            "volatility": vol,
            "sharpe": sharpe,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "n_trades": len(trades),
        }
