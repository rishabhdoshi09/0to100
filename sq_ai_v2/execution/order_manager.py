"""
Order manager — sits between signal generation and the broker.

Responsibilities:
  1. Accept trade signals from the composite signal engine.
  2. Consult RiskManager for approval and size.
  3. Apply StopLoss / PositionSizer adjustments.
  4. Place orders via the broker interface.
  5. Maintain in-memory portfolio state.
  6. Publish fills to Redis for the API to stream.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings
from data.storage.redis_client import RedisClient
from execution.broker import BrokerInterface, Order, PaperBroker, create_broker
from execution.slippage_model import SlippageModel
from risk.correlation import CorrelationManager
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager
from signals.composite_signal import SignalResult


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * (self.current_price or self.entry_price)

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price


class OrderManager:
    """
    Central hub for order lifecycle management.
    Thread-safe for concurrent signal processing.
    """

    def __init__(
        self,
        initial_capital: float = settings.backtest_initial_capital,
        live: bool = False,
    ) -> None:
        self._capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._filled_orders: List[Order] = []
        self._lock = threading.Lock()
        self._kill_switch = False
        self._daily_pnl = 0.0

        self._broker: BrokerInterface = create_broker(live=live)
        self._sizer = PositionSizer()
        self._stop_mgr = StopLossManager()
        self._corr_mgr = CorrelationManager()
        self._redis = RedisClient()

    # ── Kill switch ───────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "manual") -> None:
        with self._lock:
            self._kill_switch = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    # ── Process incoming signal ───────────────────────────────────────────

    def process_signal(
        self,
        signal: SignalResult,
        last_price: float,
        symbol_close: Optional["pd.Series"] = None,
    ) -> Optional[Order]:
        """
        Main entry point: evaluate and potentially execute a trading signal.
        """
        with self._lock:
            if self._kill_switch:
                logger.warning(f"Kill switch active — ignoring signal for {signal.symbol}")
                return None

            if self._daily_pnl < -self._capital * settings.max_daily_loss_pct:
                self.activate_kill_switch("daily_loss_limit")
                return None

        if signal.action == "HOLD":
            return None

        if signal.action == "SELL":
            return self._process_sell(signal, last_price)

        if signal.action == "BUY":
            return self._process_buy(signal, last_price, symbol_close)

        return None

    # ── Buy logic ─────────────────────────────────────────────────────────

    def _process_buy(
        self,
        signal: SignalResult,
        last_price: float,
        symbol_close=None,
    ) -> Optional[Order]:
        sym = signal.symbol

        # Guard: already long
        if sym in self._positions:
            return None

        # Guard: max open positions
        if len(self._positions) >= settings.max_open_positions:
            logger.debug(f"Max positions reached; skipping {sym}")
            return None

        # Guard: confidence
        if signal.confidence < settings.min_signal_confidence:
            logger.debug(f"{sym}: confidence {signal.confidence:.2f} < threshold")
            return None

        # Volatility metrics
        symbol_vol = 0.20
        if symbol_close is not None:
            symbol_vol = PositionSizer.compute_symbol_vol(symbol_close)

        portfolio_vol = PositionSizer.estimate_portfolio_vol(
            {s: p.market_value for s, p in self._positions.items()},
            {s: 0.20 for s in self._positions},  # simplified
            self._cash,
        )

        # Correlation penalty
        corr_penalty = self._corr_mgr.correlation_penalty(sym, list(self._positions.keys()))

        # Compute size
        from models.ensemble.hmm_regime import HMMRegime
        regime_mult = HMMRegime.regime_risk_multiplier(signal.regime_id)

        size = self._sizer.compute(
            symbol=sym,
            probability=signal.probability,
            last_price=last_price,
            portfolio_value=self._capital,
            portfolio_vol=portfolio_vol,
            symbol_vol=symbol_vol,
            regime_multiplier=regime_mult,
            correlation_penalty=corr_penalty,
        )

        if size.shares < 1:
            logger.debug(f"{sym}: zero shares computed by position sizer")
            return None

        # Check cash
        if size.trade_value > self._cash:
            size.shares = max(0, int(self._cash / last_price))
            if size.shares < 1:
                return None

        # Place order
        order = self._broker.place_order(sym, "BUY", size.shares, last_price=last_price)
        if order.status != "FILLED":
            return None

        # Update state
        fill = order.fill_price
        cost = size.shares * fill
        self._cash -= cost

        self._positions[sym] = Position(
            symbol=sym,
            quantity=size.shares,
            entry_price=fill,
            entry_time=datetime.utcnow(),
            current_price=fill,
        )

        # Register stop loss
        import numpy as np
        atr = symbol_vol * fill / np.sqrt(252)
        self._stop_mgr.open(sym, fill, datetime.utcnow().date(), atr)

        self._filled_orders.append(order)
        self._publish_fill(order, signal)
        logger.info(f"BOUGHT {size.shares} {sym} @ {fill:.2f} cost={cost:.0f}")
        return order

    # ── Sell logic ────────────────────────────────────────────────────────

    def _process_sell(self, signal: SignalResult, last_price: float) -> Optional[Order]:
        sym = signal.symbol
        if sym not in self._positions:
            return None

        pos = self._positions[sym]
        order = self._broker.place_order(sym, "SELL", pos.quantity, last_price=last_price)
        if order.status != "FILLED":
            return None

        fill = order.fill_price
        proceeds = pos.quantity * fill
        pnl = proceeds - pos.cost_basis

        self._cash += proceeds
        self._daily_pnl += pnl
        del self._positions[sym]
        self._stop_mgr.close(sym)

        self._filled_orders.append(order)
        self._publish_fill(order, signal)
        logger.info(f"SOLD {pos.quantity} {sym} @ {fill:.2f} pnl={pnl:.0f}")
        return order

    # ── Stop-loss check ───────────────────────────────────────────────────

    def check_stops(self, prices: Dict[str, float]) -> List[str]:
        """Check all stops; return list of symbols that triggered exit."""
        today = datetime.utcnow().date()
        exits = self._stop_mgr.check_all(prices, today)
        exited = []
        for sym, reason in exits.items():
            if sym in self._positions:
                logger.warning(f"Stop triggered: {sym} — {reason}")
                fake_signal = type("S", (), {"symbol": sym, "action": "SELL",
                                             "probability": 0.0, "confidence": 1.0,
                                             "regime_id": 1})()
                self._process_sell(fake_signal, prices[sym])
                exited.append(sym)
        return exited

    # ── Portfolio state ───────────────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Mark-to-market all open positions."""
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.current_price = prices[sym]
                pos.unrealised_pnl = (prices[sym] - pos.entry_price) * pos.quantity

        if isinstance(self._broker, PaperBroker):
            self._broker.update_prices(prices)

        # Cache in Redis
        portfolio = {
            "cash": self._cash,
            "positions": {
                s: {
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealised_pnl": p.unrealised_pnl,
                }
                for s, p in self._positions.items()
            },
            "total_value": self.total_value(),
            "daily_pnl": self._daily_pnl,
        }
        self._redis.cache_portfolio(portfolio)

    def total_value(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    def get_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    # ── Redis publish ─────────────────────────────────────────────────────

    def _publish_fill(self, order: Order, signal: SignalResult) -> None:
        self._redis.publish_signal({
            "type": "fill",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "fill_price": order.fill_price,
            "probability": signal.probability,
            "regime": signal.regime,
            "timestamp": datetime.utcnow().isoformat(),
        })
