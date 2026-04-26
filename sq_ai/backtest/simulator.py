"""
Simulated order broker for backtesting.

Simulates realistic Kite execution with:
  - configurable slippage (default 0.05%)
  - configurable transaction cost (default 0.10%)
  - 1-bar execution delay (order submitted at close of bar N,
    filled at open of bar N+1 — prevents lookahead bias)
  - market orders only in base mode (limit order simulation optional)

NO lookahead: orders submitted at bar-close are filled at NEXT bar's open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import settings
from logger import get_logger

log = get_logger(__name__)


@dataclass
class PendingOrder:
    symbol: str
    action: str        # BUY | SELL
    quantity: int
    submitted_at: datetime
    signal_reasoning: str = ""
    confidence: float = 0.0


@dataclass
class SimFill:
    order_id: str
    symbol: str
    action: str
    quantity: int
    fill_price: float        # after slippage
    gross_price: float       # before slippage
    slippage: float
    transaction_cost: float
    fill_time: datetime
    realized_pnl: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "fill_price": round(self.fill_price, 2),
            "gross_price": round(self.gross_price, 2),
            "slippage_cost": round(self.slippage * self.quantity, 2),
            "transaction_cost": round(self.transaction_cost, 2),
            "fill_time": self.fill_time.isoformat(),
            "realized_pnl": round(self.realized_pnl, 2),
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
        }


class SimulatedBroker:
    """
    Paper order book used by the backtester.
    Pending orders are flushed at the start of each bar with that bar's open.
    """

    def __init__(
        self,
        slippage: float = settings.backtest_slippage,
        transaction_cost: float = settings.backtest_transaction_cost,
    ) -> None:
        self._slippage = slippage
        self._tc = transaction_cost
        self._pending: List[PendingOrder] = []
        self._fill_counter = 0

    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        submitted_at: datetime,
        reasoning: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Queue an order. Filled at next bar's open."""
        if quantity < 1:
            return
        self._pending.append(
            PendingOrder(
                symbol=symbol,
                action=action,
                quantity=quantity,
                submitted_at=submitted_at,
                signal_reasoning=reasoning,
                confidence=confidence,
            )
        )
        log.debug(
            "sim_order_queued",
            symbol=symbol,
            action=action,
            quantity=quantity,
        )

    def flush_pending(
        self,
        bar_open_prices: Dict[str, float],
        bar_time: datetime,
        entry_prices: Dict[str, float],   # symbol → avg_entry_price for PnL
    ) -> List[SimFill]:
        """
        Fill all pending orders at bar_open_prices.
        entry_prices needed to compute realized PnL for SELL orders.
        Returns list of fills.
        """
        fills: List[SimFill] = []
        remaining: List[PendingOrder] = []

        for order in self._pending:
            sym = order.symbol
            gross = bar_open_prices.get(sym)
            if gross is None:
                log.warning("sim_fill_no_price", symbol=sym)
                remaining.append(order)
                continue

            # Apply slippage (adverse: BUY → higher, SELL → lower)
            if order.action == "BUY":
                fill_price = gross * (1 + self._slippage)
            else:
                fill_price = gross * (1 - self._slippage)

            trade_value = fill_price * order.quantity
            tc = trade_value * self._tc

            # Realized PnL for SELL
            realized_pnl = 0.0
            if order.action == "SELL":
                entry = entry_prices.get(sym, fill_price)
                realized_pnl = (fill_price - entry) * order.quantity - tc

            self._fill_counter += 1
            fill_id = f"SIM-{self._fill_counter:06d}"

            fills.append(SimFill(
                order_id=fill_id,
                symbol=sym,
                action=order.action,
                quantity=order.quantity,
                fill_price=fill_price,
                gross_price=gross,
                slippage=self._slippage * gross,
                transaction_cost=tc,
                fill_time=bar_time,
                realized_pnl=realized_pnl,
                reasoning=order.signal_reasoning,
                confidence=order.confidence,
            ))

        self._pending = remaining
        return fills

    def cancel_pending(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self._pending = [o for o in self._pending if o.symbol != symbol]
        else:
            self._pending.clear()
