"""
Portfolio state manager.

Tracks:
  - Cash balance
  - Open positions (symbol → Position)
  - Realized PnL
  - Trade journal (every fill)
  - Equity curve (sampled each cycle)

Thread-safe. Single source of truth for portfolio state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    side: str = "LONG"  # LONG | SHORT (MIS short supported)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "LONG":
            return (self.current_price - self.avg_entry_price) * self.quantity
        return (self.avg_entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.avg_entry_price * self.quantity
        if cost == 0:
            return 0.0
        return self.unrealized_pnl / cost * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": round(self.avg_entry_price, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 3),
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
        }


@dataclass
class TradeRecord:
    """Immutable record of a completed trade (fill)."""
    timestamp: datetime
    symbol: str
    action: str           # BUY | SELL
    quantity: int
    price: float
    value: float          # quantity × price
    transaction_cost: float
    realized_pnl: float
    order_id: str
    reasoning: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": round(self.price, 2),
            "value": round(self.value, 2),
            "transaction_cost": round(self.transaction_cost, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "order_id": self.order_id,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


class PortfolioState:
    def __init__(self, initial_capital: float) -> None:
        self._lock = threading.RLock()
        self._cash = initial_capital
        self._initial_capital = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trade_journal: List[TradeRecord] = []
        self._equity_curve: List[Dict[str, Any]] = []

    # ── Position Management ────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_id: str,
        reasoning: str = "",
        confidence: float = 0.0,
        transaction_cost_rate: float = 0.001,
        timestamp: Optional[datetime] = None,
    ) -> None:
        cost = quantity * price
        tc = cost * transaction_cost_rate
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            self._cash -= (cost + tc)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                entry_time=ts,
                current_price=price,
            )
            self._record_trade(
                timestamp=ts,
                symbol=symbol,
                action="BUY",
                quantity=quantity,
                price=price,
                transaction_cost=tc,
                realized_pnl=0.0,
                order_id=order_id,
                reasoning=reasoning,
                confidence=confidence,
            )
        log.info("position_opened", symbol=symbol, qty=quantity, price=price)

    def close_position(
        self,
        symbol: str,
        price: float,
        order_id: str,
        transaction_cost_rate: float = 0.001,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Close existing position. Returns realized PnL."""
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                log.warning("close_position_not_found", symbol=symbol)
                return 0.0
            proceeds = pos.quantity * price
            tc = proceeds * transaction_cost_rate
            realized_pnl = (price - pos.avg_entry_price) * pos.quantity - tc
            self._cash += proceeds - tc
            del self._positions[symbol]
            self._record_trade(
                timestamp=ts,
                symbol=symbol,
                action="SELL",
                quantity=pos.quantity,
                price=price,
                transaction_cost=tc,
                realized_pnl=realized_pnl,
                order_id=order_id,
            )
        log.info("position_closed", symbol=symbol, price=price, realized_pnl=realized_pnl)
        return realized_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current_price on open positions (for unrealized PnL)."""
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol].current_price = price

    # ── Equity / Snapshot ─────────────────────────────────────────────────

    def snapshot_equity(self) -> float:
        with self._lock:
            market_value = sum(p.market_value for p in self._positions.values())
            return self._cash + market_value

    def record_equity_point(self, timestamp: Optional[datetime] = None) -> None:
        equity = self.snapshot_equity()
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            self._equity_curve.append({
                "timestamp": ts.isoformat(),
                "equity": round(equity, 2),
                "cash": round(self._cash, 2),
                "open_positions": len(self._positions),
            })

    # ── State Accessors ────────────────────────────────────────────────────

    def get_state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cash": round(self._cash, 2),
                "initial_capital": round(self._initial_capital, 2),
                "total_equity": round(self.snapshot_equity(), 2),
                "open_positions": len(self._positions),
                "positions": {s: p.to_dict() for s, p in self._positions.items()},
                "total_realized_pnl": round(self._total_realized_pnl(), 2),
            }

    def get_open_positions(self) -> Dict[str, float]:
        """Return symbol → market_value for risk checks."""
        with self._lock:
            return {s: p.market_value for s, p in self._positions.items()}

    def has_position(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._positions

    def get_trade_journal(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._trade_journal]

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._equity_curve)

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        with self._lock:
            return self._cash

    # ── Internals ──────────────────────────────────────────────────────────

    def _record_trade(self, **kwargs) -> None:
        ts = kwargs.pop("timestamp", datetime.now(timezone.utc))
        record = TradeRecord(
            timestamp=ts,
            value=kwargs["quantity"] * kwargs["price"],
            **{k: v for k, v in kwargs.items()},
        )
        self._trade_journal.append(record)

    def _total_realized_pnl(self) -> float:
        return sum(t.realized_pnl for t in self._trade_journal)
