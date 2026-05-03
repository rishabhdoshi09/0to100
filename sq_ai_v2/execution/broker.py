"""
Broker abstraction layer.

Implements two modes:
  1. Paper broker  — simulates execution with realistic slippage, zero risk.
  2. Live broker   — wraps Kite Connect order placement.

All modules interact with BrokerInterface and never directly with Kite.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings
from execution.slippage_model import SlippageModel


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str             # "BUY" | "SELL"
    quantity: int
    order_type: str       # "MARKET" | "LIMIT"
    limit_price: Optional[float]
    status: str           # "PENDING" | "FILLED" | "CANCELLED" | "REJECTED"
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


class BrokerInterface:
    """Abstract interface — every broker must implement these methods."""

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
    ) -> Order:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_order(self, order_id: str) -> Optional[Order]:
        raise NotImplementedError

    def get_open_orders(self) -> List[Order]:
        raise NotImplementedError


class PaperBroker(BrokerInterface):
    """
    Simulated paper-trading broker.
    Orders are filled immediately at slippage-adjusted prices.
    """

    def __init__(self, slippage_model: Optional[SlippageModel] = None) -> None:
        self._slippage = slippage_model or SlippageModel()
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}   # last known price per symbol

    def update_prices(self, prices: Dict[str, float]) -> None:
        self._prices.update(prices)

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        last_price: Optional[float] = None,
    ) -> Order:
        oid = str(uuid.uuid4())
        price = last_price or self._prices.get(symbol, 0.0)

        if price <= 0:
            order = Order(
                order_id=oid,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status="REJECTED",
                metadata={"reason": "zero_price"},
            )
            self._orders[oid] = order
            logger.warning(f"PaperBroker: REJECTED {side} {quantity} {symbol} (zero price)")
            return order

        # Simulate fill
        if side == "BUY":
            fill_price = self._slippage.adjusted_buy_price(price, quantity)
        else:
            fill_price = self._slippage.adjusted_sell_price(price, quantity)

        order = Order(
            order_id=oid,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            status="FILLED",
            fill_price=fill_price,
            fill_time=datetime.utcnow(),
        )
        self._orders[oid] = order
        logger.info(
            f"PaperBroker: FILLED {side} {quantity} {symbol} @ {fill_price:.2f}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].status == "PENDING":
            self._orders[order_id].status = "CANCELLED"
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == "PENDING"]


class KiteLiveBroker(BrokerInterface):
    """
    Live broker wrapping Kite Connect.
    Only activated when KITE_API_KEY is set and live=True.
    """

    def __init__(self) -> None:
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=settings.kite_api_key)
            if settings.kite_access_token:
                self._kite.set_access_token(settings.kite_access_token)
            logger.info("KiteLiveBroker ready")
        except ImportError:
            logger.error("kiteconnect not installed — live broker unavailable")
            self._kite = None
        self._orders: Dict[str, Order] = {}

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        last_price: Optional[float] = None,
    ) -> Order:
        if self._kite is None:
            raise RuntimeError("Kite not initialised")

        kite_order_type = "MARKET" if order_type == "MARKET" else "LIMIT"
        kite_side = self._kite.TRANSACTION_TYPE_BUY if side == "BUY" else self._kite.TRANSACTION_TYPE_SELL

        try:
            kite_oid = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=kite_side,
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=kite_order_type,
                price=limit_price,
            )
            order = Order(
                order_id=str(kite_oid),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status="PENDING",
            )
            self._orders[str(kite_oid)] = order
            logger.info(f"Kite order placed: {kite_oid} {side} {quantity} {symbol}")
            return order
        except Exception as exc:
            logger.error(f"Kite order placement failed: {exc}")
            return Order(
                order_id="ERR",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status="REJECTED",
                metadata={"error": str(exc)},
            )

    def cancel_order(self, order_id: str) -> bool:
        if self._kite is None:
            return False
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return True
        except Exception as exc:
            logger.error(f"Cancel order failed: {exc}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == "PENDING"]


def create_broker(live: bool = False) -> BrokerInterface:
    """Factory: return PaperBroker (default) or KiteLiveBroker."""
    if live and settings.kite_api_key:
        return KiteLiveBroker()
    return PaperBroker()
