"""
Zerodha Kite Connect client wrapper.

Provides a single authenticated KiteClient used everywhere in the system.
Authentication is a two-step process:
  1.  Generate login URL → user logs in and gets a request_token
  2.  Exchange request_token for access_token (valid for one trading day)

The access_token must be stored in the .env file as KITE_ACCESS_TOKEN
(refreshed each morning before market open).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

from config import settings
from logger import get_logger

log = get_logger(__name__)


class KiteClient:
    """Thin, opinionated wrapper around KiteConnect SDK."""

    def __init__(self) -> None:
        self._kite = KiteConnect(api_key=settings.kite_api_key)
        if settings.kite_access_token:
            self._kite.set_access_token(settings.kite_access_token)
        else:
            log.warning(
                "kite_access_token not set — run generate_session() first"
            )

    # ── Authentication ─────────────────────────────────────────────────────

    def login_url(self) -> str:
        """Return the Kite login URL for manual OAuth flow."""
        return self._kite.login_url()

    def generate_session(self, request_token: str) -> str:
        """
        Exchange request_token for access_token.
        Call this once per day after manual login.
        Returns the access_token (persist to .env).
        """
        data = self._kite.generate_session(
            request_token, api_secret=settings.kite_api_secret
        )
        access_token: str = data["access_token"]
        self._kite.set_access_token(access_token)
        log.info("kite_session_created", access_token=access_token[:8] + "…")
        return access_token

    # ── Market Data ───────────────────────────────────────────────────────

    def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Full market quote for a list of symbols.
        symbols format: ["NSE:RELIANCE", "NSE:INFY"]
        """
        instruments = [f"{settings.exchange}:{s}" for s in symbols]
        return self._kite.quote(instruments)

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Return last traded price for each symbol."""
        instruments = [f"{settings.exchange}:{s}" for s in symbols]
        raw = self._kite.ltp(instruments)
        return {
            k.split(":")[1]: v["last_price"] for k, v in raw.items()
        }

    def get_ohlcv(self, symbols: List[str]) -> Dict[str, Any]:
        """OHLCV snapshot for a list of symbols."""
        instruments = [f"{settings.exchange}:{s}" for s in symbols]
        return self._kite.ohlc(instruments)

    def get_historical(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str = "day",
        continuous: bool = False,
    ) -> pd.DataFrame:
        """
        Download historical candles from Kite.

        interval: minute | 3minute | 5minute | 10minute | 15minute |
                  30minute | 60minute | day | week | month
        """
        raw = self._kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=continuous,
        )
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df.rename(
            columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    # ── Order Operations ──────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        transaction_type: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: Optional[str] = None,
        tag: str = "simplequant",
    ) -> str:
        """
        Place an order. Returns order_id on success.
        Raises KiteException on failure.
        """
        product = product or settings.product_type
        params: Dict[str, Any] = {
            "variety": KiteConnect.VARIETY_REGULAR,
            "exchange": settings.exchange,
            "tradingsymbol": symbol,
            "transaction_type": (
                KiteConnect.TRANSACTION_TYPE_BUY
                if transaction_type.upper() == "BUY"
                else KiteConnect.TRANSACTION_TYPE_SELL
            ),
            "quantity": quantity,
            "product": product,
            "order_type": (
                KiteConnect.ORDER_TYPE_MARKET
                if order_type.upper() == "MARKET"
                else KiteConnect.ORDER_TYPE_LIMIT
            ),
            "tag": tag,
        }
        if order_type.upper() == "LIMIT" and price is not None:
            params["price"] = price

        order_id: str = self._kite.place_order(**params)
        log.info(
            "order_placed",
            symbol=symbol,
            side=transaction_type,
            qty=quantity,
            order_id=order_id,
        )
        return order_id

    def cancel_order(self, order_id: str) -> str:
        return self._kite.cancel_order(
            variety=KiteConnect.VARIETY_REGULAR, order_id=order_id
        )

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        orders = self._kite.orders()
        for o in orders:
            if o["order_id"] == order_id:
                return o
        return {}

    def get_orders(self) -> List[Dict[str, Any]]:
        return self._kite.orders()

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._kite.positions()

    def get_holdings(self) -> List[Dict[str, Any]]:
        return self._kite.holdings()

    def get_margins(self) -> Dict[str, Any]:
        return self._kite.margins()

    # ── WebSocket (live ticks) ─────────────────────────────────────────────

    def get_ticker(
        self,
        on_ticks,
        on_connect,
        on_close,
        on_error=None,
    ) -> KiteTicker:
        """
        Return a configured KiteTicker (not yet connected).
        Caller is responsible for assigning tokens and calling connect().
        """
        ticker = KiteTicker(
            api_key=settings.kite_api_key,
            access_token=settings.kite_access_token,
        )
        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_close = on_close
        if on_error:
            ticker.on_error = on_error
        return ticker

    # ── Utility ───────────────────────────────────────────────────────────

    @property
    def raw(self) -> KiteConnect:
        """Escape hatch to the underlying KiteConnect object."""
        return self._kite
