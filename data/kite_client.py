"""
Zerodha Kite Connect client wrapper.

Provides a single authenticated KiteClient used everywhere in the system.
Authentication is a two-step process:
  1. Generate login URL -> user logs in and gets a request_token
  2. Exchange request_token for access_token (valid for one trading day)

All broker mutations are protected at this boundary by the canonical
live-execution interlock. Read-only market/account calls remain available.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Dict, List, Optional
import os
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

from config import settings
from logger import get_logger
from product.live_execution_interlock import assert_live_execution_allowed

log = get_logger(__name__)


# Explicit read-only allowlist for the raw KiteConnect escape hatch. Any public
# callable not named here is treated as a broker mutation and must pass the
# canonical live-execution interlock before it can reach the SDK. This is
# intentionally allow-by-enumeration: a capital-moving method added by a future
# Kite SDK release therefore fails closed by default.
_BROKER_READS = frozenset({
    "quote",
    "ltp",
    "ohlc",
    "historical_data",
    "instruments",
    "orders",
    "order_history",
    "trades",
    "positions",
    "holdings",
    "margins",
    "order_margins",
    "basket_order_margins",
    "profile",
    "get_gtts",
    "get_gtt",
    "trigger_range",
    "mf_instruments",
    "mf_orders",
    "mf_order_history",
    "mf_sips",
    "auction_instruments",
})


class _GuardedKiteProxy:
    """Read-through KiteConnect proxy; every non-allowlisted call fails closed."""

    def __init__(self, raw: KiteConnect) -> None:
        object.__setattr__(self, "_raw", raw)

    def __getattr__(self, name: str):
        attr = getattr(object.__getattribute__(self, "_raw"), name)
        if not callable(attr) or name in _BROKER_READS:
            return attr

        @wraps(attr)
        def guarded(*args, **kwargs):
            assert_live_execution_allowed(f"kite.raw.{name}")
            return attr(*args, **kwargs)

        return guarded

    def __setattr__(self, name: str, value: Any) -> None:
        # The raw proxy is a read-only facade. Callers must not replace SDK
        # methods/attributes through it and thereby create a second policy path.
        raise AttributeError(f"Cannot mutate guarded Kite proxy attribute: {name}")


def parse_request_token(raw: str) -> str:
    """Accept a bare token or the full Kite redirect URL."""
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        return ""
    if "request_token=" in text:
        from urllib.parse import parse_qs, unquote, urlparse

        query = urlparse(text).query if "://" in text else text.split("?", 1)[-1].lstrip("?")
        token = (parse_qs(query).get("request_token") or [""])[0]
        return unquote(token).strip()
    return text


def _fresh_env(name: str, default: str = "") -> str:
    """Read current credentials without relying on process-lifetime settings."""
    path = Path(__file__).resolve().parent.parent / ".env"
    file_value = ""
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == name:
                file_value = val.strip().strip('"').strip("'")
                break
    except Exception:
        pass
    if name == "KITE_ACCESS_TOKEN" and file_value:
        return file_value
    value = os.getenv(name)
    if value is not None:
        return value.strip()
    return file_value or default


class KiteClient:
    """Thin, opinionated wrapper around KiteConnect SDK."""

    def __init__(self, *, api_key: str | None = None, access_token: str | None = None,
                 api_secret: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else _fresh_env("KITE_API_KEY", settings.kite_api_key)
        self._access_token = (access_token if access_token is not None
                              else _fresh_env("KITE_ACCESS_TOKEN", settings.kite_access_token))
        self._api_secret = (api_secret if api_secret is not None
                            else _fresh_env("KITE_API_SECRET", settings.kite_api_secret))
        self._kite = KiteConnect(api_key=self._api_key)
        if self._access_token:
            self._kite.set_access_token(self._access_token)
        else:
            log.warning("kite_access_token not set — run generate_session() first")

    # -- Authentication -----------------------------------------------------

    def login_url(self) -> str:
        return self._kite.login_url()

    def generate_session(self, request_token: str) -> str:
        data = self._kite.generate_session(request_token, api_secret=self._api_secret)
        access_token: str = data["access_token"]
        self._kite.set_access_token(access_token)
        self._access_token = access_token
        log.info("kite_session_created")
        return access_token

    # -- Market Data --------------------------------------------------------

    def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        instruments = [f"{settings.exchange}:{s}" for s in symbols]
        return self._kite.quote(instruments)

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        instruments = [f"{settings.exchange}:{s}" for s in symbols]
        raw = self._kite.ltp(instruments)
        return {k.split(":")[1]: v["last_price"] for k, v in raw.items()}

    def get_ohlcv(self, symbols: List[str]) -> Dict[str, Any]:
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
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    # -- Order Operations ---------------------------------------------------

    def place_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: Optional[str] = None,
        tag: str = "simplequant",
    ) -> str:
        # This assertion deliberately happens before parameter construction or
        # any broker SDK call. Environment flags and upstream arming cannot
        # bypass it.
        assert_live_execution_allowed("kite.place_order")

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
        log.info("order_placed", symbol=symbol, side=transaction_type, qty=quantity, order_id=order_id)
        return order_id

    def cancel_order(self, order_id: str) -> str:
        # Cancellation is also a mutation; cancelling a protective order can
        # increase risk, so it is not exempted from the lock.
        assert_live_execution_allowed("kite.cancel_order")
        return self._kite.cancel_order(variety=KiteConnect.VARIETY_REGULAR, order_id=order_id)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        orders = self._kite.orders()
        for order in orders:
            if order["order_id"] == order_id:
                return order
        return {}

    def get_orders(self) -> List[Dict[str, Any]]:
        return self._kite.orders()

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._kite.positions()

    def get_holdings(self) -> List[Dict[str, Any]]:
        return self._kite.holdings()

    def get_margins(self) -> Dict[str, Any]:
        return self._kite.margins()

    # -- WebSocket ----------------------------------------------------------

    def get_ticker(self, on_ticks, on_connect, on_close, on_error=None) -> KiteTicker:
        ticker = KiteTicker(api_key=self._api_key, access_token=self._access_token)
        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_close = on_close
        if on_error:
            ticker.on_error = on_error
        return ticker

    # -- Utility ------------------------------------------------------------

    def is_connected(self) -> bool:
        return bool(self._access_token)

    def batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        try:
            instruments = [f"{settings.exchange}:{s}" if ":" not in s else s for s in symbols]
            raw = self._kite.quote(instruments)
            result: dict[str, dict] = {}
            for key, val in raw.items():
                sym = key.split(":")[-1]
                ohlc = val.get("ohlc", {})
                result[sym] = {
                    "ltp": val.get("last_price", 0.0),
                    "open": ohlc.get("open", 0.0),
                    "high": ohlc.get("high", 0.0),
                    "low": ohlc.get("low", 0.0),
                    "close": ohlc.get("close", 0.0),
                    "volume": val.get("volume", 0),
                    "change": val.get("change", 0.0),
                }
            return result
        except Exception as exc:
            log.warning("batch_quotes_failed", symbols_count=len(symbols), error=str(exc))
            return {}

    @property
    def raw(self) -> _GuardedKiteProxy:
        """Guarded escape hatch: explicit reads pass; every other call interlocks."""
        return _GuardedKiteProxy(self._kite)
