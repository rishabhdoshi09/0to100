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

from typing import Any, Dict, List, Optional
import os
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

from config import settings
from logger import get_logger

log = get_logger(__name__)


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
    """Read current credentials without relying on the process-lifetime Settings object.

    The daily access token intentionally prefers the current ``.env`` value so a running autonomy
    service sees a newly completed login even when its original process environment was stale.
    Static credentials retain the conventional process-environment override.
    """
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
            request_token, api_secret=self._api_secret
        )
        access_token: str = data["access_token"]
        self._kite.set_access_token(access_token)
        self._access_token = access_token
        log.info("kite_session_created")
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
        ticker = KiteTicker(api_key=self._api_key, access_token=self._access_token)
        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_close = on_close
        if on_error:
            ticker.on_error = on_error
        return ticker

    # ── Utility ───────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """True if access token is configured. Lightweight check — no network call."""
        return bool(self._access_token)

    def batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch full market quotes for up to 500 symbols in ONE API call.
        Returns {symbol: {ltp, open, high, low, close, volume, ...}}.
        Falls back to empty dict on error.
        """
        if not symbols:
            return {}
        try:
            from config import settings
            instruments = [f"{settings.exchange}:{s}" if ":" not in s
                           else s for s in symbols]
            raw = self._kite.quote(instruments)
            result: dict[str, dict] = {}
            for key, val in raw.items():
                sym = key.split(":")[-1]
                ohlc = val.get("ohlc", {})
                result[sym] = {
                    "ltp":    val.get("last_price", 0.0),
                    "open":   ohlc.get("open", 0.0),
                    "high":   ohlc.get("high", 0.0),
                    "low":    ohlc.get("low", 0.0),
                    "close":  ohlc.get("close", 0.0),
                    "volume": val.get("volume", 0),
                    "change": val.get("change", 0.0),
                }
            return result
        except Exception as e:
            log.warning("batch_quotes_failed", symbols_count=len(symbols), error=str(e))
            return {}

    @property
    def raw(self) -> KiteConnect:
        """Escape hatch to the underlying KiteConnect object."""
        return self._kite
