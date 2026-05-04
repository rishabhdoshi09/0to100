"""
F&O (Futures & Options) executor for NSE/NFO segment.

Wraps Kite API calls for futures and options:
  - NFO instrument discovery and caching
  - Front-month future lookup
  - Lot-size rounding
  - Margin availability checks (fail-safe: reject if API fails)
  - Futures order placement
  - Automatic rollover detection and execution
  - Option chain lookup and option order placement

Constructor accepts the existing KiteClient — same pattern as ZerodhaBroker.

WARNING: F&O positions can result in losses exceeding initial capital.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from data.kite_client import KiteClient
from logger import get_logger

log = get_logger(__name__)

_NFO_CACHE_DIR = Path("data")
_NFO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FnOExecutor:
    """Executes futures and options orders on the NFO segment via Kite."""

    def __init__(self, kite: KiteClient) -> None:
        self._kite = kite
        self._nfo_cache: Optional[pd.DataFrame] = None
        self._nfo_cache_date: Optional[date] = None

    # ── NFO Instruments ────────────────────────────────────────────────────

    def fetch_nfo_instruments(self) -> pd.DataFrame:
        """
        Return the full NFO instrument list.
        Result is cached to data/nfo_instruments_{date}.csv for intraday reuse.
        """
        today = date.today()

        # Return in-memory cache if same day
        if self._nfo_cache is not None and self._nfo_cache_date == today:
            return self._nfo_cache

        cache_path = _NFO_CACHE_DIR / f"nfo_instruments_{today}.csv"
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                self._nfo_cache = df
                self._nfo_cache_date = today
                log.debug("nfo_instruments_loaded_from_cache", path=str(cache_path))
                return df
            except Exception as exc:
                log.warning("nfo_cache_read_failed", error=str(exc))

        log.info("fetching_nfo_instruments")
        raw = self._kite.raw.instruments("NFO")
        df = pd.DataFrame(raw)

        # Normalise date columns
        for col in ("expiry",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        try:
            df.to_csv(cache_path, index=False)
            log.info("nfo_instruments_cached", path=str(cache_path), count=len(df))
        except Exception as exc:
            log.warning("nfo_cache_write_failed", error=str(exc))

        self._nfo_cache = df
        self._nfo_cache_date = today
        return df

    def get_front_month_future(self, equity_symbol: str) -> Dict[str, Any]:
        """
        Return the nearest-expiry futures contract for *equity_symbol*.

        Raises ValueError if no futures contract is found.
        """
        df = self.fetch_nfo_instruments()
        futures = df[
            (df["instrument_type"] == "FUT") &
            (df["name"].str.upper() == equity_symbol.upper())
        ].copy()

        if futures.empty:
            raise ValueError(
                f"No futures contract found for {equity_symbol} in NFO instruments"
            )

        futures = futures.sort_values("expiry")
        row = futures.iloc[0]

        return {
            "tradingsymbol": row["tradingsymbol"],
            "instrument_token": int(row["instrument_token"]),
            "expiry": row["expiry"],
            "lot_size": int(row["lot_size"]),
        }

    # ── Lot size ───────────────────────────────────────────────────────────

    @staticmethod
    def round_to_lot_size(quantity: int, lot_size: int) -> int:
        """
        Round *quantity* down to a multiple of *lot_size*.
        Returns at least one lot (lot_size) — never 0.
        """
        lots = quantity // lot_size
        result = lots * lot_size
        return result if result >= lot_size else lot_size

    # ── Margin check ───────────────────────────────────────────────────────

    def check_margin(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = "NRML",
    ) -> Tuple[bool, float, float]:
        """
        Check whether sufficient margin is available to place this order.

        Returns (sufficient: bool, required: float, available: float).
        On any API failure returns (False, 0.0, 0.0) — fail safe.
        """
        try:
            margin_resp = self._kite.raw.order_margins([{
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "variety": "regular",
                "product": product,
                "order_type": "MARKET",
                "quantity": quantity,
                "price": 0,
                "trigger_price": 0,
            }])
            required = float(margin_resp[0].get("total", {}).get("total", 0.0))
        except Exception as exc:
            log.error("margin_check_api_failed", symbol=tradingsymbol, error=str(exc))
            return False, 0.0, 0.0

        try:
            margins = self._kite.raw.margins("equity")
            available = float(
                margins.get("equity", {}).get("available", {}).get("cash", 0.0)
            )
        except Exception as exc:
            log.error("available_margin_api_failed", error=str(exc))
            return False, required, 0.0

        sufficient = available >= required
        log.info(
            "margin_check",
            symbol=tradingsymbol,
            required=round(required, 2),
            available=round(available, 2),
            sufficient=sufficient,
        )
        return sufficient, required, available

    # ── Futures orders ─────────────────────────────────────────────────────

    def place_futures_order(
        self,
        equity_symbol: str,
        action: str,
        lots: int,
        product: str = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """
        Place a futures order for *equity_symbol*.

        Parameters
        ----------
        equity_symbol : NSE equity symbol (e.g. "RELIANCE")
        action        : "BUY" or "SELL"
        lots          : number of lots to trade
        product       : "NRML" (default) or "MIS"

        Returns a status dict.
        """
        product = product or settings.fno_default_product

        try:
            contract = self.get_front_month_future(equity_symbol)
        except ValueError as exc:
            log.error("futures_contract_not_found", symbol=equity_symbol, error=str(exc))
            return {"status": "rejected", "reason": str(exc)}

        tradingsymbol = contract["tradingsymbol"]
        lot_size = contract["lot_size"]
        quantity = lots * lot_size

        sufficient, required, available = self.check_margin(
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            transaction_type=action,
            quantity=quantity,
            product=product,
        )
        if not sufficient:
            log.warning(
                "futures_order_margin_insufficient",
                symbol=tradingsymbol,
                required=required,
                available=available,
            )
            return {
                "status": "rejected",
                "reason": "insufficient_margin",
                "required": required,
                "available": available,
            }

        try:
            order_id = self._kite.raw.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=action.upper(),
                quantity=quantity,
                product=product,
                order_type="MARKET",
                tag="sq_fno",
            )
        except Exception as exc:
            log.error("futures_order_placement_failed", symbol=tradingsymbol, error=str(exc))
            return {"status": "error", "reason": str(exc)}

        log.info(
            "futures_order_placed",
            symbol=tradingsymbol,
            action=action,
            lots=lots,
            quantity=quantity,
            order_id=order_id,
        )
        return {
            "status": "placed",
            "order_id": order_id,
            "tradingsymbol": tradingsymbol,
            "quantity": quantity,
            "lots": lots,
        }

    # ── Rollover ───────────────────────────────────────────────────────────

    def should_rollover(self, expiry: date) -> bool:
        """Return True if expiry is within fno_rollover_days calendar days."""
        today = date.today()
        days_to_expiry = (expiry - today).days
        return days_to_expiry <= settings.fno_rollover_days

    def rollover_position(
        self, equity_symbol: str, current_lots: int
    ) -> Dict[str, Any]:
        """
        Roll a futures position from front-month to next-month.

        1. Close front-month (SELL current_lots)
        2. Open next-month (BUY current_lots)
        Returns dict with 'closed' and 'opened' sub-results.
        """
        df = self.fetch_nfo_instruments()
        futures = df[
            (df["instrument_type"] == "FUT") &
            (df["name"].str.upper() == equity_symbol.upper())
        ].copy()

        if len(futures) < 2:
            msg = f"Need at least 2 futures contracts to rollover {equity_symbol}"
            log.error("rollover_insufficient_contracts", symbol=equity_symbol)
            return {"status": "error", "reason": msg}

        futures = futures.sort_values("expiry")
        front = futures.iloc[0]
        next_m = futures.iloc[1]

        front_contract = {
            "tradingsymbol": front["tradingsymbol"],
            "lot_size": int(front["lot_size"]),
        }
        next_contract = {
            "tradingsymbol": next_m["tradingsymbol"],
            "lot_size": int(next_m["lot_size"]),
        }

        log.info(
            "rolling_position",
            symbol=equity_symbol,
            close_contract=front_contract["tradingsymbol"],
            open_contract=next_contract["tradingsymbol"],
            lots=current_lots,
        )

        close_result = self._place_nfo_order(
            tradingsymbol=front_contract["tradingsymbol"],
            action="SELL",
            quantity=current_lots * front_contract["lot_size"],
        )
        open_result = self._place_nfo_order(
            tradingsymbol=next_contract["tradingsymbol"],
            action="BUY",
            quantity=current_lots * next_contract["lot_size"],
        )

        return {"closed": close_result, "opened": open_result}

    # ── Options ────────────────────────────────────────────────────────────

    def get_option_chain(self, equity_symbol: str, expiry: date) -> pd.DataFrame:
        """
        Return the option chain for *equity_symbol* at *expiry*.
        Sorted by strike price ascending.
        """
        df = self.fetch_nfo_instruments()
        chain = df[
            (df["name"].str.upper() == equity_symbol.upper()) &
            (df["instrument_type"].isin(["CE", "PE"])) &
            (df["expiry"] == expiry)
        ].copy()
        return chain.sort_values("strike").reset_index(drop=True)

    def place_option_order(
        self,
        equity_symbol: str,
        strike: float,
        option_type: str,
        action: str,
        lots: int,
        expiry: date,
        product: str = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """
        Place an options order.

        Parameters
        ----------
        equity_symbol : NSE equity symbol (e.g. "NIFTY")
        strike        : strike price
        option_type   : "CE" or "PE"
        action        : "BUY" or "SELL"
        lots          : number of lots
        expiry        : expiry date
        product       : "NRML" or "MIS"
        """
        product = product or settings.fno_default_product
        chain = self.get_option_chain(equity_symbol, expiry)

        candidates = chain[
            (chain["strike"] == float(strike)) &
            (chain["instrument_type"] == option_type.upper())
        ]

        if candidates.empty:
            msg = (
                f"No option contract found: {equity_symbol} "
                f"{strike} {option_type} expiry={expiry}"
            )
            log.error("option_contract_not_found", msg=msg)
            return {"status": "rejected", "reason": msg}

        row = candidates.iloc[0]
        tradingsymbol = row["tradingsymbol"]
        lot_size = int(row["lot_size"])
        quantity = lots * lot_size

        sufficient, required, available = self.check_margin(
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            transaction_type=action,
            quantity=quantity,
            product=product,
        )
        if not sufficient:
            log.warning(
                "option_order_margin_insufficient",
                symbol=tradingsymbol,
                required=required,
                available=available,
            )
            return {
                "status": "rejected",
                "reason": "insufficient_margin",
                "required": required,
                "available": available,
            }

        try:
            order_id = self._kite.raw.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=action.upper(),
                quantity=quantity,
                product=product,
                order_type="MARKET",
                tag="sq_fno",
            )
        except Exception as exc:
            log.error("option_order_placement_failed", symbol=tradingsymbol, error=str(exc))
            return {"status": "error", "reason": str(exc)}

        log.info(
            "option_order_placed",
            symbol=tradingsymbol,
            action=action,
            lots=lots,
            quantity=quantity,
            order_id=order_id,
        )
        return {
            "status": "placed",
            "order_id": order_id,
            "tradingsymbol": tradingsymbol,
            "quantity": quantity,
            "lots": lots,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _place_nfo_order(
        self,
        tradingsymbol: str,
        action: str,
        quantity: int,
        product: str = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        product = product or settings.fno_default_product
        try:
            order_id = self._kite.raw.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=action.upper(),
                quantity=quantity,
                product=product,
                order_type="MARKET",
                tag="sq_fno",
            )
            return {
                "status": "placed",
                "order_id": order_id,
                "tradingsymbol": tradingsymbol,
                "quantity": quantity,
            }
        except Exception as exc:
            log.error("nfo_order_failed", symbol=tradingsymbol, error=str(exc))
            return {"status": "error", "reason": str(exc)}
