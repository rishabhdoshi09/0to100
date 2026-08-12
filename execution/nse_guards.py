"""
NSE-specific order guards.

Runs pre-flight checks that are specific to NSE/BSE market rules
before any order reaches the Kite API.

Checks (in sequence):
  1. Pre-open MARKET order block (09:00–09:15 IST)
  2. Circuit breaker — upper/lower price freeze
  3. CNC short-sell block — can't short-sell CNC without holding stock
  4. Minimum order value — must be at least ₹1
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple

import pytz

from logger import get_logger

log = get_logger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_PRE_OPEN_START = time(9, 0)
_PRE_OPEN_END = time(9, 15)


class NseOrderGuard:
    """Stateless guard — instantiate once and call validate() per order."""

    def validate(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        product: str,
        order_type: str,
        portfolio_positions: Dict[str, Any],
        quote: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Run all 4 NSE guard checks in sequence.

        Parameters
        ----------
        symbol              : NSE tradingsymbol (e.g. "RELIANCE")
        action              : "BUY" or "SELL"
        quantity            : number of shares
        price               : expected execution price (LTP or limit price)
        product             : "CNC", "MIS", "NRML", etc.
        order_type          : "MARKET" or "LIMIT"
        portfolio_positions : current holdings dict (symbol → any truthy value)
        quote               : optional dict with upper_circuit_limit,
                              lower_circuit_limit, last_price

        Returns
        -------
        (True, "ok")             if all checks pass
        (False, reason_string)   on the first failing check
        """

        # ── Check 1: Pre-open MARKET block ───────────────────────────────
        ok, reason = self._check_pre_open(order_type)
        if not ok:
            log.warning("nse_guard_blocked", check="pre_open", symbol=symbol, reason=reason)
            return False, reason

        # ── Check 2: Circuit breaker ──────────────────────────────────────
        if quote is not None:
            ok, reason = self._check_circuit(symbol, action, quote)
            if not ok:
                log.warning("nse_guard_blocked", check="circuit", symbol=symbol, reason=reason)
                return False, reason

        # ── Check 3: CNC short-sell block ─────────────────────────────────
        ok, reason = self._check_cnc_short(symbol, action, product, portfolio_positions)
        if not ok:
            log.warning("nse_guard_blocked", check="cnc_short", symbol=symbol, reason=reason)
            return False, reason

        # ── Check 4: Minimum order value ──────────────────────────────────
        ok, reason = self._check_min_value(symbol, quantity, price)
        if not ok:
            log.warning("nse_guard_blocked", check="min_value", symbol=symbol, reason=reason)
            return False, reason

        log.debug("nse_guard_passed", symbol=symbol, action=action, quantity=quantity)
        return True, "ok"

    # ── Individual checks ─────────────────────────────────────────────────

    @staticmethod
    def _check_pre_open(order_type: str) -> Tuple[bool, str]:
        now_ist = datetime.now(_IST).time()
        if (
            order_type.upper() == "MARKET"
            and _PRE_OPEN_START <= now_ist <= _PRE_OPEN_END
        ):
            return (
                False,
                "pre_open_market_orders_blocked: use LIMIT orders between 09:00-09:15 IST",
            )
        return True, ""

    @staticmethod
    def _check_circuit(
        symbol: str,
        action: str,
        quote: Dict[str, Any],
    ) -> Tuple[bool, str]:
        upper = quote.get("upper_circuit_limit")
        lower = quote.get("lower_circuit_limit")
        last = quote.get("last_price")

        if upper is None or lower is None or last is None:
            return True, ""

        if action.upper() == "BUY" and float(last) >= float(upper):
            return (
                False,
                f"upper_circuit_breached: {symbol} frozen at {upper}",
            )
        if action.upper() == "SELL" and float(last) <= float(lower):
            return (
                False,
                f"lower_circuit_breached: {symbol} frozen at {lower}",
            )
        return True, ""

    @staticmethod
    def _check_cnc_short(
        symbol: str,
        action: str,
        product: str,
        portfolio_positions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if (
            product.upper() == "CNC"
            and action.upper() == "SELL"
            and symbol not in portfolio_positions
        ):
            return (
                False,
                f"cnc_short_sell_blocked: {symbol} not in portfolio, "
                "short selling not permitted in CNC",
            )
        return True, ""

    @staticmethod
    def _check_min_value(
        symbol: str, quantity: int, price: float
    ) -> Tuple[bool, str]:
        order_value = quantity * price
        if order_value < 1.0:
            return (
                False,
                f"order_value_below_minimum: {quantity} * {price} = "
                f"{order_value:.4f} < ₹1",
            )
        return True, ""
