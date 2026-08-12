"""
Zerodha execution layer.

Responsibility: translate approved RiskDecision into Kite API calls
and track the full order lifecycle from placement to confirmation.

Design:
  - Every order is logged before and after placement.
  - Exceptions from Kite are caught, logged, and surfaced as OrderResult.
  - No business logic here — only plumbing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from data.kite_client import KiteClient
from execution.nse_guards import NseOrderGuard
from risk.risk_manager import RiskDecision
from logger import get_logger

log = get_logger(__name__)

_STATUS_POLL_INTERVAL = 2.0   # seconds between status polls
_STATUS_POLL_MAX_TRIES = 15   # give up after 30 seconds


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    action: str
    quantity: int
    status: str            # COMPLETE | REJECTED | CANCELLED | PENDING | ERROR
    fill_price: float
    fill_time: Optional[datetime]
    error_message: str = ""
    raw: Dict[str, Any] = None  # type: ignore[assignment]

    def is_filled(self) -> bool:
        return self.status == "COMPLETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "status": self.status,
            "fill_price": round(self.fill_price, 2),
            "fill_time": self.fill_time.isoformat() if self.fill_time else None,
            "error_message": self.error_message,
        }


class ZerodhaBroker:
    def __init__(self, kite: KiteClient) -> None:
        self._kite = kite
        self._nse_guard = NseOrderGuard()

    def execute(
        self,
        decision: RiskDecision,
        order_type: str = "MARKET",
        portfolio_positions: Dict[str, Any] = None,  # type: ignore[assignment]
        quote: Dict[str, Any] = None,  # type: ignore[assignment]
    ) -> OrderResult:
        """
        Place order from an approved RiskDecision.
        Returns OrderResult after polling for fill confirmation.
        """
        if not decision.approved:
            log.error(
                "execute_called_on_unapproved_decision",
                symbol=decision.signal.symbol,
                reason=decision.reason,
            )
            return self._error_result(
                decision.signal.symbol,
                decision.signal.action,
                int(decision.adjusted_size),
                f"unapproved_decision:{decision.reason}",
            )

        symbol = decision.signal.symbol
        action = decision.signal.action
        quantity = int(decision.adjusted_size)

        # ── NSE pre-flight checks ─────────────────────────────────────────
        from config import settings
        guard_ok, guard_reason = self._nse_guard.validate(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=quote.get("last_price", 0.0) if quote else 0.0,
            product=settings.product_type,
            order_type=order_type,
            portfolio_positions=portfolio_positions or {},
            quote=quote,
        )
        if not guard_ok:
            log.warning(
                "nse_guard_rejected_order",
                symbol=symbol,
                action=action,
                reason=guard_reason,
            )
            return self._error_result(symbol, action, quantity, guard_reason)

        log.info(
            "placing_order",
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
        )

        try:
            order_id = self._kite.place_order(
                symbol=symbol,
                transaction_type=action,
                quantity=quantity,
                order_type=order_type,
                tag="simplequant",
            )
        except Exception as exc:
            log.error("order_placement_failed", symbol=symbol, error=str(exc))
            return self._error_result(symbol, action, quantity, str(exc))

        # Poll for fill
        return self._poll_order(order_id, symbol, action, quantity)

    def cancel(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(order_id)
            log.info("order_cancelled", order_id=order_id)
            return True
        except Exception as exc:
            log.error("order_cancel_failed", order_id=order_id, error=str(exc))
            return False

    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            return [
                o for o in self._kite.get_orders()
                if o.get("status") in ("OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED")
            ]
        except Exception as exc:
            log.error("get_open_orders_failed", error=str(exc))
            return []

    def cancel_all_open_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        open_orders = self.get_open_orders()
        count = 0
        for o in open_orders:
            if self.cancel(o["order_id"]):
                count += 1
        log.info("cancelled_all_open_orders", count=count)
        return count

    # ── Internal ───────────────────────────────────────────────────────────

    def _poll_order(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: int,
    ) -> OrderResult:
        """Poll Kite for order status until terminal state or timeout."""
        for attempt in range(_STATUS_POLL_MAX_TRIES):
            time.sleep(_STATUS_POLL_INTERVAL)
            try:
                status_raw = self._kite.get_order_status(order_id)
            except Exception as exc:
                log.warning("order_status_poll_error", order_id=order_id, error=str(exc))
                continue

            if not status_raw:
                continue

            kite_status = status_raw.get("status", "")
            fill_price = float(status_raw.get("average_price") or 0)
            filled_qty = int(status_raw.get("filled_quantity") or 0)

            log.debug(
                "order_status_poll",
                order_id=order_id,
                status=kite_status,
                fill_price=fill_price,
                attempt=attempt + 1,
            )

            if kite_status == "COMPLETE":
                log.info(
                    "order_filled",
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    fill_price=fill_price,
                    quantity=filled_qty,
                )
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=filled_qty or quantity,
                    status="COMPLETE",
                    fill_price=fill_price,
                    fill_time=datetime.now(timezone.utc),
                    raw=status_raw,
                )
            elif kite_status in ("REJECTED", "CANCELLED"):
                msg = status_raw.get("status_message", "")
                log.warning(
                    "order_terminal_non_fill",
                    order_id=order_id,
                    status=kite_status,
                    msg=msg,
                )
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    status=kite_status,
                    fill_price=0.0,
                    fill_time=None,
                    error_message=msg,
                    raw=status_raw,
                )

        log.error("order_fill_timeout", order_id=order_id)
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            status="PENDING",
            fill_price=0.0,
            fill_time=None,
            error_message="fill_confirmation_timeout",
        )

    @staticmethod
    def _error_result(
        symbol: str, action: str, quantity: int, error: str
    ) -> OrderResult:
        return OrderResult(
            order_id="",
            symbol=symbol,
            action=action,
            quantity=quantity,
            status="ERROR",
            fill_price=0.0,
            fill_time=None,
            error_message=error,
        )
