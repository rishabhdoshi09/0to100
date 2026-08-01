"""Read-only Zerodha snapshot adapter for reconciliation and protection evidence.

The adapter translates Kite's orders, trades, positions, margins and GTT responses into
broker-neutral immutable snapshots. Every endpoint is captured independently: a failed or
malformed lane remains explicitly incomplete and is never interpreted as an empty account.

This module has no order-placement, modification, cancellation or GTT-creation capability.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from execution.protection.models import BrokerProtectionSnapshot
from execution.reconciliation.models import (
    BrokerAccountSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
)


@dataclass(frozen=True)
class ZerodhaSnapshotBundle:
    account: BrokerAccountSnapshot
    protections: tuple[BrokerProtectionSnapshot, ...]
    protections_complete: bool
    errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.account.complete and self.protections_complete and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_zerodha_snapshot(
    client=None,
    *,
    observed_at: datetime | str | None = None,
) -> ZerodhaSnapshotBundle:
    """Capture one read-only broker snapshot.

    ``client`` may be KiteConnect itself or QuantTerm's ``KiteClient`` wrapper. When omitted,
    the wrapper is constructed lazily. No missing endpoint is retried through a mutating API.
    """
    raw = _raw_client(client)
    stamp = _iso(observed_at)
    errors: list[str] = []

    order_rows, orders_complete = _capture_lane(
        raw, "orders", _sequence, errors, lane="orders"
    )
    trade_rows, trades_complete = _capture_lane(
        raw, "trades", _sequence, errors, lane="trades"
    )
    positions_payload, positions_complete = _capture_lane(
        raw, "positions", _mapping, errors, lane="positions"
    )
    margins_payload, account_complete = _capture_lane(
        raw, "margins", _mapping, errors, lane="margins"
    )
    gtt_rows, protections_complete = _capture_lane(
        raw, "get_gtts", _sequence, errors, lane="gtts"
    )

    orders = tuple(_order(row) for row in order_rows if isinstance(row, Mapping))
    trades = tuple(_trade(row) for row in trade_rows if isinstance(row, Mapping))
    positions = tuple(
        _position(row)
        for row in _position_rows(positions_payload)
        if isinstance(row, Mapping)
    )
    cash, available_margin = _account_values(margins_payload)
    protections = tuple(
        protection
        for row in gtt_rows
        if isinstance(row, Mapping)
        for protection in (_protection(row),)
        if protection is not None
    )

    semantic = {
        "observed_at": stamp,
        "orders": [item.as_dict() for item in orders],
        "trades": [item.as_dict() for item in trades],
        "positions": [item.as_dict() for item in positions],
        "cash": cash,
        "available_margin": available_margin,
        "completeness": {
            "orders": orders_complete,
            "trades": trades_complete,
            "positions": positions_complete,
            "account": account_complete,
            "protections": protections_complete,
        },
        "errors": errors,
    }
    snapshot_id = f"zerodha-{hashlib.sha256(_canonical(semantic).encode()).hexdigest()[:20]}"
    account_errors = tuple(
        error for error in errors if not error.startswith("gtts:")
    )
    account = BrokerAccountSnapshot(
        snapshot_id=snapshot_id,
        observed_at=stamp,
        source="zerodha_kite_read_only",
        orders=orders,
        trades=trades,
        positions=positions,
        cash=cash,
        available_margin=available_margin,
        orders_complete=orders_complete,
        trades_complete=trades_complete,
        positions_complete=positions_complete,
        account_complete=account_complete,
        errors=account_errors,
    )
    return ZerodhaSnapshotBundle(
        account=account,
        protections=protections,
        protections_complete=protections_complete,
        errors=tuple(errors),
    )


def _raw_client(client):
    if client is None:
        from data.kite_client import KiteClient

        client = KiteClient()
    return getattr(client, "raw", client)


def _capture_lane(
    raw,
    method_name: str,
    validator: Callable[[Any], Any],
    errors: list[str],
    *,
    lane: str,
):
    method = getattr(raw, method_name, None)
    if not callable(method):
        errors.append(f"{lane}:ENDPOINT_UNAVAILABLE:{method_name}")
        return validator(None), False
    try:
        value = method()
    except Exception as exc:
        errors.append(f"{lane}:{type(exc).__name__}:{exc}")
        return validator(None), False
    try:
        return validator(value), True
    except Exception as exc:
        errors.append(f"{lane}:MALFORMED_RESPONSE:{type(exc).__name__}:{exc}")
        return validator(None), False


def _sequence(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("expected a sequence")
    return tuple(value)


def _mapping(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return dict(value)


def _order(row: Mapping[str, Any]) -> BrokerOrderSnapshot:
    tag = _text(row.get("tag"))
    return BrokerOrderSnapshot(
        broker_order_id=_text(row.get("order_id")),
        status=_text(row.get("status")),
        symbol=_text(row.get("tradingsymbol") or row.get("symbol")).upper(),
        side=_text(row.get("transaction_type") or row.get("side")).upper(),
        quantity=_integer(row.get("quantity")),
        filled_quantity=_integer(row.get("filled_quantity")),
        average_price=_number(row.get("average_price")),
        client_order_ref=tag or _text(row.get("guid")),
        submission_token=tag,
        status_message=_text(
            row.get("status_message_raw") or row.get("status_message")
        ),
        updated_at=_timestamp(
            row.get("exchange_update_timestamp")
            or row.get("update_timestamp")
            or row.get("order_timestamp")
        ),
        raw=dict(row),
    )


def _trade(row: Mapping[str, Any]) -> BrokerTradeSnapshot:
    return BrokerTradeSnapshot(
        trade_id=_text(row.get("trade_id")),
        broker_order_id=_text(row.get("order_id")),
        symbol=_text(row.get("tradingsymbol") or row.get("symbol")).upper(),
        side=_text(row.get("transaction_type") or row.get("side")).upper(),
        quantity=_integer(row.get("quantity")),
        price=_number(row.get("average_price") or row.get("price")),
        executed_at=_timestamp(
            row.get("fill_timestamp")
            or row.get("exchange_timestamp")
            or row.get("order_timestamp")
        ),
        raw=dict(row),
    )


def _position_rows(payload: Mapping[str, Any]) -> tuple:
    rows = payload.get("net", ())
    return _sequence(rows)


def _position(row: Mapping[str, Any]) -> BrokerPositionSnapshot:
    return BrokerPositionSnapshot(
        symbol=_text(row.get("tradingsymbol") or row.get("symbol")).upper(),
        quantity=_integer(row.get("quantity")),
        average_price=_number(row.get("average_price")),
        product=_text(row.get("product")),
        # Exchange-side protection is reconciled from the GTT/protection lane, never inferred.
        protected_quantity=0,
        raw=dict(row),
    )


def _account_values(payload: Mapping[str, Any]) -> tuple[float, float]:
    equity = payload.get("equity", payload)
    if not isinstance(equity, Mapping):
        return 0.0, 0.0
    available = equity.get("available", {})
    if not isinstance(available, Mapping):
        available = {}
    cash = _number(
        available.get("cash")
        or available.get("live_balance")
        or equity.get("cash")
    )
    available_margin = _number(
        equity.get("net")
        or available.get("live_balance")
        or available.get("cash")
    )
    return cash, available_margin


def _protection(row: Mapping[str, Any]) -> BrokerProtectionSnapshot | None:
    trigger_id = _text(row.get("id") or row.get("trigger_id"))
    condition = row.get("condition", {})
    if not isinstance(condition, Mapping):
        condition = {}
    orders = tuple(
        item for item in (row.get("orders") or ()) if isinstance(item, Mapping)
    )
    sell_orders = tuple(
        item
        for item in orders
        if _text(item.get("transaction_type") or "SELL").upper() == "SELL"
    )
    trigger_values = tuple(
        _number(value) for value in (condition.get("trigger_values") or ())
    )
    positive_triggers = tuple(value for value in trigger_values if value > 0)

    stop_order = sell_orders[0] if sell_orders else {}
    target_order = sell_orders[-1] if len(sell_orders) > 1 else {}
    stop_price = (
        positive_triggers[0]
        if positive_triggers
        else _number(stop_order.get("trigger_price") or stop_order.get("price"))
    )
    target_price = (
        positive_triggers[-1]
        if len(positive_triggers) > 1
        else _number(target_order.get("trigger_price") or target_order.get("price"))
    )
    quantity = max((_integer(item.get("quantity")) for item in sell_orders), default=0)
    symbol = _text(
        condition.get("tradingsymbol")
        or row.get("tradingsymbol")
        or row.get("symbol")
    ).upper()
    if not trigger_id or not symbol or quantity <= 0:
        return None

    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    order_id = _text(
        row.get("order_id")
        or metadata.get("order_id")
        or row.get("tag")
        or metadata.get("tag")
    )
    status = _text(row.get("status")).lower()
    active = status in {"active", "enabled"}
    stop_reference = _text(stop_order.get("order_id")) or f"{trigger_id}:stop"
    target_reference = (
        _text(target_order.get("order_id")) or f"{trigger_id}:target"
        if target_price > 0
        else ""
    )
    return BrokerProtectionSnapshot(
        broker_protection_id=trigger_id,
        order_id=order_id,
        symbol=symbol,
        active=active,
        quantity=quantity,
        stop_price=stop_price,
        target_price=target_price,
        stop_reference=stop_reference,
        target_reference=target_reference,
        updated_at=_timestamp(
            row.get("updated_at") or row.get("created_at") or row.get("expires_at")
        ),
        raw=dict(row),
    )


def _iso(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("observed_at must be datetime, ISO string or None")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    return str(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
