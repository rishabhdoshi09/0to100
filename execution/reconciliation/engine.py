"""Pure reconciliation engine for internal OMS state versus a broker snapshot.

The engine classifies mismatches and proposes only deterministic repairs. It performs no
network access and no database mutation. Incomplete broker snapshots freeze entries instead of
being interpreted as empty orders, trades, positions, cash or margin.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Iterable, Mapping

from execution.oms import models as OM
from execution.reconciliation import models as R

_BROKER_VISIBLE = frozenset({
    OM.SUBMISSION_PENDING,
    OM.BROKER_ACKNOWLEDGED,
    OM.PARTIALLY_FILLED,
    OM.FILLED,
    OM.PROTECTION_PENDING,
    OM.PROTECTED,
    OM.EXIT_PENDING,
    OM.UNKNOWN,
    OM.QUARANTINED,
    OM.RECOVERY_REQUIRED,
})


def reconcile(
    *,
    internal_orders: Iterable[OM.OrderSnapshot],
    internal_fills: Mapping[str, Iterable[OM.FillSnapshot]] | None,
    internal_positions: Iterable[R.InternalPositionSnapshot],
    broker: R.BrokerAccountSnapshot,
    internal_cash: float | None = None,
    internal_available_margin: float | None = None,
    cash_tolerance: float = 1.0,
    margin_tolerance: float = 1.0,
) -> R.ReconciliationReport:
    orders = tuple(internal_orders)
    fills_by_order = {
        str(order_id): tuple(fills)
        for order_id, fills in dict(internal_fills or {}).items()
    }
    positions = tuple(internal_positions)
    issues: list[R.ReconciliationIssue] = []
    repairs: list[R.RepairAction] = []

    if not broker.snapshot_id:
        issues.append(_issue(
            code="BROKER_SNAPSHOT_MISSING",
            severity=R.CRITICAL,
            action=R.FREEZE_ENTRIES,
            message="Broker snapshot has no identity; no reconciliation claim is possible.",
        ))
    completeness = {
        "orders_complete": broker.orders_complete,
        "trades_complete": broker.trades_complete,
        "positions_complete": broker.positions_complete,
        "account_complete": broker.account_complete,
    }
    missing_lanes = [name for name, complete in completeness.items() if not complete]
    if missing_lanes or broker.errors:
        issues.append(_issue(
            code="BROKER_SNAPSHOT_INCOMPLETE",
            severity=R.CRITICAL,
            action=R.FREEZE_ENTRIES,
            message="Broker snapshot is incomplete; missing lanes are not treated as empty.",
            details={"missing_lanes": missing_lanes, "errors": list(broker.errors)},
        ))

    broker_orders_by_id = {
        order.broker_order_id: order for order in broker.orders if order.broker_order_id
    }
    broker_orders_by_ref: dict[str, list[R.BrokerOrderSnapshot]] = defaultdict(list)
    for order in broker.orders:
        for ref in {order.client_order_ref, order.submission_token}:
            if ref:
                broker_orders_by_ref[ref].append(order)

    internal_by_broker = {
        order.broker_order_id: order for order in orders if order.broker_order_id
    }
    matched_broker_ids: set[str] = set()
    matched_orders = 0

    trade_id_counts = Counter(trade.trade_id for trade in broker.trades if trade.trade_id)
    for trade_id, count in trade_id_counts.items():
        if count > 1:
            issues.append(_issue(
                code="DUPLICATE_BROKER_TRADE_ID",
                severity=R.CRITICAL,
                action=R.QUARANTINE,
                message=f"Broker trade id {trade_id} appears {count} times.",
                details={"trade_id": trade_id, "count": count},
            ))
    broker_trades_by_order: dict[str, list[R.BrokerTradeSnapshot]] = defaultdict(list)
    for trade in broker.trades:
        broker_trades_by_order[trade.broker_order_id].append(trade)

    for internal in orders:
        broker_order, match_issue = _match_order(
            internal,
            broker_orders_by_id=broker_orders_by_id,
            broker_orders_by_ref=broker_orders_by_ref,
        )
        if match_issue is not None:
            issues.append(match_issue)
        if broker_order is None:
            if broker.orders_complete and internal.status in _BROKER_VISIBLE:
                action = R.QUARANTINE if internal.status != OM.SUBMISSION_PENDING else R.MANUAL_REVIEW
                issues.append(_issue(
                    code="MISSING_BROKER_ORDER",
                    severity=R.CRITICAL,
                    action=action,
                    message=(
                        f"Internal order {internal.order_id} is {internal.status}, but no broker "
                        "order matches its broker id, submission token or client reference."
                    ),
                    order_id=internal.order_id,
                    symbol=internal.symbol,
                    details={
                        "internal_status": internal.status,
                        "broker_order_id": internal.broker_order_id,
                        "submission_token": internal.submission_token,
                    },
                ))
            continue

        matched_orders += 1
        matched_broker_ids.add(broker_order.broker_order_id)
        issues.extend(_validate_order_identity(internal, broker_order))
        broker_status = _broker_status(broker_order.status)
        trades = broker_trades_by_order.get(broker_order.broker_order_id, [])
        internal_fill_list = fills_by_order.get(internal.order_id, ())
        internal_fill_ids = {fill.external_fill_id for fill in internal_fill_list}
        internal_fill_qty = sum(fill.quantity for fill in internal_fill_list)
        broker_trade_qty = sum(trade.quantity for trade in trades)

        if internal.status in {OM.SUBMISSION_PENDING, OM.UNKNOWN, OM.RECOVERY_REQUIRED}:
            repairs.append(_repair(
                action_type="ACKNOWLEDGE",
                order_id=internal.order_id,
                broker_order_id=broker_order.broker_order_id,
                external_event_id=f"recon-ack:{broker.snapshot_id}:{broker_order.broker_order_id}",
                payload={"broker_status": broker_order.status},
            ))

        for trade in trades:
            if trade.trade_id and trade.trade_id not in internal_fill_ids:
                repairs.append(_repair(
                    action_type="RECORD_FILL",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    external_event_id=trade.trade_id,
                    payload=trade.as_dict(),
                ))

        if broker.trades_complete:
            if internal_fill_qty > broker_trade_qty:
                issues.append(_issue(
                    code="INTERNAL_FILL_AHEAD_OF_BROKER",
                    severity=R.CRITICAL,
                    action=R.QUARANTINE,
                    message="Internal fills exceed the complete broker trade book.",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=internal.symbol,
                    details={
                        "internal_fill_quantity": internal_fill_qty,
                        "broker_trade_quantity": broker_trade_qty,
                    },
                ))
            if broker_order.filled_quantity != broker_trade_qty:
                issues.append(_issue(
                    code="BROKER_ORDER_TRADE_DISAGREEMENT",
                    severity=R.CRITICAL,
                    action=R.FREEZE_ENTRIES,
                    message="Broker order filled quantity disagrees with its complete trade book.",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=internal.symbol,
                    details={
                        "broker_order_filled_quantity": broker_order.filled_quantity,
                        "broker_trade_quantity": broker_trade_qty,
                    },
                ))
            if broker_trade_qty > internal.approved_quantity > 0:
                issues.append(_issue(
                    code="BROKER_OVERFILL",
                    severity=R.CRITICAL,
                    action=R.QUARANTINE,
                    message="Broker trade quantity exceeds the OMS-approved quantity.",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=internal.symbol,
                    details={
                        "approved_quantity": internal.approved_quantity,
                        "broker_trade_quantity": broker_trade_qty,
                    },
                ))

        if broker_status == "REJECTED" and internal.status not in OM.TERMINAL_STATUSES:
            if broker_trade_qty == 0:
                repairs.append(_repair(
                    action_type="REJECT",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    external_event_id=f"recon-reject:{broker.snapshot_id}:{broker_order.broker_order_id}",
                    payload={"reason": broker_order.status_message or "broker rejected order"},
                ))
            else:
                issues.append(_issue(
                    code="REJECTED_ORDER_HAS_TRADES",
                    severity=R.CRITICAL,
                    action=R.QUARANTINE,
                    message="Broker reports a rejected order that also has trades.",
                    order_id=internal.order_id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=internal.symbol,
                ))
        elif broker_status == "CANCELLED" and internal.status not in OM.TERMINAL_STATUSES:
            repairs.append(_repair(
                action_type="CANCEL",
                order_id=internal.order_id,
                broker_order_id=broker_order.broker_order_id,
                external_event_id=f"recon-cancel:{broker.snapshot_id}:{broker_order.broker_order_id}",
                payload={"reason": broker_order.status_message or "broker cancelled order"},
            ))
        elif broker_status == "FILLED" and broker.trades_complete and broker_trade_qty < broker_order.quantity:
            issues.append(_issue(
                code="BROKER_COMPLETE_WITH_SHORT_FILL",
                severity=R.CRITICAL,
                action=R.QUARANTINE,
                message="Broker marks order complete while recorded trades do not reach order quantity.",
                order_id=internal.order_id,
                broker_order_id=broker_order.broker_order_id,
                symbol=internal.symbol,
                details={
                    "order_quantity": broker_order.quantity,
                    "trade_quantity": broker_trade_qty,
                },
            ))

    if broker.orders_complete:
        for broker_order in broker.orders:
            if not broker_order.broker_order_id or broker_order.broker_order_id in matched_broker_ids:
                continue
            if broker_order.broker_order_id in internal_by_broker:
                continue
            issues.append(_issue(
                code="UNKNOWN_BROKER_ORDER",
                severity=R.CRITICAL,
                action=R.FREEZE_ENTRIES,
                message="Broker contains an order that is not owned by the OMS ledger.",
                broker_order_id=broker_order.broker_order_id,
                symbol=broker_order.symbol,
                details=broker_order.as_dict(),
            ))

    if broker.positions_complete:
        issues.extend(_position_issues(positions, broker.positions))
    if broker.account_complete:
        if internal_cash is not None and abs(float(internal_cash) - broker.cash) > cash_tolerance:
            issues.append(_issue(
                code="CASH_MISMATCH",
                severity=R.CRITICAL,
                action=R.FREEZE_ENTRIES,
                message="Internal cash disagrees with broker cash.",
                details={"internal_cash": internal_cash, "broker_cash": broker.cash},
            ))
        if (
            internal_available_margin is not None
            and abs(float(internal_available_margin) - broker.available_margin) > margin_tolerance
        ):
            issues.append(_issue(
                code="MARGIN_MISMATCH",
                severity=R.CRITICAL,
                action=R.FREEZE_ENTRIES,
                message="Internal available margin disagrees with broker margin.",
                details={
                    "internal_available_margin": internal_available_margin,
                    "broker_available_margin": broker.available_margin,
                },
            ))

    entry_freeze = any(
        issue.severity == R.CRITICAL
        or issue.action in {R.QUARANTINE, R.MANUAL_REVIEW, R.FREEZE_ENTRIES}
        for issue in issues
    )
    if missing_lanes or broker.errors or not broker.snapshot_id:
        status = R.INCOMPLETE
    elif entry_freeze:
        status = R.QUARANTINED
    elif repairs:
        status = R.REPAIRABLE
    else:
        status = R.HEALTHY

    report_payload = {
        "broker_snapshot_id": broker.snapshot_id,
        "observed_at": broker.observed_at,
        "status": status,
        "entry_freeze_required": entry_freeze,
        "issues": [issue.as_dict() for issue in issues],
        "repairs": [repair.as_dict() for repair in repairs],
        "matched_orders": matched_orders,
        "internal_orders": len(orders),
        "broker_orders": len(broker.orders),
        "internal_positions": len(positions),
        "broker_positions": len(broker.positions),
    }
    report_id = f"recon-{_hash(report_payload)[:20]}"
    return R.ReconciliationReport(
        report_id=report_id,
        broker_snapshot_id=broker.snapshot_id,
        observed_at=broker.observed_at,
        status=status,
        entry_freeze_required=entry_freeze,
        issues=tuple(issues),
        repairs=tuple(_dedupe_repairs(repairs)),
        matched_orders=matched_orders,
        internal_orders=len(orders),
        broker_orders=len(broker.orders),
        internal_positions=len(positions),
        broker_positions=len(broker.positions),
        summary={
            "issue_counts": dict(Counter(issue.code for issue in issues)),
            "repair_counts": dict(Counter(repair.action_type for repair in repairs)),
            "snapshot_complete": broker.complete,
        },
    )


def _match_order(
    internal: OM.OrderSnapshot,
    *,
    broker_orders_by_id: Mapping[str, R.BrokerOrderSnapshot],
    broker_orders_by_ref: Mapping[str, list[R.BrokerOrderSnapshot]],
):
    if internal.broker_order_id:
        return broker_orders_by_id.get(internal.broker_order_id), None
    matches: dict[str, R.BrokerOrderSnapshot] = {}
    for ref in {internal.submission_token, internal.idempotency_key, internal.trade_intent_id}:
        if not ref:
            continue
        for item in broker_orders_by_ref.get(ref, []):
            matches[item.broker_order_id] = item
    if len(matches) > 1:
        return None, _issue(
            code="AMBIGUOUS_BROKER_ORDER_MATCH",
            severity=R.CRITICAL,
            action=R.QUARANTINE,
            message="Multiple broker orders match one OMS order reference.",
            order_id=internal.order_id,
            symbol=internal.symbol,
            details={"broker_order_ids": sorted(matches)},
        )
    return (next(iter(matches.values())) if matches else None), None


def _validate_order_identity(internal: OM.OrderSnapshot, broker: R.BrokerOrderSnapshot):
    issues: list[R.ReconciliationIssue] = []
    comparisons = {
        "symbol": (internal.symbol.upper(), broker.symbol.upper()),
        "side": (internal.side.upper(), broker.side.upper()),
        "quantity": (internal.approved_quantity or internal.requested_quantity, broker.quantity),
    }
    mismatches = {
        key: {"internal": left, "broker": right}
        for key, (left, right) in comparisons.items()
        if left != right
    }
    if mismatches:
        issues.append(_issue(
            code="BROKER_ORDER_IDENTITY_MISMATCH",
            severity=R.CRITICAL,
            action=R.QUARANTINE,
            message="Broker order identity disagrees with the durable OMS order.",
            order_id=internal.order_id,
            broker_order_id=broker.broker_order_id,
            symbol=internal.symbol,
            details=mismatches,
        ))
    return issues


def _position_issues(
    internal_positions: Iterable[R.InternalPositionSnapshot],
    broker_positions: Iterable[R.BrokerPositionSnapshot],
):
    internal_qty: dict[str, int] = defaultdict(int)
    internal_protected: dict[str, int] = defaultdict(int)
    broker_qty: dict[str, int] = defaultdict(int)
    broker_protected: dict[str, int] = defaultdict(int)
    for position in internal_positions:
        symbol = position.symbol.upper()
        internal_qty[symbol] += position.quantity
        internal_protected[symbol] += position.protected_quantity
    for position in broker_positions:
        symbol = position.symbol.upper()
        broker_qty[symbol] += position.quantity
        broker_protected[symbol] += position.protected_quantity

    issues: list[R.ReconciliationIssue] = []
    for symbol in sorted(set(internal_qty) | set(broker_qty)):
        if internal_qty[symbol] != broker_qty[symbol]:
            issues.append(_issue(
                code="POSITION_QUANTITY_MISMATCH",
                severity=R.CRITICAL,
                action=R.QUARANTINE,
                message="Internal and broker position quantities disagree.",
                symbol=symbol,
                details={
                    "internal_quantity": internal_qty[symbol],
                    "broker_quantity": broker_qty[symbol],
                },
            ))
        if broker_qty[symbol] > broker_protected[symbol]:
            issues.append(_issue(
                code="BROKER_POSITION_UNPROTECTED",
                severity=R.CRITICAL,
                action=R.FREEZE_ENTRIES,
                message="Broker position is not fully protected.",
                symbol=symbol,
                details={
                    "broker_quantity": broker_qty[symbol],
                    "broker_protected_quantity": broker_protected[symbol],
                    "internal_protected_quantity": internal_protected[symbol],
                },
            ))
    return issues


def _broker_status(value: str) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"COMPLETE", "FILLED"}:
        return "FILLED"
    if text in {"REJECTED"}:
        return "REJECTED"
    if text in {"CANCELLED", "CANCELED", "EXPIRED"}:
        return "CANCELLED"
    return "OPEN"


def _issue(**kwargs) -> R.ReconciliationIssue:
    payload = dict(kwargs)
    issue_id = f"issue-{_hash(payload)[:20]}"
    return R.ReconciliationIssue(issue_id=issue_id, **kwargs)


def _repair(**kwargs) -> R.RepairAction:
    payload = dict(kwargs)
    action_id = f"repair-{_hash(payload)[:20]}"
    return R.RepairAction(action_id=action_id, **kwargs)


def _dedupe_repairs(repairs: Iterable[R.RepairAction]):
    seen: set[str] = set()
    out: list[R.RepairAction] = []
    for repair in repairs:
        if repair.action_id in seen:
            continue
        seen.add(repair.action_id)
        out.append(repair)
    return out


def _hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
