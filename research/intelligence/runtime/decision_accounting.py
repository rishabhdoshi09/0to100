"""Canonical post-cycle accounting for every generated paper signal.

The runtime already records generated signals, portfolio blocks and opened paper
positions. This module closes the remaining accounting gap: after the cycle has
finished, every generated signal is assigned exactly one terminal decision outcome
(TAKEN, BLOCKED or NOT_SELECTED). Non-taken generated signals are mirrored into
``signals_rejected`` so the existing paper self-feed can shadow-test them.

No signal is invented here. We only classify signals that the authoritative
strategy runtime actually generated for the decision-time snapshot. Where the
canonical EventStore is available, the exact decision-time entry/stop/target and
signal provenance are copied into the decision ledger as well.
"""
from __future__ import annotations

from typing import Any

from research.intelligence.runtime import events as EV

TAKEN = "TAKEN"
BLOCKED = "BLOCKED"
NOT_SELECTED = "NOT_SELECTED"


def _pair(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        return str(value.get("strategy_id") or ""), str(value.get("symbol") or "").upper()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0] or ""), str(value[1] or "").upper()
    return "", ""


def _blocked(value: Any) -> tuple[str, str, str]:
    if isinstance(value, dict):
        return (
            str(value.get("strategy_id") or ""),
            str(value.get("symbol") or "").upper(),
            str(value.get("reason") or value.get("reason_code") or "TARGET_BLOCKED"),
        )
    if isinstance(value, (list, tuple)):
        sid = str(value[0] or "") if len(value) >= 1 else ""
        symbol = str(value[1] or "").upper() if len(value) >= 2 else ""
        reason = str(value[2] or "TARGET_BLOCKED") if len(value) >= 3 else "TARGET_BLOCKED"
        return sid, symbol, reason
    return "", "", "TARGET_BLOCKED"


def _allocation_actions(result) -> dict[str, str]:
    actions: dict[str, str] = {}
    for item in list(getattr(result, "allocation_decisions", ()) or ()):
        if isinstance(item, dict):
            sid = str(item.get("strategy_id") or "")
            action = str(item.get("action") or item.get("decision") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            sid, action = str(item[0] or ""), str(item[1] or "")
        else:
            continue
        if sid:
            actions[sid] = action.upper()
    return actions


def _signal_provenance(store, ctx) -> dict[tuple[str, str], dict[str, Any]]:
    """Project exact CanonicalSignal levels for this cycle from the append-only store."""
    if store is None or not hasattr(store, "all"):
        return {}
    as_of = str(getattr(ctx, "as_of_date", "") or "")
    snapshot = str(getattr(ctx, "data_snapshot_id", "") or "")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        events = list(store.all())
    except Exception:
        return {}
    for event in events:
        if type(event).__name__ != "CanonicalSignal":
            continue
        sid = str(getattr(event, "strategy_id", "") or "")
        symbol = str(getattr(event, "symbol", "") or "").upper()
        if not sid or not symbol:
            continue
        event_ts = str(getattr(event, "event_ts", "") or "")
        event_snapshot = str(getattr(event, "data_snapshot_id", "") or "")
        if as_of and event_ts and event_ts[:10] != as_of[:10]:
            continue
        if snapshot and event_snapshot and event_snapshot != snapshot:
            continue
        out[(sid, symbol)] = {
            "entry": float(getattr(event, "entry", 0.0) or 0.0),
            "stop": float(getattr(event, "stop", 0.0) or 0.0),
            "target": float(getattr(event, "target", 0.0) or 0.0),
            "max_hold": int(getattr(event, "max_hold", 0) or 0),
            "rationale": str(getattr(event, "rationale", "") or ""),
            "strategy_version": int(getattr(event, "strategy_version", 0) or 0),
            "rules_hash": str(getattr(event, "rules_hash", "") or ""),
            "signal_record_id": str(getattr(event, "record_id", "") or ""),
        }
    return out


def finalize_cycle_decisions(ctx, result, store=None):
    """Assign a terminal decision to every generated signal.

    The function is deliberately idempotent. Re-running it replaces the typed
    ``decision_outcomes`` projection and only appends missing rejection tuples.
    Canonical events dedupe in EventStore by content.
    """
    generated: list[tuple[str, str]] = []
    seen_generated: set[tuple[str, str]] = set()
    for raw in list(getattr(result, "signals_generated", ()) or ()):
        sid, symbol = _pair(raw)
        key = (sid, symbol)
        if sid and symbol and key not in seen_generated:
            seen_generated.add(key)
            generated.append(key)

    taken = {
        _pair(raw)
        for raw in list(getattr(result, "positions_opened", ()) or ())
        if all(_pair(raw))
    }
    blocked_map: dict[tuple[str, str], str] = {}
    for raw in list(getattr(result, "blocked_target_positions", ()) or ()):
        sid, symbol, reason = _blocked(raw)
        if sid and symbol:
            blocked_map[(sid, symbol)] = reason or "TARGET_BLOCKED"

    actions = _allocation_actions(result)
    signals = _signal_provenance(store, ctx)
    outcomes: list[dict[str, Any]] = []
    rejected = list(getattr(result, "signals_rejected", ()) or ())
    existing_rejected = set()
    for raw in rejected:
        if isinstance(raw, dict):
            existing_rejected.add((
                str(raw.get("symbol") or "").upper(),
                str(raw.get("strategy_id") or ""),
            ))
        elif isinstance(raw, (list, tuple)):
            symbol = str(raw[0] or "").upper() if len(raw) >= 1 else ""
            sid = str(raw[1] or "") if len(raw) >= 2 else ""
            existing_rejected.add((symbol, sid))

    for sid, symbol in generated:
        key = (sid, symbol)
        if key in taken:
            decision = TAKEN
            reason = "PAPER_POSITION_OPENED"
        elif key in blocked_map:
            decision = BLOCKED
            reason = blocked_map[key]
        else:
            decision = NOT_SELECTED
            action = actions.get(sid, "")
            reason = f"ALLOCATION_{action}" if action and action not in {"DEPLOY", "INCREASE"} else NOT_SELECTED

        signal = signals.get(key, {})
        row: dict[str, Any] = {
            "strategy_id": sid,
            "symbol": symbol,
            "decision": decision,
            "reason": reason,
            "as_of": str(getattr(ctx, "as_of_date", "") or ""),
            "data_snapshot_id": str(getattr(ctx, "data_snapshot_id", "") or ""),
            "market_regime": str(getattr(ctx, "market_regime", "") or ""),
            **signal,
        }
        outcomes.append(row)

        if decision != TAKEN:
            reject_key = (symbol, sid)
            if reject_key not in existing_rejected:
                # Keep the legacy tuple projection for compatibility; exact levels
                # remain in decision_outcomes and the canonical signal event.
                rejected.append((symbol, sid, reason))
                existing_rejected.add(reject_key)
            if store is not None:
                EV.emit(
                    store,
                    str(getattr(result, "cycle_id", "") or ctx.cycle_id()),
                    EV.SIGNAL_REJECTED,
                    strategy_id=sid,
                    data_snapshot_id=str(getattr(ctx, "data_snapshot_id", "") or ""),
                    symbol=symbol,
                    decision=decision,
                    reason=reason,
                    event_ts=str(getattr(ctx, "as_of_date", "") or ""),
                    summary={
                        "terminal_decision": decision,
                        "signal_record_id": signal.get("signal_record_id", ""),
                        "entry": signal.get("entry"),
                        "stop": signal.get("stop"),
                        "target": signal.get("target"),
                    },
                )

    result.signals_rejected = rejected
    result.decision_outcomes = outcomes
    return result
