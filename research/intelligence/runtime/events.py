"""
📡 Canonical loop events (Phase F) — the audit trail of a running cycle.

Event types are constants; `emit()` appends a deterministic `CanonicalEvent` to the store. Two
identical emits within a cycle dedupe (same content ⇒ same id), so re-running a cycle produces
no duplicate audit events. `summary` holds typed references/reasons only — never large blobs.
"""
from __future__ import annotations

from research.intelligence import schemas as SC

# lifecycle of a cycle
CYCLE_STARTED = "CYCLE_STARTED"
DATA_GATE_PASSED = "DATA_GATE_PASSED"
DATA_GATE_FAILED = "DATA_GATE_FAILED"
CYCLE_COMPLETED = "CYCLE_COMPLETED"
CYCLE_FAILED_SAFE = "CYCLE_FAILED_SAFE"
CYCLE_ALREADY_DONE = "CYCLE_ALREADY_DONE"
# strategy evaluation
STRATEGY_EVALUATION_STARTED = "STRATEGY_EVALUATION_STARTED"
STRATEGY_RUNTIME_UNSUPPORTED = "STRATEGY_RUNTIME_UNSUPPORTED"
SIGNAL_GENERATED = "SIGNAL_GENERATED"
SIGNAL_REJECTED = "SIGNAL_REJECTED"
MARKET_CONTEXT_DECODED = "MARKET_CONTEXT_DECODED"
EXECUTION_CONTEXT_DECODED = "EXECUTION_CONTEXT_DECODED"
OUTCOME_DECODED = "OUTCOME_DECODED"
# brains
EVIDENCE_CARD_CREATED = "EVIDENCE_CARD_CREATED"
EVIDENCE_CARD_UPDATED = "EVIDENCE_CARD_UPDATED"
ALLOCATION_DECISION_CREATED = "ALLOCATION_DECISION_CREATED"
# intents + execution
TRADE_INTENT_CREATED = "TRADE_INTENT_CREATED"
TRADE_INTENT_BLOCKED = "TRADE_INTENT_BLOCKED"
NEW_ENTRIES_BLOCKED = "NEW_ENTRIES_BLOCKED"
PAPER_POSITION_OPENED = "PAPER_POSITION_OPENED"
PAPER_POSITION_UPDATED = "PAPER_POSITION_UPDATED"
PAPER_POSITION_CLOSED = "PAPER_POSITION_CLOSED"
# allocation lifecycle
STRATEGY_ALLOCATION_INCREASED = "STRATEGY_ALLOCATION_INCREASED"
STRATEGY_ALLOCATION_REDUCED = "STRATEGY_ALLOCATION_REDUCED"
STRATEGY_PAUSED = "STRATEGY_PAUSED"
STRATEGY_RETIRED = "STRATEGY_RETIRED"


def emit(store, cycle_id: str, event_type: str, *, actor: str = "system",
         strategy_id: str = "", strategy_version: int = 0, rules_hash: str = "",
         data_snapshot_id: str = "", config_hash: str = "", symbol: str = "",
         decision: str = "", reason: str = "", result: str = "",
         causation_id: str = "", correlation_id: str = "", event_ts: str = "",
         summary: dict | None = None) -> SC.CanonicalEvent:
    ev = SC.CanonicalEvent(
        strategy_id=strategy_id, strategy_version=strategy_version, rules_hash=rules_hash,
        data_snapshot_id=data_snapshot_id, source="loop", event_ts=event_ts,
        event_type=event_type, cycle_id=cycle_id, causation_id=causation_id,
        correlation_id=correlation_id or cycle_id, actor=actor, config_hash=config_hash,
        symbol=symbol, decision=decision, reason=reason, result=result,
        summary=dict(summary or {}))
    store.append(ev)
    return ev
