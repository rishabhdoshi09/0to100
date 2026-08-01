"""Idempotent shadow intake from the canonical intelligence event store into the OMS.

This seam persists TradeIntents as PROPOSED orders only. It does not approve risk, prepare a
submission, select a broker, or perform network access.
"""
from __future__ import annotations

from typing import Any

from execution.oms.store import OmsStore


def ingest_event_store_intents(event_store, oms_store: OmsStore, *, cycle_id: str = "") -> dict[str, Any]:
    """Persist all eligible TradeIntent records exactly once.

    Legacy intents without Target Portfolio provenance are skipped rather than upgraded by
    inference. Repeated calls are safe because TradeIntent.record_id is the OMS idempotency key.
    """
    accepted: list[str] = []
    existing: list[str] = []
    skipped: list[dict[str, str]] = []
    for intent in event_store.of_type("TradeIntent"):
        if cycle_id and str(getattr(intent, "cycle_id", "")) != cycle_id:
            continue
        if not getattr(intent, "target_portfolio_id", "") or not getattr(intent, "target_position_id", ""):
            skipped.append({"trade_intent_id": intent.record_id, "reason": "LEGACY_UNLINKED_INTENT"})
            continue
        if int(getattr(intent, "required_quantity", 0) or 0) <= 0:
            skipped.append({"trade_intent_id": intent.record_id, "reason": "NO_POSITIVE_DELTA"})
            continue
        before = oms_store.get_by_intent(intent.record_id)
        order = oms_store.ingest_intent(intent)
        if before is None:
            accepted.append(order.order_id)
        else:
            existing.append(order.order_id)
    return {
        "accepted": accepted,
        "existing": existing,
        "skipped": skipped,
        "accepted_count": len(accepted),
        "existing_count": len(existing),
        "skipped_count": len(skipped),
    }
