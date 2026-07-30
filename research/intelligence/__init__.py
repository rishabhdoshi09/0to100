"""
🧠🧠 Two-Brain Intelligence — Brain 1 (evidence) and Brain 2 (allocation), connected only by
immutable typed records in an append-only canonical event store.

See docs/overhaul/TWO_BRAIN_ARCHITECTURE.md. Invariants: paper-only, LIVE structurally locked
(the USER_APPROVED transition is user-only — no brain or autopilot can perform it), no
synthetic-as-evidence, and no real data ⇒ no signals/experiments/deployments/positions.
"""
from research.intelligence import schemas
from research.intelligence.event_store import EventStore
from research.intelligence import decoder_registry
from research.intelligence import evidence_brain
from research.intelligence import allocation_brain
from research.intelligence import graduation
from research.intelligence import strategy_runtime

__all__ = ["schemas", "EventStore", "decoder_registry", "evidence_brain",
           "allocation_brain", "graduation", "strategy_runtime"]
