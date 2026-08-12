"""
CycleContext (Phase B/C) — the deterministic inputs to one intelligence cycle.

Carries validated data state, the frozen strategy registry, per-strategy point-in-time bar
history, the paper book, persisted runtime state, the event store, knowledge, and config. The
`cycle_id` is deterministic over the identity fields, so re-running the same cycle is detectable
and idempotent.

Phase A / A4 adds optional research seams (`market_structure`, `network_risk`,
`horizon_view`, `challenger_evidence`). They default to None, are excluded from
`cycle_id()` identity, and must not alter Brain/execution behaviour until an
evidence gate promotes a producer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research.intelligence.runtime.research_seams import (
        ChallengerEvidenceView,
        HorizonView,
        MarketStructureView,
        NetworkRiskView,
    )


@dataclass
class CycleContext:
    as_of_date: str
    cycle_type: str = "paper_session"          # paper_session / daily_growth / recovery
    mode: str = "PAPER_AUTO"
    data_ok: bool = False                      # honest default: no data ⇒ no action
    data_snapshot_id: str = ""
    market_regime: str = "RISK_ON"
    dataset_tier: str = ""                      # evidence tier stamped on cards (Phase 19)
    config_hash: str = "cfg0"
    registry_version: str = "reg0"
    strategies: list = field(default_factory=list)      # frozen StrategySpec objects
    # per-strategy point-in-time bar history: {strategy_id: {symbol: [Bar,...]}} where the last
    # bar is as_of_date (adapters only look at bars strictly before each index — no look-ahead)
    data: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)         # {strategy_id: cluster_id}
    benchmark: list = field(default_factory=list)        # [Bar,...] PIT benchmark (rel-strength)
    forward_eligible: bool = True                        # dataset may OPEN new entries
    session_phase: str = "eod"                           # premarket / opening_noise / intraday / eod
    # Operational entry permission is supplied by the autonomy supervisor BEFORE the cycle runs.
    # It is separate from data forward-eligibility: either gate can refuse new risk while the
    # cycle still manages existing positions and updates evidence.
    new_entries_allowed: bool = True
    entry_block_reason: str = ""
    capability_failures: tuple[str, ...] = ()
    live_confirmation_required: bool = False
    fresh_live_symbols: frozenset[str] = field(default_factory=frozenset)
    # ── Phase A / A4 research seams (optional; None = not computed) ──────────
    market_structure: "MarketStructureView | None" = None
    network_risk: "NetworkRiskView | None" = None
    horizon_view: "HorizonView | None" = None
    challenger_evidence: "ChallengerEvidenceView | None" = None

    def cycle_id(self) -> str:
        ident = {
            "date": self.as_of_date, "type": self.cycle_type,
            "phase": self.session_phase, "snapshot": self.data_snapshot_id,
            "registry": self.registry_version, "config": self.config_hash,
            "entry_allowed": bool(self.new_entries_allowed),
            "entry_block": self.entry_block_reason,
        }
        blob = json.dumps(ident, sort_keys=True).encode()
        return f"cyc-{hashlib.sha256(blob).hexdigest()[:16]}"

    def today_bar(self, symbol: str):
        """The (open, high, low, close) tuple for `symbol` on as_of_date, or None."""
        for sym_map in self.data.values():
            hist = sym_map.get(symbol)
            if hist:
                b = hist[-1]
                return (b.open, b.high, b.low, b.close)
        return None
