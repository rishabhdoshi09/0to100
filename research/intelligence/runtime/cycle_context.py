"""
CycleContext (Phase B/C) — the deterministic inputs to one intelligence cycle.

Carries validated data state, the frozen strategy registry, per-strategy point-in-time bar
history, the paper book, persisted runtime state, the event store, knowledge, and config. The
`cycle_id` is deterministic over the identity fields, so re-running the same cycle is detectable
and idempotent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class CycleContext:
    as_of_date: str
    cycle_type: str = "paper_session"          # paper_session / daily_growth / recovery
    mode: str = "PAPER_AUTO"
    data_ok: bool = False                      # honest default: no data ⇒ no action
    data_snapshot_id: str = ""
    market_regime: str = "RISK_ON"
    config_hash: str = "cfg0"
    registry_version: str = "reg0"
    strategies: list = field(default_factory=list)      # frozen StrategySpec objects
    # per-strategy point-in-time bar history: {strategy_id: {symbol: [Bar,...]}} where the last
    # bar is as_of_date (adapters only look at bars strictly before each index — no look-ahead)
    data: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)         # {strategy_id: cluster_id}
    session_phase: str = "eod"                           # open / intraday / eod

    def cycle_id(self) -> str:
        ident = {
            "date": self.as_of_date, "type": self.cycle_type,
            "phase": self.session_phase, "snapshot": self.data_snapshot_id,
            "registry": self.registry_version, "config": self.config_hash,
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
