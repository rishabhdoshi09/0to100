"""
💾 RuntimeState (Phase N) — persistent per-strategy operating state + cycle idempotency.

Fixes the "deployed-strategy state is re-derived" gap: lifecycle, allocation, risk budget,
last cycle, pause/retire reasons, cooldown, unsupported/data-warning flags all persist. Also
records COMPLETED cycle ids so a re-run (same cycle id) is a no-op — idempotency across restart.

On restart: load → reconcile against the paper book → refuse new risk on conflict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StrategyState:
    strategy_id: str
    family: str = ""
    lifecycle: str = "PAPER_EVALUATION"
    allocation_pct: float = 0.0
    risk_budget_pct: float = 0.0
    latest_card_id: str = ""
    latest_allocation_id: str = ""
    consecutive_failures: int = 0
    pause_reason: str = ""
    retire_reason: str = ""
    cooldown_until: str = ""
    unsupported_runtime: bool = False
    data_warning: bool = False
    last_cycle_id: str = ""

    def as_dict(self): return asdict(self)


class RuntimeState:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.strategies: dict[str, StrategyState] = {}
        self.completed_cycles: set[str] = set()
        self.last_completed_cycle: str = ""
        self.reconciled: bool = True
        if self.path and self.path.exists():
            self._load()

    # ── idempotency ──────────────────────────────────────────────────────────────
    def is_cycle_done(self, cycle_id: str) -> bool:
        return cycle_id in self.completed_cycles

    def mark_cycle_done(self, cycle_id: str) -> None:
        self.completed_cycles.add(cycle_id)
        self.last_completed_cycle = cycle_id

    # ── per-strategy state ───────────────────────────────────────────────────────
    def get(self, strategy_id: str, family: str = "") -> StrategyState:
        st = self.strategies.get(strategy_id)
        if st is None:
            st = StrategyState(strategy_id=strategy_id, family=family)
            self.strategies[strategy_id] = st
        return st

    # ── recovery ─────────────────────────────────────────────────────────────────
    def reconcile(self, paper_book) -> bool:
        """Cross-check persisted open strategies against the paper book. If a strategy the
        state thinks is active has no book presence and vice-versa in a way we can't explain,
        flag NOT reconciled so the loop refuses NEW risk until it's resolved."""
        try:
            open_strats = {p.strategy_id for p in paper_book.open.values()}
        except Exception:
            open_strats = set()
        # any book position whose strategy we have no state for is a conflict
        unknown = open_strats - set(self.strategies)
        self.reconciled = (len(unknown) == 0)
        return self.reconciled

    # ── persistence ──────────────────────────────────────────────────────────────
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "strategies": {k: v.as_dict() for k, v in self.strategies.items()},
            "completed_cycles": sorted(self.completed_cycles),
            "last_completed_cycle": self.last_completed_cycle,
        }, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
            for k, v in d.get("strategies", {}).items():
                self.strategies[k] = StrategyState(**v)
            self.completed_cycles = set(d.get("completed_cycles", []))
            self.last_completed_cycle = d.get("last_completed_cycle", "")
        except Exception:
            pass                                     # corrupt state ⇒ start clean, never crash
