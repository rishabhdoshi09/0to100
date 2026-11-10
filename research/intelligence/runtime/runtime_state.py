"""
💾 RuntimeState (Phase N) — persistent per-strategy operating state + cycle idempotency.

Fixes the "deployed-strategy state is re-derived" gap: lifecycle, allocation, risk budget,
last cycle, pause/retire reasons, cooldown, unsupported/data-warning flags all persist. Also
records COMPLETED cycle ids so a re-run (same cycle id) is a no-op — idempotency across restart.

On restart: load → reconcile against the paper book → refuse new risk on conflict.
"""
from __future__ import annotations

import json
import os
import time
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
        self.load_error: str = ""
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
        """Cross-check persisted state against the paper book before new risk."""
        if self.load_error:
            self.reconciled = False
            return False
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
        if self.load_error:
            raise RuntimeError("refusing to overwrite unrecovered intelligence runtime state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "strategies": {k: v.as_dict() for k, v in self.strategies.items()},
            "completed_cycles": sorted(self.completed_cycles),
            "last_completed_cycle": self.last_completed_cycle,
        }, ensure_ascii=False, indent=2)
        # A torn write on detachable storage must not truncate the last
        # successfully committed cycle-idempotency record.
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("runtime state must be an object")
            raw_strategies = d.get("strategies", {})
            raw_cycles = d.get("completed_cycles", [])
            if not isinstance(raw_strategies, dict) or not isinstance(raw_cycles, list):
                raise ValueError("runtime state has invalid collection types")
            strategies = {k: StrategyState(**v) for k, v in raw_strategies.items()}
            completed = set(raw_cycles)
            last = str(d.get("last_completed_cycle") or "")
            self.strategies = strategies
            self.completed_cycles = completed
            self.last_completed_cycle = last
        except Exception as exc:
            # Do not promote an empty in-memory state to a new durable truth:
            # losing completed cycle ids can replay paper entries after restart.
            self.load_error = f"{type(exc).__name__}: {exc}"[:240]
            self.reconciled = False
