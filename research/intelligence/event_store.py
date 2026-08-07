"""
🗄️ Canonical event store — append-only, single-writer, deterministically reconstructible.

Both brains read from here; only this store writes (single-writer pattern, guarded by one
lock). Appends are IDEMPOTENT: a record whose deterministic id is already present is ignored,
so reprocessing identical raw input never creates a duplicate semantic event.

State for either brain is a pure fold over the ordered event log, so the whole system can be
rebuilt from the persisted JSONL — nothing important lives only in memory.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from research.intelligence import schemas as SC


class EventStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()          # single-writer
        self._events: list = []
        self._ids: set[str] = set()
        if self.path and self.path.exists():
            self._load()

    # ── the ONLY mutation: append (idempotent) ───────────────────────────────────
    def append(self, record) -> bool:
        """Append one canonical record. Returns True if stored, False if it was a duplicate
        (same deterministic id). Thread-safe; the store is the single writer."""
        with self._lock:
            if record.record_id in self._ids:
                return False
            self._events.append(record)
            self._ids.add(record.record_id)
            if self.path:
                self._write_line(record)
            return True

    def extend(self, records) -> int:
        return sum(1 for r in records if self.append(r))

    # ── reads ─────────────────────────────────────────────────────────────────────
    def all(self) -> list:
        with self._lock:
            return list(self._events)

    def of_type(self, kind: str) -> list:
        return [e for e in self.all() if type(e).__name__ == kind]

    def for_strategy(self, strategy_id: str) -> list:
        return [e for e in self.all() if e.strategy_id == strategy_id]

    def __len__(self):
        return len(self._events)

    # ── deterministic reconstruction ─────────────────────────────────────────────
    def latest_cards(self) -> dict:
        """Latest StrategyEvidenceCard per (strategy_id, version) — a fold over the log."""
        out: dict = {}
        for e in self.all():
            if isinstance(e, SC.StrategyEvidenceCard):
                out[(e.strategy_id, e.strategy_version)] = e
        return out

    def latest_allocations(self) -> dict:
        out: dict = {}
        for e in self.all():
            if isinstance(e, SC.PaperAllocationDecision):
                out[e.strategy_id] = e
        return out

    def latest_target_portfolio(self) -> SC.TargetPortfolio | None:
        """Return the newest immutable target portfolio in append order."""
        portfolios = self.of_type("TargetPortfolio")
        return portfolios[-1] if portfolios else None

    def target_positions_for(self, portfolio: SC.TargetPortfolio | str) -> list[SC.TargetPosition]:
        """Resolve a portfolio's ordered TargetPosition records from the append-only log."""
        if isinstance(portfolio, str):
            selected = next(
                (item for item in reversed(self.of_type("TargetPortfolio"))
                 if item.record_id == portfolio),
                None,
            )
            if selected is None:
                return []
            portfolio = selected
        by_id = {item.record_id: item for item in self.of_type("TargetPosition")}
        return [by_id[position_id] for position_id in portfolio.position_ids if position_id in by_id]

    def reconstruct(self) -> dict:
        """Rebuild a compact snapshot of both brains' state purely from the event log."""
        by_type: dict[str, int] = {}
        for e in self.all():
            by_type[type(e).__name__] = by_type.get(type(e).__name__, 0) + 1
        latest_target = self.latest_target_portfolio()
        return {
            "n_events": len(self._events),
            "by_type": by_type,
            "cards": {f"{k[0]}@v{k[1]}": v.evidence_state
                      for k, v in self.latest_cards().items()},
            "allocations": {k: v.action for k, v in self.latest_allocations().items()},
            "target_portfolio": (
                {
                    "record_id": latest_target.record_id,
                    "cycle_id": latest_target.cycle_id,
                    "positions": len(latest_target.position_ids),
                    "executable": len(latest_target.executable_position_ids),
                    "blocked": len(latest_target.blocked_position_ids),
                }
                if latest_target else None
            ),
        }

    # ── persistence (append-only JSONL) ──────────────────────────────────────────
    def _write_line(self, record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": type(record).__name__, "data": record.as_dict()},
                               ensure_ascii=False) + "\n")

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rec = SC.from_dict(row["kind"], row["data"])
                if rec.record_id not in self._ids:
                    self._events.append(rec)
                    self._ids.add(rec.record_id)
            except Exception:
                continue                        # a torn/foreign line is skipped, never fatal
