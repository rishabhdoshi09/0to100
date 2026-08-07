"""
🧵 The Research Thread — an append-only chain-of-thought record.

Every autonomous cycle writes its *reasoning*, not just its conclusion, here. The thread
is how a human "watches the system think": each entry is a short, plain-language step
(observe → reason → decide → propose → conclude) with the evidence that justified it.

Design contract:
  • APPEND-ONLY. Entries are never edited or deleted. History is the audit trail.
  • DETERMINISTIC content. The wording of an entry depends only on its inputs, never on
    wall-clock time. A `clock` is injected so tests are reproducible; the timestamp is
    provenance only and is NOT part of any identity/hash.
  • HONEST. Nothing here can place an order, approve a strategy, or unlock live. It only
    records what the brain observed and proposed.

Persistence is a JSONL file (one entry per line). Appending a line never rewrites earlier
lines, so a crash mid-write can lose at most the final line — never corrupt the past.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# entry kinds — the five moves of a transparent reasoning step
OBSERVE = "OBSERVE"     # a fact the brain read from the world (data readiness, prior runs)
REASON = "REASON"       # an inference drawn from observations (why an idea is weak/strong)
DECIDE = "DECIDE"       # a choice the brain made autonomously (reject / shortlist)
PROPOSE = "PROPOSE"     # a suggestion left for the human gate (new/better strategy)
CONCLUDE = "CONCLUDE"   # the cycle's honest bottom line
KINDS = (OBSERVE, REASON, DECIDE, PROPOSE, CONCLUDE)


def _default_clock() -> str:
    """IST-naive provenance stamp (never used for identity). Falls back to UTC iso."""
    try:
        from core.market_clock import now_ist_naive
        return now_ist_naive().isoformat(timespec="seconds")
    except Exception:
        return datetime.utcnow().isoformat(timespec="seconds")


@dataclass
class ThreadEntry:
    cycle: int
    seq: int
    kind: str
    text: str
    evidence: dict = field(default_factory=dict)
    stamp: str = ""          # provenance only — NOT part of any hash/identity

    def as_dict(self) -> dict:
        return asdict(self)


class ResearchThread:
    """Append-only reasoning log. In-memory list, optionally mirrored to a JSONL file."""

    def __init__(self, path: str | Path | None = None,
                 clock: Callable[[], str] = _default_clock):
        self.path = Path(path) if path else None
        self._clock = clock
        self._entries: list[ThreadEntry] = []
        if self.path and self.path.exists():
            self._load()

    # ── writing (append-only) ───────────────────────────────────────────────────
    def add(self, cycle: int, kind: str, text: str, evidence: dict | None = None
            ) -> ThreadEntry:
        if kind not in KINDS:
            raise ValueError(f"unknown thread entry kind: {kind!r}")
        e = ThreadEntry(cycle=cycle, seq=len(self._entries) + 1, kind=kind,
                        text=text, evidence=dict(evidence or {}), stamp=self._clock())
        self._entries.append(e)
        self._append_line(e)
        return e

    def observe(self, cycle, text, evidence=None): return self.add(cycle, OBSERVE, text, evidence)
    def reason(self, cycle, text, evidence=None):  return self.add(cycle, REASON, text, evidence)
    def decide(self, cycle, text, evidence=None):  return self.add(cycle, DECIDE, text, evidence)
    def propose(self, cycle, text, evidence=None): return self.add(cycle, PROPOSE, text, evidence)
    def conclude(self, cycle, text, evidence=None):return self.add(cycle, CONCLUDE, text, evidence)

    # ── reading ─────────────────────────────────────────────────────────────────
    def all(self) -> list[ThreadEntry]:
        return list(self._entries)

    def for_cycle(self, cycle: int) -> list[ThreadEntry]:
        return [e for e in self._entries if e.cycle == cycle]

    def last_cycle(self) -> int:
        return max((e.cycle for e in self._entries), default=0)

    def proposals(self) -> list[ThreadEntry]:
        return [e for e in self._entries if e.kind == PROPOSE]

    def __len__(self) -> int:
        return len(self._entries)

    # ── persistence (append-only JSONL) ─────────────────────────────────────────
    def _append_line(self, e: ThreadEntry) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self._entries.append(ThreadEntry(
                    cycle=int(d["cycle"]), seq=int(d["seq"]), kind=d["kind"],
                    text=d["text"], evidence=d.get("evidence", {}),
                    stamp=d.get("stamp", "")))
            except Exception:
                continue  # a torn final line is skipped; the past is never corrupted
