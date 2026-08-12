"""
📈 Learning across cycles — does the brain actually get smarter?

The autonomous loop runs again and again. This module is its memory: it watches each
family's best evidence over time and answers two questions honestly —

    • IMPROVING?  a new cycle beat the family's previous best by a meaningful margin.
    • DECAYING?   a family that used to look good is now materially weaker (edge fading).

When it sees decay in a *proposed* strategy, it PROPOSES a fresh version (via
`spec.bump_version`, which forces a full re-test — old evidence never transfers). It NEVER
mutates a live/active strategy and NEVER approves anything: like everything else in the
brain, the output is advice parked for the human gate.

All thresholds are explicit and conservative. "Meaningful" is a real edge change, not
noise, so the brain does not thrash on tiny wiggles.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from research.strategy_studio import spec as S

# a change smaller than this (in R per trade) is treated as noise, not a trend
MEANINGFUL_R = 0.05


@dataclass
class FamilyTrack:
    family: str
    best_R: float
    best_cycle: int
    last_R: float
    last_cycle: int
    observations: int = 1

    def as_dict(self): return asdict(self)


@dataclass
class LearningEvent:
    cycle: int
    family: str
    kind: str                 # IMPROVED / DECAYED / STEADY / NEW
    prev_R: float
    now_R: float
    note: str

    def as_dict(self): return asdict(self)


class LearningLedger:
    """Append-only-ish memory of per-family best/last evidence across cycles."""

    def __init__(self, meaningful_R: float = MEANINGFUL_R):
        self.meaningful_R = meaningful_R
        self._tracks: dict[str, FamilyTrack] = {}
        self._events: list[LearningEvent] = []

    # ── ingest one cycle's market-evidence proposals ─────────────────────────────
    def observe_cycle(self, cycle: int, proposals) -> list[LearningEvent]:
        """Feed a cycle's proposals (market-evidence only). Returns the events detected.
        `proposals` is an iterable of objects/dicts with .family and .net_expectancy_R."""
        # keep only the best market-evidence proposal per family this cycle
        best: dict[str, float] = {}
        for p in proposals:
            fam = _get(p, "family")
            if not _get(p, "is_market_evidence", True):
                continue
            r = float(_get(p, "net_expectancy_R", 0.0))
            if fam not in best or r > best[fam]:
                best[fam] = r

        events: list[LearningEvent] = []
        for fam, now_r in sorted(best.items()):
            events.append(self._update(cycle, fam, now_r))
        self._events.extend(events)
        return events

    def _update(self, cycle: int, fam: str, now_r: float) -> LearningEvent:
        t = self._tracks.get(fam)
        if t is None:
            self._tracks[fam] = FamilyTrack(family=fam, best_R=now_r, best_cycle=cycle,
                                            last_R=now_r, last_cycle=cycle, observations=1)
            return LearningEvent(cycle, fam, "NEW", now_r, now_r,
                                 f"First evidence for {fam}: {now_r:+.2f}R.")
        prev_best = t.best_R
        kind = "STEADY"; note = f"{fam} steady near {now_r:+.2f}R."
        if now_r >= prev_best + self.meaningful_R:
            kind = "IMPROVED"; note = (f"{fam} improved: {prev_best:+.2f}R → {now_r:+.2f}R "
                                       "(beat its previous best).")
        elif now_r <= prev_best - self.meaningful_R:
            kind = "DECAYED"; note = (f"{fam} decayed: best was {prev_best:+.2f}R, now "
                                      f"{now_r:+.2f}R (edge fading).")
        t.best_R = max(t.best_R, now_r)
        t.best_cycle = cycle if now_r >= t.best_R else t.best_cycle
        t.last_R = now_r; t.last_cycle = cycle; t.observations += 1
        return LearningEvent(cycle, fam, kind, prev_best, now_r, note)

    # ── propose fresh versions for decayed families (advice only) ────────────────
    def improvement_proposals(self, cycle: int, specs_by_family: dict) -> list[S.StrategySpec]:
        """For each family that DECAYED, propose a re-tested new version of its spec (a
        real bump_version → forces a full fresh evidence gate). Never mutates the input
        spec; returns brand-new child specs the human can choose to pursue. Nothing here
        is approved or activated."""
        out: list[S.StrategySpec] = []
        for ev in self._events:
            if ev.cycle == cycle and ev.kind == "DECAYED" and ev.family in specs_by_family:
                parent = specs_by_family[ev.family]
                # a transparent, conservative response to fading edge: exit sooner (shorter
                # hold) AND trade less (tighter turnover). Always a MATERIAL change, so the
                # child gets a new config hash and MUST earn fresh evidence — the old,
                # decayed evidence never transfers. It is a proposal, not an action.
                new_hold = max(5, int(parent.max_holding_days // 2))
                new_turnover = round(max(0.1, parent.turnover_cap * 0.8), 4)
                child = parent.bump_version(max_holding_days=new_hold,
                                            turnover_cap=new_turnover)
                out.append(child)
        return out

    # ── reads ────────────────────────────────────────────────────────────────────
    def events(self): return list(self._events)
    def tracks(self): return {k: v for k, v in self._tracks.items()}
    def decayed_families(self, cycle: int | None = None):
        return [e.family for e in self._events
                if e.kind == "DECAYED" and (cycle is None or e.cycle == cycle)]
    def as_dict(self):
        return {"tracks": {k: v.as_dict() for k, v in self._tracks.items()},
                "events": [e.as_dict() for e in self._events]}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
