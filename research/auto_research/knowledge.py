"""
🧠 Knowledge — the growing memory that makes the system smarter each day.

An infant gets smarter because it *remembers* what happened and updates its priors. This is
that memory. It persists (JSON on disk) so a restart doesn't reset the child to zero — the
learning compounds day over day.

What it stores, per strategy FAMILY:
  • the latest BACKTEST edge (in-sample, historical),
  • the latest FORWARD edge (out-of-sample, real paper trades),
  • how often the family's edge was CONFIRMED forward vs turned out OVERFIT/DECAYED,
  • a derived TRUST in [0,1] that rises on forward-confirmation and falls on overfit.

Trust drives `search_weights()`, which biases tomorrow's strategy discovery toward families
that keep working out-of-sample and away from chronic decayers — the search literally gets
smarter. Trust never hits zero, so the system keeps exploring (an infant still tries things).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "logs" / "auto_research" / "knowledge.json"


@dataclass
class FamilyKnowledge:
    family: str
    backtest_R: float = 0.0          # latest in-sample edge
    forward_R: float = 0.0           # latest out-of-sample (paper) edge
    n_confirmed: int = 0             # times forward CONFIRMED the backtest
    n_overfit: int = 0               # times forward proved the backtest overfit / decayed
    days_observed: int = 0
    trust: float = 0.5               # prior: neutral

    def as_dict(self): return asdict(self)


class Knowledge:
    def __init__(self, path=None, *, learn_rate: float = 0.25):
        self.path = Path(path) if path else _default_path()
        self.learn_rate = learn_rate
        self.families: dict[str, FamilyKnowledge] = {}
        if self.path.exists():
            self._load()

    # ── record what happened ─────────────────────────────────────────────────────
    def remember_backtest(self, family: str, backtest_R: float) -> None:
        fk = self._fam(family)
        fk.backtest_R = round(float(backtest_R), 4)

    def remember_forward(self, family: str, forward_R: float, verdict: str) -> None:
        """Fold one forward-test outcome into the family's trust. `verdict` comes from
        growth.calibrate(): CONFIRMED / WEAKER_POSITIVE / OVERFIT / DECAYED."""
        fk = self._fam(family)
        fk.forward_R = round(float(forward_R), 4)
        fk.days_observed += 1
        # trust moves toward 1 on confirmation, toward 0 on overfit — EWMA, bounded [0.05,0.98]
        target = {"CONFIRMED": 1.0, "WEAKER_POSITIVE": 0.65,
                  "OVERFIT": 0.0, "DECAYED": 0.1}.get(verdict, 0.5)
        fk.trust = round(min(0.98, max(0.05, fk.trust + self.learn_rate * (target - fk.trust))), 4)
        if verdict == "CONFIRMED":
            fk.n_confirmed += 1
        elif verdict in ("OVERFIT", "DECAYED"):
            fk.n_overfit += 1

    # ── use what it knows ────────────────────────────────────────────────────────
    def family_trust(self, family: str) -> float:
        return self.families[family].trust if family in self.families else 0.5

    def search_weights(self, families) -> dict:
        """Weights for the next discovery round — proportional to trust (min 0.1 so nothing
        is starved). Families never seen default to neutral 0.5, so new ideas still get a
        fair share of attempts."""
        return {f: max(0.1, self.family_trust(f)) for f in families}

    def summary(self) -> list[dict]:
        return [fk.as_dict() for fk in sorted(self.families.values(),
                                              key=lambda x: -x.trust)]

    # ── persistence (the memory survives restarts) ───────────────────────────────
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"families": {k: v.as_dict() for k, v in self.families.items()}},
            indent=2, ensure_ascii=False))

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            for k, v in data.get("families", {}).items():
                self.families[k] = FamilyKnowledge(**v)
        except Exception:
            pass                                    # corrupt memory ⇒ start neutral, never crash

    def _fam(self, family: str) -> FamilyKnowledge:
        if family not in self.families:
            self.families[family] = FamilyKnowledge(family=family)
        return self.families[family]
