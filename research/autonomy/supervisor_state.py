"""
🧭 Operational state machine for the autonomy supervisor.

State is EXPLICIT and persisted — never inferred from whether a worker thread is alive. Every
transition records prev/next, a reason code, a plain explanation, the triggering job/event, an IST
timestamp, the snapshot id, and whether new paper risk is permitted / existing positions remain
manageable. The capability flags come from the state, not from a global green/red boolean.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

# semantic states
STARTING = "STARTING"
AUTH_REQUIRED = "AUTH_REQUIRED"
DATA_REFRESHING = "DATA_REFRESHING"
DATA_BLOCKED = "DATA_BLOCKED"
DATA_READY = "DATA_READY"
OBSERVING = "OBSERVING"
PAPER_ACTIVE = "PAPER_ACTIVE"
RESEARCHING = "RESEARCHING"
DEGRADED = "DEGRADED"
HALTED = "HALTED"

STATES = (STARTING, AUTH_REQUIRED, DATA_REFRESHING, DATA_BLOCKED, DATA_READY, OBSERVING,
          PAPER_ACTIVE, RESEARCHING, DEGRADED, HALTED)

# states in which OPENING NEW paper risk is permitted (existing positions are managed far wider)
_NEW_RISK_OK = {PAPER_ACTIVE}
# states in which existing positions can still be managed (almost always — only HALTED forbids)
_MANAGE_BLOCKED = {HALTED}


def _now_ist_iso() -> str:
    try:
        from research.intelligence.data import nse_calendar as CAL
        return CAL._now_ist().isoformat()
    except Exception:
        from datetime import datetime
        return datetime.now().isoformat()


@dataclass
class Transition:
    from_state: str
    to_state: str
    reason_code: str
    explanation: str
    trigger: str
    at_ist: str
    snapshot_id: str = ""
    new_risk_permitted: bool = False
    positions_manageable: bool = True


@dataclass
class SupervisorState:
    state: str = STARTING
    reason_code: str = "boot"
    explanation: str = "Supervisor starting."
    snapshot_id: str = ""
    updated_ist: str = field(default_factory=_now_ist_iso)
    history: list = field(default_factory=list)      # recent Transition dicts (bounded)

    @property
    def new_risk_permitted(self) -> bool:
        return self.state in _NEW_RISK_OK

    @property
    def positions_manageable(self) -> bool:
        return self.state not in _MANAGE_BLOCKED

    def transition(self, to_state: str, *, reason_code: str, explanation: str, trigger: str,
                   snapshot_id: str | None = None) -> Transition:
        if to_state not in STATES:
            raise ValueError(f"unknown state {to_state}")
        snap = self.snapshot_id if snapshot_id is None else snapshot_id
        t = Transition(from_state=self.state, to_state=to_state, reason_code=reason_code,
                       explanation=explanation, trigger=trigger, at_ist=_now_ist_iso(),
                       snapshot_id=snap, new_risk_permitted=(to_state in _NEW_RISK_OK),
                       positions_manageable=(to_state not in _MANAGE_BLOCKED))
        self.state = to_state
        self.reason_code = reason_code
        self.explanation = explanation
        self.snapshot_id = snap
        self.updated_ist = t.at_ist
        self.history.append(asdict(t))
        self.history = self.history[-50:]
        return t

    def as_dict(self) -> dict:
        d = asdict(self)
        d["new_risk_permitted"] = self.new_risk_permitted
        d["positions_manageable"] = self.positions_manageable
        return d


class StatePersistence:
    """Atomic JSON persistence for the operational state (crash-safe)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> SupervisorState:
        if not self.path.exists():
            return SupervisorState()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            st = SupervisorState(state=d.get("state", STARTING), reason_code=d.get("reason_code", ""),
                                 explanation=d.get("explanation", ""), snapshot_id=d.get("snapshot_id", ""),
                                 updated_ist=d.get("updated_ist", _now_ist_iso()),
                                 history=list(d.get("history", [])))
            return st
        except Exception:
            return SupervisorState()

    def save(self, st: SupervisorState) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.as_dict(), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
