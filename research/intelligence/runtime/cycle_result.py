"""IntelligenceCycleResult — the complete, typed record of one orchestration cycle."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# cycle status values
STATUS_OK = "OK"
STATUS_NO_ACTION = "NO_ACTION"
STATUS_ALREADY_DONE = "ALREADY_DONE"
STATUS_FAILED_SAFE = "FAILED_SAFE"


@dataclass
class IntelligenceCycleResult:
    cycle_id: str
    as_of_date: str
    mode: str
    status: str = STATUS_OK
    data_ok: bool = True
    strategies_evaluated: list = field(default_factory=list)
    signals_generated: list = field(default_factory=list)
    signals_rejected: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    events_emitted: int = 0
    cards_created: list = field(default_factory=list)     # strategy ids
    allocation_decisions: list = field(default_factory=list)  # (sid, action)
    trade_intents: list = field(default_factory=list)     # intent ids
    intents_blocked: list = field(default_factory=list)   # (sid, reason_code)
    positions_opened: list = field(default_factory=list)  # (sid, symbol)
    positions_modified: list = field(default_factory=list)
    positions_closed: list = field(default_factory=list)  # (sid, symbol, reason)
    outcomes_recorded: list = field(default_factory=list)
    strategies_paused: list = field(default_factory=list)
    strategies_retired: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    no_action_reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)
