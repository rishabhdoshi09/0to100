"""
🗣️ Typed dialogue between the brains — auditable records, not free-form agent chatter.

Every message is a typed, append-only record tied to canonical data (snapshot id, strategy id/version,
input record ids). LLM prose is only ever commentary carried inside `evidence`/`explanation`; it is
never itself a decision and can never become a `TradeIntent`. Persisted as JSONL (crash-safe append)
so the whole conversation is replayable and linkable from the UI.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path

# record types
OBSERVATION = "ObservationRecord"
EVIDENCE_GAP = "EvidenceGap"
HYPOTHESIS = "HypothesisProposal"
CHALLENGE_REQUEST = "ChallengeRequest"
CHALLENGE_REPORT = "ChallengeReport"
EXPERIMENT_REGISTRATION = "ExperimentRegistration"
EXPERIMENT_RESULT = "ExperimentResult"
PROMOTION_PROPOSAL = "PromotionProposal"
ALLOCATION_DECISION = "AllocationDecision"
RETIREMENT_DECISION = "RetirementDecision"
LEARNING_UPDATE = "LearningUpdate"
OPERATIONAL_INCIDENT = "OperationalIncident"

RECORD_TYPES = (OBSERVATION, EVIDENCE_GAP, HYPOTHESIS, CHALLENGE_REQUEST, CHALLENGE_REPORT,
                EXPERIMENT_REGISTRATION, EXPERIMENT_RESULT, PROMOTION_PROPOSAL, ALLOCATION_DECISION,
                RETIREMENT_DECISION, LEARNING_UPDATE, OPERATIONAL_INCIDENT)


def _now_ist_iso() -> str:
    try:
        from research.intelligence.data import nse_calendar as CAL
        return CAL._now_ist().isoformat()
    except Exception:
        from datetime import datetime
        return datetime.now().isoformat()


@dataclass(frozen=True)
class Record:
    record_type: str
    producer: str
    claim: str
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    consumer: str = ""
    created_at: str = field(default_factory=_now_ist_iso)
    as_of: str = ""
    input_record_ids: tuple = ()
    snapshot_id: str = ""
    strategy_id: str = ""
    strategy_version: int = 0
    evidence: dict = field(default_factory=dict)
    uncertainties: tuple = ()
    requested_action: str = ""
    decision: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class DialogueLog:
    """Append-only JSONL log of typed records."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Record) -> Record:
        if record.record_type not in RECORD_TYPES:
            raise ValueError(f"unknown record type {record.record_type}")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.as_dict(), default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def all(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def by_type(self, record_type: str) -> list:
        return [r for r in self.all() if r.get("record_type") == record_type]

    def recent(self, limit: int = 20) -> list:
        return self.all()[-limit:]
