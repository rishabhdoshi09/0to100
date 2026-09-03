"""Version stamps for historical experiments.

Old replay rows must not silently look as if they were generated under a
later committee or framework.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

from product.decision_committee import COMMITTEE_VERSION, EVIDENCE_FAMILY_VERSION
from product.evidence_families import FAMILY_SCHEMA_VERSION
from product.pit_availability import PIT_CONTRACT_VERSION

DECISION_ENGINE_VERSION = "decision_engine_v1"
REASON_CODE_VERSION = "reason_codes_v1"
RISK_POLICY_VERSION = "risk_policy_v1"
FRAMEWORK_VERSION = "business_frameworks_v1"
WAREHOUSE_SCHEMA_VERSION = "pit_warehouse_v1"
PARSER_VERSION = "pit_ingest_v1"


@dataclass(frozen=True)
class DecisionSystemVersions:
    decision_engine_version: str = DECISION_ENGINE_VERSION
    committee_version: str = COMMITTEE_VERSION
    evidence_family_version: str = EVIDENCE_FAMILY_VERSION
    family_schema_version: str = FAMILY_SCHEMA_VERSION
    framework_version: str = FRAMEWORK_VERSION
    reason_code_version: str = REASON_CODE_VERSION
    risk_policy_version: str = RISK_POLICY_VERSION
    pit_contract_version: str = PIT_CONTRACT_VERSION
    warehouse_schema_version: str = WAREHOUSE_SCHEMA_VERSION
    parser_version: str = PARSER_VERSION

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def current_versions() -> DecisionSystemVersions:
    return DecisionSystemVersions()


def experiment_fingerprint(universe: list[str], extra: dict[str, Any] | None = None) -> str:
    payload = {
        "versions": current_versions().as_dict(),
        "universe": list(universe),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return sha256(blob).hexdigest()[:16]
