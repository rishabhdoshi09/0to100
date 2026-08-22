"""Research-only forward observation design. Not wired into app.py."""
from __future__ import annotations

from typing import Any


def forward_record_template() -> dict[str, Any]:
    return {
        "kind": "SEPA_003_FORWARD_OBSERVATION",
        "execution": False,
        "paper": False,
        "autopilot": False,
        "fields": [
            "as_of", "symbol", "frozen_feature_vector",
            "hypothetical_entry", "hypothetical_stop",
            "regime_pit_v1", "sector_map_v1",
            "data_quality", "outcome_after_horizon",
        ],
        "activation": "documented_only",
        "reason": (
            "Wiring this into app.py would change production runtime. "
            "SEPA-003 only documents the ledger."
        ),
    }
