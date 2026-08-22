"""Shadow feature logging specification. Not wired into production."""
from __future__ import annotations

from typing import Any

from research.feature001.constants import EXPERIMENT, RS_VERSION, TREND_VERSION


def forward_record_template() -> dict[str, Any]:
    return {
        "kind": "FEATURE_001_SHADOW_FEATURE_LOG",
        "experiment": EXPERIMENT,
        "execution": False,
        "paper": False,
        "autopilot": False,
        "activation": "documented_only",
        "reason": (
            "Wiring passive logging into app.py / auto_scan could change "
            "production runtime reliability. FEATURE-001 only documents the "
            "ledger. Activate in a later milestone behind an explicit flag "
            "that never feeds the ticket, GTT, or autopilot."
        ),
        "eligible_after": "strictly after the FEATURE-001 experiment freeze date",
        "fields": [
            "symbol",
            "timestamp",
            "strategy",
            "production_rank",
            "production_score",
            "production_verdict",
            "actual_production_decision",
            f"{TREND_VERSION}_vector",
            f"{RS_VERSION}_vector",
            "existing_conviction_technical_only",
            "hypothetical_feature_adjusted_rank",
            "future_outcome_when_known",
        ],
        "forbidden": [
            "change BUY/WATCH",
            "change ranking used by Ready/autopilot",
            "place or cancel orders",
            "feed Telegram as a trade licence",
        ],
    }
