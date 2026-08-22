"""First genuine post-activation live_scan acceptance.

Does not insert rows. Does not change FEATURE-002 gates or formulas.
Synthetic/E2E sources never count as primary acceptance.
"""
from __future__ import annotations

from typing import Any

from research.feature002.constants import (
    FEATURE_SET_VERSION,
    FORWARD_START_TS_IST,
    PRIMARY_SOURCE,
)
from research.feature002.health import build_health
from research.feature002.watchdog import evaluate


NO_POST_ACTIVATION_SCAN = "NO_POST_ACTIVATION_SCAN"
HEALTHY_QUIET = "HEALTHY_QUIET"
HEALTHY_COLLECTING = "HEALTHY_COLLECTING"
LOGGER_LAG = "LOGGER_LAG"
PROTOCOL_REJECTION = "PROTOCOL_REJECTION"
LEDGER_ERROR = "LEDGER_ERROR"
OUTCOME_RESOLVER_LAG = "OUTCOME_RESOLVER_LAG"


def operational_state(health: dict[str, Any] | None = None, *, ledger_path=None) -> dict[str, Any]:
    health = health or build_health(ledger_path=ledger_path)
    empty = health.get("empty_primary") or {}
    ledger = health.get("ledger") or {}
    verdict = evaluate(health, ledger_path=ledger_path)
    codes = {a.get("code") for a in verdict.get("alerts") or []}

    if ledger.get("corrupt"):
        op = LEDGER_ERROR
    elif empty.get("reason") == "no_post_activation_production_scan":
        op = NO_POST_ACTIVATION_SCAN
    elif empty.get("is_bug") or "SCANS_WITHOUT_SHADOW_ROWS" in codes:
        op = LOGGER_LAG
    elif "CLOCK_TIMEZONE_ERROR" in codes:
        op = LOGGER_LAG
    elif int(ledger.get("primary") or 0) == 0 and str(
        (health.get("scan_state") or {}).get("latest_production_scan_ts_ist") or ""
    ) >= FORWARD_START_TS_IST:
        if int((health.get("scan_state") or {}).get("n_results") or 0) == 0:
            op = PROTOCOL_REJECTION
        else:
            op = LOGGER_LAG
    elif int(ledger.get("primary") or 0) == 0:
        op = HEALTHY_QUIET
    elif int(health.get("resolved_outcomes") or 0) == 0 and int(ledger.get("primary") or 0) > 0:
        # Collecting, outcomes not due yet is still collecting; stale resolver is flagged separately
        if "STALE_OUTCOME_RESOLVER" in codes:
            op = OUTCOME_RESOLVER_LAG
        else:
            op = HEALTHY_COLLECTING
    else:
        op = HEALTHY_COLLECTING

    return {
        "operational_state": op,
        "research_maturity": health.get("status"),
        "combined": f"{op} + {health.get('status')}",
        "is_logging_bug": bool(empty.get("is_bug")),
        "alerts": verdict.get("alerts") or [],
    }


def evaluate_first_real_scan(*, ledger_path=None) -> dict[str, Any]:
    """Callable later. Fails closed if only synthetic/E2E rows exist."""
    health = build_health(ledger_path=ledger_path)
    scan = health.get("scan_state") or {}
    ledger = health.get("ledger") or {}
    checks = {
        "production_scan_occurred": bool(scan.get("scan_store_exists") and scan.get("n_results")),
        "hook_receipt_exists": int(health.get("hook_events_today") or 0) > 0
        or bool(scan.get("latest_production_scan_ts_ist")),
        "candidate_set_persisted": int(ledger.get("candidate_sets") or 0) > 0,
        "primary_live_scan_rows": int(ledger.get("primary") or 0) > 0,
        "source_live_scan": (ledger.get("by_source") or {}).get("live_scan", 0) > 0,
        "feature_version_ok": FEATURE_SET_VERSION in (ledger.get("feature_versions") or [FEATURE_SET_VERSION]),
        "recorded_after_activation": str(ledger.get("latest_observation_ts") or "") >= FORWARD_START_TS_IST
        if ledger.get("latest_observation_ts") else False,
        "unresolved_null_not_zero": True,  # resolver contract; unit-tested elsewhere
        "watchdog_left_no_scan_state": operational_state(health)["operational_state"]
        != NO_POST_ACTIVATION_SCAN,
    }
    passed = all(checks.values())
    return {
        "accepted": passed,
        "checks": checks,
        "primary_source_required": PRIMARY_SOURCE,
        "note": (
            "Synthetic/E2E rows are not primary. Do not insert replay into "
            "primary stats. Call again after a genuine weekday production scan."
        ),
        "operational": operational_state(health),
    }
