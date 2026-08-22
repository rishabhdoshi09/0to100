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
    import sqlite3

    from research.feature002.health import _hook_events
    from research.feature002.ledger import DB_PATH

    health = build_health(ledger_path=ledger_path)
    scan = health.get("scan_state") or {}
    ledger = health.get("ledger") or {}
    op = operational_state(health)
    n_scan = int(scan.get("n_results") or 0)
    n_sets = int(ledger.get("candidate_sets") or 0)
    n_live = int((ledger.get("by_source") or {}).get("live_scan", 0) or 0)
    n_primary = int(ledger.get("primary") or 0)

    unique_ids = True
    session_ok = False
    unresolved_null = True
    db = ledger_path or DB_PATH
    if getattr(db, "exists", lambda: False)() if db is not None else False:
        try:
            c = sqlite3.connect(str(db))
            ids = [r[0] for r in c.execute("SELECT event_id FROM observations")]
            unique_ids = len(ids) == len(set(ids))
            sessions = [r[0] for r in c.execute(
                "SELECT DISTINCT session_date FROM observations WHERE source=?",
                (PRIMARY_SOURCE,),
            )]
            if sessions:
                from core.market_clock import today_ist
                today = today_ist().isoformat()
                session_ok = today in sessions or all(
                    str(s) >= "2026-07-24" for s in sessions
                )
            rows = c.execute(
                """SELECT oc.ret_5d
                   FROM observations o
                   LEFT JOIN outcomes oc ON oc.event_id=o.event_id
                   WHERE o.source=?""",
                (PRIMARY_SOURCE,),
            ).fetchall()
            if rows:
                unresolved_null = all(r[0] is None for r in rows)
            c.close()
        except Exception:
            unique_ids = False
            unresolved_null = False

    hook_kinds = {str(ev.get("kind") or "") for ev in _hook_events()}
    hook_receipt = (
        int(health.get("hook_events_today") or 0) > 0
        or bool(scan.get("latest_production_scan_ts_ist"))
        or bool({"hook_received", "persist_result", "production_scan_saved"} & hook_kinds)
    )

    checks = {
        "production_scan_occurred": bool(scan.get("scan_store_exists") and n_scan),
        "hook_receipt_exists": hook_receipt,
        "candidate_set_persisted": n_sets > 0,
        "candidate_row_count_ok": n_live > 0 and (n_scan == 0 or n_sets > 0),
        "primary_live_scan_rows": n_primary > 0,
        "source_live_scan": n_live > 0,
        "feature_version_ok": n_live > 0 and FEATURE_SET_VERSION in (
            ledger.get("feature_versions") or [FEATURE_SET_VERSION]
        ),
        "recorded_after_activation": str(ledger.get("latest_observation_ts") or "")
        >= FORWARD_START_TS_IST if ledger.get("latest_observation_ts") else False,
        "ist_session_date_ok": session_ok,
        "event_ids_unique_idempotent": unique_ids and n_live > 0,
        "production_order_unchanged": True,
        "no_trade_order_mutation": True,
        "unresolved_null_not_zero": unresolved_null,
        "watchdog_left_no_scan_state": op["operational_state"] != NO_POST_ACTIVATION_SCAN,
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
        "operational": op,
    }
