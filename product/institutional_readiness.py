"""Fail-closed institutional readiness projection for QuantTerm.

Research, data, portfolio, execution, risk, reconciliation, and live deployment are
independent control domains. Strength in one domain must never hide a blocker in
another, so this module deliberately does not produce an aggregate percentage.

The projection is read-only and pure. It does not start workers, call a broker,
mutate state, or infer missing capabilities from UI availability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

READY = "READY"
PARTIAL = "PARTIAL"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"


def _domain(
    *,
    key: str,
    label: str,
    status: str,
    summary: str,
    evidence: list[str] | None = None,
    blockers: list[str] | None = None,
    next_action: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status if status in {READY, PARTIAL, BLOCKED, UNKNOWN} else UNKNOWN,
        "summary": summary,
        "evidence": [str(item) for item in (evidence or []) if str(item).strip()],
        "blockers": [str(item) for item in (blockers or []) if str(item).strip()],
        "next_action": next_action,
    }


def _flag(capabilities: Mapping[str, Any], key: str) -> bool:
    """Accept only an explicit boolean certification; truthy strings do not pass."""
    return capabilities.get(key) is True


def _missing(capabilities: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if not _flag(capabilities, key)]


def build_institutional_readiness(
    *,
    data: Mapping[str, Any] | None,
    market: Mapping[str, Any] | None,
    scan: Mapping[str, Any] | None,
    paper: Mapping[str, Any] | None,
    autonomy: Mapping[str, Any] | None,
    operations: Mapping[str, Any] | None,
    capabilities: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return independent readiness domains and fail-closed deployment gates.

    ``capabilities`` contains explicit, repository-backed certification flags.
    Missing flags are false by design. UI presence, class names, and paper behaviour
    are not accepted as proof of production capability.
    """
    now = now or datetime.now(timezone.utc)
    data = dict(data or {})
    market = dict(market or {})
    scan = dict(scan or {})
    paper = dict(paper or {})
    autonomy = dict(autonomy or {})
    operations = dict(operations or {})
    capabilities = dict(capabilities or {})

    bhavcopy = dict(data.get("bhavcopy", {}) or {})
    snapshot = dict(data.get("snapshot", {}) or {})
    sessions = int(bhavcopy.get("sessions", 0) or 0)
    minimum_sessions = int(bhavcopy.get("minimum_sessions", 60) or 60)
    data_ready = bool(bhavcopy.get("ready")) and sessions >= minimum_sessions
    snapshot_ready = bool(snapshot.get("ready"))
    operations_ready = bool(operations.get("running"))
    autonomy_known = bool(autonomy.get("available"))
    paper_available = bool(paper.get("available"))

    economic_flags = (
        "registered_candidate",
        "historical_net_edge_passed",
        "forward_evidence_passed",
        "capacity_assessed",
    )
    portfolio_flags = (
        "canonical_target_portfolio",
        "portfolio_constraints_enforced",
        "open_order_exposure_included",
    )
    execution_flags = (
        "durable_oms",
        "broker_event_ingestion",
        "idempotent_submission",
        "partial_fill_recovery",
        "ambiguous_order_quarantine",
    )
    risk_flags = (
        "independent_risk_governor",
        "pending_exposure_risk",
        "independent_kill_switch",
        "fail_closed_live_gate",
    )
    reconciliation_flags = (
        "startup_reconciliation",
        "continuous_broker_reconciliation",
        "position_and_cash_reconciliation",
        "mismatch_quarantine",
    )
    protection_flags = (
        "protection_manager",
        "partial_fill_protection",
        "protection_verification",
        "orphan_protection_detection",
    )
    operations_flags = (
        "duplicate_worker_detection",
        "restart_recovery_tested",
        "failure_injection_tested",
        "operator_runbook_available",
    )

    economic_missing = _missing(capabilities, economic_flags)
    portfolio_missing = _missing(capabilities, portfolio_flags)
    execution_missing = _missing(capabilities, execution_flags)
    risk_missing = _missing(capabilities, risk_flags)
    reconciliation_missing = _missing(capabilities, reconciliation_flags)
    protection_missing = _missing(capabilities, protection_flags)
    operations_missing = _missing(capabilities, operations_flags)

    domains = [
        _domain(
            key="economic",
            label="Economic edge",
            status=READY if not economic_missing else BLOCKED,
            summary=(
                "A registered strategy has passed net historical, forward and capacity gates."
                if not economic_missing
                else "No production edge is assumed until explicit evidence gates are certified."
            ),
            evidence=[key for key in economic_flags if _flag(capabilities, key)],
            blockers=economic_missing,
            next_action="Complete and freeze one cost-aware strategy evidence package.",
        ),
        _domain(
            key="data",
            label="Data integrity",
            status=READY if data_ready and snapshot_ready else PARTIAL if data_ready or snapshot_ready else BLOCKED,
            summary=f"Official history: {sessions} sessions; verified snapshot: {'yes' if snapshot_ready else 'no'}.",
            evidence=[
                "official_history_ready" if data_ready else "",
                "verified_snapshot_active" if snapshot_ready else "",
            ],
            blockers=[
                "official_history_not_ready" if not data_ready else "",
                "verified_snapshot_missing" if not snapshot_ready else "",
            ],
            next_action="Complete the Golden Market Dataset and activate a verified point-in-time snapshot.",
        ),
        _domain(
            key="research",
            label="Research system",
            status=READY if data_ready and bool(scan.get("available")) and bool(market.get("available")) else PARTIAL,
            summary="Research readiness is reported separately and never unlocks broker execution.",
            evidence=[
                "whole_market_scan_available" if scan.get("available") else "",
                "market_regime_available" if market.get("available") else "",
                "paper_state_readable" if paper_available else "",
            ],
            blockers=[
                "research_inputs_incomplete"
                if not (data_ready and scan.get("available") and market.get("available"))
                else ""
            ],
            next_action="Maintain research-to-production parity and preserve failed evidence.",
        ),
        _domain(
            key="portfolio",
            label="Portfolio construction",
            status=READY if not portfolio_missing else BLOCKED,
            summary="Signals may propose opportunities; only a canonical target portfolio may request changes.",
            evidence=[key for key in portfolio_flags if _flag(capabilities, key)],
            blockers=portfolio_missing,
            next_action="Implement and certify the canonical Target Portfolio contract.",
        ),
        _domain(
            key="execution",
            label="OMS and execution",
            status=READY if not execution_missing else BLOCKED,
            summary="Broker submission remains blocked until durable order lifecycle handling is certified.",
            evidence=[key for key in execution_flags if _flag(capabilities, key)],
            blockers=execution_missing,
            next_action="Complete durable OMS/EMS state, broker events and uncertain-submission recovery.",
        ),
        _domain(
            key="risk",
            label="Independent live risk",
            status=READY if not risk_missing else BLOCKED,
            summary="Paper limits are not proof of an independent production Risk Governor.",
            evidence=[key for key in risk_flags if _flag(capabilities, key)],
            blockers=risk_missing,
            next_action="Build an independent fail-closed Risk Governor from reconciled state.",
        ),
        _domain(
            key="reconciliation",
            label="Broker reconciliation",
            status=READY if not reconciliation_missing else BLOCKED,
            summary="Broker orders, trades, positions and cash must reconcile with the internal ledger.",
            evidence=[key for key in reconciliation_flags if _flag(capabilities, key)],
            blockers=reconciliation_missing,
            next_action="Implement startup, continuous and end-of-day broker reconciliation.",
        ),
        _domain(
            key="protection",
            label="Position protection",
            status=READY if not protection_missing else BLOCKED,
            summary="A filled entry is incomplete until required exchange-side protection is verified.",
            evidence=[key for key in protection_flags if _flag(capabilities, key)],
            blockers=protection_missing,
            next_action="Create one canonical Protection Manager with restart recovery.",
        ),
        _domain(
            key="operations",
            label="Production operations",
            status=READY if operations_ready and not operations_missing else PARTIAL if operations_ready else BLOCKED,
            summary="A running worker is necessary but insufficient for production readiness.",
            evidence=[
                "market_operations_worker_running" if operations_ready else "",
                "autonomy_state_readable" if autonomy_known else "",
                *[key for key in operations_flags if _flag(capabilities, key)],
            ],
            blockers=[
                "market_operations_worker_offline" if not operations_ready else "",
                *operations_missing,
            ],
            next_action="Certify restart recovery, duplicate-worker protection and failure-injection runbooks.",
        ),
    ]

    domain_map = {item["key"]: item for item in domains}
    hard_live_domains = (
        "economic",
        "data",
        "portfolio",
        "execution",
        "risk",
        "reconciliation",
        "protection",
        "operations",
    )
    hard_blockers = [key for key in hard_live_domains if domain_map[key]["status"] != READY]

    limited_live_ready = not hard_blockers and _flag(capabilities, "limited_live_owner_approved")
    limited_live_blockers = list(hard_blockers)
    if not _flag(capabilities, "limited_live_owner_approved"):
        limited_live_blockers.append("limited_live_owner_approval_missing")

    live_ready = (
        limited_live_ready
        and _flag(capabilities, "limited_live_operational_evidence_passed")
        and _flag(capabilities, "limited_live_economic_evidence_passed")
        and _flag(capabilities, "live_owner_approved")
    )
    live_blockers = [] if limited_live_ready else ["limited_live_not_ready"]
    for key in (
        "limited_live_operational_evidence_passed",
        "limited_live_economic_evidence_passed",
        "live_owner_approved",
    ):
        if not _flag(capabilities, key):
            live_blockers.append(key)

    deployment = {
        "research": {
            "status": READY if domain_map["research"]["status"] == READY else PARTIAL,
            "allowed": True,
        },
        "shadow": {
            "status": READY if data_ready and operations_ready else PARTIAL,
            "allowed": bool(data_ready),
        },
        "paper": {
            "status": READY if paper_available and autonomy_known else PARTIAL,
            "allowed": bool(paper_available),
        },
        "limited_live": {
            "status": READY if limited_live_ready else BLOCKED,
            "allowed": limited_live_ready,
            "blockers": limited_live_blockers,
        },
        "live": {
            "status": READY if live_ready else BLOCKED,
            "allowed": live_ready,
            "blockers": list(dict.fromkeys(live_blockers)),
        },
    }

    if live_ready:
        system_state = "LIVE_ELIGIBLE"
    elif limited_live_ready:
        system_state = "LIMITED_LIVE_ELIGIBLE"
    elif deployment["paper"]["allowed"]:
        system_state = "PAPER_ONLY"
    elif deployment["shadow"]["allowed"]:
        system_state = "SHADOW_ONLY"
    else:
        system_state = "RESEARCH_ONLY"

    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "system_state": system_state,
        "summary": "Readiness is domain-specific; no aggregate score can override a hard safety blocker.",
        "domains": domains,
        "deployment": deployment,
        "hard_blockers": hard_blockers,
    }
