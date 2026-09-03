"""Canonical decision truth for the current Recommendations desk.

The recommendation ensemble may nominate and rank setups before the autonomous
committee has judged them. This module keeps those two concepts separate:

* recommendation/card fields describe the setup and supporting evidence;
* candidate_lifecycle is the persisted source of truth for BUY / WAIT / AVOID;
* only a decision whose id belongs to the exact current scan run is canonical.

That exact-lineage check matters because a same-session rescan can update a
candidate's ``scan_run_id`` before the committee has revisited it. Older decision
fields may still be present on the durable row during that interval. They must
never leak into the new scan's UI.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from product import candidate_lifecycle as CL
from product.scan_store import default_scan_path, load_scan

CANONICAL_DECISIONS = frozenset({"BUY", "WAIT", "AVOID"})
_ACTION_BADGE = {
    "BUY": "Buy",
    "WAIT": "Wait",
    "AVOID": "Avoid",
    "NO_JUDGMENT": "No judgment",
}


def _mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return -1


@lru_cache(maxsize=8)
def _scan_run_for_file(path_text: str, mtime_ns: int) -> str:
    del mtime_ns  # cache invalidator only
    payload = load_scan(Path(path_text)) or {}
    return str(payload.get("scanned_at") or "")


def current_scan_run_id() -> str:
    path = default_scan_path()
    return _scan_run_for_file(str(path), _mtime_ns(path))


@lru_cache(maxsize=16)
def _candidate_map_for_scan(scan_run_id: str, db_mtime_ns: int) -> dict[str, dict[str, Any]]:
    """Load exact-scan candidates once per DB revision, not once per card."""
    del db_mtime_ns  # cache invalidator only
    if not scan_run_id or not CL.DB_PATH.exists():
        return {}
    con = CL._connect()  # product-internal store; also applies schema migrations
    try:
        rows = con.execute(
            "SELECT * FROM candidates WHERE scan_run_id=? ORDER BY updated_at DESC",
            (scan_run_id,),
        ).fetchall()
    finally:
        con.close()
    out: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        symbol = str(row.get("symbol") or "").upper()
        if symbol and symbol not in out:
            out[symbol] = row
    return out


def candidates_for_current_scan(scan_run_id: str) -> dict[str, dict[str, Any]]:
    return _candidate_map_for_scan(scan_run_id, _mtime_ns(CL.DB_PATH))


def _wait_trigger(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = candidate.get("wait_trigger_json")
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def project_candidate_truth(
    card: Mapping[str, Any],
    *,
    scan_run_id: str,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return UI fields for one card without mutating the recommendation evidence.

    A row is canonical only when all three pieces line up with the current scan:
    ``scan_run_id``, ``recommendation_id`` and the decision-id suffix written by
    ``autonomous_loop._consume_paper``. This deliberately treats retained fields
    from an earlier same-session decision as NO_JUDGMENT.

    Buy-zone wording is also decision truth. A scanner/ensemble candidate may
    carry useful entry geometry, but the UI must not call it a ``Buy Zone`` until
    the committee has frozen BUY for this exact scan. WAIT/AVOID/NO_JUDGMENT keep
    the raw ``entry``/stop/target geometry for explanation while buy-zone fields
    are cleared so the frontend labels the level neutrally as Entry.
    """
    symbol = str(card.get("symbol") or "").upper()
    raw_badge = str(card.get("action_badge") or "")
    raw_buy_zone_low = card.get("buy_zone_low")
    raw_buy_zone_high = card.get("buy_zone_high")
    base: dict[str, Any] = {
        "raw_action_badge": raw_badge,
        "canonical_decision": "NO_JUDGMENT",
        "decision_truth_status": "NO_CURRENT_SCAN_JUDGMENT",
        "decision_match_scope": "NONE",
        "decision_scan_run_id": scan_run_id or None,
        "candidate_id": None,
        "opportunity_id": None,
        "recommendation_id": None,
        "decision_id": None,
        "paper_intent_id": None,
        "outcome_id": None,
        "canonical_candidate_state": "UNJUDGED",
        "canonical_entry_state": "",
        "canonical_execution_state": "",
        "decision_reason_code": "",
        "wait_trigger": {},
        "action_badge": _ACTION_BADGE["NO_JUDGMENT"],
        "buy_zone_low": None,
        "buy_zone_high": None,
        "buy_zone_authorized": False,
    }
    if not symbol or not scan_run_id:
        base["decision_truth_status"] = "SCAN_LINEAGE_UNAVAILABLE"
        return base
    if not candidate:
        return base

    row_scan = str(candidate.get("scan_run_id") or "")
    decision = str(candidate.get("decision") or "").upper()
    decision_id = str(candidate.get("decision_id") or "")
    recommendation_id = str(candidate.get("recommendation_id") or "")
    exact_scan = row_scan == scan_run_id
    exact_decision = decision_id.endswith(f"|{scan_run_id}")
    exact_recommendation = recommendation_id.startswith(f"{scan_run_id}:{symbol}:")

    if not exact_scan:
        base["decision_truth_status"] = "CANDIDATE_SCAN_MISMATCH"
        return base
    if decision not in CANONICAL_DECISIONS or not exact_decision or not exact_recommendation:
        # The candidate belongs to this scan, but the committee has not yet frozen
        # a decision for this exact scan. Do not expose retained READY/BUY fields.
        base["decision_truth_status"] = "COMMITTEE_PENDING_FOR_SCAN"
        base["decision_match_scope"] = "EXACT_SCAN_CANDIDATE_ONLY"
        return base

    base.update({
        "canonical_decision": decision,
        "decision_truth_status": "CANONICAL_CURRENT_SCAN",
        "decision_match_scope": "EXACT_SCAN_RUN",
        "candidate_id": candidate.get("candidate_id"),
        "opportunity_id": candidate.get("opportunity_id"),
        "recommendation_id": recommendation_id,
        "decision_id": decision_id,
        "paper_intent_id": candidate.get("paper_intent_id"),
        "outcome_id": candidate.get("outcome_id"),
        "canonical_candidate_state": str(candidate.get("state") or ""),
        "canonical_entry_state": str(candidate.get("entry_state") or ""),
        "canonical_execution_state": str(candidate.get("execution_state") or ""),
        "decision_reason_code": str(candidate.get("reason") or ""),
        "wait_trigger": _wait_trigger(candidate),
        "action_badge": _ACTION_BADGE[decision],
        "buy_zone_low": raw_buy_zone_low if decision == "BUY" else None,
        "buy_zone_high": raw_buy_zone_high if decision == "BUY" else None,
        "buy_zone_authorized": decision == "BUY",
    })
    return base


def decorate_current_recommendation(card: Mapping[str, Any]) -> dict[str, Any]:
    """Attach current-scan committee truth to a recommendation card."""
    scan_run_id = current_scan_run_id()
    symbol = str(card.get("symbol") or "").upper()
    candidate = candidates_for_current_scan(scan_run_id).get(symbol) if scan_run_id and symbol else None
    return project_candidate_truth(card, scan_run_id=scan_run_id, candidate=candidate)
