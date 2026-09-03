"""Operational attribution after an outcome matures.

Not philosophical causality. Records which families/methods supported the
decision, what was weak, and — for WAIT/AVOID — whether later price action
was a rational miss or a system failure to wake a valid entry.
"""
from __future__ import annotations

from typing import Any, Mapping

from product import decision_taxonomy as T
from product.counterfactual_learning import (
    AVOIDED_LOSER as CF_AVOIDED_LOSER,
    CORRECT_REJECTION as CF_CORRECT_REJECTION,
    GOOD_WAIT,
    MISSED_WINNER as CF_MISSED_WINNER,
    RAN_AWAY,
)
from product.missed_winner import analyze_decision_quality
from product.risk_audit import r_multiple

WAIT_RATIONALLY_MAINTAINED = "WAIT_RATIONALLY_MAINTAINED"
GOOD_WAIT_LABEL = "GOOD_WAIT"
WAIT_THEN_VALID_ENTRY = "WAIT_THEN_VALID_ENTRY"
MISSED_REENTRY = "MISSED_REENTRY"
WAIT_TRIGGER_FAILED = "WAIT_TRIGGER_FAILED"
RAN_AWAY_WITHOUT_VALID_ENTRY = "RAN_AWAY_WITHOUT_VALID_ENTRY"
FALSE_POSITIVE = "FALSE_POSITIVE"
BUY_WINNER = "BUY_WINNER"
BUY_LOSER = "BUY_LOSER"
CORRECT_REJECTION = "CORRECT_REJECTION"
AVOIDED_LOSER = "AVOIDED_LOSER"
MISSED_WINNER = "MISSED_WINNER"
RATIONAL_REJECTION_DESPITE_RALLY = "RATIONAL_REJECTION_DESPITE_RALLY"
EVIDENCE_REJECTION_CORRECT = "EVIDENCE_REJECTION_CORRECT"
OVERSTRICT_VETO = "OVERSTRICT_VETO"


def _families(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("evidence_family_votes") or row.get("families") or {}
    if isinstance(raw, Mapping):
        return {str(k): str(v).upper() for k, v in raw.items()}
    return {}


def attribute_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(row.get("decision") or "")
    families = _families(row)
    supporting = [k for k, v in families.items() if v in {"SUPPORTIVE", "PASS", "BUY"}]
    weak = [k for k, v in families.items() if v in {"UNKNOWN", "NEUTRAL", "WAIT"}]
    failed = [k for k, v in families.items() if v in {"OPPOSED", "FAIL", "AVOID"}]
    entry = row.get("entry")
    stop = row.get("stop")
    fwd = row.get("forward_return_pct")
    exit_px = None
    try:
        if entry is not None and fwd is not None:
            exit_px = float(entry) * (1.0 + float(fwd) / 100.0)
    except (TypeError, ValueError):
        exit_px = None
    r_mult = r_multiple(entry=_f(entry), stop=_f(stop), exit_price=exit_px)
    classification = str(row.get("classification") or "")
    later_valid = bool(row.get("later_valid_entry") or row.get("later_entered"))
    wake_failed = bool(row.get("wake_failed"))

    wait_label = None
    avoid_label = None
    if decision in {T.WAIT_DECISION, "WAIT"}:
        quality = analyze_decision_quality(
            row,
            classification=classification or MISSED_WINNER,
            forward_return_pct=_f(fwd),
            later_entered=later_valid,
        )
        rational = quality.get("original_decision_rational") is True
        trigger_failed = bool(row.get("wait_trigger_failed"))
        if later_valid and not wake_failed:
            wait_label = WAIT_THEN_VALID_ENTRY
        elif later_valid and wake_failed:
            wait_label = MISSED_REENTRY
        elif trigger_failed:
            wait_label = WAIT_TRIGGER_FAILED
        elif classification == RAN_AWAY or str(classification) == "RAN_AWAY_WITHOUT_ENTRY":
            wait_label = RAN_AWAY_WITHOUT_VALID_ENTRY
        elif rational and not later_valid:
            wait_label = WAIT_RATIONALLY_MAINTAINED
        elif classification == GOOD_WAIT:
            wait_label = GOOD_WAIT_LABEL
        else:
            wait_label = str(quality.get("note") or classification or WAIT_RATIONALLY_MAINTAINED)

    if decision == T.AVOID:
        quality = analyze_decision_quality(
            row,
            classification=classification or MISSED_WINNER,
            forward_return_pct=_f(fwd),
            later_entered=later_valid,
        )
        vetoes = [str(v.get("code") if isinstance(v, Mapping) else v) for v in (row.get("vetoes") or [])]
        rally = _f(fwd) is not None and float(fwd) >= 8
        if classification == CORRECT_REJECTION or str(row.get("classification") or "") == "CORRECT_REJECTION":
            avoid_label = CORRECT_REJECTION
        elif classification == AVOIDED_LOSER or (_f(fwd) is not None and float(fwd) <= -5):
            avoid_label = AVOIDED_LOSER
        elif any(c in {T.INSUFFICIENT_EVIDENCE, T.INSUFFICIENT_INDEPENDENT_EVIDENCE, T.FINANCIAL_QUALITY_FAIL} for c in vetoes + [str(row.get("reason_code") or "")]):
            avoid_label = EVIDENCE_REJECTION_CORRECT if not rally else RATIONAL_REJECTION_DESPITE_RALLY
        elif rally and quality.get("original_decision_rational") is True:
            avoid_label = RATIONAL_REJECTION_DESPITE_RALLY
        elif rally and any("VETO" in c or c in {T.BUSINESS_QUALITY_FAIL, T.WEAK_SECTOR} for c in vetoes):
            avoid_label = OVERSTRICT_VETO
        elif classification == MISSED_WINNER or rally:
            avoid_label = MISSED_WINNER
        else:
            avoid_label = str(quality.get("note") or classification or CORRECT_REJECTION)

    buy_label = None
    if decision == T.BUY:
        if r_mult is not None and r_mult > 0:
            buy_label = BUY_WINNER
        elif r_mult is not None and r_mult < 0:
            buy_label = BUY_LOSER
        if str(row.get("breakout_failed") or "").lower() in {"1", "true", "yes"}:
            buy_label = FALSE_POSITIVE

    return {
        "symbol": row.get("symbol"),
        "decision_id": row.get("decision_id"),
        "decision": decision,
        "supporting_families": supporting,
        "weak_unknown_families": weak,
        "failed_families": failed,
        "hard_vetoes": row.get("vetoes") or [],
        "entry_state": row.get("entry_state"),
        "regime": row.get("regime") or (row.get("references") or {}).get("regime"),
        "r_multiple": r_mult,
        "forward_return_pct": fwd,
        "buy_attribution": buy_label,
        "wait_attribution": wait_label,
        "avoid_attribution": avoid_label,
        "decision_quality_vs_price": (
            "Distinguish WAIT quality from subsequent direction. "
            "A +15% rally after ENTRY_EXTENDED is not automatically a missed winner."
        ),
        "updates_policy": False,
        "learning_level": 1,
    }


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
