"""Decision-quality analysis for later rallies. A rise is not automatically an error."""
from __future__ import annotations

from typing import Any, Mapping

from product import decision_taxonomy as T
from product.counterfactual_learning import GOOD_WAIT, MISSED_WINNER

WAIT_RATIONALLY_MAINTAINED = "WAIT_RATIONALLY_MAINTAINED"
MISSED_REENTRY = "MISSED_REENTRY"

# Gates that mean the machine never offered an acceptable risk entry.
RATIONAL_NO_ENTRY = frozenset(
    {
        T.ENTRY_TOO_EXTENDED,
        T.NO_VALID_ENTRY,
        T.ENTRY_NOT_TRIGGERED,
        T.LOW_LIQUIDITY,
        T.LIQUIDITY_FAILED,
        T.UNACCEPTABLE_DOWNSIDE,
        T.DATA_INTEGRITY,
    }
)

# Hard quality vetoes: later price action does not prove they were wrong.
RATIONAL_QUALITY = frozenset(
    {
        T.BUSINESS_QUALITY_FAIL,
        T.FINANCIAL_QUALITY_FAIL,
        T.DD_GATE_FAILED,
        T.LOW_QUALITY_SETUP,
        T.NO_SETUP,
    }
)

# Gates that may be too conservative; still do not auto-change policy.
POSSIBLY_STRICT = frozenset(
    {
        T.INSUFFICIENT_EVIDENCE,
        T.WEAK_SECTOR,
        T.EMPIRICAL_GATE_FAILED,
        T.EVIDENCE_POLICY_BLOCK,
        T.REGIME_INCOMPATIBLE,
    }
)


def analyze_decision_quality(
    row: Mapping[str, Any],
    *,
    classification: str,
    forward_return_pct: float | None,
    later_entered: bool = False,
) -> dict[str, Any]:
    reason = str(row.get("reason_code") or "")
    decision = str(row.get("decision") or "")
    out = {
        "symbol": row.get("symbol"),
        "original_decision": decision,
        "primary_veto": reason,
        "classification": classification,
        "subsequent_return_pct": forward_return_pct,
        "later_entered": later_entered,
        "original_decision_rational": None,
        "note": "",
        "updates_policy": False,
        "wait_attribution": "",
    }
    if classification == GOOD_WAIT or later_entered:
        out["original_decision_rational"] = True
        out["note"] = "YES — wait resolved into a later valid entry"
        out["wait_attribution"] = GOOD_WAIT
        return out
    if classification != MISSED_WINNER:
        out["note"] = "Not a missed-winner path. Classification stands on its own."
        return out
    if reason in RATIONAL_NO_ENTRY:
        out["original_decision_rational"] = True
        out["note"] = "YES — never offered acceptable risk entry"
        out["wait_attribution"] = WAIT_RATIONALLY_MAINTAINED
        return out
    if reason in RATIONAL_QUALITY:
        out["original_decision_rational"] = True
        out["note"] = f"YES — {reason} was a quality veto; a later rally does not reverse it"
        out["wait_attribution"] = WAIT_RATIONALLY_MAINTAINED
        return out
    if reason in POSSIBLY_STRICT:
        out["original_decision_rational"] = None
        out["note"] = "UNCLEAR — later rally does not prove the gate was wrong; sample first"
        return out
    out["original_decision_rational"] = False
    out["note"] = "NO — gate may be too conservative"
    if bool(row.get("later_valid_entry")) and bool(row.get("wake_failed")):
        out["wait_attribution"] = MISSED_REENTRY
    return out
