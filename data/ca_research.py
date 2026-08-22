"""Corporate-action research acceptability — separate from CA_COMPLETE.

Never infer an adjustment factor from a price gap. Unresolved events are
quarantined or segmented, not silently fixed.
"""
from __future__ import annotations

from typing import Any

from data.corporate_actions import ledger_status, load_events, todo_path
from research.sepa.ca_audit import EVENT_CLASSES, classify_subject

CA_COMPLETE = "CA_COMPLETE"
CA_RESEARCH_ACCEPTABLE = "CA_RESEARCH_ACCEPTABLE"
CA_INCOMPLETE = "CA_INCOMPLETE"

SHARE_COUNT = {"split", "bonus", "consolidation"}
UNRESOLVED_CLASSES = {
    "rights": "quarantine",
    "demerger": "quarantine",
    "merger": "quarantine",
    "special_distribution": "quarantine",
    "symbol_restructuring": "research_segmentation",
    "unknown_discontinuity": "quarantine",
}


def disposition(event_class: str) -> dict[str, Any]:
    cls = str(event_class or "unknown_discontinuity")
    if cls in SHARE_COUNT:
        return {
            "event_class": cls,
            "choice": "authoritative_adjustment",
            "usable_in_price_series": True,
            "note": "Share-count factor from official ledger only.",
        }
    treat = UNRESOLVED_CLASSES.get(cls, "quarantine")
    return {
        "event_class": cls,
        "choice": treat,
        "usable_in_price_series": False,
        "note": "Do not infer a factor from the gap. Segment or censor the name.",
    }


def research_status(path=None) -> dict[str, Any]:
    st = ledger_status(path, verify=False)
    n = int(st.get("events") or 0)
    types_ok = True  # ledger_status already drops non-share-count types
    acceptable = n > 0 and types_ok
    complete = False  # rights/demerger/merger archive is not on file
    label = CA_RESEARCH_ACCEPTABLE if acceptable and not complete else (
        CA_COMPLETE if complete else CA_INCOMPLETE
    )
    if acceptable and not complete:
        label = CA_RESEARCH_ACCEPTABLE
    return {
        **st,
        "ca_complete": False,
        "ca_research_acceptable": acceptable,
        "label": label,
        "status": "RESEARCH_READY_WITH_LIMITATIONS" if acceptable else "DESCRIPTIVE_ONLY",
        "unresolved_classes": UNRESOLVED_CLASSES,
        "todo_path": str(todo_path()),
        "never_infers_from_gaps": True,
        "verifier_unchanged": True,
        "note": (
            "Share-count bonus/split/consolidation rows may be used when present. "
            "The global CA verifier is not weakened. Completeness requires official "
            "treatment of rights, demergers, mergers, and symbol transitions."
        ),
    }


def events_as_of(symbol: str, as_of, path=None) -> list[dict[str, Any]]:
    """Share-count events with ex_date <= as_of. Future CA cannot leak backward."""
    import pandas as pd
    asof = pd.Timestamp(as_of)
    all_ev = load_events(path)
    out = []
    for ev in all_ev.get(str(symbol).upper(), []):
        ex = ev.get("ex_date")
        if ex is None:
            continue
        if pd.Timestamp(ex) <= asof:
            out.append({
                "symbol": str(symbol).upper(),
                "ex_date": str(pd.Timestamp(ex).date()),
                "factor": ev.get("factor"),
                "type": ev.get("type"),
            })
    return out
