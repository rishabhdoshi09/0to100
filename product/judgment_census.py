"""Scoped census. Funnel stages never mix silently with overlapping diagnostics."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

SHORTLIST_TIERS = {"high_conviction", "good_setup"}


def unique_shortlist_symbols(reco: Mapping[str, Any]) -> list[str]:
    """Unique HC+GS names. Ensemble category counts may overlap; this does not."""
    names: list[str] = []
    seen: set[str] = set()
    for cat in reco.get("categories") or []:
        if not isinstance(cat, Mapping):
            continue
        cid = str(cat.get("id") or cat.get("key") or "")
        for card in cat.get("cards") or []:
            if not isinstance(card, Mapping):
                continue
            tier = str(card.get("reco_tier") or "")
            if tier not in SHORTLIST_TIERS and cid not in SHORTLIST_TIERS:
                continue
            symbol = str(card.get("symbol") or "").upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                names.append(symbol)
    return names


def build_census(
    *,
    scan: Mapping[str, Any],
    reco: Mapping[str, Any],
    committee: Sequence[Mapping[str, Any]],
    session: str,
    scan_run_id: str,
    generated_at: str,
    researched_symbols: Sequence[str] = (),
    candidate_states: Mapping[str, int] | None = None,
    population: str = "",
    scope_kind: str = "CURRENT_SCAN",
) -> dict[str, Any]:
    coverage = dict(scan.get("coverage") or {})
    reasons = dict(coverage.get("reason_counts") or {})
    summary = dict(scan.get("summary") or {})
    ensemble = dict(reco.get("ensemble") or {})
    records = list(committee or [])
    buy = [r for r in records if r.get("decision") == "BUY"]
    wait = [r for r in records if r.get("decision") == "WAIT"]
    avoid = [r for r in records if r.get("decision") == "AVOID"]
    ready = [r for r in records if r.get("candidate_state") == "READY"]
    exec_blocked = [r for r in records if str(r.get("execution_state") or "").startswith("BLOCKED")]
    veto_n = sum(len(r.get("vetoes") or []) for r in records)
    disagree = sum(1 for r in records if r.get("disagreement"))

    raw = int(coverage.get("requested") or scan.get("requested_universe") or 0)
    eligible = int(coverage.get("checked") or scan.get("scanned") or 0)
    setup = int(coverage.get("qualified") or summary.get("qualified") or 0)
    unique_shortlist = unique_shortlist_symbols(reco)
    ensemble_shortlist = int(ensemble.get("high_conviction_count") or 0) + int(
        ensemble.get("good_setup_count") or 0
    )
    shortlist_n = len(unique_shortlist) or ensemble_shortlist
    shortlist_set = set(unique_shortlist)
    evaluated_n = len(records)
    committee_on_shortlist = (
        sum(1 for r in records if str(r.get("symbol") or "").upper() in shortlist_set)
        if shortlist_set
        else min(evaluated_n, shortlist_n)
    )
    serious = sum(
        1
        for r in records
        if r.get("tier") in SHORTLIST_TIERS and r.get("decision") in {"BUY", "WAIT"}
    )
    researched = len([s for s in researched_symbols if s])

    # Strict funnel must decline. Deep research is a side path, not a stage
    # between shortlist and committee (we evaluate the shortlist, research a subset).
    funnel = [
        {
            "id": "RAW_INSTRUMENTS",
            "n": raw,
            "source": "scan.coverage.requested",
            "scope": scope_kind,
            "overlapping": False,
            "note": "instrument master walk, includes non-cash rows",
        },
        {
            "id": "ELIGIBLE",
            "n": eligible,
            "source": "scan.coverage.checked",
            "scope": scope_kind,
            "overlapping": False,
            "note": "names the scanner actually evaluated",
        },
        {
            "id": "SETUP_CANDIDATES",
            "n": setup,
            "source": "scan.coverage.qualified",
            "scope": scope_kind,
            "overlapping": False,
            "note": "any technical setup",
        },
        {
            "id": "RECOMMENDATION_SHORTLIST",
            "n": shortlist_n,
            "source": "unique flatten of high_conviction+good_setup cards",
            "scope": scope_kind,
            "overlapping": False,
            "note": "unique symbols; ensemble category counts may double-count",
        },
        {
            "id": "COMMITTEE",
            "n": min(int(committee_on_shortlist), int(shortlist_n)),
            "source": "decision_committee.evaluate_many ∩ SHORTLIST",
            "scope": scope_kind,
            "overlapping": False,
            "note": "Evaluated shortlist names only. Remembered extra names are a side path.",
        },
        {
            "id": "SERIOUS_CANDIDATES",
            "n": serious,
            "source": "committee BUY|WAIT on HC/GS",
            "scope": scope_kind,
            "overlapping": False,
            "note": "survived committee as investable or wait",
        },
        {
            "id": "BUY",
            "n": len(buy),
            "source": "committee.decision",
            "scope": scope_kind,
            "overlapping": False,
            "note": "investment judgment",
        },
        {
            "id": "READY",
            "n": len(ready),
            "source": "committee.candidate_state",
            "scope": scope_kind,
            "overlapping": False,
            "note": "BUY + entry + required evidence + no hard veto",
        },
    ]

    return {
        "schema_version": 2,
        "scope": {
            "kind": scope_kind,
            "session": session,
            "scan_run_id": scan_run_id,
            "generated_at": generated_at,
            "population": population
            or "latest_momentum_scan + latest_recommendations + this committee pass",
        },
        "funnel": funnel,
        "judgments": {
            "BUY": len(buy),
            "WAIT": len(wait),
            "AVOID": len(avoid),
            "READY": len(ready),
            "EXECUTION_BLOCKED": len(exec_blocked),
            "PAPER_ENTERED": 0,
        },
        "overlapping_diagnostics": {
            "scope": "scan.coverage.reason_counts — overlapping, not a funnel",
            "reason_counts": reasons,
            "extended": int(summary.get("extended") or 0),
            "watch_tier": int(ensemble.get("watch_count") or 0),
            "ensemble_shortlist_possibly_overlapping": ensemble_shortlist,
            "unique_shortlist": shortlist_n,
            "deep_researched": researched,
            "committee_evaluated_including_memory": {
                "count": evaluated_n,
                "scope": "CURRENT_SCAN+REMEMBERED_WAIT",
                "overlapping": True,
                "note": "May exceed SHORTLIST when opportunity memory adds names.",
            },
        },
        "side_paths": {
            "deep_research": {
                "n": researched,
                "symbols": list(researched_symbols),
                "scope": scope_kind,
                "overlapping": True,
                "note": "Information-value subset of the shortlist. Not a funnel stage.",
            }
        },
        "committee_stats": {
            "evaluated": evaluated_n,
            "on_shortlist": committee_on_shortlist,
            "hard_vetoes": veto_n,
            "method_disagreements": disagree,
            "researched_symbols": list(researched_symbols),
        },
        "candidate_states": dict(candidate_states or {}),
        "monotonic_funnel": all(
            funnel[i]["n"] >= funnel[i + 1]["n"] for i in range(len(funnel) - 1)
        ),
    }
