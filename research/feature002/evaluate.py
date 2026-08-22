"""FEATURE-002 maturity and rank metrics. No graduation before gates."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from research.feature002.constants import (
    DECISION_MONTHS,
    DECISION_MULTI_SETS,
    DECISION_PER_FAMILY,
    DECISION_RESOLVED,
    EARLY_MAX,
    INTERIM_MAX,
    QUIET_MAX,
    UNTIL_MATURE,
)
from research.feature002.ledger import counts, list_primary_observations


def _resolved(rows: list[dict[str, Any]], field: str = "ret_5d") -> list[dict[str, Any]]:
    return [r for r in rows if r.get(field) is not None]


def maturity(rows: list[dict[str, Any]] | None = None, *, path=None) -> dict[str, Any]:
    rows = rows if rows is not None else list_primary_observations(path=path)
    resolved = _resolved(rows)
    by_set: dict[str, list] = defaultdict(list)
    for r in resolved:
        by_set[str(r.get("candidate_set_id") or "")].append(r)
    multi = sum(1 for v in by_set.values() if len(v) >= 2)
    dates = sorted({str(r.get("session_date") or "")[:7] for r in resolved if r.get("session_date")})
    months = len(dates)
    fam_n: dict[str, int] = defaultdict(int)
    for r in resolved:
        try:
            fams = json.loads(r["families"]) if isinstance(r.get("families"), str) else (r.get("families") or [])
        except Exception:
            fams = []
        for f in fams:
            fam_n[str(f)] += 1
    n = len(resolved)
    if n <= QUIET_MAX:
        stage = "QUIET"
    elif n <= EARLY_MAX:
        stage = "EARLY"
    elif n <= INTERIM_MAX:
        stage = "INTERIM"
    else:
        stage = "INTERIM"
    capable = (
        n >= DECISION_RESOLVED
        and multi >= DECISION_MULTI_SETS
        and months >= DECISION_MONTHS
    )
    if capable:
        stage = "DECISION-CAPABLE"
    return {
        "stage": stage,
        "n_primary": len(rows),
        "n_resolved_5d": n,
        "n_multi_candidate_sets": multi,
        "n_months": months,
        "family_resolved": dict(fam_n),
        "decision_capable": capable,
        "gates": {
            "resolved": DECISION_RESOLVED,
            "multi_sets": DECISION_MULTI_SETS,
            "per_family": DECISION_PER_FAMILY,
            "months": DECISION_MONTHS,
        },
        "verdict": None if capable else UNTIL_MATURE,
    }


def spearman_rank(ranks: list[float], ys: list[float]) -> dict[str, Any]:
    if len(ranks) < 8:
        return {"n": len(ranks), "rho": None}
    try:
        from scipy.stats import spearmanr
        # lower rank number = better pick; invert so higher predicted quality
        pred = [-float(x) for x in ranks]
        r = spearmanr(pred, ys)
        return {"n": len(ranks), "rho": float(r.statistic), "p": float(r.pvalue)}
    except Exception:
        return {"n": len(ranks), "rho": None}


def top_minus_bottom(rows: list[dict[str, Any]], rank_key: str, outcome_key: str = "ret_5d") -> dict[str, Any]:
    usable = [r for r in rows if r.get(rank_key) is not None and r.get(outcome_key) is not None]
    by_set: dict[str, list] = defaultdict(list)
    for r in usable:
        by_set[str(r.get("candidate_set_id") or "")].append(r)
    spreads = []
    for chunk in by_set.values():
        if len(chunk) < 4:
            continue
        k = max(1, len(chunk) // 4)
        ranked = sorted(chunk, key=lambda r: int(r[rank_key]))
        top = ranked[:k]
        bot = ranked[-k:]
        mt = sum(float(r[outcome_key]) for r in top) / len(top)
        mb = sum(float(r[outcome_key]) for r in bot) / len(bot)
        spreads.append(mt - mb)
    if not spreads:
        return {"n_sets": 0, "mean": None}
    return {"n_sets": len(spreads), "mean": float(sum(spreads) / len(spreads))}


def summarize(*, path=None) -> dict[str, Any]:
    rows = list_primary_observations(path=path)
    mat = maturity(rows, path=path)
    resolved = _resolved(rows)
    out = {
        "maturity": mat,
        "ledger": counts(path=path),
        "status": mat["verdict"] or UNTIL_MATURE,
        "rank_metrics": None,
        "note": (
            "Primary FEATURE-002 statistics require live_scan rows recorded "
            "on or after the protocol timestamp. Implementation-test rows are excluded."
        ),
    }
    if mat["stage"] in {"INTERIM", "DECISION-CAPABLE"} and resolved:
        out["rank_metrics"] = {
            "spearman_rs": spearman_rank(
                [r.get("rs_rank") for r in resolved],
                [float(r["ret_5d"]) for r in resolved],
            ),
            "spearman_trend": spearman_rank(
                [r.get("trend_rank") for r in resolved],
                [float(r["ret_5d"]) for r in resolved],
            ),
            "spearman_production": spearman_rank(
                [r.get("production_rank") for r in resolved],
                [float(r["ret_5d"]) for r in resolved],
            ),
            "tmb_rs": top_minus_bottom(resolved, "rs_rank"),
            "tmb_trend": top_minus_bottom(resolved, "trend_rank"),
            "tmb_production": top_minus_bottom(resolved, "production_rank"),
        }
    return out
