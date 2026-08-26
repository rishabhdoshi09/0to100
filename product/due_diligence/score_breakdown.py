"""Inspectable sector-weighted fundamental score. Missing KPIs are skipped."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# Display buckets. KPI `pillar` values map into these.
PILLAR_GROUPS: dict[str, tuple[str, ...]] = {
    "Growth": ("growth",),
    "Profitability": ("profitability",),
    "Asset quality": ("asset_quality",),
    "Balance sheet": ("leverage", "liquidity"),
    "Cash flow": ("cash",),
    "Capital strength": ("capital",),
    "Funding / deposits": ("funding",),
    "Governance": ("governance",),
    "Sector KPIs": ("sector",),
    "Consistency": ("consistency",),
}

SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "bank": {
        "Growth": 15, "Asset quality": 20, "Profitability": 15,
        "Capital strength": 15, "Funding / deposits": 15,
        "Consistency": 10, "Governance": 10,
    },
    "nbfc": {
        "Growth": 15, "Asset quality": 20, "Profitability": 15,
        "Capital strength": 10, "Funding / deposits": 15,
        "Cash flow": 5, "Consistency": 10, "Governance": 10,
    },
    "nbfc_gold": {
        "Growth": 15, "Asset quality": 20, "Profitability": 15,
        "Capital strength": 10, "Funding / deposits": 15,
        "Cash flow": 5, "Consistency": 10, "Governance": 10,
    },
    "nbfc_housing": {
        "Growth": 15, "Asset quality": 20, "Profitability": 15,
        "Capital strength": 10, "Funding / deposits": 15,
        "Cash flow": 5, "Consistency": 10, "Governance": 10,
    },
    "default": {
        "Growth": 20, "Profitability": 20, "Balance sheet": 15,
        "Cash flow": 15, "Consistency": 10, "Governance": 10,
        "Sector KPIs": 10,
    },
}


def _group_for(pillar: str) -> str:
    for label, members in PILLAR_GROUPS.items():
        if pillar in members:
            return label
    return "Sector KPIs"


def score_breakdown(
    findings: Sequence[Mapping[str, Any]],
    *,
    framework_id: str,
    coverage: float,
    overall: int | None,
) -> dict[str, Any]:
    weights = SECTOR_WEIGHTS.get(framework_id) or SECTOR_WEIGHTS["default"]
    by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in weights}
    for finding in findings:
        group = _group_for(str(finding.get("pillar") or ""))
        if group not in by_group:
            if group == "Asset quality" and "Asset quality" not in weights:
                group = "Sector KPIs"
            elif group == "Capital strength" and "Capital strength" not in weights:
                group = "Balance sheet"
            elif group == "Funding / deposits" and "Funding / deposits" not in weights:
                group = "Sector KPIs"
            else:
                group = "Sector KPIs" if "Sector KPIs" in by_group else next(iter(by_group))
        by_group.setdefault(group, []).append(dict(finding))

    pillars: list[dict[str, Any]] = []
    awarded_total = 0.0
    possible_total = 0.0
    for name, max_pts in weights.items():
        items = by_group.get(name) or []
        usable = [f for f in items if f.get("points") is not None]
        possible_total += max_pts
        if not usable:
            pillars.append({
                "id": name.lower().replace(" ", "_").replace("/", "_"),
                "label": name,
                "awarded": None,
                "max": max_pts,
                "display": f"Data unavailable / {max_pts:.0f}",
                "explain": "No measured KPI in this bucket — points are skipped, not guessed.",
                "kpis": [
                    {
                        "id": f.get("id"),
                        "label": f.get("label"),
                        "points": None,
                        "weight": f.get("weight"),
                        "trend": f.get("trend"),
                        "available": False,
                    }
                    for f in items
                ],
            })
            continue
        used_w = sum(float(f.get("weight") or 0) for f in usable)
        weighted = sum(float(f["points"]) * float(f["weight"]) for f in usable) / used_w
        awarded = round(weighted / 100.0 * max_pts, 1)
        awarded_total += awarded
        steps = []
        for finding in usable:
            kpi_share = float(finding["weight"]) / used_w * max_pts
            kpi_pts = round(float(finding["points"]) / 100.0 * kpi_share, 2)
            steps.append(
                f"{finding['label']}: trend {finding.get('trend')} → "
                f"{finding['points']:.0f}/100 × bucket share {kpi_share:.1f} = {kpi_pts}"
            )
        pillars.append({
            "id": name.lower().replace(" ", "_").replace("/", "_"),
            "label": name,
            "awarded": awarded,
            "max": max_pts,
            "display": f"{awarded:.0f}/{max_pts:.0f}",
            "explain": (
                f"{len(usable)} KPI(s). Bucket max {max_pts:.0f}. "
                + " ".join(steps)
            ),
            "formula": (
                f"bucket = Σ(kpi_points × kpi_weight) / Σ(kpi_weight) / 100 × {max_pts:.0f}"
            ),
            "kpis": [
                {
                    "id": f.get("id"),
                    "label": f.get("label"),
                    "points": f.get("points"),
                    "weight": f.get("weight"),
                    "trend": f.get("trend"),
                    "available": f.get("available"),
                    "fact": f.get("fact"),
                }
                for f in items
            ],
        })

    measured = [p for p in pillars if p.get("awarded") is not None]
    measured_max = sum(float(p["max"]) for p in measured) or 1.0
    scaled = int(round(awarded_total / measured_max * 100)) if measured else None
    return {
        "overall": overall,
        "scaled_from_buckets": scaled,
        "coverage_pct": round(coverage * 100.0, 1),
        "weights_for": framework_id,
        "pillars": pillars,
        "explain": (
            f"Overall {overall}/100 from measured sector KPIs. "
            f"Bucket weights are {framework_id}-specific. Missing buckets are skipped."
            if overall is not None else
            "Fundamental quality is Unmeasured — score coverage below 40% or no KPI values."
        ),
    }
