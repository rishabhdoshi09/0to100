"""
Position Sizing Engine — Institutional Recommendation Framework.

Bridges research → action. Takes the full forensic/quant output and produces
a portfolio bucket with a suggested position weight.

Design principles:
  1. Hard disqualifiers first. No score can override a CRITICAL flag,
     distress-zone Z, manipulator M-Score, or high blow-up similarity.
  2. Soft caps second. Grey-zone Z, elevated blow-up match, or HIGH flags
     limit the ceiling even if the composite score is strong.
  3. Score-based placement last. Only after the gates pass does the IQS
     composite drive the bucket and weight.

Buckets:
  Core Compounder  8–10%   Exceptional quality across every dimension
  High Conviction  5–8%    Strong research with manageable risk
  Standard         3–5%    Adequate quality, some concerns
  Speculative      1–3%    Material risk but non-zero thesis
  Tracking         0.1–1%  Watch only — do not size up
  Avoid            0%      Hard disqualified
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.models import (
    FundamentalData, LayerResult, RedFlag, Severity, SEVERITY_ORDER,
)
from forensics.scoring import CompositeResult

GRADE_SCORE: Dict[str, float] = {
    "A+": 95, "A": 88, "B+": 78, "B": 68, "C": 52, "D": 35, "F": 15, "N/A": 50,
}

# (min_score, bucket_name, (weight_lo, weight_hi))
BUCKET_MAP: List[Tuple[float, str, Tuple[float, float]]] = [
    (85.0, "Core Compounder", (8.0, 10.0)),
    (72.0, "High Conviction",  (5.0,  8.0)),
    (58.0, "Standard",         (3.0,  5.0)),
    (42.0, "Speculative",      (1.0,  3.0)),
    (20.0, "Tracking",         (0.1,  1.0)),
    ( 0.0, "Avoid",            (0.0,  0.0)),
]


@dataclass
class SizingResult:
    bucket: str
    weight_lo: float
    weight_hi: float
    suggested_weight: float
    sizing_score: float               # 0–100 composite used for placement
    hard_gate: str                    # which gate fired (empty if none)
    soft_cap: str                     # which soft cap applied (empty if none)
    reasoning: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)


def _bucket_from_score(score: float) -> Tuple[str, float, float]:
    for thresh, name, (lo, hi) in BUCKET_MAP:
        if score >= thresh:
            return name, lo, hi
    return "Avoid", 0.0, 0.0


def recommend(
    composite: CompositeResult,
    flags: List[RedFlag],
    layers: Dict[str, "LayerResult"],
) -> SizingResult:
    fraud = layers.get("fraud")
    quant = layers.get("quant")
    blowup = layers.get("blowup")
    cap = layers.get("capital_allocation")
    micro = layers.get("microstructure")

    m_score: Optional[float] = fraud.extras.get("m_score") if fraud else None
    z_score: Optional[float] = fraud.extras.get("z_score") if fraud else None
    top_sim: Optional[float] = (blowup.extras.get("matches", [[None, 0]])[0][1]
                                if blowup and blowup.extras.get("matches") else None)
    top_case: str = (blowup.extras.get("matches", [["?"]])[0][0]
                     if blowup and blowup.extras.get("matches") else "")
    ann_vol: Optional[float] = None
    max_dd: Optional[float] = None
    if quant:
        for m in quant.metrics:
            if m.name == "Annualized Volatility" and m.value is not None:
                ann_vol = m.value
            if m.name == "Max Drawdown (5y)" and m.value is not None:
                max_dd = m.value
    grade = cap.extras.get("grade", "N/A") if cap else "N/A"
    accum = micro.extras.get("accumulation_score") if micro else None
    n_critical = sum(1 for f in flags if f.severity == Severity.CRITICAL)
    n_high = sum(1 for f in flags if f.severity == Severity.HIGH)
    iqs = composite.score

    reasoning: List[str] = []
    inputs = {
        "IQS": f"{iqs:.0f}/100" if iqs is not None else "n/a",
        "Fraud Risk (M-Score)": (f"{m_score:.2f} — {'MANIPULATOR' if m_score > -1.78 else 'clean'}"
                                 if m_score is not None else "n/a"),
        "Distress Risk (Z-Score)": (f"{z_score:.2f} — {'DISTRESS' if z_score < 1.81 else 'grey' if z_score < 2.99 else 'safe'}"
                                    if z_score is not None else "n/a"),
        "Blow-Up Similarity": f"{top_sim:.0%} to {top_case}" if top_sim is not None else "n/a",
        "Capital Allocation": grade,
        "Accumulation Score": f"{accum:.0f}/100" if accum is not None else "n/a",
        "Annualized Volatility": f"{ann_vol:.1%}" if ann_vol is not None else "n/a",
        "Max Drawdown": f"{max_dd:.1%}" if max_dd is not None else "n/a",
        "Critical / High Flags": f"{n_critical} critical / {n_high} high",
    }

    # ── HARD DISQUALIFIERS ────────────────────────────────────────────────────
    gate = ""
    if n_critical >= 1:
        gate = f"{n_critical} CRITICAL flag(s) active"
        reasoning.append(f"Hard gate: {gate}")
    elif m_score is not None and m_score > -1.78:
        gate = f"Beneish M-Score {m_score:.2f} in manipulator zone"
        reasoning.append(f"Hard gate: {gate}")
    elif z_score is not None and z_score < 1.81:
        gate = f"Altman Z-Score {z_score:.2f} in distress zone"
        reasoning.append(f"Hard gate: {gate}")
    elif top_sim is not None and top_sim >= 0.75:
        gate = f"{top_sim:.0%} blow-up similarity to {top_case}"
        reasoning.append(f"Hard gate: {gate} — profile matches pre-collapse forensics")

    if gate:
        return SizingResult(
            bucket="Avoid", weight_lo=0.0, weight_hi=0.0, suggested_weight=0.0,
            sizing_score=0.0, hard_gate=gate, soft_cap="",
            reasoning=reasoning, inputs=inputs,
        )

    # ── COMPOSITE SIZING SCORE ────────────────────────────────────────────────
    parts: List[Tuple[float, float]] = []
    if iqs is not None:
        parts.append((iqs, 0.35))
    grade_s = GRADE_SCORE.get(grade, 50.0)
    parts.append((grade_s, 0.20))
    if accum is not None:
        parts.append((accum, 0.15))
    if ann_vol is not None:
        vol_s = max(0.0, min(100.0, 100 - (ann_vol - 0.15) * 200))
        parts.append((vol_s, 0.15))
    if max_dd is not None:
        dd_s = max(0.0, min(100.0, 100 + max_dd * 130))
        parts.append((dd_s, 0.15))

    if parts:
        total_w = sum(w for _, w in parts)
        sizing_score = sum(s * w for s, w in parts) / total_w
    else:
        sizing_score = 50.0

    # Flag penalty
    sizing_score = max(0.0, sizing_score - n_high * 4)

    # ── SOFT CAPS ─────────────────────────────────────────────────────────────
    soft_cap = ""
    cap_ceiling = 100.0
    if top_sim is not None and top_sim >= 0.50:
        cap_ceiling = min(cap_ceiling, 25.0)
        soft_cap = f"Blow-up similarity {top_sim:.0%} to {top_case} → max Tracking"
        reasoning.append(f"Soft cap: {soft_cap}")
    elif z_score is not None and z_score < 2.99:
        cap_ceiling = min(cap_ceiling, 42.0)
        soft_cap = f"Z-Score {z_score:.2f} in grey zone → max Speculative"
        reasoning.append(f"Soft cap: {soft_cap}")
    if n_high >= 3 and cap_ceiling > 58.0:
        cap_ceiling = min(cap_ceiling, 58.0)
        msg = f"{n_high} high-severity flags → cap at Standard"
        soft_cap = f"{soft_cap}; {msg}" if soft_cap else msg
        reasoning.append(f"Soft cap: {msg}")

    effective_score = min(sizing_score, cap_ceiling)

    # ── BUCKET PLACEMENT ──────────────────────────────────────────────────────
    bucket, w_lo, w_hi = _bucket_from_score(effective_score)
    mid = (w_lo + w_hi) / 2
    # Nudge within range based on confidence level
    conf_adj = {"High": 0.5, "Medium": 0.0, "Low": -0.3}.get(composite.confidence, 0.0)
    suggested = max(w_lo, min(w_hi, mid + conf_adj * (w_hi - w_lo) / 2))

    # Reasoning narrative
    reasoning.append(f"Sizing score: {sizing_score:.1f}"
                     + (f" → capped to {effective_score:.1f}" if effective_score < sizing_score else ""))
    reasoning.append(f"Placed in '{bucket}' bucket ({w_lo}–{w_hi}%)")
    reasoning.append(f"Confidence {composite.confidence} → suggested {suggested:.1f}%")

    return SizingResult(
        bucket=bucket, weight_lo=w_lo, weight_hi=w_hi, suggested_weight=round(suggested, 1),
        sizing_score=effective_score, hard_gate="", soft_cap=soft_cap,
        reasoning=reasoning, inputs=inputs,
    )
