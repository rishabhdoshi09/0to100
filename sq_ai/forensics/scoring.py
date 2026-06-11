"""
Composite scoring + verdict engine.

Institutional Quality Score (0–100), weighted across layers, with weights
renormalized over layers that actually produced a score (no silent neutral
filling). Verdict is rules-based on score + red flag severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.models import LayerResult, RedFlag, Severity, SEVERITY_ORDER, Verdict

# layer key → (weight, layer name used in results dict)
WEIGHTS: Dict[str, float] = {
    "forensics": 0.20,        # Financial Quality
    "governance": 0.15,
    "quant": 0.15,            # Quant Stability
    "valuation": 0.10,
    "fraud": 0.15,            # Fraud Risk
    "cashflow": 0.10,         # Cash Flow Strength (sub-score of forensics layer)
    "altdata": 0.05,
    "capital_allocation": 0.10,
}


@dataclass
class CompositeResult:
    score: Optional[float]
    coverage: float                       # fraction of weight backed by real data
    components: List[Tuple[str, float, Optional[float]]]  # (label, weight, score)
    # Explainability graph: signed points each component moved the score
    # away from the neutral 50 baseline.
    contributions: List[Tuple[str, float]] = field(default_factory=list)
    verdict: Verdict = Verdict.NEUTRAL
    confidence: str = "Low"
    strengths: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    hidden_signals: List[str] = field(default_factory=list)


LABELS = {
    "forensics": "Financial Quality", "governance": "Governance",
    "quant": "Quant Stability", "valuation": "Valuation",
    "fraud": "Fraud Risk", "cashflow": "Cash Flow Strength",
    "altdata": "Alternative Data", "capital_allocation": "Capital Allocation",
}


def _cashflow_subscore(forensics: LayerResult) -> Optional[float]:
    cf_metrics = [m for m in forensics.metrics
                  if m.score is not None and
                  m.name in ("Cash Conversion (CFO/NI)", "Free Cash Flow (3y)")]
    if not cf_metrics:
        return None
    return sum(m.score for m in cf_metrics) / len(cf_metrics)


def compose(layers: Dict[str, LayerResult], all_flags: List[RedFlag]) -> CompositeResult:
    layer_scores: Dict[str, Optional[float]] = {
        k: layers[k].score for k in WEIGHTS if k in layers}
    layer_scores["cashflow"] = _cashflow_subscore(layers["forensics"]) \
        if "forensics" in layers else None
    layer_scores["capital_allocation"] = layers["capital_allocation"].score \
        if "capital_allocation" in layers else None

    components, weighted, wsum = [], 0.0, 0.0
    for key, w in WEIGHTS.items():
        s = layer_scores.get(key)
        components.append((LABELS[key], w, s))
        if s is not None:
            weighted += s * w
            wsum += w

    score = weighted / wsum if wsum > 0 else None
    coverage = wsum / sum(WEIGHTS.values())

    res = CompositeResult(score=score, coverage=coverage, components=components)
    if wsum > 0:
        res.contributions = sorted(
            ((LABELS[key], (s - 50) * w / wsum)
             for key, w in WEIGHTS.items()
             if (s := layer_scores.get(key)) is not None),
            key=lambda t: -t[1])
    res.confidence = ("High" if coverage >= 0.85 else
                      "Medium" if coverage >= 0.6 else "Low")

    # ── Verdict rules: score sets the base, severity caps it ──────────────────
    n_crit = sum(1 for f in all_flags if f.severity == Severity.CRITICAL)
    n_high = sum(1 for f in all_flags if f.severity == Severity.HIGH)

    if score is None:
        res.verdict = Verdict.NEUTRAL
    elif n_crit >= 2:
        res.verdict = Verdict.AVOID
    elif n_crit == 1:
        res.verdict = Verdict.HIGH_RISK
    elif n_high >= 3:
        res.verdict = Verdict.HIGH_RISK if score < 50 else Verdict.CAUTION
    elif score >= 75 and n_high == 0:
        res.verdict = Verdict.STRONG_CANDIDATE
    elif score >= 60:
        res.verdict = Verdict.WATCHLIST
    elif score >= 45:
        res.verdict = Verdict.NEUTRAL
    else:
        res.verdict = Verdict.CAUTION

    # ── Narrative inputs ───────────────────────────────────────────────────────
    all_metrics = [m for L in layers.values() for m in L.metrics]
    strong = sorted((m for m in all_metrics if m.score is not None and m.score >= 75),
                    key=lambda m: -m.score)[:4]
    res.strengths = [f"{m.name}: {m.display} — {m.implication}" for m in strong]

    top_flags = sorted(all_flags, key=lambda f: -SEVERITY_ORDER[f.severity])[:4]
    res.risks = [f"[{f.severity.value}] {f.title} — {f.evidence}" for f in top_flags]

    # Hidden signals: things retail dashboards don't show
    hidden_names = {"Accrual Ratio", "Receivables vs Revenue Growth",
                    "Beneish M-Score", "Revenue Volatility (QoQ)",
                    "Short Interest", "Insider Transactions (12m)"}
    res.hidden_signals = [
        f"{m.name}: {m.display} — {m.implication}"
        for m in all_metrics if m.name in hidden_names][:5]

    return res
