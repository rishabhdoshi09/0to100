"""
Management Credibility Tracker — promise vs delivery.

For every quantified management statement (guidance), store:
    fiscal year, metric, guided value, actual value, source.

After enough history, compute Management Accuracy: the fraction of guidance
items that were substantially met. A management that guides 25% growth and
delivers 12% — repeatedly — earns a credibility discount that no amount of
polished commentary can offset.

Data source: DAL category "concalls", payload schema:
  {
    "guidance": [
      {"fy": "FY24", "metric": "revenue_growth", "statement": "We expect 25% growth",
       "guided": 0.25, "actual": 0.12, "source": "Q4FY23 concall"},
      ...
    ]
  }
Legacy fallback: logs/{SYMBOL}_guidance.json with the same schema.

An item counts as MET if actual >= guided − tolerance, where tolerance is
20% of |guided| (relative) or 0.02 absolute, whichever is larger. Items
without an `actual` yet are PENDING and excluded from accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from forensics.models import (FundamentalData, LayerResult, Metric,
                               RedFlag, Severity)
from forensics.data_reliability import CoverageTracker, source_metric

_LOGS = Path(__file__).parent.parent / "logs"

MIN_ITEMS_FOR_SCORE = 3      # need ≥3 resolved items before judging
MIN_ITEMS_FOR_FLAG = 4       # and ≥4 before flagging chronic over-promising


@dataclass
class GuidanceItem:
    fy: str
    metric: str
    statement: str
    guided: float
    actual: Optional[float] = None
    source: str = ""

    @property
    def resolved(self) -> bool:
        return self.actual is not None

    @property
    def met(self) -> Optional[bool]:
        if self.actual is None:
            return None
        tolerance = max(abs(self.guided) * 0.20, 0.02)
        return self.actual >= self.guided - tolerance

    @property
    def miss_magnitude(self) -> Optional[float]:
        """How far short of guidance, as a fraction of the guided value."""
        if self.actual is None or self.guided == 0:
            return None
        return (self.guided - self.actual) / abs(self.guided)


def _load_guidance(symbol: str) -> List[GuidanceItem]:
    raw: Optional[Dict] = None
    # Preferred: DAL
    try:
        from forensics import dal
        rec = dal.get_latest("concalls", symbol)
        if rec is not None and isinstance(rec.payload, dict):
            raw = rec.payload
    except Exception:
        pass
    # Legacy fallback
    if raw is None:
        path = _LOGS / f"{symbol}_guidance.json"
        if path.exists():
            try:
                with open(path) as f:
                    raw = json.load(f)
            except Exception:
                raw = None
    if not raw:
        return []
    items = []
    for g in raw.get("guidance", []):
        try:
            items.append(GuidanceItem(
                fy=str(g["fy"]), metric=str(g["metric"]),
                statement=str(g.get("statement", "")),
                guided=float(g["guided"]),
                actual=float(g["actual"]) if g.get("actual") is not None else None,
                source=str(g.get("source", "")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return items


def analyze(d: FundamentalData,
            tracker: Optional[CoverageTracker] = None) -> LayerResult:
    items = _load_guidance(d.symbol)
    resolved = [g for g in items if g.resolved]
    pending = [g for g in items if not g.resolved]

    metrics: List[Metric] = []
    flags: List[RedFlag] = []
    notes: List[str] = []
    extras: Dict = {"n_items": len(items), "n_resolved": len(resolved),
                    "n_pending": len(pending)}

    if not items:
        if tracker:
            tracker.mark_missing("Alternative Data", "Management guidance history")
        notes.append("No guidance history on file. Add records via the DAL "
                     "(category 'concalls') to activate this tracker.")
        return LayerResult(layer="Management Credibility", score=None,
                           metrics=metrics, flags=flags, notes=notes,
                           extras=extras)

    if tracker:
        tracker.declare("Alternative Data", "Concall Transcript",
                        "Management guidance")

    if len(resolved) < MIN_ITEMS_FOR_SCORE:
        notes.append(f"Only {len(resolved)} resolved guidance item(s) — "
                     f"need {MIN_ITEMS_FOR_SCORE} before scoring credibility.")
        return LayerResult(layer="Management Credibility", score=None,
                           metrics=metrics, flags=flags, notes=notes,
                           extras=extras)

    hits = sum(1 for g in resolved if g.met)
    accuracy = hits / len(resolved)
    misses = [g for g in resolved if not g.met]
    avg_miss = (sum(g.miss_magnitude for g in misses if g.miss_magnitude is not None)
                / len(misses)) if misses else 0.0

    score = round(accuracy * 100, 1)
    extras.update({"accuracy": accuracy, "avg_miss_magnitude": avg_miss,
                   "hits": hits})

    m_acc = Metric(
        name="Management Accuracy",
        value=accuracy,
        display=f"{accuracy:.0%} ({hits}/{len(resolved)} guidance items met)",
        what="Fraction of quantified management guidance that was substantially "
             "delivered (within 20% relative tolerance).",
        why="Managements that systematically over-promise are re-rated downward "
            "by institutions; chronic misses also correlate with aggressive "
            "accounting to 'bridge' the gap.",
        good="≥ 75% over a full cycle. Below 50% is a structural credibility problem.",
        implication=(
            "Management guidance has been reliable." if accuracy >= 0.75
            else "Mixed delivery — discount forward guidance."
            if accuracy >= 0.5
            else "Management chronically over-promises — treat guidance as marketing."
        ),
        score=score,
    )
    source_metric(m_acc, "Alternative Data", "Concall Transcript", tracker)
    metrics.append(m_acc)

    if misses:
        m_miss = Metric(
            name="Average Miss Magnitude",
            value=avg_miss,
            display=f"{avg_miss:.0%} below guidance",
            what="When guidance was missed, by how much on average "
                 "(as a fraction of the guided value).",
            why="Small misses are forecasting noise; large misses suggest "
                "guidance is aspirational rather than grounded.",
            good="< 20%. Misses above 40% mean the guidance process is broken.",
            implication=f"Typical miss falls {avg_miss:.0%} short of what was guided.",
        )
        source_metric(m_miss, "Alternative Data", "Concall Transcript", tracker)
        metrics.append(m_miss)

    if accuracy < 0.5 and len(resolved) >= MIN_ITEMS_FOR_FLAG:
        worst = max(misses, key=lambda g: g.miss_magnitude or 0)
        flags.append(RedFlag(
            title="Management chronically over-promises",
            severity=Severity.HIGH,
            evidence=f"Only {accuracy:.0%} of {len(resolved)} guidance items "
                     f"delivered; average miss {avg_miss:.0%} below guidance.",
            why_it_matters="Guidance that consistently exceeds delivery is a "
                           "credibility signal institutions track formally — and "
                           "it pressures management toward aggressive accounting.",
            precedent="Yes Bank guided aggressive growth every year 2017-19 "
                      "while asset quality deteriorated.",
            sources=["Concall Transcript"],
            evidence_rows=(
                [("Accuracy", f"{accuracy:.0%} ({hits}/{len(resolved)})"),
                 ("Avg miss magnitude", f"{avg_miss:.0%}"),
                 ("Worst miss", f"{worst.fy} {worst.metric}: guided "
                                f"{worst.guided:.0%} → actual {worst.actual:.0%}")]
            ),
        ))

    if pending:
        notes.append(f"{len(pending)} guidance item(s) pending resolution: "
                     + ", ".join(f"{g.fy} {g.metric}" for g in pending[:4]))

    return LayerResult(layer="Management Credibility", score=score,
                       metrics=metrics, flags=flags, notes=notes,
                       extras=extras)
