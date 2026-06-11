"""
Data Reliability Framework.

Every analysis layer declares the data sources it consumed and their
reliability tier. The framework aggregates these into:

  • Per-dimension coverage scores
  • An overall Coverage Score fed into the confidence level

Coverage dimensions and their ideal sources:
  Fundamentals     — Audited Annual Report / Exchange Filing (XBRL)
  Governance       — Exchange Filing / NSE Shareholding Pattern
  Ownership        — NSE Shareholding Pattern / BSE Filing
  Market Structure — NSE Bhavcopy / Exchange Filing
  Alternative Data — Concall Transcript / Alternative / Web Data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.models import SOURCE_RELIABILITY

# dimension → list of (source_used, weight)
DIMENSION_IDEAL_SOURCES: Dict[str, List[str]] = {
    "Fundamentals": [
        "Audited Annual Report", "Exchange Filing (XBRL)", "Market Data (yfinance)"],
    "Governance": [
        "Exchange Filing (XBRL)", "NSE Shareholding Pattern",
        "Market Data (yfinance)", "Not Available"],
    "Ownership": [
        "NSE Shareholding Pattern", "BSE Filing",
        "Market Data (yfinance)", "Not Available"],
    "Market Structure": [
        "Exchange Filing (XBRL)", "Market Data (yfinance)", "Not Available"],
    "Alternative Data": [
        "Concall Transcript", "Alternative / Web Data", "Not Available"],
}

DIMENSION_WEIGHT: Dict[str, float] = {
    "Fundamentals":     0.35,
    "Governance":       0.25,
    "Ownership":        0.20,
    "Market Structure": 0.12,
    "Alternative Data": 0.08,
}


@dataclass
class DimensionCoverage:
    dimension: str
    sources_used: List[Tuple[str, float]]   # (source_label, reliability)
    score: float                            # 0–100
    missing: List[str]                      # signals not available


@dataclass
class CoverageReport:
    dimensions: List[DimensionCoverage]
    overall: float
    breakdown: Dict[str, float]             # dimension → score


class CoverageTracker:
    """Collects source declarations from layers and computes coverage scores."""

    def __init__(self) -> None:
        # dimension → list of (source, reliability)
        self._entries: Dict[str, List[Tuple[str, float]]] = {
            d: [] for d in DIMENSION_IDEAL_SOURCES}
        self._missing: Dict[str, List[str]] = {d: [] for d in DIMENSION_IDEAL_SOURCES}

    def declare(self, dimension: str, source: str,
                metric_name: str = "") -> float:
        """Register a data source for a dimension. Returns its reliability."""
        rel = SOURCE_RELIABILITY.get(source, 50.0)
        if dimension in self._entries:
            self._entries[dimension].append((source, rel))
        return rel

    def mark_missing(self, dimension: str, what: str) -> None:
        """Record a data point that is unavailable."""
        if dimension in self._missing:
            self._missing[dimension].append(what)
            self._entries[dimension].append(("Not Available", 0.0))

    def report(self) -> CoverageReport:
        dim_results: List[DimensionCoverage] = []
        breakdown: Dict[str, float] = {}

        for dim, entries in self._entries.items():
            if not entries:
                score = 0.0
            else:
                reliabilities = [r for _, r in entries]
                # Score is the mean reliability of all declared sources,
                # penalised for missing data points
                n_missing = sum(1 for s, _ in entries if s == "Not Available")
                n_total = len(entries)
                mean_rel = sum(reliabilities) / n_total
                missing_penalty = (n_missing / n_total) * 30
                score = max(0.0, mean_rel - missing_penalty)
            breakdown[dim] = score
            dim_results.append(DimensionCoverage(
                dimension=dim,
                sources_used=[(s, r) for s, r in entries if s != "Not Available"],
                score=score,
                missing=self._missing.get(dim, []),
            ))

        overall = sum(breakdown[d] * DIMENSION_WEIGHT[d]
                      for d in DIMENSION_WEIGHT if d in breakdown)
        return CoverageReport(dimensions=dim_results, overall=overall,
                              breakdown=breakdown)


def source_metric(metric, dimension: str, source: str,
                  tracker: Optional[CoverageTracker]) -> None:
    """Tag an existing Metric with source info and register it."""
    metric.source = source
    metric.reliability = SOURCE_RELIABILITY.get(source, 50.0)
    if tracker:
        tracker.declare(dimension, source, metric.name)
