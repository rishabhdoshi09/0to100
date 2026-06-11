"""
Quant Red Flag Analyst — orchestrator.

Runs all analysis layers, cross-layer flag checks, position sizing, stress
testing, and optionally persists predictions to the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from logger import get_logger
from forensics import (
    altdata, blowup, capital_allocation, committee, data_source,
    fraud_models, governance, microstructure, quant_risk,
    statement_forensics, valuation,
)
from forensics import (stress_test, position_sizing, prediction_ledger,
                       promoter_intelligence, auditor_intelligence,
                       management_credibility, event_timeline)
from forensics.committee import CommitteeResult
from forensics.models import (
    FundamentalData, LayerResult, RedFlag, Severity, SEVERITY_ORDER,
    stmt_row, val, pct_change,
)
from forensics.scoring import CompositeResult, compose
from forensics.statement_forensics import REVENUE, CFO
from forensics.position_sizing import SizingResult
from forensics.stress_test import ScenarioResult
from forensics.data_reliability import CoverageTracker, CoverageReport
from forensics import knowledge_graph as kg

log = get_logger("forensics.analyzer")


@dataclass
class AnalysisReport:
    symbol: str
    ticker: str
    company: str
    layers: Dict[str, LayerResult] = field(default_factory=dict)
    flags: List[RedFlag] = field(default_factory=list)
    composite: CompositeResult = None  # type: ignore[assignment]
    committee: CommitteeResult = None  # type: ignore[assignment]
    sizing: Optional[SizingResult] = None
    stress: List[ScenarioResult] = field(default_factory=list)
    predictions_saved: int = 0
    coverage: Optional[CoverageReport] = None
    graph: Optional[kg.KnowledgeGraph] = None
    timeline: list = field(default_factory=list)  # List[TimelineEvent]


def _cross_layer_flags(d: FundamentalData) -> List[RedFlag]:
    flags: List[RedFlag] = []
    rev = stmt_row(d.income, REVENUE)
    cfo = stmt_row(d.cashflow, CFO)
    rev_g = pct_change(val(rev), val(rev, 1))
    cfo_g = pct_change(val(cfo), val(cfo, 1))
    period = str(d.years()[0].year) if d.years() else ""

    if rev_g is not None and cfo_g is not None and rev_g > 0.10 and cfo_g < 0:
        flags.append(RedFlag(
            title="Revenue growing while operating cash flow shrinks",
            severity=Severity.HIGH, period=period,
            evidence=f"Revenue {rev_g:+.1%} YoY but CFO {cfo_g:+.1%} YoY.",
            why_it_matters="Sales the company can't collect cash on aren't sales — "
                           "this divergence is the #1 screen used by activist shorts.",
            precedent="Both Luckin Coffee and Gensol Engineering showed booming "
                      "revenue with deteriorating cash flow before fraud surfaced.",
            evidence_rows=[
                ("Revenue (current year)", f"{val(rev):,.0f}"),
                ("Revenue (prior year)", f"{val(rev, 1):,.0f}"),
                ("Revenue growth YoY", f"{rev_g:+.1%}"),
                ("Operating Cash Flow (current year)", f"{val(cfo):,.0f}"),
                ("Operating Cash Flow (prior year)", f"{val(cfo, 1):,.0f}"),
                ("CFO growth YoY", f"{cfo_g:+.1%}"),
            ],
            sources=["Income Statement", "Cash Flow Statement"],
        ))
    return flags


def run_pipeline(d: FundamentalData,
                 save_predictions: bool = False) -> AnalysisReport:
    """Run every analysis layer on already-fetched data.

    Separated from fetching so Research Replay can run the identical
    pipeline on point-in-time clipped data."""
    tracker = CoverageTracker()

    layers: Dict[str, LayerResult] = {
        "forensics": statement_forensics.analyze(d),
        "quant": quant_risk.analyze(d),
        "fraud": fraud_models.analyze(d),
        "governance": governance.analyze_governance(d),
        "smart_money": governance.analyze_smart_money(d),
        "valuation": valuation.analyze(d),
        "altdata": altdata.analyze(d),
        "microstructure": microstructure.analyze(d),
        "capital_allocation": capital_allocation.analyze(d),
    }
    layers["blowup"] = blowup.analyze(
        d,
        m_score=layers["fraud"].extras.get("m_score"),
        z_score=layers["fraud"].extras.get("z_score"),
    )
    layers["promoter"] = promoter_intelligence.analyze(d, tracker)
    layers["auditor"] = auditor_intelligence.analyze(d, tracker)
    layers["credibility"] = management_credibility.analyze(d, tracker)

    flags = _cross_layer_flags(d)
    for L in layers.values():
        flags.extend(L.flags)
    flags.sort(key=lambda f: -SEVERITY_ORDER[f.severity])

    coverage = tracker.report()
    graph = kg.build(flags)
    timeline = event_timeline.build(d.symbol, layers, flags)

    composite = compose(layers, flags, data_reliability=coverage.overall)
    committee_result = committee.convene(d, layers, flags)
    sizing = position_sizing.recommend(composite, flags, layers)
    scenarios = stress_test.run(d)

    n_saved = 0
    if save_predictions:
        baseline = prediction_ledger.extract_baseline_metrics(layers)
        preds = prediction_ledger.generate_predictions(
            d.symbol, d.ticker, flags, baseline)
        n_saved = prediction_ledger.save_predictions(preds)
        log.info("predictions_saved", symbol=d.symbol, count=n_saved)

    log.info("analysis_complete", symbol=d.symbol,
             score=composite.score, verdict=composite.verdict.value,
             bucket=sizing.bucket, flags=len(flags))

    return AnalysisReport(
        symbol=d.symbol,
        ticker=d.ticker,
        company=d.info.get("longName") or d.symbol,
        layers=layers,
        flags=flags,
        composite=composite,
        committee=committee_result,
        sizing=sizing,
        stress=scenarios,
        predictions_saved=n_saved,
        coverage=coverage,
        graph=graph,
        timeline=timeline,
    )


class QuantRedFlagAnalyst:

    def analyze(self, symbol: str, save_predictions: bool = False) -> AnalysisReport:
        d = data_source.fetch_fundamentals(symbol)
        log.info("analysis_started", symbol=symbol, ticker=d.ticker)
        return run_pipeline(d, save_predictions=save_predictions)
