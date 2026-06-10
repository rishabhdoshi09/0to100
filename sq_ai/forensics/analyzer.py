"""
Quant Red Flag Analyst — orchestrator.

Runs all analysis layers against one symbol, performs cross-layer red flag
checks, and produces the composite Institutional Quality Score + verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from logger import get_logger
from forensics import (
    altdata, data_source, fraud_models, governance, quant_risk,
    statement_forensics, valuation,
)
from forensics.models import (
    FundamentalData, LayerResult, RedFlag, Severity, SEVERITY_ORDER,
    stmt_row, val, pct_change,
)
from forensics.scoring import CompositeResult, compose
from forensics.statement_forensics import REVENUE, CFO

log = get_logger("forensics.analyzer")


@dataclass
class AnalysisReport:
    symbol: str
    ticker: str
    company: str
    layers: Dict[str, LayerResult] = field(default_factory=dict)
    flags: List[RedFlag] = field(default_factory=list)
    composite: CompositeResult = None  # type: ignore[assignment]


def _cross_layer_flags(d: FundamentalData) -> List[RedFlag]:
    """Critical checks that span statements (the classic short-seller screen)."""
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
        ))
    return flags


class QuantRedFlagAnalyst:
    """Institutional enhancement layer — run all forensic/quant engines on a symbol."""

    def analyze(self, symbol: str) -> AnalysisReport:
        d = data_source.fetch_fundamentals(symbol)
        log.info("analysis_started", symbol=symbol, ticker=d.ticker)

        layers: Dict[str, LayerResult] = {
            "forensics": statement_forensics.analyze(d),
            "quant": quant_risk.analyze(d),
            "fraud": fraud_models.analyze(d),
            "governance": governance.analyze_governance(d),
            "smart_money": governance.analyze_smart_money(d),
            "valuation": valuation.analyze(d),
            "altdata": altdata.analyze(d),
        }

        flags = _cross_layer_flags(d)
        for L in layers.values():
            flags.extend(L.flags)
        flags.sort(key=lambda f: -SEVERITY_ORDER[f.severity])

        composite = compose(layers, flags)
        log.info("analysis_complete", symbol=symbol,
                 score=composite.score, verdict=composite.verdict.value,
                 flags=len(flags))

        return AnalysisReport(
            symbol=d.symbol,
            ticker=d.ticker,
            company=d.info.get("longName") or d.symbol,
            layers=layers,
            flags=flags,
            composite=composite,
        )
