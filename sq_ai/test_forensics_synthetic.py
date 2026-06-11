"""
Offline smoke test for the Quant Red Flag Analyst.

Builds a synthetic company with deliberately planted red flags (high accruals,
receivables spike, negative FCF, debt surge, dilution, distress-zone Z) and
verifies every layer, the flag engine, scoring and the Rich renderer run
end-to-end without network access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from forensics import analyzer as az
from forensics.models import FundamentalData
from forensics.scoring import compose
from forensics.report import render
from forensics import (statement_forensics, quant_risk, fraud_models,
                       governance, valuation, altdata, microstructure,
                       blowup, committee, capital_allocation, tearsheet,
                       stress_test, position_sizing, prediction_ledger,
                       promoter_intelligence, auditor_intelligence,
                       management_credibility, event_timeline,
                       dal, replay, knowledge_graph as kg)
from forensics.data_reliability import CoverageTracker

years = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31", "2023-03-31"])

income = pd.DataFrame({
    years[0]: {"Total Revenue": 1500, "Net Income": 200, "Gross Profit": 450,
               "Operating Income": 250, "EBIT": 250,
               "Selling General And Administration": 150},
    years[1]: {"Total Revenue": 1000, "Net Income": 150, "Gross Profit": 380,
               "Operating Income": 200, "EBIT": 200,
               "Selling General And Administration": 90},
    years[2]: {"Total Revenue": 900, "Net Income": 140, "Gross Profit": 350,
               "Operating Income": 190, "EBIT": 190,
               "Selling General And Administration": 85},
    years[3]: {"Total Revenue": 850, "Net Income": 130, "Gross Profit": 330,
               "Operating Income": 180, "EBIT": 180,
               "Selling General And Administration": 80},
})

balance = pd.DataFrame({
    years[0]: {"Total Assets": 2000, "Current Assets": 900,
               "Current Liabilities": 700, "Accounts Receivable": 600,
               "Inventory": 300, "Goodwill": 900, "Total Debt": 1200,
               "Long Term Debt": 900, "Stockholders Equity": 500,
               "Ordinary Shares Number": 120, "Net PPE": 400,
               "Retained Earnings": 100},
    years[1]: {"Total Assets": 1500, "Current Assets": 700,
               "Current Liabilities": 450, "Accounts Receivable": 300,
               "Inventory": 180, "Goodwill": 850, "Total Debt": 700,
               "Long Term Debt": 500, "Stockholders Equity": 480,
               "Ordinary Shares Number": 100, "Net PPE": 380,
               "Retained Earnings": 90},
})

cashflow = pd.DataFrame({
    years[0]: {"Operating Cash Flow": -10, "Free Cash Flow": -110,
               "Capital Expenditure": -100, "Depreciation And Amortization": 40},
    years[1]: {"Operating Cash Flow": 120, "Free Cash Flow": -10,
               "Capital Expenditure": -130, "Depreciation And Amortization": 38},
    years[2]: {"Operating Cash Flow": 110, "Free Cash Flow": -5,
               "Capital Expenditure": -115, "Depreciation And Amortization": 36},
})

q_dates = pd.date_range("2024-06-30", periods=8, freq="QE")[::-1]
q_rev = [400, 380, 200, 260, 250, 245, 240, 235]  # planted outlier quarter
quarterly_income = pd.DataFrame({d: {"Total Revenue": r}
                                 for d, r in zip(q_dates, q_rev)})

rng = np.random.default_rng(7)
idx = pd.bdate_range("2021-06-01", "2026-06-01")
close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.025, len(idx))))
vol = rng.lognormal(13, 0.4, len(idx))
vol[-15] *= 8  # planted volume spike
px = pd.DataFrame({
    "Close": close,
    "High": close * (1 + rng.uniform(0, 0.02, len(idx))),
    "Low": close * (1 - rng.uniform(0, 0.02, len(idx))),
    "Volume": vol,
}, index=idx)
bench = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(
    rng.normal(0.0004, 0.011, len(idx))))}, index=idx)

info = {
    "longName": "Synthetic Industries Ltd", "marketCap": 1800,
    "trailingPE": 9.0, "forwardPE": 8.0, "priceToBook": 3.6,
    "priceToSalesTrailing12Months": 1.2, "enterpriseToEbitda": 10.0,
    "sharesOutstanding": 120, "heldPercentInsiders": 0.72,
    "heldPercentInstitutions": 0.04, "recommendationMean": 3.8,
    "numberOfAnalystOpinions": 4, "shortPercentOfFloat": 0.12,
    "targetMeanPrice": 90.0, "currentPrice": 100.0,
    "fullTimeEmployees": 4200, "auditRisk": 9, "boardRisk": 7,
    "compensationRisk": 5, "shareHolderRightsRisk": 8, "overallRisk": 8,
    "companyOfficers": [{"name": "A. Promoter", "title": "Managing Director & CEO"}],
}

d = FundamentalData(
    symbol="SYNTH", ticker="SYNTH.NS", info=info,
    income=income, balance=balance, cashflow=cashflow,
    quarterly_income=quarterly_income, prices=px, benchmark=bench,
)

layers = {
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
    d, m_score=layers["fraud"].extras.get("m_score"),
    z_score=layers["fraud"].extras.get("z_score"))
# ── Plant DAL records: a resigned auditor with a fee spike, and a management
#    team that chronically over-promises ─────────────────────────────────────
dal.put("auditors", "SYNTH", {
    "auditor_name": "Shady & Associates",
    "tenure_years": 1,
    "change_type": "resigned",
    "qualified_opinion": False,
    "emphasis_of_matter": True,
    "caro_observations": ["Delayed statutory dues", "Inventory records incomplete"],
    "audit_fees": [{"year": 2026, "fee": 21.0}, {"year": 2025, "fee": 12.0}],
}, source="Alternative / Web Data", period="2026-03-31")

dal.put("concalls", "SYNTH", {"guidance": [
    {"fy": "FY23", "metric": "revenue_growth", "statement": "We expect 30% growth",
     "guided": 0.30, "actual": 0.06},
    {"fy": "FY24", "metric": "revenue_growth", "statement": "We expect 25% growth",
     "guided": 0.25, "actual": 0.11},
    {"fy": "FY25", "metric": "ebitda_margin", "statement": "Margins will expand to 22%",
     "guided": 0.22, "actual": 0.14},
    {"fy": "FY26", "metric": "revenue_growth", "statement": "We expect 28% growth",
     "guided": 0.28, "actual": 0.05},
    {"fy": "FY27", "metric": "revenue_growth", "statement": "We expect 20% growth",
     "guided": 0.20},
]}, source="Concall Transcript", period="2026-03-31")

tracker = CoverageTracker()
layers["promoter"] = promoter_intelligence.analyze(d, tracker)
layers["auditor"] = auditor_intelligence.analyze(d, tracker)
layers["credibility"] = management_credibility.analyze(d, tracker)
coverage = tracker.report()
flags = az._cross_layer_flags(d)
for L in layers.values():
    flags.extend(L.flags)
from forensics.models import SEVERITY_ORDER
flags.sort(key=lambda f: -SEVERITY_ORDER[f.severity])
composite = compose(layers, flags, data_reliability=coverage.overall)
committee_result = committee.convene(d, layers, flags)
sizing_result = position_sizing.recommend(composite, flags, layers)
scenarios = stress_test.run(d)
graph = kg.build(flags)
timeline = event_timeline.build("SYNTH", layers, flags)
report = az.AnalysisReport(symbol="SYNTH", ticker="SYNTH.NS",
                           company=info["longName"], layers=layers,
                           flags=flags, composite=composite,
                           committee=committee_result,
                           sizing=sizing_result,
                           stress=scenarios,
                           coverage=coverage,
                           graph=graph,
                           timeline=timeline)
render(report, explain=True)

# ── Assertions: planted flags must be detected ─────────────────────────────────
titles = [f.title for f in flags]
expected = [
    "Revenue growing while operating cash flow shrinks",
    "High accruals — earnings not backed by cash",
    "Weak cash conversion",
    "Receivables growing unusually fast",
    "Debt increasing rapidly",
    "Share dilution",
    "Goodwill concentration — write-down risk",
    "Persistent negative free cash flow",
]
missed = [t for t in expected if t not in titles]
assert not missed, f"Planted flags not detected: {missed}"
assert composite.score is not None and composite.score < 55, composite.score
assert composite.verdict.value in ("High Risk", "Avoid", "Caution"), composite.verdict
assert any("anomaly" in t.lower() for t in titles), "QoQ outlier not detected"

# New layers: blow-up similarity, committee, explainability, microstructure
matches = layers["blowup"].extras.get("matches", [])
assert matches, "Blow-up engine produced no matches"
top_name, top_sim, _ = matches[0]
assert top_sim > 0.6, f"Planted DHFL-style profile under-matched: {top_name} {top_sim:.0%}"
assert committee_result.consensus == "Sell", committee_result.tally
assert composite.contributions, "Explainability contributions missing"
assert layers["microstructure"].score is not None, "Accumulation score missing"

# L9: capital allocation — synthetic company has goodwill=45% of assets + falling ROIC
cap_layer = layers["capital_allocation"]
assert cap_layer.score is not None, "Capital allocation score missing"
grade = cap_layer.extras.get("grade")
assert grade in ("D", "F", "C"), f"Expected poor grade for empire-builder: {grade}"
emp_sigs = cap_layer.extras.get("empire_signals", [])
assert emp_sigs, "Empire building signals not detected"

# Evidence Locker: key flags should carry structured evidence_rows
auditable = [f for f in flags if f.evidence_rows]
assert len(auditable) >= 3, f"Only {len(auditable)} flags have structured evidence"

# L15: tear sheet generation
bundle = tearsheet.generate(report)
assert "INSTITUTIONAL TEAR SHEET" in bundle.institutional
assert "Capital Allocation:" in bundle.one_pager
assert "Institutional Quality Score" in bundle.social_card
assert "╔" in bundle.social_card, "Social card missing box art"
assert "Position:" in bundle.social_card or "Avoid" in bundle.social_card

# Position sizing: synthetic fraud company must be Avoid or Tracking at most
assert sizing_result.bucket in ("Avoid", "Tracking", "Speculative"), \
    f"Expected Avoid/Tracking for fraud-flagged company, got: {sizing_result.bucket}"
assert sizing_result.hard_gate or sizing_result.soft_cap, \
    "Expected hard gate or soft cap for company with CRITICAL flag"

# Stress testing: 4 scenarios, Severe Recession must show distress
assert len(scenarios) == 4, f"Expected 4 scenarios, got {len(scenarios)}"
severe = next((s for s in scenarios if s.name == "Severe Recession"), None)
assert severe is not None
assert severe.outcome in ("High Risk", "Critical", "Stressed"), \
    f"Severe Recession outcome too optimistic: {severe.outcome}"

# Prediction ledger: predictions generated from flags
baseline = prediction_ledger.extract_baseline_metrics(layers)
preds = prediction_ledger.generate_predictions("SYNTH", "SYNTH.NS", flags, baseline)
assert len(preds) >= 3, f"Expected ≥3 predictions from {len(flags)} flags, got {len(preds)}"
# verify each has confidence between 0 and 1
for p in preds:
    assert 0 < p.confidence < 1, f"Bad confidence: {p.confidence}"
    assert p.horizon_days > 0

# Coverage-Adjusted IQS: raw score should exist and adjusted score should be
# shrunk toward 50 when evidence is imperfect
assert composite.raw_score is not None
assert composite.data_reliability == coverage.overall
if composite.raw_score < 50:
    assert composite.score >= composite.raw_score, \
        "Weak-evidence low score should regress toward neutral 50"

# Reliability-weighted flags: statement-sourced flags carry more force
rev_cfo = next(f for f in flags
               if f.title == "Revenue growing while operating cash flow shrinks")
assert rev_cfo.reliability > 0.9, f"Statement flag under-weighted: {rev_cfo.reliability}"
for f in flags:
    assert 0 < f.reliability <= 1.0

# Reliability-aware predictions: confidence haircut applied per source
for p in preds:
    assert 0 < p.source_reliability <= 1.0
    assert p.confidence <= 0.95

# Evidence chain rendering
ev_lines = kg.explain_verdict(graph, flags, composite.verdict.value)
assert any("VERDICT" in ln for ln in ev_lines), "Evidence chain missing verdict"

# Data Reliability Framework
assert coverage.overall >= 0, "Coverage overall score should be numeric"
assert len(coverage.dimensions) == 5, f"Expected 5 coverage dimensions, got {len(coverage.dimensions)}"

# Knowledge Graph
assert graph is not None, "Knowledge graph not built"
# The synthetic company fires many flags so the graph should have chains
if graph.nodes:
    assert len(graph.nodes) >= 3, f"Expected ≥3 active nodes, got {len(graph.nodes)}"
    # render_text should work without error
    lines = kg.render_text(graph)
    assert lines, "Knowledge graph render produced no output"

# Promoter and Auditor layers exist
assert "promoter" in layers, "Promoter intelligence layer missing"
assert "auditor" in layers, "Auditor intelligence layer missing"
assert layers["auditor"].extras.get("audit_risk_score") is not None

# DAL: roundtrip + point-in-time query
rec = dal.get_latest("auditors", "SYNTH")
assert rec is not None and rec.payload["auditor_name"] == "Shady & Associates"
assert rec.reliability == 60.0  # Alternative / Web Data tier
from datetime import date as _date
assert dal.get_as_of("auditors", "SYNTH", _date(2025, 1, 1)) is None, \
    "Point-in-time query leaked a future record"
assert dal.get_as_of("auditors", "SYNTH", _date(2026, 6, 1)) is not None

# Auditor Intelligence picked up the DAL record: resignation + fee spike + CARO
assert "Auditor changed or resigned" in titles, "Auditor resignation not flagged"
assert "Audit fee spike" in titles, "Audit fee spike (+75% YoY) not flagged"
assert "CARO observations flagged" in titles, "CARO observations not flagged"
assert layers["auditor"].extras["audit_risk_score"] < 60, \
    f"Audit risk score too lenient: {layers['auditor'].extras['audit_risk_score']}"

# Management Credibility: 0/4 resolved items met → chronic over-promising
cred = layers["credibility"]
assert cred.score is not None and cred.score < 50, f"Credibility score: {cred.score}"
assert "Management chronically over-promises" in titles, \
    "Chronic over-promising not flagged"
assert cred.extras["n_pending"] == 1

# Corporate Event Timeline merges auditor + guidance + forensic events
assert timeline, "Event timeline empty"
cats = {e.category for e in timeline}
assert "Auditor" in cats and "Guidance" in cats, f"Timeline categories: {cats}"

# Research Replay: point-in-time clipping must not leak the future
clipped = replay.clip(d, _date(2025, 6, 1))
assert len(clipped.income.columns) == 3, \
    f"Expected 3 fiscal periods as of 2025-06-01, got {len(clipped.income.columns)}"
assert clipped.prices.index[-1].date() <= _date(2025, 6, 1)
assert "marketCap" in clipped.info  # rescaled, not dropped
fwd = replay._fwd_return(d.prices, _date(2025, 6, 1), 6)
assert fwd is not None, "Forward return computation failed"

print(f"\nSMOKE TEST PASSED — {len(flags)} flags, score "
      f"{composite.score:.1f}, verdict {composite.verdict.value}, "
      f"committee {committee_result.tally} → {committee_result.consensus}, "
      f"position bucket {sizing_result.bucket}, "
      f"severe recession → {severe.outcome} ({severe.survival_probability:.0f}%), "
      f"{len(preds)} predictions generated, "
      f"top blow-up {top_name} {top_sim:.0%}, "
      f"capital grade {grade}, {len(auditable)} auditable flags")
