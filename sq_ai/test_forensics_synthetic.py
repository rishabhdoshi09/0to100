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
                       blowup, committee)

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
}
layers["blowup"] = blowup.analyze(
    d, m_score=layers["fraud"].extras.get("m_score"),
    z_score=layers["fraud"].extras.get("z_score"))
flags = az._cross_layer_flags(d)
for L in layers.values():
    flags.extend(L.flags)
from forensics.models import SEVERITY_ORDER
flags.sort(key=lambda f: -SEVERITY_ORDER[f.severity])
composite = compose(layers, flags)
committee_result = committee.convene(d, layers, flags)
report = az.AnalysisReport(symbol="SYNTH", ticker="SYNTH.NS",
                           company=info["longName"], layers=layers,
                           flags=flags, composite=composite,
                           committee=committee_result)
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
print(f"\nSMOKE TEST PASSED — {len(flags)} flags, score "
      f"{composite.score:.1f}, verdict {composite.verdict.value}, "
      f"committee {committee_result.tally} → {committee_result.consensus}, "
      f"top blow-up match {top_name} {top_sim:.0%}")
