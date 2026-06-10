"""
Layer 11 — Historical Blow-Up Similarity Engine.

Compares the company's forensic fingerprint against the pre-collapse profiles
of famous failures. Each historical case is defined by the features it
exhibited in the 1–3 years before collapse; similarity is the weighted share
of those features the current company also shows.

Features are graded 0–1 (smooth ramps, not hard booleans), so a company that
is *approaching* a DHFL-style profile scores partially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
    stmt_row, val, pct_change,
)
from forensics.statement_forensics import (
    REVENUE, NET_INCOME, CFO, FCF, TOTAL_ASSETS, RECEIVABLES, TOTAL_DEBT,
    EQUITY, GOODWILL, INTANGIBLES,
)


def _ramp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """0 below lo, 1 above hi, linear in between. None propagates."""
    if x is None:
        return None
    if hi > lo:
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return max(0.0, min(1.0, (lo - x) / (lo - hi)))  # inverted ramp


FEATURE_DESC = {
    "debt_surge": "rapid debt growth",
    "high_leverage": "high leverage (D/E)",
    "weak_cash_conversion": "deteriorating cash conversion",
    "high_accruals": "profits not backed by cash (accruals)",
    "receivables_gap": "receivables outpacing revenue",
    "hypergrowth": "abnormally fast revenue growth",
    "negative_fcf": "persistent negative free cash flow",
    "distress_z": "Altman Z in/near distress zone",
    "manipulator_m": "Beneish M-Score near manipulator zone",
    "goodwill_heavy": "acquisition-inflated balance sheet (goodwill)",
    "governance_risk": "weak governance / audit oversight",
}


@dataclass
class BlowupCase:
    name: str
    summary: str
    features: Dict[str, float]   # feature → weight in this case's fingerprint


CASES: List[BlowupCase] = [
    BlowupCase("Enron (2001)",
               "Off-balance-sheet leverage and accrual-stuffed earnings.",
               {"high_accruals": 3, "manipulator_m": 3, "weak_cash_conversion": 2,
                "hypergrowth": 1, "governance_risk": 2}),
    BlowupCase("Satyam (2009)",
               "Fabricated cash and receivables; promoter-controlled board.",
               {"receivables_gap": 3, "high_accruals": 2, "manipulator_m": 2,
                "governance_risk": 3}),
    BlowupCase("DHFL (2019)",
               "Debt-fueled loan book, weakening cash generation, ALM mismatch.",
               {"debt_surge": 3, "high_leverage": 3, "weak_cash_conversion": 2,
                "distress_z": 2, "negative_fcf": 1}),
    BlowupCase("IL&FS (2018)",
               "Opaque leverage pyramid funded by short-term paper.",
               {"high_leverage": 3, "debt_surge": 2, "negative_fcf": 2,
                "distress_z": 2, "governance_risk": 1}),
    BlowupCase("Luckin Coffee (2020)",
               "Fabricated sales: hypergrowth with no matching cash.",
               {"hypergrowth": 3, "weak_cash_conversion": 3, "receivables_gap": 2,
                "negative_fcf": 2}),
    BlowupCase("Wirecard (2020)",
               "Fictitious profits parked offshore; audit oversight failure.",
               {"manipulator_m": 3, "high_accruals": 2, "governance_risk": 3,
                "hypergrowth": 1}),
    BlowupCase("Carillion (2018)",
               "Goodwill-heavy roll-up paying dividends from negative FCF.",
               {"goodwill_heavy": 3, "negative_fcf": 3, "high_accruals": 2,
                "high_leverage": 1}),
    BlowupCase("Evergrande (2021)",
               "Property leverage spiral; growth entirely debt-funded.",
               {"debt_surge": 3, "high_leverage": 3, "negative_fcf": 2,
                "distress_z": 2}),
]


def _company_features(d: FundamentalData, m_score: Optional[float],
                      z_score: Optional[float]) -> Dict[str, Optional[float]]:
    inc, bal, cf = d.income, d.balance, d.cashflow
    rev = stmt_row(inc, REVENUE)
    ni = stmt_row(inc, NET_INCOME)
    cfo = stmt_row(cf, CFO)
    fcf = stmt_row(cf, FCF)
    ta = stmt_row(bal, TOTAL_ASSETS)
    recv = stmt_row(bal, RECEIVABLES)
    debt = stmt_row(bal, TOTAL_DEBT)
    eq = stmt_row(bal, EQUITY)
    gw0 = val(stmt_row(bal, GOODWILL)) or val(stmt_row(bal, INTANGIBLES))

    debt_g = pct_change(val(debt), val(debt, 1))
    de = (val(debt) / val(eq)) if (val(debt) is not None and val(eq)) else None
    conv = (val(cfo) / val(ni)) if (val(cfo) is not None and (val(ni) or 0) > 0) else None
    accrual = ((val(ni) - val(cfo)) / val(ta)) \
        if None not in (val(ni), val(cfo), val(ta)) and val(ta) else None
    rev_g = pct_change(val(rev), val(rev, 1))
    recv_g = pct_change(val(recv), val(recv, 1))
    gap = (recv_g - rev_g) if None not in (recv_g, rev_g) else None
    fcf_vals = [val(fcf, i) for i in range(min(3, len(cf.columns)))]
    fcf_known = [v for v in fcf_vals if v is not None]
    neg_share = (sum(1 for v in fcf_known if v < 0) / len(fcf_known)) \
        if fcf_known else None
    gw_ratio = (gw0 / val(ta)) if (gw0 is not None and val(ta)) else None
    audit = d.info.get("auditRisk")
    overall = d.info.get("overallRisk")
    gov = max(audit or 0, overall or 0) if (audit or overall) else None

    return {
        "debt_surge": _ramp(debt_g, 0.15, 0.50),
        "high_leverage": _ramp(de, 1.0, 3.0),
        "weak_cash_conversion": _ramp(conv, 0.8, 0.2),
        "high_accruals": _ramp(accrual, 0.03, 0.12),
        "receivables_gap": _ramp(gap, 0.05, 0.30),
        "hypergrowth": _ramp(rev_g, 0.25, 0.60),
        "negative_fcf": neg_share,
        "distress_z": _ramp(z_score, 2.99, 1.5),
        "manipulator_m": _ramp(m_score, -2.22, -1.5),
        "goodwill_heavy": _ramp(gw_ratio, 0.20, 0.50),
        "governance_risk": _ramp(float(gov), 5, 9) if gov is not None else None,
    }


def analyze(d: FundamentalData, m_score: Optional[float],
            z_score: Optional[float]) -> LayerResult:
    res = LayerResult(layer="Blow-Up Similarity Engine", score=None)
    if d.income.empty or d.balance.empty:
        res.notes.append("Financial statements unavailable — layer skipped.")
        return res

    feats = _company_features(d, m_score, z_score)
    matches = []
    for case in CASES:
        wsum = hit = 0.0
        drivers = []
        for fname, w in case.features.items():
            fv = feats.get(fname)
            if fv is None:
                continue
            wsum += w
            hit += w * fv
            if fv >= 0.5:
                drivers.append(FEATURE_DESC[fname])
        if wsum < sum(case.features.values()) * 0.5:
            continue  # too little data to compare against this case
        matches.append((case, hit / wsum, drivers))

    matches.sort(key=lambda t: -t[1])
    res.extras["matches"] = [(c.name, sim, drv) for c, sim, drv in matches[:3]]

    for case, sim, drivers in matches[:3]:
        res.metrics.append(Metric(
            name=case.name, value=sim, display=f"{sim:.0%} similar",
            what=f"Fingerprint overlap with {case.name}: {case.summary}",
            why="Companies rarely fail in new ways — pre-collapse profiles "
                "rhyme across decades and markets.",
            good="Below ~30% similarity to every case.",
            implication=("Driven by: " + ", ".join(drivers) + ".") if drivers
            else "No individual feature strongly matched.",
        ))

    if matches:
        top_case, top_sim, top_drivers = matches[0]
        res.score = max(0.0, 100 - top_sim * 100)
        if top_sim >= 0.60:
            res.flags.append(RedFlag(
                title=f"Blow-up profile match: {top_case.name}",
                severity=Severity.CRITICAL if top_sim >= 0.75 else Severity.HIGH,
                evidence=f"Current profile shares {top_sim:.0%} similarity with "
                         f"{top_case.name} prior to collapse, due to "
                         f"{', '.join(top_drivers) or 'multiple partial matches'}.",
                why_it_matters=top_case.summary,
                precedent=f"{top_case.name} — equity holders were effectively "
                          "wiped out.",
            ))
    else:
        res.notes.append("Insufficient data to fingerprint against historical "
                         "failure cases.")
    return res
