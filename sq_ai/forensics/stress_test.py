"""
Scenario Stress Testing — Institutional Survival Framework.

Applies four macro shocks to the actual financial model and recomputes
Altman Z, interest coverage, FCF, and debt/EBITDA from first principles.
Survival probability is derived from the recomputed numbers, not heuristics.

Scenarios:
  Mild Recession    Revenue -10%, margin -2pp, CFO -15%
  Severe Recession  Revenue -20%, margin -5pp, CFO -30%
  Credit Shock      Interest cost +200bps on outstanding debt, CFO -10%
  Liquidity Crunch  CFO -30%, revenue -5%, margin -1pp, rates +100bps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.models import FundamentalData, stmt_row, val
from forensics.statement_forensics import (
    REVENUE, TOTAL_ASSETS, CURRENT_ASSETS, CURRENT_LIABILITIES,
    TOTAL_DEBT, EQUITY, CFO, FCF,
)
from forensics.capital_allocation import EBIT_

DEPREC = ["Depreciation And Amortization", "Depreciation Amortization Depletion",
          "Reconciled Depreciation"]
INTEREST = ["Interest Expense", "Interest Expense Non Operating",
            "Net Interest Income"]
RETAINED = ["Retained Earnings"]
SHARES_PRICE = None  # market cap supplied from info dict


@dataclass
class ScenarioResult:
    name: str
    revenue_shock: float
    margin_shock: float       # absolute pp shift on operating margin
    interest_shock: float     # absolute rate increase on outstanding debt
    cfo_shock: float
    stressed_ebit: Optional[float]
    stressed_interest: Optional[float]
    stressed_cfo: Optional[float]
    stressed_fcf: Optional[float]
    interest_coverage: Optional[float]
    debt_ebitda: Optional[float]
    z_score: Optional[float]
    survival_probability: float      # 0–100
    outcome: str                     # Safe / Stressed / High Risk / Critical
    details: List[Tuple[str, str]] = field(default_factory=list)


SCENARIOS: Dict[str, Dict] = {
    "Mild Recession": {
        "revenue_shock": -0.10,
        "margin_shock": -0.02,
        "interest_shock": 0.0,
        "cfo_shock": -0.15,
    },
    "Severe Recession": {
        "revenue_shock": -0.20,
        "margin_shock": -0.05,
        "interest_shock": 0.0,
        "cfo_shock": -0.30,
    },
    "Credit Shock": {
        "revenue_shock": 0.0,
        "margin_shock": 0.0,
        "interest_shock": 0.02,
        "cfo_shock": -0.10,
    },
    "Liquidity Crunch": {
        "revenue_shock": -0.05,
        "margin_shock": -0.01,
        "interest_shock": 0.01,
        "cfo_shock": -0.30,
    },
}


def _survival(
    interest_coverage: Optional[float],
    stressed_fcf: Optional[float],
    z: Optional[float],
    debt_ebitda: Optional[float],
) -> float:
    prob = 85.0
    if interest_coverage is not None:
        if interest_coverage < 1.0:
            prob -= 45
        elif interest_coverage < 1.5:
            prob -= 30
        elif interest_coverage < 2.5:
            prob -= 15
        elif interest_coverage < 3.5:
            prob -= 5
    if stressed_fcf is not None and stressed_fcf < 0:
        prob -= 18
    if z is not None:
        if z < 1.81:
            prob -= 30
        elif z < 2.99:
            prob -= 10
    if debt_ebitda is not None:
        if debt_ebitda > 8:
            prob -= 20
        elif debt_ebitda > 5:
            prob -= 10
        elif debt_ebitda > 3:
            prob -= 3
    return max(0.0, min(100.0, prob))


def _outcome(prob: float) -> str:
    if prob >= 75:
        return "Safe"
    if prob >= 55:
        return "Stressed"
    if prob >= 35:
        return "High Risk"
    return "Critical"


def run(d: FundamentalData) -> List[ScenarioResult]:
    inc, bal, cf = d.income, d.balance, d.cashflow
    results = []

    rev0 = val(stmt_row(inc, REVENUE))
    ebit0 = val(stmt_row(inc, EBIT_))
    ta0 = val(stmt_row(bal, TOTAL_ASSETS))
    ca0 = val(stmt_row(bal, CURRENT_ASSETS))
    cl0 = val(stmt_row(bal, CURRENT_LIABILITIES))
    debt0 = val(stmt_row(bal, TOTAL_DEBT))
    eq0 = val(stmt_row(bal, EQUITY))
    re0 = val(stmt_row(bal, RETAINED))
    cfo0 = val(stmt_row(cf, CFO))
    fcf0 = val(stmt_row(cf, FCF))
    dep0 = val(stmt_row(cf, DEPREC)) or val(stmt_row(inc, DEPREC))
    mcap = d.info.get("marketCap")

    # Interest expense: try direct, fallback to EBIT - EBT
    int_raw = stmt_row(inc, INTEREST)
    interest0 = val(int_raw)
    if interest0 is None or interest0 >= 0:
        # yfinance signs interest expense positive sometimes; or try EBT route
        ebt_row = stmt_row(inc, ["Pretax Income", "Earnings Before Tax"])
        if ebt_row is not None and ebit0 is not None:
            ebt0 = val(ebt_row)
            if ebt0 is not None:
                interest0 = max(0.0, ebit0 - ebt0)
    if interest0 is None:
        interest0 = (debt0 * 0.07) if debt0 else 0.0  # fallback: assume 7% cost

    # Operating margin for margin shock translation
    op_margin = (ebit0 / rev0) if (ebit0 is not None and rev0 and rev0 > 0) else 0.0

    for name, shocks in SCENARIOS.items():
        rs = shocks["revenue_shock"]
        ms = shocks["margin_shock"]
        is_ = shocks["interest_shock"]
        cs = shocks["cfo_shock"]

        stressed_rev = (rev0 * (1 + rs)) if rev0 is not None else None
        stressed_ebit = None
        if ebit0 is not None and rev0 is not None:
            # shocked EBIT = base EBIT + revenue Δ × operating margin + revenue × margin_shock
            stressed_ebit = ebit0 + rev0 * rs * op_margin + (rev0 if rev0 else 0) * ms

        stressed_interest = None
        if interest0 is not None:
            extra = (debt0 * is_) if (debt0 is not None and is_) else 0.0
            stressed_interest = interest0 * (1 + is_) + extra

        stressed_cfo = (cfo0 * (1 + cs)) if cfo0 is not None else None
        stressed_fcf = None
        if fcf0 is not None and cfo0 is not None and stressed_cfo is not None:
            stressed_fcf = fcf0 + (stressed_cfo - cfo0)

        ic = None
        if stressed_ebit is not None and stressed_interest and stressed_interest > 0:
            ic = stressed_ebit / stressed_interest
        elif stressed_ebit is not None:
            ic = float("inf")  # no debt

        ebitda = None
        if stressed_ebit is not None and dep0 is not None:
            ebitda = stressed_ebit + dep0
        debt_ebitda = (debt0 / ebitda) if (ebitda and ebitda > 0 and debt0) else None

        # Recompute Altman Z under stress
        z = None
        if all(v is not None for v in [ta0, ca0, cl0, re0, stressed_ebit, stressed_rev, eq0]):
            wc = ca0 - cl0
            liab = (debt0 or 0) + cl0
            if mcap and liab and ta0:
                z = (1.2 * wc / ta0 + 1.4 * re0 / ta0 + 3.3 * stressed_ebit / ta0
                     + 0.6 * mcap / liab + 1.0 * stressed_rev / ta0)

        prob = _survival(ic, stressed_fcf, z, debt_ebitda)
        outcome = _outcome(prob)

        details = []
        if stressed_rev is not None:
            details.append(("Stressed Revenue", f"{stressed_rev:,.0f}  ({rs:+.0%} shock)"))
        if stressed_ebit is not None:
            details.append(("Stressed EBIT", f"{stressed_ebit:,.0f}"))
        if ic is not None and ic != float("inf"):
            details.append(("Interest Coverage", f"{ic:.2f}x  {'✓' if ic > 2 else '⚠' if ic > 1 else '✗'}"))
        elif ic == float("inf"):
            details.append(("Interest Coverage", "∞ (no debt)"))
        if stressed_fcf is not None:
            details.append(("Stressed FCF", f"{stressed_fcf:,.0f}  {'✓' if stressed_fcf > 0 else '✗'}"))
        if debt_ebitda is not None:
            details.append(("Debt / EBITDA", f"{debt_ebitda:.1f}x  {'✓' if debt_ebitda < 3 else '⚠' if debt_ebitda < 5 else '✗'}"))
        if z is not None:
            zone = "safe ✓" if z > 2.99 else ("grey ⚠" if z > 1.81 else "DISTRESS ✗")
            details.append(("Stressed Altman Z", f"{z:.2f}  ({zone})"))
        details.append(("Survival Probability", f"{prob:.0f}%"))

        results.append(ScenarioResult(
            name=name,
            revenue_shock=rs, margin_shock=ms,
            interest_shock=is_, cfo_shock=cs,
            stressed_ebit=stressed_ebit,
            stressed_interest=stressed_interest,
            stressed_cfo=stressed_cfo,
            stressed_fcf=stressed_fcf,
            interest_coverage=ic if ic != float("inf") else None,
            debt_ebitda=debt_ebitda,
            z_score=z,
            survival_probability=prob,
            outcome=outcome,
            details=details,
        ))

    return results
