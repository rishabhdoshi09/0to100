"""
Layer 3 — Fraud Probability Engine.

Beneish M-Score (earnings manipulation), Altman Z-Score (bankruptcy distress),
Piotroski F-Score (fundamental strength).
"""

from __future__ import annotations

from typing import List, Optional

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity, stmt_row, val,
)
from forensics.statement_forensics import (
    REVENUE, NET_INCOME, GROSS_PROFIT, CFO, TOTAL_ASSETS, CURRENT_ASSETS,
    CURRENT_LIABILITIES, RECEIVABLES, TOTAL_DEBT, EQUITY, SHARES,
)

SGA = ["Selling General And Administration", "Selling General And Administrative"]
DEPREC = ["Depreciation And Amortization", "Depreciation Amortization Depletion",
          "Reconciled Depreciation"]
PPE = ["Net PPE", "Gross PPE", "Properties"]
LTDEBT = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
RETAINED = ["Retained Earnings"]
EBIT = ["EBIT", "Operating Income"]


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Fraud Probability Engine", score=None)
    inc, bal, cf = d.income, d.balance, d.cashflow
    if inc.empty or bal.empty or len(inc.columns) < 2:
        res.notes.append("Need two years of statements — layer skipped.")
        return res

    rev = stmt_row(inc, REVENUE)
    ni = stmt_row(inc, NET_INCOME)
    gp = stmt_row(inc, GROSS_PROFIT)
    cfo = stmt_row(cf, CFO)
    ta = stmt_row(bal, TOTAL_ASSETS)
    ca = stmt_row(bal, CURRENT_ASSETS)
    cl = stmt_row(bal, CURRENT_LIABILITIES)
    recv = stmt_row(bal, RECEIVABLES)
    debt = stmt_row(bal, TOTAL_DEBT)
    eq = stmt_row(bal, EQUITY)
    sga = stmt_row(inc, SGA)
    dep = stmt_row(cf, DEPREC) if stmt_row(cf, DEPREC) is not None else stmt_row(inc, DEPREC)
    ppe = stmt_row(bal, PPE)
    ltd = stmt_row(bal, LTDEBT)
    re_ = stmt_row(bal, RETAINED)
    ebit = stmt_row(inc, EBIT)
    shares = stmt_row(bal, SHARES)
    period = str(d.years()[0].year) if d.years() else ""
    scores: List[float] = []

    # ── Beneish M-Score ────────────────────────────────────────────────────────
    missing: List[str] = []

    def idx(name: str, curr: Optional[float], prev: Optional[float]) -> float:
        r = _ratio(curr, prev)
        if r is None:
            missing.append(name)
            return 1.0  # neutral substitution
        return r

    dsri = idx("DSRI", _ratio(val(recv), val(rev)), _ratio(val(recv, 1), val(rev, 1)))
    gm0, gm1 = _ratio(val(gp), val(rev)), _ratio(val(gp, 1), val(rev, 1))
    gmi = idx("GMI", gm1, gm0)  # prior/current — deterioration > 1
    sq0 = None if None in (val(ca), val(ppe), val(ta)) or not val(ta) else \
        1 - (val(ca) + val(ppe)) / val(ta)
    sq1 = None if None in (val(ca, 1), val(ppe, 1), val(ta, 1)) or not val(ta, 1) else \
        1 - (val(ca, 1) + val(ppe, 1)) / val(ta, 1)
    aqi = idx("AQI", sq0, sq1)
    sgi = idx("SGI", val(rev), val(rev, 1))
    depi = idx("DEPI",
               _ratio(val(dep, 1), (val(dep, 1) or 0) + (val(ppe, 1) or 0) or None),
               _ratio(val(dep), (val(dep) or 0) + (val(ppe) or 0) or None))
    sgai = idx("SGAI", _ratio(val(sga), val(rev)), _ratio(val(sga, 1), val(rev, 1)))
    lvgi = idx("LVGI",
               _ratio((val(cl) or 0) + (val(ltd) or 0) or None, val(ta)),
               _ratio((val(cl, 1) or 0) + (val(ltd, 1) or 0) or None, val(ta, 1)))
    tata = _ratio((val(ni) or 0) - (val(cfo) or 0), val(ta))
    if tata is None:
        missing.append("TATA")
        tata = 0.0

    m_score = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
               + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    manip = m_score > -1.78
    s_m = max(0.0, min(100.0, (-m_score - 1.78) * 30 + 60))
    scores.append(s_m)
    res.metrics.append(Metric(
        name="Beneish M-Score", value=m_score, display=f"{m_score:.2f}",
        what="8-variable statistical model estimating the probability that earnings "
             "are manipulated.",
        why="Trained on actual SEC enforcement cases; famously flagged Enron before "
            "its collapse (by a student project).",
        good="Below −2.22 is safe. Above −1.78 = likely manipulator profile.",
        implication="No manipulator profile detected." if not manip
        else "The financial profile resembles known earnings manipulators.",
        score=s_m,
    ))
    if missing:
        res.notes.append(f"M-Score components unavailable (neutral 1.0 used): {', '.join(missing)}")
    if manip:
        res.flags.append(RedFlag(
            title="Beneish M-Score in manipulator zone",
            severity=Severity.CRITICAL, period=period,
            evidence=f"M-Score {m_score:.2f} (threshold −1.78).",
            why_it_matters="Firms above the threshold are ~5x more likely to be "
                           "subsequently charged with accounting fraud.",
            precedent="Enron scored as a manipulator in 1998 — three years before "
                      "bankruptcy.",
        ))

    # ── Altman Z-Score ─────────────────────────────────────────────────────────
    ta0 = val(ta)
    mcap = d.info.get("marketCap")
    z = None
    if ta0 and None not in (val(ca), val(cl), val(re_), val(ebit), val(rev)):
        wc = val(ca) - val(cl)
        liab = (val(debt) or 0) + (val(cl) or 0)
        if mcap and liab:
            z = (1.2 * wc / ta0 + 1.4 * val(re_) / ta0 + 3.3 * val(ebit) / ta0
                 + 0.6 * mcap / liab + 1.0 * val(rev) / ta0)
    if z is not None:
        zone = "Safe" if z > 2.99 else ("Grey" if z > 1.81 else "Distress")
        s_z = max(0.0, min(100.0, z / 4 * 100))
        scores.append(s_z)
        res.metrics.append(Metric(
            name="Altman Z-Score", value=z, display=f"{z:.2f} ({zone} zone)",
            what="Bankruptcy-prediction score combining liquidity, profitability, "
                 "leverage, solvency and activity.",
            why="The standard credit-desk screen for financial distress two years out.",
            good="Above 2.99 = safe; 1.81–2.99 = grey; below 1.81 = distress zone.",
            implication=f"Z-Score places the firm in the {zone.lower()} zone.",
            score=s_z,
        ))
        if z < 1.81:
            res.flags.append(RedFlag(
                title="Altman Z-Score in distress zone",
                severity=Severity.CRITICAL, period=period,
                evidence=f"Z-Score {z:.2f} < 1.81.",
                why_it_matters="Historically ~80% of firms in this zone faced default, "
                               "restructuring or deep dilution within two years.",
                precedent="DHFL and Jet Airways sat in the distress zone for years "
                          "before defaulting.",
            ))
    else:
        res.notes.append("Altman Z-Score: insufficient inputs (banks/financials are "
                         "also out of model scope).")

    # ── Piotroski F-Score ──────────────────────────────────────────────────────
    f = 0
    checks = 0

    def check(cond: Optional[bool]) -> None:
        nonlocal f, checks
        if cond is None:
            return
        checks += 1
        f += int(cond)

    roa0 = _ratio(val(ni), val(ta))
    roa1 = _ratio(val(ni, 1), val(ta, 1))
    check(roa0 is not None and roa0 > 0)                              # 1 ROA > 0
    check(None if val(cfo) is None else val(cfo) > 0)                 # 2 CFO > 0
    check(None if None in (roa0, roa1) else roa0 > roa1)              # 3 ΔROA
    check(None if None in (val(cfo), val(ni)) else val(cfo) > val(ni))  # 4 accruals
    lev0, lev1 = _ratio(val(ltd), val(ta)), _ratio(val(ltd, 1), val(ta, 1))
    check(None if None in (lev0, lev1) else lev0 <= lev1)             # 5 leverage ↓
    cr0, cr1 = _ratio(val(ca), val(cl)), _ratio(val(ca, 1), val(cl, 1))
    check(None if None in (cr0, cr1) else cr0 > cr1)                  # 6 liquidity ↑
    sh_g = _ratio(val(shares), val(shares, 1))
    check(None if sh_g is None else sh_g <= 1.0)                      # 7 no dilution
    check(None if None in (gm0, gm1) else gm0 > gm1)                  # 8 margin ↑
    at0, at1 = _ratio(val(rev), val(ta)), _ratio(val(rev, 1), val(ta, 1))
    check(None if None in (at0, at1) else at0 > at1)                  # 9 turnover ↑

    if checks >= 5:
        s_f = f / checks * 100
        scores.append(s_f)
        res.metrics.append(Metric(
            name="Piotroski F-Score", value=float(f), display=f"{f}/{checks}",
            what="9-point checklist of profitability, leverage and efficiency trends.",
            why="Separates improving businesses from deteriorating ones; low scorers "
                "underperform high scorers by ~7.5%/yr in the original study.",
            good="7–9 strong; 0–3 weak.",
            implication="Fundamental momentum is positive." if f / checks >= 0.66
            else "Fundamentals are stagnant or deteriorating.",
            score=s_f,
        ))
        if f / checks <= 0.34:
            res.flags.append(RedFlag(
                title="Weak Piotroski F-Score",
                severity=Severity.MEDIUM, period=period,
                evidence=f"F-Score {f}/{checks}.",
                why_it_matters="Most of the firm's fundamental trends point the wrong "
                               "way simultaneously.",
                precedent="Low-F value traps (e.g. legacy telecoms) kept getting "
                          "cheaper for years.",
            ))

    res.score = sum(scores) / len(scores) if scores else None
    res.extras.update({"m_score": m_score, "z_score": z, "f_score": f, "f_checks": checks})
    return res
