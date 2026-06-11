"""
Layer 1 — Financial Statement Forensics.

Earnings quality, balance sheet stress, and cash flow integrity checks of the
kind run by forensic accountants and activist short sellers.
"""

from __future__ import annotations

from typing import List, Optional

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
    stmt_row, val, pct_change,
)

# Statement line-item label variants (yfinance naming)
REVENUE = ["Total Revenue", "Operating Revenue"]
NET_INCOME = ["Net Income", "Net Income Common Stockholders",
              "Net Income Continuous Operations"]
GROSS_PROFIT = ["Gross Profit"]
OPERATING_INCOME = ["Operating Income", "EBIT"]
CFO = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
FCF = ["Free Cash Flow"]
CAPEX = ["Capital Expenditure"]
TOTAL_ASSETS = ["Total Assets"]
CURRENT_ASSETS = ["Current Assets", "Total Current Assets"]
CURRENT_LIABILITIES = ["Current Liabilities", "Total Current Liabilities"]
RECEIVABLES = ["Accounts Receivable", "Receivables", "Net Receivables"]
INVENTORY = ["Inventory"]
GOODWILL = ["Goodwill"]
INTANGIBLES = ["Goodwill And Other Intangible Assets", "Other Intangible Assets"]
TOTAL_DEBT = ["Total Debt"]
EQUITY = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]
SHARES = ["Ordinary Shares Number", "Share Issued"]
DEBT_ISSUED = ["Issuance Of Debt", "Long Term Debt Issuance"]


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def _fmt_ratio(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}x"


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Financial Statement Forensics", score=None)
    inc, bal, cf = d.income, d.balance, d.cashflow

    if inc.empty or bal.empty or cf.empty:
        res.notes.append("Financial statements unavailable — layer skipped.")
        return res

    rev = stmt_row(inc, REVENUE)
    ni = stmt_row(inc, NET_INCOME)
    cfo = stmt_row(cf, CFO)
    fcf = stmt_row(cf, FCF)
    assets = stmt_row(bal, TOTAL_ASSETS)
    recv = stmt_row(bal, RECEIVABLES)
    inv = stmt_row(bal, INVENTORY)
    gw = stmt_row(bal, GOODWILL)
    intang = stmt_row(bal, INTANGIBLES)
    debt = stmt_row(bal, TOTAL_DEBT)
    equity = stmt_row(bal, EQUITY)
    shares = stmt_row(bal, SHARES)
    period = str(d.years()[0].year) if d.years() else ""

    scores: List[float] = []

    # ── Earnings quality: accrual ratio ───────────────────────────────────────
    ni0, cfo0, assets0 = val(ni), val(cfo), val(assets)
    if None not in (ni0, cfo0, assets0) and assets0:
        accrual = (ni0 - cfo0) / assets0
        s = max(0.0, min(100.0, 50 - accrual * 500))  # 0 accruals → 50; negative better
        scores.append(s)
        res.metrics.append(Metric(
            name="Accrual Ratio", value=accrual, display=_fmt_pct(accrual),
            what="(Net income − operating cash flow) ÷ total assets.",
            why="High accruals mean profits are booked on paper, not collected in cash — "
                "the classic precursor to earnings manipulation.",
            good="Below 0% (cash exceeds reported profit). Above +10% is aggressive.",
            implication="Earnings are cash-backed." if accrual <= 0
            else "A meaningful share of reported profit has not converted to cash.",
            score=s,
        ))
        if accrual > 0.10:
            res.flags.append(RedFlag(
                title="High accruals — earnings not backed by cash",
                severity=Severity.HIGH, period=period,
                evidence=f"Accrual ratio {_fmt_pct(accrual)} "
                         f"(NI {ni0:,.0f} vs CFO {cfo0:,.0f}).",
                why_it_matters="Sloan (1996) showed high-accrual firms systematically "
                               "underperform; accruals reverse into future earnings misses.",
                precedent="Enron and Satyam both showed years of profits with weak "
                          "operating cash flow before collapse.",
                evidence_rows=[
                    ("Net Income", f"{ni0:,.0f}"),
                    ("Operating Cash Flow", f"{cfo0:,.0f}"),
                    ("NI − CFO", f"{ni0 - cfo0:,.0f}"),
                    ("Total Assets", f"{assets0:,.0f}"),
                    ("Accrual Ratio (NI−CFO)/Assets", _fmt_pct(accrual) + "  (flag >10%)"),
                ],
                sources=["Income Statement", "Cash Flow Statement", "Balance Sheet"],
            ))

    # ── Cash conversion: CFO / Net Income ─────────────────────────────────────
    if None not in (ni0, cfo0) and ni0 and ni0 > 0:
        conv = cfo0 / ni0
        s = max(0.0, min(100.0, conv * 70))
        scores.append(s)
        res.metrics.append(Metric(
            name="Cash Conversion (CFO/NI)", value=conv, display=_fmt_ratio(conv),
            what="Operating cash flow divided by net income.",
            why="Profit is an opinion; cash is a fact. Persistent gaps signal aggressive "
                "revenue recognition or channel stuffing.",
            good="Above 1.0x sustained over multiple years.",
            implication="Strong: every rupee of profit arrives as cash." if conv >= 1
            else "Reported profit is running ahead of cash collection.",
            score=s,
        ))
        if conv < 0.5:
            res.flags.append(RedFlag(
                title="Weak cash conversion",
                severity=Severity.HIGH, period=period,
                evidence=f"CFO is only {conv:.0%} of net income.",
                why_it_matters="Less than half of profit converts to cash — earnings "
                               "quality is poor and may be unsustainable.",
                precedent="Manpasand Beverages reported strong profits with weak cash "
                          "conversion before its auditor resigned (2018).",
                evidence_rows=[
                    ("Net Income", f"{ni0:,.0f}"),
                    ("Operating Cash Flow", f"{cfo0:,.0f}"),
                    ("CFO / NI ratio", f"{conv:.2f}x  (threshold 0.50x)"),
                ],
                sources=["Income Statement", "Cash Flow Statement"],
            ))

    # ── Receivables vs revenue growth (DSO-style check) ───────────────────────
    rev_g = pct_change(val(rev), val(rev, 1))
    recv_g = pct_change(val(recv), val(recv, 1))
    if rev_g is not None and recv_g is not None:
        gap = recv_g - rev_g
        s = max(0.0, min(100.0, 75 - gap * 150))
        scores.append(s)
        res.metrics.append(Metric(
            name="Receivables vs Revenue Growth", value=gap,
            display=f"recv {_fmt_pct(recv_g)} vs rev {_fmt_pct(rev_g)}",
            what="YoY growth of accounts receivable compared to revenue growth.",
            why="Receivables growing much faster than sales suggests revenue is being "
                "booked before customers pay — or to customers who never will.",
            good="Receivables growth at or below revenue growth.",
            implication="Collections keep pace with sales." if gap <= 0.05
            else "Receivables are outpacing sales — possible aggressive recognition.",
            score=s,
        ))
        if gap > 0.25 and recv_g > 0.20:
            res.flags.append(RedFlag(
                title="Receivables growing unusually fast",
                severity=Severity.HIGH, period=period,
                evidence=f"Receivables {_fmt_pct(recv_g)} vs revenue {_fmt_pct(rev_g)} YoY.",
                why_it_matters="A widening receivables gap is one of the most reliable "
                               "single predictors of restatements and SEC enforcement.",
                precedent="Sunbeam (1998) and Ricoh India (2016) both stuffed channels, "
                          "showing receivables spikes before restating revenue.",
                evidence_rows=[
                    ("Revenue (current year)", f"{val(rev):,.0f}"),
                    ("Revenue (prior year)", f"{val(rev, 1):,.0f}"),
                    ("Revenue growth YoY", _fmt_pct(rev_g)),
                    ("Receivables (current year)", f"{val(recv):,.0f}"),
                    ("Receivables (prior year)", f"{val(recv, 1):,.0f}"),
                    ("Receivables growth YoY", _fmt_pct(recv_g)),
                    ("Gap (recv − rev growth)", _fmt_pct(gap)),
                ],
                sources=["Balance Sheet", "Income Statement"],
            ))

    # ── Inventory build-up ─────────────────────────────────────────────────────
    inv_g = pct_change(val(inv), val(inv, 1))
    if rev_g is not None and inv_g is not None:
        gap = inv_g - rev_g
        if gap > 0.25 and inv_g > 0.20:
            res.flags.append(RedFlag(
                title="Inventory exploding relative to sales",
                severity=Severity.MEDIUM, period=period,
                evidence=f"Inventory {_fmt_pct(inv_g)} vs revenue {_fmt_pct(rev_g)} YoY.",
                why_it_matters="Unsold stock piling up forecasts margin-destroying "
                               "discounts or write-downs, and can hide cost capitalization.",
                precedent="Under Armour's 2017 inventory glut preceded its first revenue "
                          "decline and an SEC accounting probe.",
            ))
        res.metrics.append(Metric(
            name="Inventory vs Revenue Growth", value=gap,
            display=f"inv {_fmt_pct(inv_g)} vs rev {_fmt_pct(rev_g)}",
            what="YoY inventory growth compared to revenue growth.",
            why="Inventory rising faster than sales means demand is weakening or costs "
                "are being parked on the balance sheet.",
            good="Inventory growth at or below revenue growth.",
            implication="Inventory discipline intact." if gap <= 0.10
            else "Stock is building faster than it sells.",
        ))

    # ── Goodwill / intangibles concentration ─────────────────────────────────
    gw0 = val(gw) or val(intang)
    if gw0 is not None and assets0:
        ratio = gw0 / assets0
        res.metrics.append(Metric(
            name="Goodwill & Intangibles / Assets", value=ratio, display=f"{ratio:.0%}",
            what="Share of total assets that is goodwill or other intangibles.",
            why="Goodwill is the residue of acquisitions; a heavy concentration means "
                "book value depends on deals continuing to perform.",
            good="Below ~20% of assets. Above 40% carries material write-down risk.",
            implication="Asset base is largely tangible." if ratio < 0.2
            else "Significant write-down risk if acquired businesses underperform.",
        ))
        if ratio > 0.4:
            res.flags.append(RedFlag(
                title="Goodwill concentration — write-down risk",
                severity=Severity.MEDIUM, period=period,
                evidence=f"Goodwill+intangibles are {ratio:.0%} of total assets.",
                why_it_matters="One impairment can wipe out years of reported earnings "
                               "and breach debt covenants tied to net worth.",
                precedent="Kraft Heinz wrote down $15.4B of goodwill in 2019, cutting "
                          "the stock ~27% in a day.",
            ))

    # ── Cash flow integrity: FCF & financing dependence ───────────────────────
    fcf_vals = [val(fcf, i) for i in range(min(3, len(cf.columns)))]
    fcf_known = [v for v in fcf_vals if v is not None]
    if fcf_known:
        neg_years = sum(1 for v in fcf_known if v < 0)
        s = max(0.0, 100 - neg_years * 35)
        scores.append(s)
        res.metrics.append(Metric(
            name="Free Cash Flow (3y)", value=float(neg_years),
            display=f"{len(fcf_known) - neg_years}/{len(fcf_known)} years positive",
            what="Operating cash flow minus capital expenditure, over the last 3 years.",
            why="FCF is the cash actually available to owners. Persistent negative FCF "
                "means growth is funded by lenders and new shareholders, not customers.",
            good="Consistently positive and growing.",
            implication="Self-funding business." if neg_years == 0
            else f"Negative FCF in {neg_years} of last {len(fcf_known)} years — "
                 "dependent on external financing.",
            score=s,
        ))
        if neg_years >= 2:
            res.flags.append(RedFlag(
                title="Persistent negative free cash flow",
                severity=Severity.HIGH, period=period,
                evidence=f"FCF negative in {neg_years} of last {len(fcf_known)} fiscal years.",
                why_it_matters="The business consumes cash; any tightening of credit or "
                               "equity markets becomes an existential threat.",
                precedent="WeWork and most 2021-vintage cash-burners saw valuations "
                          "collapse when financing windows closed.",
            ))

    # ── Debt trajectory ────────────────────────────────────────────────────────
    debt_g = pct_change(val(debt), val(debt, 1))
    de = None
    if val(debt) is not None and val(equity):
        de = val(debt) / val(equity)
        res.metrics.append(Metric(
            name="Debt / Equity", value=de, display=_fmt_ratio(de),
            what="Total debt divided by shareholders' equity.",
            why="Leverage amplifies both returns and distress; rapidly rising leverage "
                "often funds the gap left by weak operating cash flow.",
            good="Below 1.0x for most sectors (banks/NBFCs excluded).",
            implication="Conservative balance sheet." if de < 1
            else "Leverage is elevated — equity is junior to a large debt stack.",
        ))
    if debt_g is not None and debt_g > 0.40 and (de or 0) > 0.5:
        res.flags.append(RedFlag(
            title="Debt increasing rapidly",
            severity=Severity.HIGH, period=period,
            evidence=f"Total debt up {_fmt_pct(debt_g)} YoY; D/E {_fmt_ratio(de)}.",
            why_it_matters="Debt-fueled growth is the most common path from 'growth "
                           "story' to restructuring.",
            precedent="IL&FS and DHFL both grew the balance sheet on short-term debt "
                      "before defaulting in 2018-19.",
        ))

    # ── Share dilution ─────────────────────────────────────────────────────────
    sh_g = pct_change(val(shares), val(shares, 1))
    if sh_g is not None and sh_g > 0.05:
        res.flags.append(RedFlag(
            title="Share dilution",
            severity=Severity.MEDIUM, period=period,
            evidence=f"Shares outstanding up {_fmt_pct(sh_g)} YoY.",
            why_it_matters="Each new share dilutes existing owners; serial issuers "
                           "transfer value from shareholders to the company's cash needs.",
            precedent="Chronic diluters (e.g. Yes Bank post-2018 rescues) rarely "
                      "recover per-share value even when the business survives.",
        ))

    # ── Margin deterioration ──────────────────────────────────────────────────
    gp = stmt_row(inc, GROSS_PROFIT)
    gm0 = (val(gp) / val(rev)) if (val(gp) is not None and val(rev)) else None
    gm1 = (val(gp, 1) / val(rev, 1)) if (val(gp, 1) is not None and val(rev, 1)) else None
    if gm0 is not None and gm1 is not None:
        delta = gm0 - gm1
        res.metrics.append(Metric(
            name="Gross Margin Trend", value=delta,
            display=f"{gm1:.1%} → {gm0:.1%}",
            what="Change in gross margin versus the prior fiscal year.",
            why="Gross margin is the hardest line to manipulate; sustained erosion "
                "signals pricing power loss or input cost stress.",
            good="Stable or expanding.",
            implication="Pricing power holding." if delta >= -0.01
            else "Margins are compressing.",
        ))
        if delta < -0.03:
            res.flags.append(RedFlag(
                title="Margin deterioration",
                severity=Severity.MEDIUM, period=period,
                evidence=f"Gross margin fell {abs(delta):.1%} pts YoY ({gm1:.1%} → {gm0:.1%}).",
                why_it_matters="Margin erosion compounds: it pressures earnings, then "
                               "covenants, then management's incentive to massage numbers.",
                precedent="Sustained gross-margin slides preceded blowups at GE Capital "
                          "era GE and at Vakrangee.",
            ))

    res.score = sum(scores) / len(scores) if scores else None
    return res
