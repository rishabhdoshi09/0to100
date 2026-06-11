"""
Layer 9 — Capital Allocation Intelligence.

Most screeners show ROIC. This layer goes further:

  Incremental ROIC — how much additional NOPAT per additional invested capital?
  That's where compounders separate from empire builders.

Also covers: ROIC trend, ROE quality (DuPont decomposition), empire building
detection, buyback effectiveness, dividend sustainability, and a composite
Shareholder Friendliness Score — combined into a Capital Allocation Grade.
"""

from __future__ import annotations

from typing import List, Optional

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
    stmt_row, val, pct_change,
)
from forensics.statement_forensics import (
    REVENUE, NET_INCOME, GROSS_PROFIT, CFO, FCF, TOTAL_ASSETS,
    CURRENT_ASSETS, TOTAL_DEBT, EQUITY, SHARES, GOODWILL, INTANGIBLES,
    CAPEX,
)

TAX_RATE = 0.25          # conservative; used when explicit tax data absent
WACC_HURDLE = 0.10       # 10% — typical emerging-market cost of capital hurdle


DIVIDENDS_PAID = ["Cash Dividends Paid", "Payment Of Dividends",
                  "Common Stock Dividend Paid"]
BUYBACKS = ["Repurchase Of Capital Stock", "Common Stock Repurchase",
            "Purchase Of Business"]
CASH = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
EBIT_ = ["EBIT", "Operating Income"]
TAX = ["Tax Provision", "Income Tax Expense", "Provision For Income Taxes"]


def _nopat(ebit: Optional[float], tax_rate: float = TAX_RATE) -> Optional[float]:
    if ebit is None:
        return None
    return ebit * (1 - tax_rate)


def _invested_capital(bal: "pd.DataFrame", year: int = 0) -> Optional[float]:
    """Invested Capital = Equity + Debt − Cash."""
    eq = val(stmt_row(bal, EQUITY), year)
    debt = val(stmt_row(bal, TOTAL_DEBT), year)
    cash = val(stmt_row(bal, CASH), year)
    if eq is None or debt is None:
        return None
    return eq + debt - (cash or 0)


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Capital Allocation Intelligence", score=None)
    inc, bal, cf = d.income, d.balance, d.cashflow
    if inc.empty or bal.empty or len(inc.columns) < 2:
        res.notes.append("Need two years of statements — layer skipped.")
        return res

    period = str(d.years()[0].year) if d.years() else ""
    scores: List[float] = []

    ebit = stmt_row(inc, EBIT_)
    ni = stmt_row(inc, NET_INCOME)
    rev = stmt_row(inc, REVENUE)
    fcf_ = stmt_row(cf, FCF)
    shares = stmt_row(bal, SHARES)
    gw_raw = stmt_row(bal, GOODWILL)
    gw = gw_raw if gw_raw is not None else stmt_row(bal, INTANGIBLES)

    # ── ROIC ──────────────────────────────────────────────────────────────────
    ic0 = _invested_capital(bal, 0)
    ic1 = _invested_capital(bal, 1)
    nopat0 = _nopat(val(ebit))
    nopat1 = _nopat(val(ebit, 1))

    roic0 = (nopat0 / ic0) if (nopat0 is not None and ic0 and ic0 > 0) else None
    roic1 = (nopat1 / ic1) if (nopat1 is not None and ic1 and ic1 > 0) else None

    if roic0 is not None:
        s_roic = max(0.0, min(100.0, roic0 / 0.25 * 100))
        scores.append(s_roic)
        res.metrics.append(Metric(
            name="ROIC", value=roic0, display=f"{roic0:.1%}",
            what="Net Operating Profit After Tax ÷ Invested Capital.",
            why="ROIC above WACC means the business creates value with every "
                "incremental rupee invested; below WACC it destroys it.",
            good=f"Above {WACC_HURDLE:.0%}. World-class compounders sustain 20–30%+.",
            implication=(f"Value-creating: ROIC {roic0:.1%} above {WACC_HURDLE:.0%} hurdle."
                         if roic0 > WACC_HURDLE
                         else f"Value-destroying: ROIC {roic0:.1%} below "
                              f"{WACC_HURDLE:.0%} cost of capital."),
            score=s_roic,
        ))

    # ── Incremental ROIC — the compounder test ─────────────────────────────────
    inc_roic = None
    if None not in (nopat0, nopat1, ic0, ic1):
        d_nopat = nopat0 - nopat1
        d_ic = ic0 - ic1
        if abs(d_ic) > abs(ic0) * 0.01:      # ignore noise (<1% IC change)
            inc_roic = d_nopat / d_ic if d_ic != 0 else None
    if inc_roic is not None:
        s_inc = max(0.0, min(100.0, inc_roic / 0.30 * 100))
        scores.append(s_inc)
        res.metrics.append(Metric(
            name="Incremental ROIC", value=inc_roic, display=f"{inc_roic:.1%}",
            what="Change in NOPAT ÷ change in Invested Capital year-over-year.",
            why="ROIC tells you the stock; Incremental ROIC tells you the movie. "
                "A business re-investing at 30%+ incrementally will compound at 30%+ "
                "regardless of where it started.",
            good="Consistently above WACC (~10%). "
                 "Asian Paints / Nestlé-type compounders sustain 25–40%+.",
            implication=(
                f"Each additional rupee of capital deployed generated "
                f"{inc_roic:.1%} return — "
                + ("exceptional compounder territory." if inc_roic > 0.25
                   else "healthy reinvestment." if inc_roic > WACC_HURDLE
                   else "value-diluting reinvestment.")),
            score=s_inc,
        ))

    # ── ROIC trend ────────────────────────────────────────────────────────────
    roic_trend = pct_change(roic0, roic1)

    # ── ROE quality — DuPont decomposition ────────────────────────────────────
    eq0, eq1_v = val(stmt_row(bal, EQUITY)), val(stmt_row(bal, EQUITY), 1)
    ni0 = val(ni)
    rev0 = val(rev)
    ta0 = val(stmt_row(inc, ["Total Assets"])) or val(stmt_row(bal, TOTAL_ASSETS))
    roe = (ni0 / eq0) if (ni0 is not None and eq0) else None
    net_margin = (ni0 / rev0) if (ni0 is not None and rev0) else None
    asset_turn = (rev0 / ta0) if (rev0 is not None and ta0) else None
    leverage = (ta0 / eq0) if (ta0 is not None and eq0) else None

    if roe is not None:
        s_roe = max(0.0, min(100.0, roe / 0.20 * 100))
        scores.append(s_roe)
        decomp_parts = []
        if net_margin is not None:
            decomp_parts.append(f"margin {net_margin:.1%}")
        if asset_turn is not None:
            decomp_parts.append(f"× turnover {asset_turn:.2f}x")
        if leverage is not None:
            decomp_parts.append(f"× leverage {leverage:.1f}x")
        res.metrics.append(Metric(
            name="ROE (DuPont)", value=roe,
            display=f"{roe:.1%}"
                    + (f" = {' '.join(decomp_parts)}" if decomp_parts else ""),
            what="Return on Equity decomposed into net margin × asset turnover "
                 "× leverage (DuPont).",
            why="A high ROE built on leverage is fragile; one built on high margins "
                "and turnover is durable. The decomposition exposes which.",
            good="Above 15%, driven by margin and turnover — not leverage.",
            implication=(
                f"ROE {roe:.1%} is " +
                ("leverage-inflated — quality is lower than the headline suggests."
                 if (leverage or 0) > 3 and roe > 0.15
                 else "high quality — margin/turnover-driven.")
                if roe > 0.12 else "Below the earnings-power benchmark."),
            score=s_roe,
        ))
        if (leverage or 0) > 3 and roe > 0.15:
            res.flags.append(RedFlag(
                title="ROE inflated by leverage, not business quality",
                severity=Severity.MEDIUM, period=period,
                evidence=(f"ROE {roe:.1%} but leverage factor {leverage:.1f}x; "
                          f"margin {net_margin:.1%}" if net_margin else
                          f"ROE {roe:.1%} but leverage {leverage:.1f}x."),
                why_it_matters="Leveraged ROE can be high at a firm heading for "
                               "distress; the DuPont breakdown is why analysts look "
                               "beyond the headline number.",
                precedent="DHFL and IL&FS both reported healthy ROEs right up to "
                          "the quarter before default.",
                evidence_rows=[
                    ("ROE", f"{roe:.1%}"),
                    ("Net Margin", f"{net_margin:.1%}" if net_margin else "n/a"),
                    ("Asset Turnover", f"{asset_turn:.2f}x" if asset_turn else "n/a"),
                    ("Leverage (Assets/Equity)", f"{leverage:.1f}x"),
                    ("Rule of thumb: leverage >3x = warning", ">3.0x triggered"),
                ],
                sources=["Income Statement", "Balance Sheet"],
            ))

    # ── Empire Building Detector ───────────────────────────────────────────────
    gw0 = val(gw)
    gw1 = val(gw, 1)
    gw_growth = pct_change(gw0, gw1)
    ta0_bal = val(stmt_row(bal, TOTAL_ASSETS))
    empire_signals = []
    if gw_growth is not None and gw_growth > 0.25:
        empire_signals.append(f"goodwill +{gw_growth:.0%} YoY")
    if roic_trend is not None and roic_trend < -0.10 and roic0 is not None:
        empire_signals.append(f"ROIC falling {roic_trend:+.0%} YoY")
    if inc_roic is not None and inc_roic < WACC_HURDLE:
        empire_signals.append(
            f"incremental ROIC {inc_roic:.1%} below WACC hurdle")

    if len(empire_signals) >= 2:
        res.flags.append(RedFlag(
            title="Empire Building Detector: scale over value",
            severity=Severity.HIGH, period=period,
            evidence="; ".join(empire_signals) + ".",
            why_it_matters="Management may be pursuing scale rather than shareholder "
                           "value — acquisitions are destroying the return profile while "
                           "balance sheet grows.",
            precedent="GE, Tata Steel (Corus), and Vodafone Idea all expanded balance "
                      "sheets while ROIC deteriorated, preceding major write-downs.",
            evidence_rows=[
                ("Goodwill growth YoY", f"{gw_growth:+.0%}" if gw_growth else "n/a"),
                ("Goodwill absolute (current)", f"{gw0:,.0f}" if gw0 else "n/a"),
                ("ROIC current year", f"{roic0:.1%}" if roic0 else "n/a"),
                ("ROIC prior year", f"{roic1:.1%}" if roic1 else "n/a"),
                ("ROIC trend", f"{roic_trend:+.1%}" if roic_trend else "n/a"),
                ("Incremental ROIC", f"{inc_roic:.1%}" if inc_roic else "n/a"),
            ],
            sources=["Balance Sheet", "Income Statement"],
        ))
        res.metrics.append(Metric(
            name="Empire Building Signal", value=float(len(empire_signals)),
            display=f"{len(empire_signals)}/3 signals active",
            what="Combined indicator: rising goodwill + falling ROIC + sub-WACC "
                 "incremental returns.",
            why="Each signal alone is explainable; all three together is the "
                "signature of an acquirer that buys growth instead of earning it.",
            good="0 active signals.",
            implication=f"Management capital allocation is suspect: {'; '.join(empire_signals)}.",
        ))

    # ── Shareholder Friendliness Score ────────────────────────────────────────
    sf_scores: List[float] = []

    sh0, sh1 = val(shares), val(shares, 1)
    if sh0 is not None and sh1 is not None and sh1 > 0:
        dilution = (sh0 - sh1) / sh1
        sf_s = max(0.0, min(100.0, 80 - dilution * 300))
        sf_scores.append(sf_s)
        res.metrics.append(Metric(
            name="Dilution (12m)", value=dilution, display=f"{dilution:+.1%}",
            what="Year-over-year change in shares outstanding.",
            why="Issuing shares silently transfers value from existing shareholders "
                "to new capital needs.",
            good="Flat or declining (buybacks). Every 1% of dilution is a silent "
                 "1% cut to per-share value.",
            implication="Shareholders not being diluted." if dilution <= 0.01
            else f"{dilution:.1%} of shareholder value transferred via dilution.",
            score=sf_s,
        ))

    div_paid = val(stmt_row(cf, DIVIDENDS_PAID))
    fcf0 = val(fcf_)
    if div_paid is not None and fcf0 is not None and fcf0 != 0:
        payout_fcf = abs(div_paid) / abs(fcf0)
        sf_s2 = max(0.0, min(100.0, (1 - max(0, payout_fcf - 0.5)) * 100))
        sf_scores.append(sf_s2)
        res.metrics.append(Metric(
            name="Dividend FCF Payout Ratio", value=payout_fcf,
            display=f"{payout_fcf:.0%} of FCF",
            what="Dividends paid as a fraction of free cash flow.",
            why="Dividends funded from FCF are sustainable; from debt they are a "
                "slow value transfer to exiting shareholders.",
            good="Below 60% of FCF.",
            implication="Dividend is well-covered by cash generation." if payout_fcf < 0.6
            else "Dividend is consuming most/all of free cash flow — sustainability risk.",
            score=sf_s2,
        ))
        if payout_fcf > 1.0:
            res.flags.append(RedFlag(
                title="Dividend exceeds free cash flow",
                severity=Severity.MEDIUM, period=period,
                evidence=f"Dividends {abs(div_paid):,.0f} vs FCF {fcf0:,.0f} "
                         f"({payout_fcf:.0%} payout).",
                why_it_matters="A dividend that can't be funded from operations "
                               "must be funded by debt — a structural wealth "
                               "transfer from the balance sheet.",
                precedent="Carillion paid dividends from debt for years before "
                          "collapsing with £7B of liabilities.",
                evidence_rows=[
                    ("Free Cash Flow", f"{fcf0:,.0f}"),
                    ("Dividends Paid", f"{div_paid:,.0f}"),
                    ("FCF Payout Ratio", f"{payout_fcf:.0%}  (flag >100%)"),
                ],
                sources=["Cash Flow Statement"],
            ))

    buyback = val(stmt_row(cf, BUYBACKS))
    if buyback is not None and buyback < 0:  # cash outflow for repurchases
        buyback_abs = abs(buyback)
        mcap = d.info.get("marketCap") or 0
        if mcap > 0:
            bb_yield = buyback_abs / mcap
            sf_s3 = max(0.0, min(100.0, bb_yield * 2000))
            sf_scores.append(sf_s3)
            res.metrics.append(Metric(
                name="Buyback Yield", value=bb_yield, display=f"{bb_yield:.1%}",
                what="Value of shares repurchased as a fraction of market cap.",
                why="Buybacks at below intrinsic value are the highest-ROI capital "
                    "allocation possible; at above intrinsic value they destroy value.",
                good="2–5%+ with stock trading below historical average multiples.",
                implication=f"Company bought back {bb_yield:.1%} of its own market cap.",
                score=sf_s3,
            ))

    ins_own = d.info.get("heldPercentInsiders")
    if ins_own is not None:
        sf_s4 = max(0.0, min(100.0, ins_own * 120)) if ins_own < 0.65 \
            else max(0.0, 100 - (ins_own - 0.65) * 200)
        sf_scores.append(sf_s4)

    shareholder_friendliness = sum(sf_scores) / len(sf_scores) if sf_scores else None
    if shareholder_friendliness is not None:
        scores.append(shareholder_friendliness)
        res.metrics.append(Metric(
            name="Shareholder Friendliness Score", value=shareholder_friendliness,
            display=f"{shareholder_friendliness:.0f}/100",
            what="Composite of: dilution history, dividend sustainability, buyback "
                 "yield, and insider ownership alignment.",
            why="Capital returned efficiently to shareholders compounds over decades; "
                "chronic diluters and over-distributors slowly consume equity.",
            good="Above 65.",
            implication=(
                "Management is treating shareholders well." if shareholder_friendliness >= 65
                else "Capital return to shareholders needs scrutiny."),
        ))

    # ── Capital Allocation Grade ───────────────────────────────────────────────
    roic_score = scores[0] if scores else 50.0
    empire_penalty = 20 * len(empire_signals) / 3 if empire_signals else 0
    raw = (sum(scores) / len(scores) - empire_penalty) if scores else None

    if raw is None:
        grade = "N/A"
    elif raw >= 85:
        grade = "A+"
    elif raw >= 75:
        grade = "A"
    elif raw >= 65:
        grade = "B+"
    elif raw >= 55:
        grade = "B"
    elif raw >= 45:
        grade = "C"
    elif raw >= 35:
        grade = "D"
    else:
        grade = "F"

    res.extras["grade"] = grade
    res.extras["roic"] = roic0
    res.extras["inc_roic"] = inc_roic
    res.extras["empire_signals"] = empire_signals
    res.extras["shareholder_friendliness"] = shareholder_friendliness
    res.score = max(0.0, min(100.0, raw)) if raw is not None else None
    res.extras["roic_trend"] = roic_trend
    return res
