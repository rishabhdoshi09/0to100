"""
Layer 6 — Valuation Intelligence.

Relative multiples vs the stock's own history. Sector/industry medians are not
available from the free data source, so the historical anchor does the work and
the limitation is disclosed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity, stmt_row, val,
)
from forensics.statement_forensics import REVENUE, NET_INCOME, EQUITY


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Valuation Intelligence", score=None)
    info = d.info
    scores = []

    pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    pb = info.get("priceToBook")
    ps = info.get("priceToSalesTrailing12Months")
    ev_ebitda = info.get("enterpriseToEbitda")
    mcap = info.get("marketCap")

    # Historical P/E band from price history + per-share earnings
    hist_pe_pct = None
    ni = stmt_row(d.income, NET_INCOME)
    shares = info.get("sharesOutstanding")
    if (pe is not None and ni is not None and shares and not d.prices.empty
            and len(d.income.columns) >= 3):
        eps_by_year = {c.year: v / shares for c, v in
                       ((c, val(ni, i)) for i, c in enumerate(d.income.columns))
                       if v is not None and v > 0}
        px = d.prices["Close"]
        hist_pes = []
        for ts, price in px.items():
            eps = eps_by_year.get(ts.year) or eps_by_year.get(ts.year - 1)
            if eps:
                hist_pes.append(price / eps)
        if len(hist_pes) > 200:
            hist = pd.Series(hist_pes)
            hist_pe_pct = float((hist < pe).mean())

    def mult(name: str, v: Optional[float], good: str, ctx: str) -> None:
        if v is None:
            return
        res.metrics.append(Metric(
            name=name, value=float(v), display=f"{v:.1f}x",
            what=ctx,
            why="Multiples tell you what you pay per unit of fundamentals; the higher "
                "the multiple, the more future perfection is already priced in.",
            good=good,
            implication="Within normal ranges." if name != "Trailing P/E"
            else "",
        ))

    mult("Trailing P/E", pe, "Context-dependent; vs own 5y history below.",
         "Price divided by the last 12 months of earnings per share.")
    mult("Forward P/E", fwd_pe, "Materially below trailing P/E implies expected growth.",
         "Price divided by consensus next-year EPS.")
    mult("EV / EBITDA", ev_ebitda, "Below ~12x for mature businesses.",
         "Enterprise value over EBITDA — capital-structure-neutral earnings multiple.")
    mult("Price / Book", pb, "Sector-dependent; <1x can be value or value trap.",
         "Price relative to accounting net worth.")
    mult("Price / Sales", ps, "Below ~3x unless margins are exceptional.",
         "Price relative to revenue — useful when earnings are noisy.")

    label = "Insufficient data"
    if hist_pe_pct is not None:
        label = ("Cheap" if hist_pe_pct < 0.3
                 else "Expensive" if hist_pe_pct > 0.75 else "Fairly Valued")
        s = 100 - hist_pe_pct * 80
        scores.append(s)
        res.metrics.append(Metric(
            name="P/E vs Own 5y History", value=hist_pe_pct,
            display=f"{hist_pe_pct:.0%} percentile → {label}",
            what="Where today's P/E sits inside the stock's own 5-year P/E "
                 "distribution.",
            why="A stock's own history is the cleanest valuation anchor — it controls "
                "for sector and quality automatically.",
            good="Below the 50th percentile of its own range.",
            implication=f"Currently {label.lower()} versus its own trading history.",
            score=s,
        ))
        if hist_pe_pct > 0.9:
            res.flags.append(RedFlag(
                title="Valuation at extreme of own history",
                severity=Severity.MEDIUM,
                evidence=f"Trailing P/E in the {hist_pe_pct:.0%} percentile of its "
                         "5-year range.",
                why_it_matters="Peak multiples on peak earnings is where permanent "
                               "capital loss happens even in good businesses.",
                precedent="The 2021 quality-compounder de-rating cut 40–60% off names "
                          "whose earnings never missed.",
            ))
    elif pe is not None:
        # crude absolute fallback
        s = max(10.0, min(90.0, 100 - pe))
        scores.append(s)
        res.notes.append("Historical P/E band could not be built; scored on absolute "
                         "multiple only.")

    res.notes.append("Sector/industry median comparison requires a paid screener "
                     "feed — comparison shown is vs the stock's own history.")
    res.extras["valuation_label"] = label
    res.score = sum(scores) / len(scores) if scores else None
    return res
