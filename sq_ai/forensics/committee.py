"""
Layer 12 — AI Investment Committee.

Five virtual analysts, each with a single mandate, vote Buy / Hold / Sell from
the already-computed layer outputs. Deterministic rules — every vote is
reproducible and comes with the rationale that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from forensics.models import (
    FundamentalData, LayerResult, Severity, stmt_row, val, pct_change,
)
from forensics.statement_forensics import REVENUE, GROSS_PROFIT


@dataclass
class Vote:
    analyst: str
    mandate: str
    vote: str           # Buy | Hold | Sell
    rationale: str


@dataclass
class CommitteeResult:
    votes: List[Vote]
    consensus: str      # Buy | Hold | Sell
    tally: Dict[str, int]


def _vote_from_score(s: Optional[float], buy_at: float = 65,
                     sell_at: float = 45) -> str:
    if s is None:
        return "Hold"
    return "Buy" if s >= buy_at else ("Sell" if s < sell_at else "Hold")


def convene(d: FundamentalData, layers: Dict[str, LayerResult],
            flags) -> CommitteeResult:
    votes: List[Vote] = []

    # ── Quant Fund — data only ────────────────────────────────────────────────
    q = layers.get("quant")
    micro = layers.get("microstructure")
    q_score = q.score if q else None
    ann_ret = (q.extras.get("ann_return") if q else None) or 0.0
    accum = micro.extras.get("accumulation_score") if micro else None
    composite_q = None
    parts = [s for s in (q_score, accum) if s is not None]
    if parts:
        composite_q = sum(parts) / len(parts)
        if ann_ret < 0:
            composite_q -= 10
    v = _vote_from_score(composite_q)
    votes.append(Vote(
        "Quant Fund", "Price data, volatility, flow — no stories.",
        v, f"Stability {q_score:.0f}" if q_score is not None else "No price data"))
    if composite_q is not None:
        votes[-1].rationale = (
            f"Quant stability {q_score:.0f}/100"
            + (f", accumulation {accum:.0f}/100" if accum is not None else "")
            + f", trailing return {ann_ret:+.0%}. Signal: {v.lower()}.")

    # ── Value Investor — valuation only ───────────────────────────────────────
    val_layer = layers.get("valuation")
    label = (val_layer.extras.get("valuation_label")
             if val_layer else None) or "Insufficient data"
    vv = {"Cheap": "Buy", "Fairly Valued": "Hold",
          "Expensive": "Sell"}.get(label, "Hold")
    votes.append(Vote(
        "Value Investor", "Price paid versus value received.",
        vv, f"Stock screens {label.lower()} against its own 5-year multiple "
            f"history."))

    # ── Short Seller — fraud and forensics only ───────────────────────────────
    n_crit = sum(1 for f in flags if f.severity == Severity.CRITICAL)
    n_high = sum(1 for f in flags if f.severity == Severity.HIGH)
    fraud = layers.get("fraud")
    m = fraud.extras.get("m_score") if fraud else None
    if n_crit >= 1 or n_high >= 3 or (m is not None and m > -1.78):
        sv, sr = "Sell", (f"{n_crit} critical / {n_high} high-severity flags"
                          + (f"; M-Score {m:.2f} in manipulator zone"
                             if m is not None and m > -1.78 else "")
                          + ". This is the profile I short.")
    elif n_high >= 1:
        sv, sr = "Hold", (f"{n_high} high-severity flag(s) — watching, "
                          "not yet a short.")
    else:
        sv, sr = "Buy", "Nothing to short here — forensics are clean."
    votes.append(Vote("Short Seller",
                      "Find the fraud before the market does.", sv, sr))

    # ── Credit Analyst — solvency only ────────────────────────────────────────
    z = fraud.extras.get("z_score") if fraud else None
    if z is None:
        cv, cr = "Hold", "Z-Score not computable (insufficient inputs or a financial)."
    elif z < 1.81:
        cv, cr = "Sell", f"Z-Score {z:.2f}: distress zone. Equity is an option, not an asset."
    elif z < 2.99:
        cv, cr = "Hold", f"Z-Score {z:.2f}: grey zone — solvency adequate, not comfortable."
    else:
        cv, cr = "Buy", f"Z-Score {z:.2f}: safe zone. Balance sheet survives a downturn."
    votes.append(Vote("Credit Analyst",
                      "Will the balance sheet survive stress?", cv, cr))

    # ── Growth Investor — expansion only ──────────────────────────────────────
    rev = stmt_row(d.income, REVENUE)
    gp = stmt_row(d.income, GROSS_PROFIT)
    rev_g = pct_change(val(rev), val(rev, 1))
    gm0 = (val(gp) / val(rev)) if (val(gp) is not None and val(rev)) else None
    gm1 = (val(gp, 1) / val(rev, 1)) if (val(gp, 1) is not None and val(rev, 1)) else None
    margin_ok = (gm0 is not None and gm1 is not None and gm0 >= gm1 - 0.01)
    if rev_g is None:
        gv, gr = "Hold", "Revenue trend unavailable."
    elif rev_g > 0.15 and margin_ok:
        gv, gr = "Buy", f"Revenue {rev_g:+.0%} YoY with margins holding — growth is real."
    elif rev_g > 0.15:
        gv, gr = "Hold", f"Revenue {rev_g:+.0%} YoY but margins eroding — growth is being bought."
    elif rev_g > 0.05:
        gv, gr = "Hold", f"Revenue {rev_g:+.0%} YoY — pedestrian growth."
    else:
        gv, gr = "Sell", f"Revenue {rev_g:+.0%} YoY — no growth story to own."
    votes.append(Vote("Growth Investor",
                      "Is the business getting bigger, profitably?", gv, gr))

    tally = {"Buy": 0, "Hold": 0, "Sell": 0}
    for vt in votes:
        tally[vt.vote] += 1
    if tally["Sell"] > tally["Buy"] and tally["Sell"] >= 2:
        consensus = "Sell"
    elif tally["Buy"] > tally["Sell"] and tally["Buy"] >= 3:
        consensus = "Buy"
    else:
        consensus = "Hold"

    return CommitteeResult(votes=votes, consensus=consensus, tally=tally)
