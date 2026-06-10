"""
Layer 7 — Alternative Data Intelligence.

Free-source proxies: headcount, analyst recommendation momentum, short interest,
and news-flow sentiment via the platform's existing RSS pipeline. Converted into
a single Alternative Momentum Score; unavailable signals are disclosed.
"""

from __future__ import annotations

from forensics.models import FundamentalData, LayerResult, Metric


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Alternative Data Intelligence", score=None)
    info = d.info
    scores = []

    emp = info.get("fullTimeEmployees")
    if emp:
        res.metrics.append(Metric(
            name="Headcount", value=float(emp), display=f"{emp:,}",
            what="Reported full-time employees.",
            why="Hiring leads revenue by 2–4 quarters; freezes and silent layoffs "
                "show up here before guidance cuts.",
            good="Growing in line with revenue (trend not available from this source).",
            implication="Single snapshot only — track QoQ via LinkedIn/filings for "
                        "the momentum signal.",
        ))

    rec = info.get("recommendationMean")
    if rec is not None:
        s = max(0.0, min(100.0, (5 - rec) / 4 * 100))
        scores.append(s)
        res.metrics.append(Metric(
            name="Analyst Consensus", value=float(rec),
            display=f"{rec:.1f} (1=Strong Buy, 5=Sell), "
                    f"{info.get('numberOfAnalystOpinions', '?')} analysts",
            what="Mean sell-side recommendation.",
            why="Not for the rating itself — but rapid downgrades cluster around "
                "deteriorating channel checks the street sees first.",
            good="Below 2.5 with stable or improving trend.",
            implication="Street is constructive." if rec < 2.5
            else "Sell-side conviction is weak.",
            score=s,
        ))

    short = info.get("shortPercentOfFloat")
    if short is not None:
        s = max(0.0, min(100.0, 100 - short * 500))
        scores.append(s)
        res.metrics.append(Metric(
            name="Short Interest", value=float(short), display=f"{short:.1%} of float",
            what="Percentage of tradable shares sold short.",
            why="Shorts are the market's forensic accountants; double-digit short "
                "interest means professionals are betting on a specific thesis "
                "against the company.",
            good="Below ~3% of float.",
            implication="No meaningful short thesis." if short < 0.05
            else "A significant short bet exists — find and read the thesis.",
            score=s,
        ))

    tgt, px = info.get("targetMeanPrice"), info.get("currentPrice")
    if tgt and px:
        upside = tgt / px - 1
        res.metrics.append(Metric(
            name="Consensus Target Upside", value=upside, display=f"{upside:+.0%}",
            what="Mean analyst price target versus current price.",
            why="Large implied downside means the street already expects de-rating.",
            good="Positive, but treat as sentiment not valuation.",
            implication="Street sees upside." if upside > 0 else "Street targets sit "
            "below the current price.",
        ))

    res.notes.append("Web traffic, app usage, search trends and social sentiment "
                     "need external feeds (Similarweb/Google Trends) — not wired in. "
                     "News sentiment can reuse the existing news/ RSS pipeline.")
    res.score = sum(scores) / len(scores) if scores else None
    return res
