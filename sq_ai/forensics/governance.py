"""
Layer 4 — Corporate Governance Engine.
Layer 5 — Insider & Smart Money Analysis.

yfinance exposes limited governance data (officers, audit risk scores for US
names, holders, insider transactions). Everything available is scored;
everything unavailable is explicitly noted, never silently assumed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
)


def analyze_governance(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Corporate Governance Engine", score=None)
    info = d.info
    scores = []

    officers = info.get("companyOfficers") or []
    ceo = next((o for o in officers
                if "ceo" in (o.get("title") or "").lower()
                or "managing director" in (o.get("title") or "").lower()), None)
    if ceo:
        res.metrics.append(Metric(
            name="Chief Executive", value=None,
            display=f"{ceo.get('name', '?')} ({ceo.get('title', '')})",
            what="The current CEO / Managing Director on record.",
            why="Leadership continuity and tenure shape capital allocation; abrupt "
                "CEO/CFO churn is among the strongest fraud-adjacent signals.",
            good="Long, stable tenure with clean exits of predecessors.",
            implication="Tenure history is not available from this data source — "
                        "verify CFO/auditor changes in the annual report.",
        ))

    # ISS-style risk scores (US coverage; 1 = best decile, 10 = worst)
    iss = {k: info.get(k) for k in
           ("auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk",
            "overallRisk")}
    labels = {"auditRisk": "Audit Risk", "boardRisk": "Board Risk",
              "compensationRisk": "Compensation Risk",
              "shareHolderRightsRisk": "Shareholder Rights Risk",
              "overallRisk": "Overall Governance Risk"}
    for key, v in iss.items():
        if v is None:
            continue
        s = (10 - v) / 9 * 100
        scores.append(s)
        res.metrics.append(Metric(
            name=labels[key], value=float(v), display=f"{v}/10 (1=best)",
            what="ISS governance risk decile (1 = lowest risk, 10 = highest).",
            why="Aggregates board independence, audit quality, pay structure and "
                "shareholder rights — the items activists attack first.",
            good="1–3. Scores of 8–10 are bottom-decile governance.",
            implication="Governance screens clean." if v <= 4
            else "Governance is in the weak tail of coverage.",
            score=s,
        ))
        if key == "auditRisk" and v >= 8:
            res.flags.append(RedFlag(
                title="Elevated audit risk score",
                severity=Severity.HIGH,
                evidence=f"ISS audit risk decile {v}/10.",
                why_it_matters="Weak audit oversight is the soil accounting fraud "
                               "grows in.",
                precedent="Wirecard's audit red flags were public for years before "
                          "the €1.9B hole surfaced.",
            ))

    held_insiders = info.get("heldPercentInsiders")
    if held_insiders is not None:
        res.metrics.append(Metric(
            name="Promoter / Insider Ownership", value=held_insiders,
            display=f"{held_insiders:.1%}",
            what="Share of equity held by insiders/promoters.",
            why="Skin in the game aligns incentives; but extreme concentration can "
                "enable related-party abuse and low float manipulation.",
            good="Roughly 20–60%. Near-zero or above ~75% both warrant scrutiny.",
            implication="Healthy alignment." if 0.15 <= held_insiders <= 0.65
            else "Ownership structure is at an extreme — check pledging and "
                 "related-party transactions.",
        ))
        scores.append(80.0 if 0.15 <= held_insiders <= 0.65 else 45.0)

    if not res.metrics:
        res.notes.append("No governance data exposed for this listing — review "
                         "auditor opinions, RPTs and board composition manually.")
    res.notes.append("Not coverable from this source: auditor changes, qualified "
                     "opinions, related-party transactions, CFO turnover, pledging. "
                     "These require annual-report / exchange-filing review.")
    res.score = sum(scores) / len(scores) if scores else None
    return res


def analyze_smart_money(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Insider & Smart Money", score=None)
    info = d.info
    scores = []

    inst = info.get("heldPercentInstitutions")
    if inst is not None:
        s = min(100.0, inst * 200)
        scores.append(s)
        res.metrics.append(Metric(
            name="Institutional Ownership", value=inst, display=f"{inst:.1%}",
            what="Share of equity held by institutions (FII/DII/funds).",
            why="Institutions do the diligence retail cannot; near-zero institutional "
                "presence in a liquid stock is itself a warning.",
            good="Above ~15% for a liquid mid/large cap.",
            implication="Smart money is present." if inst > 0.15
            else "Institutions are largely absent — ask why.",
            score=s,
        ))

    classification = "Neutral"
    txns = d.insider_transactions
    if isinstance(txns, pd.DataFrame) and not txns.empty:
        recent = txns
        if "Start Date" in txns.columns:
            cutoff = datetime.now() - timedelta(days=365)
            dates = pd.to_datetime(txns["Start Date"], errors="coerce")
            recent = txns[dates > cutoff]
        text_col = next((c for c in ("Text", "Transaction") if c in recent.columns), None)
        buys = sells = 0
        if text_col is not None:
            t = recent[text_col].astype(str).str.lower()
            buys = int(t.str.contains("purchase|buy").sum())
            sells = int(t.str.contains("sale|sell").sum())
        if buys + sells > 0:
            ratio = buys / (buys + sells)
            classification = ("Bullish" if ratio > 0.6
                              else "Warning" if ratio < 0.25 else "Neutral")
            scores.append(ratio * 100)
            res.metrics.append(Metric(
                name="Insider Transactions (12m)", value=ratio,
                display=f"{buys} buys / {sells} sells → {classification}",
                what="Count of insider buy vs sell filings in the last year.",
                why="Insiders sell for many reasons but buy for only one. Clustered "
                    "selling before bad news is a recurring pattern in blowups.",
                good="Net buying, or at least no clustered selling.",
                implication=f"Insider activity classifies as {classification}.",
                score=ratio * 100,
            ))
            if classification == "Warning":
                res.flags.append(RedFlag(
                    title="Heavy insider selling",
                    severity=Severity.MEDIUM,
                    evidence=f"{sells} insider sales vs {buys} purchases in 12 months.",
                    why_it_matters="Management is reducing exposure to the equity it "
                                   "knows best.",
                    precedent="Luckin Coffee insiders pledged/monetized stakes ahead of "
                              "the 2020 fabricated-sales disclosure.",
                ))
    else:
        res.notes.append("Insider transaction feed unavailable for this listing "
                         "(common for NSE symbols) — check exchange disclosures.")

    res.extras["classification"] = classification
    res.score = sum(scores) / len(scores) if scores else None
    return res
