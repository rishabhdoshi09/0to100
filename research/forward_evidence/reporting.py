"""Separated HISTORICAL / PAPER / LIVE statistics — never silently average sources."""
from __future__ import annotations

from typing import Any

from research.forward_evidence.maturity import classify_maturity, plain_label
from research.forward_evidence.outcome_ledger import load_outcomes
from research.forward_evidence.sources import (
    HISTORICAL_BACKTEST,
    LIMITED_LIVE,
    LIVE,
    PAPER_FORWARD,
)


def _stats(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "expectancy_r": None,
            "net_pnl": 0.0,
            "hit_rate": None,
            "profit_factor": None,
            "plain": "Not enough trades yet.",
        }
    rs = [float(r.get("r_outcome", 0) or 0) for r in rows]
    pnls = [float(r.get("net_pnl", 0) or 0) for r in rows]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = (gp / gl) if gl > 0 else (None if gp <= 0 else float("inf"))
    exp = sum(rs) / len(rs)
    hit = len(wins) / len(rs)
    plain = (
        f"{len(rows)} trades so far; average result about {exp:+.2f}R."
        if len(rows) >= 10
        else f"Only {len(rows)} trades — evidence is still too small to trust."
    )
    return {
        "n": len(rows),
        "expectancy_r": round(exp, 4),
        "net_pnl": round(sum(pnls), 2),
        "hit_rate": round(hit, 4),
        "profit_factor": (None if pf is None or pf == float("inf") else round(pf, 4)),
        "plain": plain,
    }


def policy_report(policy_id: str, *, historical_rows: list[dict] | None = None) -> dict:
    paper = load_outcomes(evidence_source=PAPER_FORWARD, policy_id=policy_id)
    live = load_outcomes(evidence_source=LIVE, policy_id=policy_id)
    limited = load_outcomes(evidence_source=LIMITED_LIVE, policy_id=policy_id)
    hist = historical_rows or load_outcomes(evidence_source=HISTORICAL_BACKTEST, policy_id=policy_id)
    hs, ps, ls = _stats(hist), _stats(paper), _stats(live)
    maturity = classify_maturity(
        historical_n=hs["n"], paper_n=ps["n"], limited_live_n=len(limited), live_n=ls["n"],
        paper_expectancy_r=ps["expectancy_r"], live_expectancy_r=ls["expectancy_r"],
    )
    combined_note = (
        "Historical and paper numbers are shown separately on purpose. "
        "They are not averaged into one fake edge."
    )
    if hs["n"] and ps["n"] and (ps["expectancy_r"] or 0) < (hs["expectancy_r"] or 0):
        conclusion = "Historical edge has not yet reproduced forward."
    elif ps["n"] < 10:
        conclusion = "Forward paper sample is still too small."
    elif (ps["expectancy_r"] or 0) <= 0:
        conclusion = "Forward paper results are not showing an advantage yet."
    else:
        conclusion = "Forward paper results look promising but are not live-validated."
    return {
        "policy_id": policy_id,
        "historical": hs,
        "paper_forward": ps,
        "limited_live": _stats(limited),
        "live": ls,
        "maturity": maturity,
        "maturity_plain": plain_label(maturity),
        "combined_methodology": combined_note,
        "scientific_conclusion": conclusion,
        "live_authorized": False,
    }


def learning_blurb(policy_id: str, display_name: str | None = None) -> str:
    rep = policy_report(policy_id)
    name = display_name or policy_id
    n = rep["paper_forward"]["n"]
    if n == 0:
        return f"{name} has not taken forward paper trades yet."
    exp = rep["paper_forward"]["expectancy_r"]
    if n < 10:
        return (
            f"{name} has taken {n} forward paper trades. "
            "Evidence is still too small to trust."
        )
    if exp is not None and exp <= 0:
        return (
            f"{name} has taken {n} forward paper trades. "
            "So far it has not shown a reliable advantage."
        )
    return (
        f"{name} has taken {n} forward paper trades "
        f"(about {exp:+.2f}R average). Still observation-only — not real money."
    )
