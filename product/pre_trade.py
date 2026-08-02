"""Retail pre-trade cockpit — compose existing risk/data facts into GO/CAUTION/NO_GO.

Read-only. Never places orders, never invents setups, never arms LIVE.
A GO means the plan is risk-coherent under current book/market/data — not “buy now”.
"""
from __future__ import annotations

from typing import Any, Mapping

GO = "GO"
CAUTION = "CAUTION"
NO_GO = "NO_GO"


def build_pre_trade(
    *,
    symbol: str,
    plan: Mapping[str, Any] | None,
    market: Mapping[str, Any] | None = None,
    scan_record: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
    book_correlation: Mapping[str, Any] | None = None,
    paper: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    plan = dict(plan or {})
    market = dict(market or {})
    scan_record = dict(scan_record or {}) if scan_record else None
    readiness = dict(readiness or {})
    book_correlation = dict(book_correlation or {})
    paper = dict(paper or {})

    blockers: list[str] = []
    warnings: list[str] = []

    available = bool(plan.get("available"))
    tradeable = bool(plan.get("tradeable")) if available else False

    if not available:
        blockers.append(str(plan.get("message") or "No risk plan — need a scanned setup with entry and stop."))
    elif not tradeable:
        blockers.append(str(plan.get("reason") or "Plan is not tradeable at current capital/risk."))

    heat = str(plan.get("heat_verdict") or "OK").upper()
    if heat == "DANGER":
        blockers.append("Book open-risk would breach a safety limit — close or reduce something first.")
    elif heat == "CAUTION":
        warnings.append("Book heat is elevated — size down or wait.")

    health = str(
        plan.get("market_health")
        or market.get("health")
        or ""
    ).strip().lower()
    risk_factor = plan.get("market_risk_factor")
    try:
        risk_factor_f = float(risk_factor) if risk_factor is not None else None
    except (TypeError, ValueError):
        risk_factor_f = None
    if health == "weak" or (risk_factor_f is not None and risk_factor_f <= 0.5):
        warnings.append("Market regime is weak — risk is throttled; new entries need extra caution.")
    elif health == "mixed" or (risk_factor_f is not None and risk_factor_f < 1.0):
        warnings.append("Market regime is mixed — position size is throttled.")

    corr_status = str(plan.get("correlation_status") or "unknown")
    correlated = list(plan.get("correlated_with") or [])
    if corr_status == "adds_to_bet" and correlated:
        warnings.append(
            f"Not a new independent bet — moves with {', '.join(str(s) for s in correlated[:4])}."
        )

    if scan_record:
        if scan_record.get("chase_risk") or scan_record.get("extended"):
            warnings.append("Scan flags chase/extension risk — late entry quality is weaker.")
        if str(scan_record.get("verdict") or "").upper() in {"AVOID", "SELL", "REJECT"}:
            warnings.append(f"Scanner verdict is {scan_record.get('verdict')} — treat as caution, not confirmation.")

    checklist = dict(readiness.get("retail_research_checklist") or {})
    gaps = list(checklist.get("gaps") or [])
    critical_gap_keys = {
        "bhav_history",
        "verified_snapshot",
        "corporate_actions",
        "universe_history",
        "live_edge",
    }
    hard_gaps = [g for g in gaps if str(g.get("key") or "") in critical_gap_keys and g.get("status") == "MISSING"]
    soft_gaps = [g for g in gaps if g not in hard_gaps]
    for gap in hard_gaps[:4]:
        blockers.append(f"Data gap: {gap.get('label')} — {gap.get('next_action') or 'fix before trusting edge'}")
    for gap in soft_gaps[:3]:
        if gap.get("status") in {"MISSING", "PARTIAL"}:
            warnings.append(f"Research incomplete: {gap.get('label')}")

    lanes = list(readiness.get("lanes") or [])
    for lane in lanes:
        if lane.get("key") in {"history", "scanner", "snapshot"} and lane.get("status") == "MISSING":
            label = lane.get("label") or lane.get("key")
            msg = f"Product lane missing: {label}"
            if msg not in blockers:
                blockers.append(msg)

    try:
        cost_drag = float(plan["cost_drag_r"]) if plan.get("cost_drag_r") is not None else None
    except (TypeError, ValueError):
        cost_drag = None
    if cost_drag is not None and cost_drag >= 0.25:
        warnings.append(f"Round-trip costs ≈ {cost_drag:.2f}R — edge must clear a fat cost hurdle.")

    open_positions = list(paper.get("open_positions") or [])
    n_open = len(open_positions)
    n_bets = int(book_correlation.get("n_bets") or plan.get("effective_bets_after") or n_open or 0)
    n_pos = int(book_correlation.get("n_positions") or n_open or 0)
    if n_pos >= 2 and n_bets and n_bets < n_pos:
        warnings.append(f"Book concentration: {n_pos} positions behave like {n_bets} bet(s).")

    if blockers:
        verdict = NO_GO
    elif warnings:
        verdict = CAUTION
    elif available and tradeable:
        verdict = GO
    else:
        verdict = NO_GO
        if not blockers:
            blockers.append("Insufficient facts for a coherent pre-trade decision.")

    meaning = {
        GO: "Risk plan is coherent under current book, market throttle and known data — still not a buy order.",
        CAUTION: "Tradable only with eyes open — warnings must be accepted before any paper/live rehearsal.",
        NO_GO: "Do not add risk — fix blockers first (setup, book heat, or critical data gaps).",
    }[verdict]

    return {
        "schema_version": 1,
        "symbol": sym,
        "available": available,
        "verdict": verdict,
        "meaning": meaning,
        "tradeable": bool(available and tradeable and verdict != NO_GO),
        "blockers": blockers,
        "warnings": warnings,
        "plan": plan,
        "plan_summary": str(plan.get("summary") or ""),
        "cost_drag_r": cost_drag,
        "round_trip_cost_pct": plan.get("round_trip_cost_pct"),
        "correlation": {
            "status": corr_status,
            "correlated_with": correlated,
            "effective_bets_before": plan.get("effective_bets_before"),
            "effective_bets_after": plan.get("effective_bets_after"),
            "n_positions": n_pos,
            "n_bets": n_bets,
            "message": book_correlation.get("message") or "",
        },
        "market_throttle": {
            "health": health or "unknown",
            "market_risk_factor": risk_factor_f,
            "suggested_risk_pct": plan.get("suggested_risk_pct"),
            "trade_stance": market.get("trade_stance") or market.get("summary") or "",
        },
        "data_gaps": [
            {
                "key": g.get("key"),
                "label": g.get("label"),
                "status": g.get("status"),
                "next_action": g.get("next_action"),
            }
            for g in gaps
            if g.get("status") in {"MISSING", "PARTIAL"}
        ],
        "paper_snapshot": {
            "open_positions": n_open,
            "capital": plan.get("capital") or paper.get("capital"),
            "open_risk_pct": plan.get("open_risk_pct_before"),
        },
        "scan": {
            "available": bool(scan_record),
            "verdict": (scan_record or {}).get("verdict"),
            "score": (scan_record or {}).get("score"),
            "signals": (scan_record or {}).get("signals") or [],
            "entry": (scan_record or {}).get("entry") or plan.get("entry"),
            "stop": (scan_record or {}).get("stop") or plan.get("stop"),
            "target": (scan_record or {}).get("target") or plan.get("target"),
        },
        "read_only": True,
        "places_orders": False,
        "honesty": (
            "Pre-trade cockpit composes size, book heat, costs, correlation and data gaps. "
            "GO is not a signal and does not arm LIVE trading."
        ),
    }
