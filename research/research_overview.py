"""
🛰️ Research Overview — mission control for QuantTerm's scientific process.

This is the query/aggregation layer behind the internal Research Dashboard. It
answers ONE question in under 30 seconds: *is my research organisation healthy?*
— not "is the market bullish", not "what's the best stock", but the health of the
research itself.

Crucially it is a LAYER, not a page: the dashboard renders it, and JARVIS QUERIES
it ("show me every gate that got less profitable over 90 days") instead of
computing anything itself. All statistics live in the subsystems below; this
module only composes their reads into structured sections:

    🧪 research_health   beliefs by status · promoted/retired this week ·
                         experiments awaiting validation · calibration · debt
    📉 edge_health       signals drifting / recovering / durable / dead ·
                         median recovery time
    ⚖️ gate_scorecard    per-reason saved / cost / net / confidence / trend
    📊 data_health       observation counts · schema versions · feature coverage
    🧾 research_debt      the engineering-style backlog of unfinished science
    🕰️ time_machine      what the system believed on ANY past date

Every section is fail-open: a missing or broken subsystem yields an empty/……
section, never an exception — mission control must render even when a feed is down.
"""
from __future__ import annotations


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════════════════
# 🧪 Research health
# ══════════════════════════════════════════════════════════════════════════════

def research_health() -> dict:
    from research import scientific_memory as SM
    beliefs = _safe(SM.list_beliefs, [])
    by_status: dict[str, int] = {}
    for b in beliefs:
        by_status[b["status"]] = by_status.get(b["status"], 0) + 1
    activity = _safe(lambda: SM.recent_activity(7), {})
    overdue = _safe(lambda: SM.overdue_for_review(30), [])

    experiments = _safe(lambda: _experiments(), {"registered": 0, "promoted": 0,
                                                 "rejected": 0, "recent_rejected": []})
    calib = _safe(_calibration_score, {"n": 0})
    return {
        "beliefs_total": len(beliefs),
        "beliefs_active": by_status.get(SM.ACTIVE, 0),
        "beliefs_watch": by_status.get(SM.WATCH, 0),
        "beliefs_retired": by_status.get(SM.RETIRED, 0),
        "beliefs_rejected": by_status.get(SM.REJECTED, 0),
        "promoted_this_week": activity.get("promoted", 0),
        "retired_this_week": activity.get("retired", 0),
        "to_watch_this_week": activity.get("to_watch", 0),
        "experiments_awaiting_validation": experiments["registered"],
        "experiments_promoted": experiments["promoted"],
        "experiments_rejected": experiments["rejected"],
        "recently_rejected_hypotheses": experiments["recent_rejected"][:5],
        "beliefs_overdue_review": len(overdue),
        "calibration": calib,
    }


def _experiments() -> dict:
    from research import registry as REG
    allx = REG.list_experiments()
    registered = [e for e in allx if e.get("status") == "REGISTERED"]
    promoted = [e for e in allx if e.get("status") == "PROMOTED"]
    rejected = [e for e in allx if e.get("status") == "REJECTED"]
    return {"registered": len(registered), "promoted": len(promoted),
            "rejected": len(rejected),
            "recent_rejected": [e.get("name", "") for e in rejected]}


def _calibration_score() -> dict:
    from research.calibration import calibration_report
    rep = calibration_report()
    return {"n": rep.get("n", 0), "ece": rep.get("ece"),
            "brier_skill": rep.get("brier_skill"),
            "insight": rep.get("insight", "")}


# ══════════════════════════════════════════════════════════════════════════════
# 📉 Edge health
# ══════════════════════════════════════════════════════════════════════════════

def edge_health() -> dict:
    from research.edge_timeline import timeline_report
    profiles = _safe(timeline_report, [])
    counts: dict[str, int] = {}
    recovery_times: list[float] = []
    for p in profiles:
        counts[p["profile"]] = counts.get(p["profile"], 0) + 1
        mrt = p.get("median_recovery_trades")
        if mrt:
            recovery_times.append(mrt)
    from research.drift import drift_report
    drifting = _safe(drift_report, [])
    return {
        "cyclical": counts.get("CYCLICAL", 0),
        "dead": counts.get("DEAD", 0),
        "durable": counts.get("DURABLE", 0),
        "decaying": counts.get("DECAYING", 0),
        "recovering": counts.get("RECOVERING", 0),
        "strengthening": counts.get("STRENGTHENING", 0),
        "median_recovery_trades": round(sorted(recovery_times)[len(recovery_times)//2], 1)
        if recovery_times else None,
        "signals_in_drift": [d for d in drifting if d.get("status") == "DECAYING"][:5],
        "tracked_signals": len(profiles),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ⚖️ Gate scorecard
# ══════════════════════════════════════════════════════════════════════════════

def gate_scorecard() -> list[dict]:
    """Per rejection reason: saved (correctly avoided) vs cost (missed winners),
    net observed edge, confidence, and a recent trend. Ranked by how much a
    reason is COSTING (too-conservative first — the ones to loosen)."""
    from research.non_event import rejection_analysis, reason_trend
    from research.explainability import REASON_LABELS
    rows = _safe(rejection_analysis, [])
    out = []
    for a in rows:
        n = a.get("n", 0)
        conf = ("High" if n >= 100 and a.get("p_value") is not None
                and a["p_value"] < 0.05 else "Medium" if n >= 30 else "Low")
        out.append({
            "gate": REASON_LABELS.get(a["reason"], a["reason"]),
            "reason": a["reason"],
            "saved": a.get("correctly_avoided", 0),
            "cost": a.get("missed_winners", 0),
            "net_fwd_pct": a.get("avg_fwd_pct"),          # observed, canonical
            "modeled_avg_r": a.get("modeled_avg_r"),      # modeled, labelled
            "confidence": conf,
            "verdict": a.get("verdict"),
            "trend": _safe(lambda a=a: reason_trend(a["reason"]), "→"),
            "n": n,
        })
    rank = {"TOO_CONSERVATIVE": 0, "NEUTRAL": 1, "EARNING": 2, "INSUFFICIENT": 3}
    return sorted(out, key=lambda d: (rank.get(d["verdict"], 9), -(d["cost"] or 0)))


# ══════════════════════════════════════════════════════════════════════════════
# 📊 Data health
# ══════════════════════════════════════════════════════════════════════════════

def data_health() -> dict:
    from research.feature_store import observation_counts, feature_coverage
    counts = _safe(observation_counts, {})
    coverage = _safe(feature_coverage, [])
    thin = [c for c in coverage if c.get("fill_rate", 1.0) < 0.5][:8]
    impossible = sum(c.get("problems", {}).get("IMPOSSIBLE", 0) for c in coverage)
    stale = sum(c.get("problems", {}).get("STALE", 0) for c in coverage)
    return {
        "total_observations": counts.get("total", 0),
        "by_kind": counts.get("by_kind", {}),
        "schema_versions": counts.get("schema_versions", []),
        "current_schema": counts.get("current_schema"),
        "on_current_schema": counts.get("on_current_schema", True),
        "thin_features": thin,
        "impossible_values": impossible,
        "stale_values": stale,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 🧾 Research debt — treat research like software engineering
# ══════════════════════════════════════════════════════════════════════════════

def research_debt() -> dict:
    from research import scientific_memory as SM
    exp = _safe(_experiments, {"registered": 0})
    overdue = _safe(lambda: SM.overdue_for_review(30), [])
    drift_unresolved = _safe(lambda: [d for d in _drift_report()
                                      if d.get("status") == "DECAYING"], [])
    data = _safe(data_health, {})
    schema_debt = 0 if data.get("on_current_schema", True) else 1
    return {
        "experiments_awaiting_validation": exp.get("registered", 0),
        "beliefs_overdue_review": len(overdue),
        "overdue_detail": overdue[:5],
        "drift_alerts_unresolved": len(drift_unresolved),
        "schemas_awaiting_migration": schema_debt,
    }


def _drift_report():
    from research.drift import drift_report
    return drift_report()


# ══════════════════════════════════════════════════════════════════════════════
# 🕰️ Time Machine
# ══════════════════════════════════════════════════════════════════════════════

def time_machine(iso_date: str) -> dict:
    """Reconstruct what the system BELIEVED on a past date — that day's beliefs
    (from the belief-event history), not today's. Frozen features + versioned
    schemas + belief history make this faithful, not a re-simulation."""
    from research.scientific_memory import beliefs_as_of, ACTIVE, WATCH
    beliefs = _safe(lambda: beliefs_as_of(iso_date), [])
    by_status: dict[str, int] = {}
    for b in beliefs:
        by_status[b["status"]] = by_status.get(b["status"], 0) + 1
    return {
        "as_of": iso_date,
        "beliefs": beliefs,
        "active": by_status.get(ACTIVE, 0),
        "watch": by_status.get(WATCH, 0),
        "total": len(beliefs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Top-level composition
# ══════════════════════════════════════════════════════════════════════════════

def knowledge_growth(days: int = 30) -> dict:
    """📈 The metric that matters most now — is the Research OS actually LEARNING?
    Not profit, not Sharpe: net validated knowledge gained. Beliefs validated vs
    retired over the window (normalised per-30-days), plus the average evidence
    behind an active belief. This is the health of the flywheel itself. Fail-open."""
    from research import scientific_memory as SM
    beliefs = _safe(SM.list_beliefs, [])
    active = [b for b in beliefs if b["status"] == SM.ACTIVE]
    avg_ev = round(sum((b.get("evidence_n") or 0) for b in active) / len(active), 1) \
        if active else 0.0
    act = _safe(lambda: SM.recent_activity(days), {})
    promoted, retired = act.get("promoted", 0), act.get("retired", 0)
    scale = 30.0 / days if days else 1.0
    net = promoted - retired
    return {
        "window_days": days,
        "beliefs_total": len(beliefs),
        "beliefs_active": len(active),
        "avg_evidence_per_belief": avg_ev,
        "validated_in_window": promoted,
        "retired_in_window": retired,
        "net_knowledge_gain": net,
        "validated_per_month": round(promoted * scale, 1),
        "retired_per_month": round(retired * scale, 1),
        "net_per_month": round(net * scale, 1),
        "learning": net > 0,
    }


def overview() -> dict:
    """The whole mission-control read in one call — for the dashboard and for
    JARVIS to query. Every section fail-open."""
    return {
        "research_health": _safe(research_health, {}),
        "knowledge_growth": _safe(knowledge_growth, {}),
        "edge_health": _safe(edge_health, {}),
        "gate_scorecard": _safe(gate_scorecard, []),
        "data_health": _safe(data_health, {}),
        "research_debt": _safe(research_debt, {}),
    }
