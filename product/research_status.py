"""User-facing Research/Learning status assembled from existing evidence systems.

This is intentionally an observer.  It does not run experiments, promote models,
change production strategy, or synthesize performance.  Its job is to make the
already-existing Research OS legible from the product.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _production_strategy_status() -> dict[str, Any]:
    from product.strategy_contract import strategy_registry_contract
    return strategy_registry_contract()


def _research_overview() -> dict[str, Any]:
    from research.research_overview import overview
    value = overview()
    return dict(value) if isinstance(value, Mapping) else {}


def _decision_status() -> dict[str, Any]:
    from product.evidence_authority import build_decision_journal
    journal = build_decision_journal(limit=250)
    counts = dict(journal.get("counts") or {})
    perf = dict(journal.get("performance") or {})
    return {
        "surfaced_history": int(counts.get("surfaced_history") or 0),
        "latest_scan_decisions": int(counts.get("latest_scan_decisions") or 0),
        "settled_sample_size": int(perf.get("sample_size") or 0),
        "hit_rate_pct": perf.get("hit_rate_pct"),
        "expectancy_pct": perf.get("expectancy_pct"),
        "average_gain_pct": perf.get("average_gain_pct"),
        "average_loss_pct": perf.get("average_loss_pct"),
        "max_drawdown_pct": perf.get("max_drawdown_pct"),
        "performance_claim_allowed": bool(perf.get("sample_size")),
        "performance_label": perf.get("label") or perf.get("status") or "UNAVAILABLE",
    }


def build_research_status() -> dict[str, Any]:
    overview = _safe(_research_overview, {})
    strategies = _safe(_production_strategy_status, {
        "production_strategy_count": 0,
        "verified_backtest_parity_count": 0,
        "unverified_backtest_parity_count": 0,
        "strategies": [],
    })
    decisions = _safe(_decision_status, {
        "surfaced_history": 0,
        "latest_scan_decisions": 0,
        "settled_sample_size": 0,
        "performance_claim_allowed": False,
        "performance_label": "UNAVAILABLE",
    })

    health = dict(overview.get("research_health") or {})
    edge = dict(overview.get("edge_health") or {})
    growth = dict(overview.get("knowledge_growth") or {})
    debt = dict(overview.get("research_debt") or {})
    data = dict(overview.get("data_health") or {})

    blockers: list[str] = []
    if int(strategies.get("unverified_backtest_parity_count") or 0):
        blockers.append(
            f"{int(strategies.get('unverified_backtest_parity_count') or 0)} active production lane(s) have BACKTEST PARITY: UNVERIFIED."
        )
    if int(health.get("experiments_awaiting_validation") or 0):
        blockers.append(
            f"{int(health.get('experiments_awaiting_validation') or 0)} research experiment(s) await validation."
        )
    if int(debt.get("drift_alerts_unresolved") or 0):
        blockers.append(
            f"{int(debt.get('drift_alerts_unresolved') or 0)} unresolved drift alert(s)."
        )
    if not int(decisions.get("settled_sample_size") or 0):
        blockers.append("No settled decision sample is available for a measured performance claim.")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "ATTENTION" if blockers else "HEALTHY",
        "production": {
            "active_strategies": int(strategies.get("production_strategy_count") or 0),
            "verified_backtest_parity": int(strategies.get("verified_backtest_parity_count") or 0),
            "unverified_backtest_parity": int(strategies.get("unverified_backtest_parity_count") or 0),
            "strategies": list(strategies.get("strategies") or []),
        },
        "experiments": {
            "awaiting_validation": int(health.get("experiments_awaiting_validation") or 0),
            "promoted": int(health.get("experiments_promoted") or 0),
            "rejected": int(health.get("experiments_rejected") or 0),
            "recently_rejected": list(health.get("recently_rejected_hypotheses") or []),
        },
        "learning": {
            "beliefs_active": int(health.get("beliefs_active") or 0),
            "beliefs_watch": int(health.get("beliefs_watch") or 0),
            "beliefs_retired": int(health.get("beliefs_retired") or 0),
            "promoted_this_week": int(health.get("promoted_this_week") or 0),
            "retired_this_week": int(health.get("retired_this_week") or 0),
            "net_knowledge_gain": growth.get("net_knowledge_gain"),
            "avg_evidence_per_active_belief": growth.get("avg_evidence_per_belief"),
            "calibration": dict(health.get("calibration") or {}),
        },
        "edge_health": {
            "tracked_signals": int(edge.get("tracked_signals") or 0),
            "durable": int(edge.get("durable") or 0),
            "decaying": int(edge.get("decaying") or 0),
            "dead": int(edge.get("dead") or 0),
            "recovering": int(edge.get("recovering") or 0),
            "signals_in_drift": list(edge.get("signals_in_drift") or []),
        },
        "decisions": decisions,
        "data": {
            "total_observations": int(data.get("total_observations") or 0),
            "on_current_schema": bool(data.get("on_current_schema", True)),
            "thin_features": list(data.get("thin_features") or []),
            "stale_values": int(data.get("stale_values") or 0),
            "impossible_values": int(data.get("impossible_values") or 0),
        },
        "research_debt": debt,
        "blockers": blockers,
        "invariant": (
            "QuantTerm may say it is learning only through measured observations, experiments, "
            "calibration/drift state, and settled decisions. Unmatched backtests never decorate production recommendations."
        ),
    }
