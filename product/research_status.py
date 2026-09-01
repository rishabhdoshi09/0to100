"""One Research / Learning projection from systems that already exist.

Does not start research jobs. Does not invent sample sizes or 'AI is learning'.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from product.strategy_catalog import production_registry, research_only_strategies


def build_research_status(
    *,
    paper_learning: Mapping[str, Any] | None = None,
    decision_journal: Mapping[str, Any] | None = None,
    autonomy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if paper_learning is None:
        try:
            from product.paper_learning import public_memory
            paper_learning = public_memory()
        except Exception:
            paper_learning = {}
    if decision_journal is None:
        try:
            from product.evidence_authority import build_decision_journal
            decision_journal = build_decision_journal(limit=80)
        except Exception:
            decision_journal = {}
    if autonomy is None:
        autonomy = {}

    production = production_registry()
    research = research_only_strategies()
    feed = dict((paper_learning or {}).get("self_feed") or {})
    taken = list(feed.get("taken") or [])
    skipped = list(feed.get("skipped") or [])
    tests = list(feed.get("candidate_tests") or [])
    journal = dict(decision_journal or {})
    perf = dict(journal.get("performance") or {})
    learning_status = str((autonomy or {}).get("learning_status") or "UNKNOWN")
    closed = int((paper_learning or {}).get("closed_trades") or 0)

    lines: list[str] = []
    ensemble = production["ensemble"]
    lines.append(
        f"{ensemble['label']} v{ensemble['strategy_version']}: "
        f"BACKTEST PARITY {ensemble['backtest_parity']}. "
        "No production promotion from an unverified hash."
    )
    sample = int(perf.get("sample_size") or 0)
    if sample:
        hit = perf.get("hit_rate_pct")
        exp = perf.get("expectancy_pct")
        lines.append(
            "Tracked scanner outcomes: "
            f"{sample} settled"
            + (f" · hit rate {hit}%" if hit is not None else "")
            + (f" · expectancy {exp}%" if exp is not None else "")
            + ("" if perf.get("sufficient_sample") else " · confidence insufficient")
            + ". Paper/tracked research — not broker-verified live P&L."
        )
    else:
        lines.append("No settled tracked outcomes yet. QuantTerm makes no performance claim.")
    if closed:
        lines.append(f"Paper book: {closed} closed trades in local memory. Live orders stay locked.")
    else:
        lines.append("Paper book has no closed trades in local memory.")
    if learning_status and learning_status != "UNKNOWN":
        lines.append(f"Autonomy learning status: {learning_status}.")
    try:
        from product.learning_policy_store import load_policies
        policies = [
            p for p in (load_policies().get("policies") or [])
            if str(p.get("production_status") or "") in {"ACTIVE", "ELIGIBLE", "EXPERIMENTAL"}
        ]
        if policies:
            lines.append(
                f"{len(policies)} explicit learning polic"
                f"{'ies' if len(policies) != 1 else 'y'} on file. "
                "Insufficient samples stay INSUFFICIENT EVIDENCE."
            )
    except Exception:
        policies = []

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production": production,
        "research_only": research,
        "active_count": 1 if ensemble.get("active") else 0,
        "evaluating_count": len(research),
        "rejected_count": 0,
        "paper": {
            "available": bool((paper_learning or {}).get("available")),
            "as_of": (paper_learning or {}).get("as_of") or "",
            "closed_trades": closed,
            "taken": taken[:20],
            "skipped": skipped[:20],
            "candidate_tests": tests[:20],
            "cooldown": list((paper_learning or {}).get("cooldown") or [])[:12],
            "summary": (paper_learning or {}).get("summary") or "",
            "live_locked": True,
        },
        "decision_journal": {
            "generated_at": journal.get("generated_at"),
            "counts": journal.get("counts") or {},
            "scan_summary": journal.get("scan_summary") or {},
            "performance": perf,
            "entries": list(journal.get("entries") or [])[:40],
            "note": journal.get("note") or "",
        },
        "learning_status": learning_status,
        "headlines": lines,
        "disclaimer": (
            "Measurable evidence only. 'AI is learning' is not a status. "
            "Research-only strategies never change today's BUY list."
        ),
    }
