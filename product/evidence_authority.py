"""Operator-facing evidence authority for QuantTerm.

This module does not create another stock-selection engine. It explains the
existing Recommendations methods, the canonical scan coverage ledger, and the
tracked signal outcomes in one truthful contract.

Rules:
- unknown evidence is never converted to zero or a pass;
- the Evidence Score is descriptive, never a win probability or money-path gate;
- score coverage is shown separately from score strength;
- performance uses only settled tracked outcomes;
- the decision journal includes both surfaced ideas and names that were checked
  but rejected / unavailable in the latest whole-market scan.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.reco_methods import METHOD_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]

_COMPONENTS: tuple[tuple[str, str, float, tuple[str, ...]], ...] = (
    ("price_structure", "Price & structure", 30.0, ("tape", "sepa", "trend", "rs")),
    ("fundamentals", "Fundamentals", 20.0, ("funds",)),
    ("market_sector", "Market / sector context", 10.0, ("sector",)),
    ("empirical", "Empirical evidence", 20.0, ("ev", "case")),
    ("setup_risk", "Setup / entry quality", 20.0, ("conviction",)),
)


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def evidence_scorecard(card: Mapping[str, Any]) -> dict[str, Any]:
    """Explain one recommendation card using its existing method panel.

    Score is normalized over evidence that is actually known. Coverage tells the
    operator how much of the full score model was measurable. A high score with
    low coverage is therefore visibly incomplete and is never promoted here.
    """
    methods = [dict(m) for m in (card.get("methods") or []) if isinstance(m, Mapping)]
    by_id = {str(m.get("id") or ""): m for m in methods}
    components: list[dict[str, Any]] = []
    total_known_capacity = 0.0
    total_weighted_score = 0.0

    for cid, label, maximum, member_ids in _COMPONENTS:
        members = [by_id[mid] for mid in member_ids if mid in by_id]
        possible_method_weight = sum(float(METHOD_WEIGHTS.get(mid, 0.0)) for mid in member_ids)
        known = [
            m for m in members
            if str(m.get("status") or "unknown") != "unknown"
            and _f(m.get("points")) is not None
        ]
        known_method_weight = sum(
            float(METHOD_WEIGHTS.get(str(m.get("id") or ""), 0.0)) for m in known
        )
        component_coverage = (
            known_method_weight / possible_method_weight * 100.0
            if possible_method_weight else 0.0
        )
        if known_method_weight > 0:
            normalized = sum(
                float(METHOD_WEIGHTS.get(str(m.get("id") or ""), 0.0))
                * max(0.0, min(100.0, float(_f(m.get("points")) or 0.0)))
                for m in known
            ) / known_method_weight
            earned_known = maximum * normalized / 100.0
            known_capacity = maximum * known_method_weight / possible_method_weight
            total_known_capacity += known_capacity
            total_weighted_score += known_capacity * normalized / 100.0
            score = round(normalized, 1)
        else:
            earned_known = 0.0
            known_capacity = 0.0
            score = None
        statuses = {"pass": 0, "fail": 0, "unknown": 0}
        details: list[dict[str, Any]] = []
        for mid in member_ids:
            method = by_id.get(mid)
            status = str((method or {}).get("status") or "unknown")
            if status not in statuses:
                status = "unknown"
            statuses[status] += 1
            details.append({
                "id": mid,
                "label": str((method or {}).get("label") or mid),
                "status": status,
                "points": _f((method or {}).get("points")),
                "detail": str((method or {}).get("detail") or "Evidence unavailable"),
            })
        components.append({
            "id": cid,
            "label": label,
            "max_points": maximum,
            "score": score,
            "known_capacity": round(known_capacity, 2),
            "earned_known_points": round(earned_known, 2),
            "coverage_pct": round(component_coverage, 1),
            "passed": statuses["pass"],
            "failed": statuses["fail"],
            "unknown": statuses["unknown"],
            "methods": details,
        })

    total_capacity = sum(item[2] for item in _COMPONENTS)
    coverage_pct = round(
        (total_known_capacity / total_capacity * 100.0) if total_capacity else 0.0, 1
    )
    score = (
        round(total_weighted_score / total_known_capacity * 100.0, 1)
        if total_known_capacity else None
    )
    passed = sum(1 for m in methods if str(m.get("status")) == "pass")
    failed = sum(1 for m in methods if str(m.get("status")) == "fail")
    unknown = max(0, len(METHOD_WEIGHTS) - passed - failed)
    if coverage_pct >= 80:
        quality = "WELL_COVERED"
    elif coverage_pct >= 55:
        quality = "PARTIAL"
    else:
        quality = "THIN"
    return {
        "score": score,
        "coverage_pct": coverage_pct,
        "quality": quality,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "checks_total": len(METHOD_WEIGHTS),
        "components": components,
        "market_support": card.get("market_support"),
        "entry_state": card.get("entry_state"),
        "blockers": list(card.get("blockers") or []),
        "disclaimer": (
            "Evidence Score is not a win probability and never overrides entry, "
            "risk, portfolio, or live-safety gates."
        ),
    }


def _read_reco_ledger(path: Path | None = None, *, max_lines: int = 500) -> list[dict[str, Any]]:
    target = path or (ROOT / "logs" / "product" / "reco_ledger.jsonl")
    if not target.exists():
        return []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()[-max(1, max_lines):]
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _recommendation_history(symbol: str = "", *, limit: int = 80) -> list[dict[str, Any]]:
    want = str(symbol or "").strip().upper()
    rows: list[dict[str, Any]] = []
    for batch in reversed(_read_reco_ledger()):
        recorded_at = str(batch.get("recorded_at") or "")
        scan_at = str(batch.get("scan_scanned_at") or "")
        for raw in batch.get("cards") or []:
            if not isinstance(raw, Mapping):
                continue
            sym = str(raw.get("symbol") or "").upper()
            if not sym or (want and sym != want):
                continue
            frozen = dict(raw.get("evidence_scorecard") or {})
            rows.append({
                "kind": "SURFACED",
                "symbol": sym,
                "recorded_at": recorded_at,
                "scan_scanned_at": scan_at,
                "decision": str(raw.get("tier") or raw.get("action_badge") or "surfaced"),
                "reason": str(raw.get("thesis") or "Recommendation evidence qualified."),
                "entry_state": raw.get("entry_state"),
                "timing": raw.get("timing"),
                "stock_quality": raw.get("stock_quality"),
                "family_confirms": raw.get("family_confirms"),
                "families": list(raw.get("families") or []),
                "conflicts": list(raw.get("conflicts") or []),
                "entry": raw.get("entry"),
                "stop": raw.get("stop"),
                "target": raw.get("target"),
                "cmp": raw.get("cmp"),
                "evidence_score": frozen.get("score"),
                "evidence_coverage_pct": frozen.get("coverage_pct"),
                "evidence_quality": frozen.get("quality"),
                "evidence_passed": frozen.get("passed"),
                "evidence_failed": frozen.get("failed"),
                "evidence_unknown": frozen.get("unknown"),
                "evidence_components": list(frozen.get("components") or []),
                "score_frozen_at_decision": bool(frozen),
                "source": "recommendation_ledger",
            })
            if len(rows) >= limit:
                return rows
    return rows


def _scan_decisions(symbol: str = "", *, limit: int = 250) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from scan.scan_coverage import load_audit
        audit = load_audit()
    except Exception:
        audit = {"summary": {}, "ledger": []}
    want = str(symbol or "").strip().upper()
    decisions: list[dict[str, Any]] = []
    for raw in audit.get("ledger") or []:
        if not isinstance(raw, Mapping):
            continue
        sym = str(raw.get("symbol") or "").upper()
        if not sym or (want and sym != want):
            continue
        status = str(raw.get("status") or "UNKNOWN")
        decisions.append({
            "kind": "SCAN_DECISION",
            "symbol": sym,
            "decision": status,
            "reason": str(raw.get("reason") or "No reason recorded."),
            "error": str(raw.get("error") or ""),
            "source": "latest_scan_audit",
            "generated_at": audit.get("generated_at"),
        })
        if len(decisions) >= limit:
            break
    return dict(audit.get("summary") or {}), decisions


def _max_drawdown_from_returns(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for pct in returns:
        equity *= max(0.0, 1.0 + float(pct) / 100.0)
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) * 100.0 if peak > 0 else 0.0
        worst = min(worst, dd)
    return round(worst, 2)


def performance_summary() -> dict[str, Any]:
    """Performance of settled tracked scanner signals only; never marketing P&L."""
    try:
        from core.signal_outcome_tracker import get_accuracy_report, get_recent_signals
        stats = dict(get_accuracy_report() or {})
        recent = list(get_recent_signals(limit=5000) or [])
    except Exception:
        stats, recent = {}, []
    wins = int(stats.get("wins") or 0)
    losses = int(stats.get("losses") or 0)
    closed = wins + losses
    settled = [
        r for r in recent
        if r.get("worked") in (0, 1) and _f(r.get("outcome_pct")) is not None
    ]
    settled.sort(key=lambda r: str(r.get("logged_at") or ""))
    returns = [float(r["outcome_pct"]) for r in settled]
    return {
        "source": "tracked_signal_outcomes",
        "scope": "paper/tracked research signals; not broker-verified live P&L",
        "sample_size": closed,
        "open_signals": int(stats.get("open_signals") or 0),
        "wins": wins,
        "losses": losses,
        "hit_rate_pct": round(wins / closed * 100.0, 1) if closed else None,
        "expectancy_pct": _f(stats.get("system_edge")) if closed else None,
        "avg_gain_pct": _f(stats.get("avg_win_pct")) if wins else None,
        "avg_loss_pct": _f(stats.get("avg_loss_pct")) if losses else None,
        "max_drawdown_pct": _max_drawdown_from_returns(returns),
        "benchmark_comparison": None,
        "benchmark_note": (
            "Unavailable until a publication-dated benchmark series is attached "
            "to the same settled sample."
        ),
        "sufficient_sample": closed >= 30,
        "sample_note": (
            "Descriptive only — fewer than 30 settled outcomes."
            if 0 < closed < 30 else
            ("No settled outcomes yet." if closed == 0 else "At least 30 settled outcomes available.")
        ),
    }


def build_decision_journal(*, symbol: str = "", limit: int = 120) -> dict[str, Any]:
    scan_summary, scan_rows = _scan_decisions(symbol, limit=limit)
    reco_rows = _recommendation_history(symbol, limit=limit)
    combined = (reco_rows + scan_rows)[: max(1, min(int(limit or 120), 1000))]
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol or "").strip().upper(),
        "scan_summary": scan_summary,
        "performance": performance_summary(),
        "entries": combined,
        "counts": {
            "surfaced_history": len(reco_rows),
            "latest_scan_decisions": len(scan_rows),
        },
        "note": (
            "Surfaced ideas preserve their decision-time score/coverage when the "
            "ledger version supports it. Latest scan rows include rejected, excluded, "
            "or unavailable names. Missing evidence remains missing."
        ),
    }


def build_authority_contract(scan_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scan = dict(scan_payload or {})
    coverage = dict(scan.get("coverage") or {})
    return {
        "schema_version": 1,
        "product": "QuantTerm",
        "positioning": "Evidence-Driven Market Intelligence for Retail Traders",
        "principle": (
            "QuantTerm does not predict stocks. It discovers, tests, ranks, and "
            "tracks opportunities using evidence."
        ),
        "methodology": [
            "Build the current NSE cash-equity universe",
            "Verify data coverage and freshness",
            "Evaluate independent evidence methods",
            "Reject weak, extended, risky, or incomplete candidates",
            "Rank surviving opportunities and apply portfolio/risk gates",
            "Record surfaced and rejected decisions",
            "Settle outcomes on official future data",
            "Feed measured outcomes back into research and paper behavior",
        ],
        "scan_coverage": {
            "state": str(scan.get("coverage_state") or coverage.get("state") or "UNKNOWN"),
            "requested": int(coverage.get("requested") or scan.get("requested_universe") or 0),
            "checked": int(coverage.get("checked") or scan.get("universe_size") or 0),
            "qualified": int(coverage.get("qualified") or 0),
            "no_setup": int(coverage.get("no_setup") or 0),
            "policy_excluded": int(coverage.get("policy_excluded") or 0),
            "data_unavailable": int(coverage.get("data_unavailable") or 0),
            "analysis_errors": int(coverage.get("analysis_errors") or 0),
            "history_coverage_pct": _f(coverage.get("history_coverage_pct")),
            "coverage_pct": _f(coverage.get("coverage_pct")),
            "scanned_at": scan.get("scanned_at"),
        },
        "performance": performance_summary(),
        "score_semantics": (
            "Evidence Score summarizes known research checks. Unknown checks do not "
            "become zero; coverage is reported separately. It is not a win probability."
        ),
        "regulatory_language": (
            "Market intelligence and research tooling; no claim of SEBI approval, "
            "certification, or regulatory endorsement."
        ),
    }
