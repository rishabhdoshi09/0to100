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
from core.runtime_paths import logs_dir

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
    target = path or (logs_dir() / "product" / "reco_ledger.jsonl")
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


_BENCH_CACHE: dict[str, Any] = {"key": None, "at": 0.0, "value": None, "building": False}
_BENCH_LOCK = __import__("threading").Lock()
_BENCH_TTL_S = 30 * 60.0
_BENCH_MIN_N = 30


def _nifty_closes():
    """Nifty 50 closes by ISO session date from the local index cache (no network)."""
    import pandas as pd
    from data.index_store import get_index_ohlcv_as_of_cached

    frame = get_index_ohlcv_as_of_cached("^NSEI", datetime.now().date().isoformat())
    if frame is None or getattr(frame, "empty", True):
        return {}
    col = next((c for c in frame.columns if str(c).lower() == "close"), None)
    if col is None:
        return {}
    out: dict[str, float] = {}
    for stamp, value in frame[col].items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0:
            out[pd.Timestamp(stamp).date().isoformat()] = price
    return out


def benchmark_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Settled signals vs Nifty 50 over exactly the sessions each was held.

    Each signal's fill and exit sessions are recomputed from official bars by
    the canonical first-touch resolver; Nifty is measured from the close before
    the fill session to the close of the exit session. A row whose window the
    index cache does not cover exactly is skipped, never approximated.
    """
    from statistics import median

    from core.outcome_resolver import first_touch_detail

    closes = _nifty_closes()
    if not closes:
        return None
    sessions = sorted(closes)
    pairs: list[tuple[float, float]] = []
    for row in rows:
        try:
            got = first_touch_detail(
                str(row.get("symbol") or ""), str(row.get("logged_at") or "")[:10],
                float(row.get("entry_price") or 0), float(row.get("stop_price") or 0),
                float(row.get("target_price") or 0),
            )
        except Exception:
            got = None
        if not got or got.get("worked") not in (0, 1):
            continue
        fill, exit_ = got.get("fill_date"), got.get("exit_date")
        if not fill or not exit_ or exit_ not in closes:
            continue
        before = [d for d in sessions if d < fill]
        if not before or before[-1] < str(row.get("logged_at") or "")[:10]:
            continue  # the session before the fill is missing from the cache
        bench = (closes[exit_] / closes[before[-1]] - 1.0) * 100.0
        pairs.append((float(got["pct"]), bench))
    n = len(pairs)
    if n == 0:
        return None
    excess = [sig - bench for sig, bench in pairs]
    return {
        "benchmark": "Nifty 50",
        "method": "same sessions as each settled signal (close before fill -> exit close)",
        "n": n,
        "sufficient_sample": n >= _BENCH_MIN_N,
        "mean_signal_pct": round(sum(s for s, _ in pairs) / n, 2),
        "mean_benchmark_pct": round(sum(b for _, b in pairs) / n, 2),
        "mean_excess_pct": round(sum(excess) / n, 2),
        "median_excess_pct": round(median(excess), 2),
        "beat_benchmark_pct": round(sum(1 for x in excess if x > 0) / n * 100.0, 1),
    }


def _cached_benchmark(settled: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Never computes inside a request: a stale/missing value is rebuilt by one
    background thread and the last value (or None) is returned meanwhile."""
    import threading
    import time as _t

    key = (
        len(settled),
        str(max((str(r.get("outcome_checked_at") or "") for r in settled), default="")),
    )
    now = _t.monotonic()
    with _BENCH_LOCK:
        current = _BENCH_CACHE["key"] == key and now - float(_BENCH_CACHE["at"]) < _BENCH_TTL_S
        if current or _BENCH_CACHE["building"]:
            return _BENCH_CACHE["value"]
        _BENCH_CACHE["building"] = True
    rows = [dict(r) for r in settled]

    def _build() -> None:
        try:
            value = benchmark_comparison(rows)
        except Exception:
            value = None
        with _BENCH_LOCK:
            _BENCH_CACHE.update({"key": key, "at": _t.monotonic(), "value": value, "building": False})

    try:
        threading.Thread(target=_build, daemon=True, name="qt-benchmark").start()
    except Exception:
        with _BENCH_LOCK:
            _BENCH_CACHE["building"] = False
    return _BENCH_CACHE["value"]


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
    bench = _cached_benchmark(settled) if len(settled) >= 1 else None
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
        "benchmark_comparison": bench,
        "benchmark_note": (
            "Computing the Nifty comparison over each signal's exact holding sessions…"
            if bench is None else
            (
                f"Nifty 50 over the same sessions: signals {bench['mean_signal_pct']:+.2f}% vs "
                f"Nifty {bench['mean_benchmark_pct']:+.2f}% (excess {bench['mean_excess_pct']:+.2f}%, "
                f"n={bench['n']})."
                + ("" if bench["sufficient_sample"] else " Descriptive only — under 30 matched outcomes.")
            )
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
