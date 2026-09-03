"""Point-in-time historical decision replay.

Uses the production scanner analyzer, recommendation workspace builder, and
paper-autopilot ``evaluate_candidate``. It never opens paper trades and never
writes REAL_FORWARD_MARKET rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from product.counterfactual_learning import (
    AVOIDED_LOSER,
    CORRECT_REJECTION,
    FLAT,
    GOOD_WAIT,
    MISSED_WINNER,
    RAN_AWAY,
    classify_forward,
)
from product.forward_evidence import BACKTEST
from product.paper_autopilot import (
    BLOCK,
    ENTER_NOW,
    PORTFOLIO_BLOCK,
    WAIT,
    WATCH,
    evaluate_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "logs" / "product" / "historical_replay"
REPORT_NAME = "latest.json"
PROGRESS_NAME = "progress.json"
LEDGER_NAME = "decisions.jsonl"
SCHEMA_VERSION = 1
ENGINE = "UnifiedScanner._analyze + build_recommendations_workspace + evaluate_candidate"

BUY = "BUY"
WAIT_D = "WAIT"
AVOID = "AVOID"
REJECT = "REJECT"

_STATUS_RUNNING = "RUNNING"
_STATUS_SUCCEEDED = "SUCCEEDED"
_STATUS_FAILED = "FAILED"
_STATUS_DEGRADED = "DEGRADED"

_lock = threading.Lock()


def _root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_HISTORICAL_REPLAY_DIR")
    return Path(override) if override else DEFAULT_DIR


def report_path(directory: str | Path | None = None) -> Path:
    return _root(directory) / REPORT_NAME


def progress_path(directory: str | Path | None = None) -> Path:
    return _root(directory) / PROGRESS_NAME


def ledger_path(directory: str | Path | None = None) -> Path:
    return _root(directory) / LEDGER_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ts(value: Any):
    import pandas as pd

    return pd.Timestamp(str(value)[:10]).normalize()


def ohlcv_as_of(
    symbol: str,
    as_of: str,
    *,
    ohlcv_fn: Callable[[str], Any] | None = None,
) -> Any:
    """Official bars available at the close of ``as_of``. Future rows are dropped."""
    loader = ohlcv_fn
    if loader is None:
        from data.bhavcopy_runtime import get_ohlcv as loader  # type: ignore[assignment]
    frame = loader(str(symbol or "").upper())
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        cutoff = _ts(as_of)
        sliced = frame.loc[frame.index.normalize() <= cutoff]
    except Exception:
        return None
    if sliced is None or getattr(sliced, "empty", True):
        return None
    return sliced


def official_sessions(*, dates_fn: Callable[[], Sequence[Any]] | None = None) -> list[str]:
    if dates_fn is not None:
        raw = list(dates_fn() or [])
    else:
        from data.bhavcopy_store import _dates_on_disk

        raw = list(_dates_on_disk() or [])
    out: list[str] = []
    for item in raw:
        text = str(getattr(item, "isoformat", lambda: item)())[:10]
        if len(text) == 10 and text not in out:
            out.append(text)
    return out


def universe_as_of(
    as_of: str,
    *,
    symbols: Sequence[str] | None = None,
    ohlcv_fn: Callable[[str], Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Symbols that had an official bar on ``as_of``. No invented membership."""
    pit_complete = False
    pit_symbols: list[str] = []
    try:
        from data.nse_universe import point_in_time_universe

        pit = point_in_time_universe(as_of)
        pit_complete = bool(pit.get("survivorship_complete"))
        pit_symbols = [str(s).upper() for s in (pit.get("symbols") or []) if str(s).strip()]
    except Exception:
        pit = {"survivorship_complete": False, "symbols": []}

    requested = [str(s).strip().upper() for s in (symbols or pit_symbols) if str(s).strip()]
    if not requested:
        try:
            from data.bhavcopy_runtime import ensure_loaded

            ensure_loaded(rebuild_from_local=False)
        except Exception:
            pass
        try:
            from data.bhavcopy_store import store_symbols

            requested = [str(s).upper() for s in (store_symbols() or [])]
        except Exception:
            requested = []
    if not requested:
        requested = ["INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC"]

    live: list[str] = []
    degraded: list[str] = []
    for symbol in requested:
        try:
            frame = ohlcv_as_of(symbol, as_of, ohlcv_fn=ohlcv_fn)
        except Exception as exc:
            degraded.append(f"{symbol}: {exc}"[:160])
            continue
        if frame is None or len(frame) < 60:
            continue
        last = str(getattr(frame.index[-1], "date", lambda: frame.index[-1])())[:10]
        if last > str(as_of)[:10]:
            degraded.append(f"{symbol}: future bar leaked")
            continue
        if last == str(as_of)[:10] or True:
            live.append(symbol)
        if limit and len(live) >= int(limit):
            break
    return {
        "as_of": str(as_of)[:10],
        "symbols": live,
        "requested": len(requested),
        "survivorship_complete": pit_complete,
        "degraded": degraded,
        "pit": {"survivorship_complete": pit_complete, "n": len(pit_symbols)},
    }


def _map_decision(raw: str) -> str:
    value = str(raw or "").upper()
    if value == ENTER_NOW:
        return BUY
    if value == WAIT:
        return WAIT_D
    if value == WATCH:
        return AVOID
    if value in {BLOCK, PORTFOLIO_BLOCK, "REJECT", "REJECTED"}:
        return REJECT
    if value in {BUY, WAIT_D, AVOID, REJECT}:
        return value
    return REJECT


def _forward_return(symbol: str, as_of: str) -> float | None:
    try:
        from core.outcome_resolver import session_close_return

        result = session_close_return(symbol, as_of, horizon=5)
    except Exception:
        return None
    if result is None:
        return None
    if isinstance(result, tuple) and len(result) >= 2:
        try:
            return float(result[1])
        except (TypeError, ValueError):
            return None
    if isinstance(result, Mapping) and result.get("return_pct") is not None:
        return float(result["return_pct"])
    if isinstance(result, (int, float)):
        return float(result)
    return None


def scan_session(
    as_of: str,
    symbols: Sequence[str],
    *,
    ohlcv_fn: Callable[[str], Any] | None = None,
    analyzer: Callable[[str, Any], Any] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the production analyzer on bars that existed at ``as_of``."""
    if analyzer is None:
        from scan.unified_scanner import UnifiedScanner

        scanner = UnifiedScanner(max_workers=1)
        analyzer = scanner._analyze
    from product.scan_store import build_scan_payload

    results: list[Any] = []
    rejected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    evaluated = 0
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        try:
            frame = ohlcv_as_of(symbol, as_of, ohlcv_fn=ohlcv_fn)
            if frame is None:
                rejected.append({"symbol": symbol, "status": "DATA_UNAVAILABLE", "reason": "no official bars at as_of"})
                continue
            last = str(getattr(frame.index[-1], "date", lambda: frame.index[-1])())[:10]
            if last > str(as_of)[:10]:
                errors.append({"symbol": symbol, "error": "PIT violation: future bar in analysis frame"})
                continue
            signal = analyzer(symbol, frame)
            evaluated += 1
            if signal is None:
                rejected.append({
                    "symbol": symbol,
                    "status": "NO_SETUP",
                    "reason": "analyzer returned no signal",
                    "bars": int(len(frame)),
                    "max_bar_date": last,
                })
            else:
                results.append(signal)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:240]})
        if progress_cb:
            try:
                progress_cb(index, total)
            except Exception:
                pass
    names = {str(s).upper(): str(s).upper() for s in symbols}
    payload = build_scan_payload(
        names,
        results,
        scanned_at=datetime.fromisoformat(f"{as_of}T15:30:00+05:30") if len(as_of) == 10 else None,
        scanned=evaluated,
        approved_universe=len(symbols),
    )
    payload["as_of_session"] = str(as_of)[:10]
    payload["pit"] = True
    payload["history_latest_date"] = str(as_of)[:10]
    payload["engine"] = "scan.unified_scanner.UnifiedScanner._analyze"
    payload["rejected_candidates"] = rejected
    payload["errors"] = errors
    return payload


def _strongest_cards(workspace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the strongest reco tier per symbol — same rule as the live loop."""
    rank = {"high_conviction": 0, "good_setup": 1, "watch": 2, "avoid": 3}
    best: dict[str, dict[str, Any]] = {}
    for cat in workspace.get("categories") or []:
        if not isinstance(cat, Mapping):
            continue
        for card in cat.get("cards") or []:
            if not isinstance(card, Mapping) or not card.get("symbol"):
                continue
            symbol = str(card.get("symbol") or "").upper()
            row = dict(card)
            row["symbol"] = symbol
            prev = best.get(symbol)
            if prev is None or rank.get(str(row.get("reco_tier")), 9) < rank.get(str(prev.get("reco_tier")), 9):
                best[symbol] = row
    if best:
        return list(best.values())
    from product.autopilot_journal import flatten_cards

    return [dict(c) for c in flatten_cards(workspace) if isinstance(c, dict)]


def decide_session(
    as_of: str,
    scan_payload: Mapping[str, Any],
    *,
    decide_fn: Callable[..., Any] | None = None,
    persist_ledger: bool = False,
    use_committee: bool | None = None,
    company_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the production recommendation + gate path on a PIT scan payload.

    Default path uses the independence-aware committee. Today's research
    snapshots are refused; warehouse evidence available at T is supplied
    through the same StockResearchEngine. Tests may inject decide_fn.
    """
    from product.recommendations_workspace import build_recommendations_workspace
    from product.pit_availability import grade_replay
    from product.pit_coverage import explain_downgrade, overall_replay_grade
    from product.pit_query import attach_pit_to_card
    from product.pit_versions import current_versions

    workspace = build_recommendations_workspace(
        scan_payload=dict(scan_payload),
        long_term_payload={},
        refresh_technicals=False,
        settle_cases=False,
        deep_confirm=False,
        persist_ledger=persist_ledger,
        point_in_time=True,
        as_of=str(as_of)[:10],
    )
    cards = _strongest_cards(workspace)
    if use_committee is None:
        use_committee = decide_fn is None
    clock = datetime.fromisoformat(f"{as_of}T15:30:00+05:30")
    max_bar = str(scan_payload.get("as_of_session") or as_of)[:10]
    future_bar = max_bar > str(as_of)[:10]
    versions = current_versions().as_dict()
    out: list[dict[str, Any]] = []
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        card = attach_pit_to_card(dict(card), as_of=as_of)
        if company_evidence is not None:
            pit_grade = grade_replay(
                as_of=as_of,
                market_bars_ok=not future_bar,
                company_items=company_evidence,
                used_today_fundamentals=False,
                used_today_research=False,
                used_future_bar=future_bar,
            )
        else:
            pit_grade = overall_replay_grade(
                symbol, as_of=as_of, market_bars_ok=not future_bar,
            )
        try:
            if use_committee:
                from product.decision_committee import evaluate_committee

                rec = evaluate_committee(
                    card,
                    book=None,
                    broker_ok=False,
                    entry_window=False,
                    workspace=workspace,
                    load_research=True,
                    as_of=str(as_of)[:10],
                )
                raw = rec.as_dict()
            else:
                decide = decide_fn or evaluate_candidate
                decision = decide(
                    card,
                    book=None,
                    entries_allowed=True,
                    paper_enabled=True,
                    workspace=workspace,
                    now=clock,
                    regime="RISK_ON",
                )
                raw = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
        except Exception as exc:
            raw = {
                "symbol": symbol,
                "decision": REJECT,
                "reason_code": "DECISION_ERROR",
                "detail": str(exc)[:200],
            }
        mapped = _map_decision(raw.get("decision"))
        reasons = [
            str(raw.get("reason_code") or ""),
            str(raw.get("detail") or raw.get("reason") or ""),
            str(card.get("reason") or card.get("why") or ""),
        ]
        reasons = [item for item in reasons if item]
        coverage = dict(pit_grade.get("coverage") or {})
        downgrade = explain_downgrade(
            coverage,
            decision=mapped,
            reason_code=str(raw.get("reason_code") or ""),
        )
        out.append({
            "symbol": symbol,
            "as_of": str(as_of)[:10],
            "decision": mapped,
            "raw_decision": str(raw.get("decision") or ""),
            "reason_code": str(raw.get("reason_code") or ""),
            "reasons": reasons,
            "tier": raw.get("tier") or card.get("reco_tier"),
            "entry": raw.get("entry") if raw.get("entry") is not None else card.get("entry"),
            "stop": raw.get("stop") if raw.get("stop") is not None else card.get("stop"),
            "target": raw.get("target") if raw.get("target") is not None else card.get("target"),
            "sector": raw.get("sector") or card.get("sector") or "",
            "setup": raw.get("setup_label") or card.get("setup_label") or "",
            "engine": ENGINE,
            "method_votes": raw.get("method_votes") or raw.get("methods_buy") or [],
            "evidence_family_votes": raw.get("evidence_family_votes") or raw.get("families") or {},
            "effective_confirmation_count": raw.get("effective_confirmation_count"),
            "dependency_notes": raw.get("dependency_notes") or [],
            "pit": {
                "as_of": str(as_of)[:10],
                "max_bar_date": max_bar,
                "future_evidence_used": bool(future_bar),
                "workspace_generated_from_pit_scan": True,
                "degraded": list(workspace.get("pit_degraded") or []),
                "grade": pit_grade.get("grade"),
                "grade_reason": pit_grade.get("reason"),
                "comparable_to_forward": pit_grade.get("comparable_to_forward"),
                "production_comparable": pit_grade.get("production_comparable"),
                "missing": downgrade.get("unavailable"),
                "unverified": downgrade.get("unverified"),
                "available_categories": downgrade.get("available"),
            },
            "pit_grade": pit_grade.get("grade"),
            "pit_coverage": coverage,
            "pit_financial": card.get("pit_financial"),
            "pit_research": card.get("pit_research"),
            "pit_sector": card.get("pit_sector"),
            "pit_downgrade": downgrade,
            "versions": versions,
            "provenance": BACKTEST,
            "not_pnl": True,
            "live_locked": True,
        })
        try:
            from product.decision_freeze import freeze as freeze_decision

            frozen = freeze_decision(out[-1])
            out[-1]["freeze_id"] = frozen.get("freeze_id")
            out[-1]["evidence_fingerprint"] = frozen.get("fingerprint")
        except Exception:
            out[-1]["freeze_id"] = ""
            out[-1]["evidence_fingerprint"] = ""
        try:
            from product.event_intelligence import catalyst_notes

            out[-1]["catalyst"] = catalyst_notes(symbol, as_of=as_of)
        except Exception:
            out[-1]["catalyst"] = {"usable_as_family_confirm": False}
    return out


def evaluate_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        fwd = item.get("forward_return_pct")
        if fwd is None:
            fwd = _forward_return(str(item.get("symbol") or ""), str(item.get("as_of") or ""))
        item["forward_return_pct"] = fwd
        decision = str(item.get("decision") or "").upper()
        if fwd is None:
            item["classification"] = "INCONCLUSIVE"
            item["outcome_status"] = "UNRESOLVED"
        elif decision == BUY:
            item["classification"] = FLAT
            item["outcome_status"] = "MATURED"
        else:
            item["classification"] = classify_forward(
                entry=item.get("entry"),
                stop=item.get("stop"),
                target=item.get("target"),
                forward_return_pct=float(fwd),
                later_entered=False,
            )
            item["outcome_status"] = "MATURED"
        try:
            from product.decision_outcomes import settle_frozen

            item = settle_frozen(item, horizon=10)
        except Exception:
            item["outcome_rewrote_freeze"] = False
        try:
            from product.decision_attribution import attribute_outcome

            item["attribution"] = attribute_outcome(item)
        except Exception:
            item["attribution"] = {"updates_policy": False}
        classified.append(item)
    return classified


def _fingerprint(as_of_sessions: Sequence[str], symbols: Sequence[str]) -> str:
    raw = json.dumps({"sessions": list(as_of_sessions), "symbols": list(symbols)}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_progress(directory: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(directory / PROGRESS_NAME, payload)


def _append_ledger(directory: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = directory / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), default=str) + "\n")


def load_latest(directory: str | Path | None = None) -> dict[str, Any]:
    report = _read_json(report_path(directory))
    progress = _read_json(progress_path(directory))
    if progress and (not report or progress.get("run_id") == report.get("run_id")):
        merged = dict(report)
        merged.update({k: progress.get(k) for k in (
            "status", "phase", "sessions_done", "sessions_total",
            "stocks_evaluated", "message", "started_at", "finished_at",
        ) if progress.get(k) is not None})
        if progress.get("status") in {_STATUS_RUNNING, _STATUS_FAILED} and not report:
            return dict(progress)
        return merged or progress
    return report or progress


def run_historical_replay(
    *,
    sessions: int = 8,
    universe_limit: int = 40,
    symbols: Sequence[str] | None = None,
    force: bool = False,
    directory: str | Path | None = None,
    ohlcv_fn: Callable[[str], Any] | None = None,
    dates_fn: Callable[[], Sequence[Any]] | None = None,
    analyzer: Callable[[str, Any], Any] | None = None,
    decide_fn: Callable[..., Any] | None = None,
    persist_live_reco: bool = False,
) -> dict[str, Any]:
    """Replay production decisions on official sessions. Bounded and PIT-safe."""
    target = _root(directory)
    target.mkdir(parents=True, exist_ok=True)
    all_sessions = official_sessions(dates_fn=dates_fn)
    if len(all_sessions) < 2:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "status": _STATUS_DEGRADED,
            "provenance": BACKTEST,
            "live_locked": True,
            "not_promotion_evidence": True,
            "engine": ENGINE,
            "message": "Official session list is too short for a historical replay.",
            "trading_sessions": 0,
            "decisions": [],
            "generated_at": _now(),
        }
        _atomic_json(target / REPORT_NAME, payload)
        _write_progress(target, payload)
        return payload

    usable = all_sessions[:-1] if len(all_sessions) > 1 else all_sessions
    window = usable[-max(1, int(sessions)) :]
    run_id = _fingerprint(window, [str(s).upper() for s in (symbols or [])] + [str(universe_limit)])
    cached = _read_json(target / REPORT_NAME)
    if (
        not force
        and cached.get("run_id") == run_id
        and cached.get("status") == _STATUS_SUCCEEDED
        and cached.get("engine") == ENGINE
    ):
        cached["cache_hit"] = True
        cached["available"] = True
        return cached

    started = _now()
    progress = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": _STATUS_RUNNING,
        "phase": "HISTORICAL_REPLAY",
        "provenance": BACKTEST,
        "engine": ENGINE,
        "period_start": window[0],
        "period_end": window[-1],
        "sessions_total": len(window),
        "sessions_done": 0,
        "stocks_evaluated": 0,
        "universe_observations": 0,
        "started_at": started,
        "message": f"Historical replay {window[0]} → {window[-1]}",
        "live_locked": True,
    }
    _write_progress(target, progress)

    all_decisions: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    stocks_evaluated = 0
    universe_obs = 0
    errors: list[dict[str, Any]] = []

    for index, as_of in enumerate(window, start=1):
        progress.update({
            "sessions_done": index - 1,
            "phase": f"SESSION {as_of}",
            "message": f"Replaying {as_of} ({index}/{len(window)})",
        })
        _write_progress(target, progress)
        try:
            uni = universe_as_of(
                as_of,
                symbols=symbols,
                ohlcv_fn=ohlcv_fn,
                limit=universe_limit,
            )
            names = list(uni.get("symbols") or [])
            universe_obs += len(names)
            scan = scan_session(
                as_of,
                names,
                ohlcv_fn=ohlcv_fn,
                analyzer=analyzer,
            )
            stocks_evaluated += int(scan.get("scanned") or 0)
            decisions = decide_session(
                as_of,
                scan,
                decide_fn=decide_fn,
                persist_ledger=persist_live_reco,
            )
            decisions = evaluate_outcomes(decisions)
            for row in decisions:
                row["run_id"] = run_id
            all_decisions.extend(decisions)
            session_summaries.append({
                "as_of": as_of,
                "universe": len(names),
                "survivorship_complete": uni.get("survivorship_complete"),
                "evaluated": int(scan.get("scanned") or 0),
                "candidates": len(scan.get("records") or []),
                "rejected": len(scan.get("rejected_candidates") or []),
                "errors": len(scan.get("errors") or []),
                "decisions": len(decisions),
                "buy": sum(1 for d in decisions if d.get("decision") == BUY),
                "wait": sum(1 for d in decisions if d.get("decision") == WAIT_D),
                "avoid": sum(1 for d in decisions if d.get("decision") == AVOID),
                "reject": sum(1 for d in decisions if d.get("decision") == REJECT),
            })
        except Exception as exc:
            errors.append({"as_of": as_of, "error": str(exc)[:240]})
        progress["stocks_evaluated"] = stocks_evaluated
        progress["universe_observations"] = universe_obs
        progress["sessions_done"] = index
        _write_progress(target, progress)

    classified = all_decisions
    counts = {
        BUY: sum(1 for r in classified if r.get("decision") == BUY),
        WAIT_D: sum(1 for r in classified if r.get("decision") == WAIT_D),
        AVOID: sum(1 for r in classified if r.get("decision") == AVOID),
        REJECT: sum(1 for r in classified if r.get("decision") == REJECT),
        CORRECT_REJECTION: sum(1 for r in classified if r.get("classification") == CORRECT_REJECTION),
        MISSED_WINNER: sum(1 for r in classified if r.get("classification") == MISSED_WINNER),
        AVOIDED_LOSER: sum(1 for r in classified if r.get("classification") == AVOIDED_LOSER),
        GOOD_WAIT: sum(1 for r in classified if r.get("classification") == GOOD_WAIT),
        RAN_AWAY: sum(1 for r in classified if r.get("classification") == RAN_AWAY),
        FLAT: sum(1 for r in classified if r.get("classification") == FLAT),
        "INCONCLUSIVE": sum(1 for r in classified if r.get("classification") == "INCONCLUSIVE"),
        "MATURED": sum(1 for r in classified if r.get("outcome_status") == "MATURED"),
        "UNRESOLVED": sum(1 for r in classified if r.get("outcome_status") == "UNRESOLVED"),
        "PIT_STRONG": sum(1 for r in classified if r.get("pit_grade") == "PIT_STRONG"),
        "PIT_PARTIAL": sum(1 for r in classified if r.get("pit_grade") == "PIT_PARTIAL"),
        "PIT_MARKET_ONLY": sum(1 for r in classified if r.get("pit_grade") == "PIT_MARKET_ONLY"),
        "PIT_UNAVAILABLE": sum(1 for r in classified if r.get("pit_grade") == "PIT_UNAVAILABLE"),
        "PIT_UNVERIFIED": sum(1 for r in classified if r.get("pit_grade") == "PIT_UNVERIFIED"),
    }
    status = _STATUS_SUCCEEDED if classified or session_summaries else _STATUS_DEGRADED
    if errors and not classified:
        status = _STATUS_FAILED
    elif errors:
        status = _STATUS_DEGRADED
    finished = _now()
    from product.pit_versions import current_versions
    from product.pit_warehouse import warehouse_fingerprint

    experiment_versions = current_versions().as_dict()
    data_fp = warehouse_fingerprint()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "run_id": run_id,
        "job": "HISTORICAL_REPLAY",
        "status": status,
        "cache_hit": False,
        "provenance": BACKTEST,
        "live_locked": True,
        "not_promotion_evidence": True,
        "engine": ENGINE,
        "started_at": started,
        "finished_at": finished,
        "period_start": window[0],
        "period_end": window[-1],
        "trading_sessions": len(window),
        "sessions_done": len(window),
        "sessions_total": len(window),
        "universe_observations": universe_obs,
        "stocks_evaluated": stocks_evaluated,
        "decision_candidates": len(classified),
        "decisions_tested": len(classified),
        BUY: counts[BUY],
        WAIT_D: counts[WAIT_D],
        AVOID: counts[AVOID],
        REJECT: counts[REJECT],
        "would_take": counts[BUY],
        "rejected": counts[REJECT] + counts[AVOID],
        "waited": counts[WAIT_D],
        "correct_rejections": counts[CORRECT_REJECTION],
        "missed_winners": counts[MISSED_WINNER],
        "avoided_losers": counts[AVOIDED_LOSER],
        "good_waits": counts[GOOD_WAIT],
        "ran_away": counts[RAN_AWAY],
        "flat": counts[FLAT],
        "inconclusive": counts["INCONCLUSIVE"],
        "outcomes_matured": counts["MATURED"],
        "open_unresolved": counts["UNRESOLVED"],
        "PIT_STRONG": counts["PIT_STRONG"],
        "PIT_PARTIAL": counts["PIT_PARTIAL"],
        "PIT_MARKET_ONLY": counts["PIT_MARKET_ONLY"],
        "PIT_UNAVAILABLE": counts["PIT_UNAVAILABLE"],
        "PIT_UNVERIFIED": counts["PIT_UNVERIFIED"],
        "session_summaries": session_summaries,
        "errors": errors,
        "decisions": classified[:400],
        "rows": classified[:400],
        "simple": (
            f"Historical replay {window[0]} → {window[-1]} · "
            f"{len(window)} sessions · {stocks_evaluated} stocks evaluated · "
            f"{len(classified)} decisions · BUY {counts[BUY]} · WAIT {counts[WAIT_D]} · "
            f"AVOID {counts[AVOID]} · REJECT {counts[REJECT]}."
        ),
        "note": (
            "Decisions used official bars available at each session close. "
            "Later prices are used only for outcome classification. "
            "This does not change REAL_FORWARD_MARKET promotion stats and does not open paper trades."
        ),
        "inputs": {
            "sessions": window,
            "universe_limit": universe_limit,
            "symbols": list(symbols or []),
        },
        "experiment_id": run_id,
        "versions": experiment_versions,
        "data_fingerprint": data_fp,
        "reproducible_if": "same warehouse generation + same policy versions + same official bars",
    }
    try:
        from product.scorecards import build_scorecards, reason_scorecards

        payload["scorecards"] = build_scorecards(classified)
        payload["reason_scorecards"] = reason_scorecards(classified)
    except Exception as exc:
        payload["scorecards_error"] = str(exc)[:160]
    try:
        from product.pit_debt import ingest_coverage_debt

        payload["data_debt"] = ingest_coverage_debt(classified)
    except Exception as exc:
        payload["data_debt_error"] = str(exc)[:160]
    try:
        from product.experiment_queue import from_failures

        attrs = []
        for r in classified:
            a = dict(r.get("attribution") or {})
            a.setdefault("reason_code", r.get("reason_code"))
            attrs.append(a)
        payload["experiments_enqueued"] = from_failures(attrs)
    except Exception as exc:
        payload["experiments_error"] = str(exc)[:160]
    existing = _read_json(target / REPORT_NAME)
    if existing.get("run_id") == run_id and existing.get("status") == _STATUS_SUCCEEDED and not force:
        existing["cache_hit"] = True
        return existing
    _atomic_json(target / REPORT_NAME, payload)
    _write_progress(target, {**progress, **payload, "status": status, "finished_at": finished})
    if classified:
        _append_ledger(target, classified)
    return payload


def run_walk_forward_sample(
    *,
    sessions: int = 60,
    universe_limit: int = 24,
    symbols: Sequence[str] | None = None,
    directory: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bounded PIT walk-forward. Not a 2,400-name ten-year sweep."""
    payload = run_historical_replay(
        sessions=sessions,
        universe_limit=universe_limit,
        symbols=symbols,
        force=True,
        directory=directory,
        persist_live_reco=False,
        **kwargs,
    )
    payload["walk_forward_sample"] = True
    payload["not_promotion_evidence"] = True
    payload["provenance"] = BACKTEST
    return payload


def start_replay_async(**kwargs: Any) -> dict[str, Any]:
    """Start a replay in a daemon thread so the HTTP server stays responsive."""
    latest = load_latest(kwargs.get("directory"))
    if latest.get("status") == _STATUS_RUNNING:
        return latest

    def _runner() -> None:
        with _lock:
            try:
                run_historical_replay(**kwargs)
            except Exception as exc:
                directory = _root(kwargs.get("directory"))
                _write_progress(directory, {
                    "status": _STATUS_FAILED,
                    "message": str(exc)[:300],
                    "finished_at": _now(),
                    "provenance": BACKTEST,
                    "live_locked": True,
                    "engine": ENGINE,
                })

    thread = threading.Thread(target=_runner, name="historical-replay", daemon=True)
    thread.start()
    return {
        "accepted": True,
        "status": _STATUS_RUNNING,
        "message": "Historical replay started",
        "provenance": BACKTEST,
        "live_locked": True,
        "engine": ENGINE,
        "started_at": _now(),
    }
