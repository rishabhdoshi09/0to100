"""Live-forward evidence for the FINAL ``QT_RECO_ENSEMBLE`` decision.

Two experiments are kept separate:

TAKEN / recommended
    Frozen entry/stop/target settles through official NSE bhavcopy first-touch.

NOT TAKEN / Watch / Avoid
    The decision-time reference price settles through canonical ``wait_patience``
    plus the fixed-session close return. This measures whether patience offered a
    better entry, the stock ran away without us, or the rejected setup failed.

Rejected outcomes are learning/control evidence, NEVER mixed into recommended
win-rate or expectancy. Nothing is reconstructed with future fundamentals.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_PATH = ROOT / "logs" / "product" / "reco_replay.jsonl"
DEFAULT_ARTIFACT_PATH = ROOT / "logs" / "product" / "qt_reco_ensemble_backtest.json"

MIN_VERIFIED_SAMPLE = 30
MIN_VERIFIED_SCAN_DAYS = 10
MIN_REJECTION_LEARNING_SAMPLE = 30
_TIMESTAMP_TOLERANCE = timedelta(minutes=10)

OutcomeResolver = Callable[[str, str, float, float, float], tuple[float, float, int] | None]
WaitResolver = Callable[[str, str, float], tuple[float, float, str] | None]
CloseResolver = Callable[[str, str], tuple[float, float] | None]


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = datetime.fromisoformat(text[:10])
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _current_hash() -> str:
    try:
        from product.strategy_catalog import current_rules_hash
        return current_rules_hash()
    except Exception:
        return ""


def _default_resolver(symbol: str, opened_at: str, entry: float, stop: float, target: float):
    from core.outcome_resolver import first_touch_path
    return first_touch_path(symbol, opened_at, entry, stop, target)


def _default_wait_resolver(symbol: str, opened_at: str, reference: float):
    from core.outcome_resolver import wait_patience
    return wait_patience(symbol, opened_at, reference)


def _default_close_resolver(symbol: str, opened_at: str):
    from core.outcome_resolver import session_close_return
    return session_close_return(symbol, opened_at)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if isinstance(raw, dict):
                    rows.append(raw)
    except Exception:
        return []
    return rows


def _future_timestamp_violations(value: Any, captured_at: datetime, path: str = "") -> list[str]:
    violations: list[str] = []
    timestamp_keys = {
        "published_at", "event_date", "filing_date", "reported_at",
        "observed_at", "as_of", "source_as_of", "timestamp", "ts",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key).lower() in timestamp_keys and item not in (None, ""):
                parsed = _parse_dt(item)
                if parsed is not None and parsed > captured_at + _TIMESTAMP_TOLERANCE:
                    violations.append(f"{child}={item}")
            elif isinstance(item, (Mapping, list, tuple)):
                violations.extend(_future_timestamp_violations(item, captured_at, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if isinstance(item, (Mapping, list, tuple)):
                violations.extend(_future_timestamp_violations(item, captured_at, f"{path}[{index}]"))
    return violations


def _record_pit_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    captured = _parse_dt(record.get("captured_at"))
    scanned = _parse_dt(record.get("scan_scanned_at"))
    reasons: list[str] = []
    if not bool(record.get("captured_live")):
        reasons.append("capture is not marked live")
    if captured is None:
        reasons.append("captured_at missing/unparseable")
    if scanned is None:
        reasons.append("scan_scanned_at missing/unparseable")
    if captured is not None and scanned is not None and scanned > captured + _TIMESTAMP_TOLERANCE:
        reasons.append("scan timestamp occurs after capture timestamp")
    future: list[str] = []
    if captured is not None:
        for candidate in record.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            raw_input = candidate.get("input")
            if isinstance(raw_input, Mapping):
                future.extend(_future_timestamp_violations(raw_input, captured))
    if future:
        reasons.append(f"future evidence timestamp(s): {', '.join(future[:5])}")
    return {
        "verified": not reasons,
        "captured_at": record.get("captured_at"),
        "scan_scanned_at": record.get("scan_scanned_at"),
        "violations": reasons,
    }


def _parts(candidate: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    decision = candidate.get("decision_at_capture")
    raw_input = candidate.get("input")
    selection = candidate.get("selection_snapshot")
    return (
        decision if isinstance(decision, Mapping) else {},
        raw_input if isinstance(raw_input, Mapping) else {},
        selection if isinstance(selection, Mapping) else {},
    )


def _candidate_key(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    decision, raw_input, _ = _parts(candidate)
    symbol = str(decision.get("symbol") or raw_input.get("symbol") or "").upper()
    category = str(decision.get("category_id") or raw_input.get("category_id") or "")
    scan_date = str(record.get("scan_scanned_at") or "")[:10]
    return f"{scan_date}|{symbol}|{category}"


def _opened_at(record: Mapping[str, Any]) -> str:
    return str(record.get("scan_scanned_at") or record.get("captured_at") or "")[:10]


def _selection_reasons(candidate: Mapping[str, Any]) -> list[str]:
    _, _, selection = _parts(candidate)
    out: list[str] = []
    learning = selection.get("learning")
    if isinstance(learning, Mapping) and bool(learning.get("negative")):
        out.append("OBSERVED_EDGE_NEGATIVE")
    dd = selection.get("due_diligence")
    if isinstance(dd, Mapping) and not bool(dd.get("passed")):
        reasons = [str(x) for x in (dd.get("reasons") or []) if str(x).strip()]
        if reasons:
            out.extend(f"DD:{reason}" for reason in reasons[:4])
        else:
            out.append("DD:FAILED")
    return out or ["BASE_ENSEMBLE_NOT_TAKEN"]


def _settle_taken(record: Mapping[str, Any], candidate: Mapping[str, Any], *, resolver: OutcomeResolver) -> dict[str, Any]:
    decision, raw_input, _ = _parts(candidate)
    symbol = str(decision.get("symbol") or raw_input.get("symbol") or "").upper()
    if not symbol:
        return {"status": "INVALID_CAPTURE", "reason": "symbol missing"}
    entry = _f(raw_input.get("entry") or raw_input.get("entry_price"))
    stop = _f(raw_input.get("stop") or raw_input.get("stop_price"))
    target = _f(raw_input.get("target") or raw_input.get("target_price"))
    if entry is None or stop is None or target is None or entry <= 0 or stop <= 0 or target <= entry or stop >= entry:
        return {"status": "MISSING_GEOMETRY", "symbol": symbol, "entry": entry, "stop": stop, "target": target}
    opened_at = _opened_at(record)
    if not opened_at:
        return {"status": "INVALID_CAPTURE", "symbol": symbol, "reason": "decision date missing"}
    try:
        outcome = resolver(symbol, opened_at, entry, stop, target)
    except Exception as exc:
        return {"status": "RESOLVER_ERROR", "symbol": symbol, "reason": str(exc)[:200]}
    if outcome is None:
        return {"status": "PENDING", "symbol": symbol, "opened_at": opened_at, "entry": entry, "stop": stop, "target": target}
    exit_price, outcome_pct, worked = outcome
    if int(worked) == -1:
        return {"status": "NO_FILL", "symbol": symbol, "opened_at": opened_at, "entry": entry, "stop": stop, "target": target}
    return {
        "status": "SETTLED", "symbol": symbol, "opened_at": opened_at,
        "entry": entry, "stop": stop, "target": target,
        "exit_price": float(exit_price), "outcome_pct": float(outcome_pct),
        "worked": 1 if int(worked) == 1 else 0,
        "category_id": decision.get("category_id") or raw_input.get("category_id"),
        "reco_tier": decision.get("reco_tier"),
        "primary_thesis_id": decision.get("primary_thesis_id"),
    }


def _settle_rejected(
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    wait_resolver: WaitResolver,
    close_resolver: CloseResolver,
) -> dict[str, Any]:
    decision, raw_input, _ = _parts(candidate)
    symbol = str(decision.get("symbol") or raw_input.get("symbol") or "").upper()
    if not symbol:
        return {"status": "INVALID_CAPTURE", "reason": "symbol missing"}
    opened_at = _opened_at(record)
    reference = _f(
        raw_input.get("entry") or raw_input.get("entry_price")
        or raw_input.get("cmp") or raw_input.get("price") or raw_input.get("close")
    )
    if not opened_at:
        return {"status": "INVALID_CAPTURE", "symbol": symbol, "reason": "decision date missing"}
    if reference is None or reference <= 0:
        return {"status": "MISSING_REFERENCE", "symbol": symbol, "opened_at": opened_at}
    try:
        wait = wait_resolver(symbol, opened_at, reference)
    except Exception as exc:
        return {"status": "WAIT_RESOLVER_ERROR", "symbol": symbol, "reason": str(exc)[:200]}
    try:
        close = close_resolver(symbol, opened_at)
    except Exception:
        close = None
    if wait is None:
        return {
            "status": "REJECTED_PENDING", "symbol": symbol, "opened_at": opened_at,
            "reference_price": reference, "gate_reasons": _selection_reasons(candidate),
        }
    exit_price, patience_pct, patience = wait
    forward_price = forward_pct = None
    if close is not None:
        forward_price, forward_pct = float(close[0]), float(close[1])
    return {
        "status": "REJECTED_SETTLED", "symbol": symbol, "opened_at": opened_at,
        "reference_price": reference, "patience_result": str(patience),
        "patience_exit_price": float(exit_price), "patience_forward_pct": float(patience_pct),
        "forward_price": forward_price, "forward_pct": forward_pct,
        "category_id": decision.get("category_id") or raw_input.get("category_id"),
        "reco_tier": decision.get("reco_tier"),
        "primary_thesis_id": decision.get("primary_thesis_id"),
        "gate_reasons": _selection_reasons(candidate),
    }


def _max_consecutive_losses(rows: list[Mapping[str, Any]]) -> int:
    worst = run = 0
    for row in sorted(rows, key=lambda r: (str(r.get("opened_at") or ""), str(r.get("symbol") or ""))):
        if int(row.get("worked") or 0) == 0:
            run += 1; worst = max(worst, run)
        else:
            run = 0
    return worst


def _taken_metrics(settled: list[dict[str, Any]], *, no_fill: int, pending: int, missing_geometry: int, recommended: int, distinct_scan_dates: int) -> dict[str, Any]:
    wins = [r for r in settled if int(r.get("worked") or 0) == 1]
    losses = [r for r in settled if int(r.get("worked") or 0) == 0]
    returns = [float(r.get("outcome_pct") or 0.0) for r in settled]
    win_returns = [float(r.get("outcome_pct") or 0.0) for r in wins]
    loss_returns = [float(r.get("outcome_pct") or 0.0) for r in losses]
    n = len(settled)
    return {
        "sample_size": n, "filled_settled": n, "recommended_captures": recommended,
        "wins": len(wins), "losses": len(losses), "no_fill": no_fill,
        "pending": pending, "missing_geometry": missing_geometry,
        "hit_rate_pct": round(len(wins) / n * 100.0, 2) if n else None,
        "expectancy_pct": round(mean(returns), 4) if returns else None,
        "average_outcome_pct": round(mean(returns), 4) if returns else None,
        "avg_win_pct": round(mean(win_returns), 4) if win_returns else None,
        "avg_loss_pct": round(mean(loss_returns), 4) if loss_returns else None,
        "best_outcome_pct": round(max(returns), 4) if returns else None,
        "worst_outcome_pct": round(min(returns), 4) if returns else None,
        "max_consecutive_losses": _max_consecutive_losses(settled),
        "distinct_scan_dates": distinct_scan_dates,
        "costs_included": False,
        "portfolio_drawdown_pct": None,
        "portfolio_drawdown_note": "Unavailable: signal outcomes are not a capital-weighted execution portfolio.",
        "return_basis": "gross first-touch signal outcome on official NSE bhavcopy",
    }


def _rejection_group_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for reason in row.get("gate_reasons") or ["UNKNOWN"]:
            grouped[str(reason)].append(row)
    out: list[dict[str, Any]] = []
    for reason, items in grouped.items():
        n = len(items)
        offered = sum(1 for x in items if x.get("patience_result") == "OFFERED")
        ran = sum(1 for x in items if x.get("patience_result") == "RAN_AWAY")
        failed = sum(1 for x in items if x.get("patience_result") == "FAILED")
        forwards = [float(x["forward_pct"]) for x in items if x.get("forward_pct") is not None]
        out.append({
            "reason": reason, "n": n, "sample_sufficient": n >= MIN_REJECTION_LEARNING_SAMPLE,
            "offered": offered, "ran_away": ran, "failed": failed,
            "ran_away_rate_pct": round(ran / n * 100.0, 2) if n else None,
            "offered_rate_pct": round(offered / n * 100.0, 2) if n else None,
            "avg_forward_pct": round(mean(forwards), 4) if forwards else None,
        })
    out.sort(key=lambda x: (-int(x.get("n") or 0), str(x.get("reason") or "")))
    return out


def _rejected_metrics(settled: list[dict[str, Any]], *, captures: int, pending: int, missing_reference: int) -> dict[str, Any]:
    n = len(settled)
    offered = sum(1 for x in settled if x.get("patience_result") == "OFFERED")
    ran = sum(1 for x in settled if x.get("patience_result") == "RAN_AWAY")
    failed = sum(1 for x in settled if x.get("patience_result") == "FAILED")
    forwards = [float(x["forward_pct"]) for x in settled if x.get("forward_pct") is not None]
    positives = [x for x in forwards if x > 0]
    return {
        "rejected_captures": captures, "rejected_settled": n,
        "rejected_pending": pending, "rejected_missing_reference": missing_reference,
        "wait_offered": offered, "wait_ran_away": ran, "wait_failed": failed,
        "missed_opportunity_rate_pct": round(ran / n * 100.0, 2) if n else None,
        "better_entry_offered_rate_pct": round(offered / n * 100.0, 2) if n else None,
        "rejected_avg_forward_pct": round(mean(forwards), 4) if forwards else None,
        "rejected_positive_forward_rate_pct": round(len(positives) / len(forwards) * 100.0, 2) if forwards else None,
        "definition": "Control-group learning only; rejected outcomes are not recommendation P&L.",
        "by_gate_reason": _rejection_group_stats(settled),
    }


def build_production_signal_evidence(
    *, replay_path: Path | None = None, current_rules_hash: str | None = None,
    resolver: OutcomeResolver | None = None, wait_resolver: WaitResolver | None = None,
    close_resolver: CloseResolver | None = None,
) -> dict[str, Any]:
    target = replay_path or DEFAULT_REPLAY_PATH
    rules_hash = current_rules_hash if current_rules_hash is not None else _current_hash()
    resolve = resolver or _default_resolver
    resolve_wait = wait_resolver or _default_wait_resolver
    resolve_close = close_resolver or _default_close_resolver
    records = _load_jsonl(target)

    same_hash_records: list[Mapping[str, Any]] = []
    pit_audits: list[dict[str, Any]] = []
    for record in records:
        strategy = record.get("production_strategy") or {}
        captured_hash = str(strategy.get("rules_hash") or "") if isinstance(strategy, Mapping) else ""
        if rules_hash and captured_hash == rules_hash:
            same_hash_records.append(record); pit_audits.append(_record_pit_audit(record))

    point_in_time_verified = bool(same_hash_records) and all(a.get("verified") for a in pit_audits)
    seen_keys: set[str] = set(); scan_dates: set[str] = set()
    settled: list[dict[str, Any]] = []; rejected_settled: list[dict[str, Any]] = []
    no_fill = pending = missing_geometry = recommended = resolver_errors = invalid = 0
    rejected = rejected_pending = rejected_missing_reference = rejected_errors = 0

    for record in same_hash_records:
        pit = _record_pit_audit(record)
        if not pit.get("verified"): continue
        scan_date = str(record.get("scan_scanned_at") or "")[:10]
        if scan_date: scan_dates.add(scan_date)
        for candidate in record.get("candidates") or []:
            if not isinstance(candidate, Mapping): continue
            key = _candidate_key(record, candidate)
            if key in seen_keys: continue
            seen_keys.add(key)
            decision, _, _ = _parts(candidate)
            if bool(decision.get("allows_recommend")):
                recommended += 1
                outcome = _settle_taken(record, candidate, resolver=resolve)
                status = str(outcome.get("status") or "")
                if status == "SETTLED": outcome["key"] = key; settled.append(outcome)
                elif status == "NO_FILL": no_fill += 1
                elif status == "PENDING": pending += 1
                elif status == "MISSING_GEOMETRY": missing_geometry += 1
                elif status == "RESOLVER_ERROR": resolver_errors += 1
                elif status == "INVALID_CAPTURE": invalid += 1
            else:
                rejected += 1
                outcome = _settle_rejected(record, candidate, wait_resolver=resolve_wait, close_resolver=resolve_close)
                status = str(outcome.get("status") or "")
                if status == "REJECTED_SETTLED": outcome["key"] = key; rejected_settled.append(outcome)
                elif status == "REJECTED_PENDING": rejected_pending += 1
                elif status == "MISSING_REFERENCE": rejected_missing_reference += 1
                elif status in {"WAIT_RESOLVER_ERROR", "INVALID_CAPTURE"}: rejected_errors += 1

    metrics = _taken_metrics(settled, no_fill=no_fill, pending=pending, missing_geometry=missing_geometry,
                             recommended=recommended, distinct_scan_dates=len(scan_dates))
    rejection_metrics = _rejected_metrics(rejected_settled, captures=rejected, pending=rejected_pending,
                                          missing_reference=rejected_missing_reference)
    sample_ok = int(metrics.get("sample_size") or 0) >= MIN_VERIFIED_SAMPLE
    days_ok = len(scan_dates) >= MIN_VERIFIED_SCAN_DAYS
    evidence_ready = bool(point_in_time_verified and sample_ok and days_ok)
    reasons: list[str] = []
    if not same_hash_records: reasons.append("no live capture uses the current executable rules hash")
    if same_hash_records and not point_in_time_verified: reasons.append("point-in-time capture audit failed")
    if not sample_ok: reasons.append(f"filled settled sample {metrics.get('sample_size', 0)} < {MIN_VERIFIED_SAMPLE}")
    if not days_ok: reasons.append(f"distinct scan dates {len(scan_dates)} < {MIN_VERIFIED_SCAN_DAYS}")
    if resolver_errors: reasons.append(f"{resolver_errors} taken-outcome resolver error(s)")
    if rejected_errors: reasons.append(f"{rejected_errors} rejected-outcome resolver error(s)")

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": "QT_RECO_ENSEMBLE", "rules_hash": rules_hash or None,
        "scope": "PRODUCTION_SIGNAL_OUTCOMES", "completed": True,
        "point_in_time_verified": point_in_time_verified, "evidence_ready": evidence_ready,
        "minimum_sample": MIN_VERIFIED_SAMPLE, "minimum_scan_days": MIN_VERIFIED_SCAN_DAYS,
        "dataset": {
            "kind": "LIVE_FORWARD_CAPTURE", "source": str(target), "reconstructed": False,
            "same_hash_records": len(same_hash_records), "distinct_scan_dates": len(scan_dates),
            "first_scan_date": min(scan_dates) if scan_dates else None,
            "last_scan_date": max(scan_dates) if scan_dates else None,
            "eligible_candidate_keys": len(seen_keys),
            "pit_violations": [a for a in pit_audits if not a.get("verified")][:10],
        },
        "metrics": metrics,
        "rejection_learning": rejection_metrics,
        "walk_forward": {
            "kind": "LIVE_FORWARD_CAPTURE", "out_of_sample": True,
            "historical_reconstruction": False, "same_hash_only": True,
            "first_touch_official_bhavcopy": True, "rejected_wait_patience": True,
            "rejected_session_close_control": True, "costs_included": False,
        },
        "settled_examples": settled[-20:], "rejected_examples": rejected_settled[-20:],
        "blockers": reasons,
        "detail": (
            "Same-hash live-forward evidence is ready; rejected decisions are tracked separately as a control/learning group."
            if evidence_ready else
            "Production evidence is collecting: " + ("; ".join(reasons) if reasons else "insufficient settled evidence")
        ),
    }


def save_production_signal_evidence(payload: Mapping[str, Any], *, artifact_path: Path | None = None) -> Path:
    target = artifact_path or DEFAULT_ARTIFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def refresh_production_signal_evidence(
    *, replay_path: Path | None = None, artifact_path: Path | None = None,
    current_rules_hash: str | None = None, resolver: OutcomeResolver | None = None,
    wait_resolver: WaitResolver | None = None, close_resolver: CloseResolver | None = None,
) -> dict[str, Any]:
    payload = build_production_signal_evidence(
        replay_path=replay_path, current_rules_hash=current_rules_hash, resolver=resolver,
        wait_resolver=wait_resolver, close_resolver=close_resolver,
    )
    save_production_signal_evidence(payload, artifact_path=artifact_path)
    return payload
