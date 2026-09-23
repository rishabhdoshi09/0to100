"""Durable progress and plateau control for autonomous evidence acquisition.

An evidence request is an investigation, not an instruction to replay forever.
This module records batch yield idempotently, closes requests when their stopping
conditions are met, and marks them PLATEAUED when repeated batches stop changing
the measurable evidence state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_path
from research.autonomy import evidence_acquisition as EA

SCHEMA_VERSION = 1
DEFAULT_PROGRESS_PATH = logs_path("research", "evidence_progress.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "requests": {}}
    if not isinstance(payload, dict):
        return {"schema_version": SCHEMA_VERSION, "requests": {}}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("requests", {})
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _decision_is_terminal(value: str) -> bool:
    decision = str(value or "").upper()
    if not decision:
        return False
    return decision not in {
        "EVIDENCE_ACQUISITION",
        "RETEST_WITH_MORE_DATA",
        "DATA_TASK",
        "NO_ACTION",
        "NO GAP",
    }


def _historical_yield(batch_result: Mapping[str, Any]) -> int:
    # Trade samples are the unit used by the current research gates. Decisions
    # are not substituted when no virtual trade was formed; doing so would
    # falsely claim that a rejected/no-geometry decision increased trade N.
    try:
        return max(0, int(batch_result.get("historical_paper_trades") or 0))
    except Exception:
        return 0


def _update_request_file(
    request_id: str,
    status: str,
    progress: Mapping[str, Any],
    *,
    request_path: str | Path | None = None,
) -> None:
    path = Path(request_path) if request_path is not None else EA.DEFAULT_REQUEST_PATH
    payload = EA.load_request(path)
    if str(payload.get("request_id") or "") != str(request_id):
        return
    payload["status"] = str(status)
    payload["progress"] = dict(progress)
    payload["updated_at"] = _now()
    if status in {EA.SATISFIED, EA.PLATEAUED, EA.BLOCKED}:
        payload["closed_at"] = _now()
    # Reuse the evidence-acquisition writer so file semantics stay atomic.
    EA.save_request(payload, path=path)


def record_historical_batch(
    request: Mapping[str, Any] | None,
    batch_result: Mapping[str, Any] | None,
    research_result: Mapping[str, Any] | None,
    *,
    path: str | Path | None = None,
    request_path: str | Path | None = None,
    plateau_batches: int = 3,
    resolved_metrics: Sequence[str] = (),
) -> dict[str, Any]:
    """Record one completed historical research batch exactly once."""
    req = dict(request or {})
    request_id = str(req.get("request_id") or "")
    if not request_id:
        return {}
    allowed = {str(x).upper() for x in (req.get("allowed_lanes") or [])}
    if "HISTORICAL_REPLAY" not in allowed:
        return {
            "request_id": request_id,
            "status": EA.BLOCKED,
            "reason": "HISTORICAL_REPLAY_NOT_ALLOWED_FOR_REQUEST",
        }

    batch = dict(batch_result or {})
    research = dict(research_result or {})
    batch_id = str(batch.get("batch_id") or research.get("historical_batch_id") or "")
    target_path = Path(path) if path is not None else DEFAULT_PROGRESS_PATH
    store = _read(target_path)
    request_rows = dict(store.get("requests") or {})
    previous = dict(request_rows.get(request_id) or {})
    seen = list(previous.get("seen_batch_ids") or [])
    if batch_id and batch_id in seen:
        return previous

    acquired = _historical_yield(batch)
    acquired_total = int(previous.get("samples_acquired") or 0) + acquired
    baseline = max(0, int(req.get("current_samples") or 0))
    target = max(baseline, int(req.get("target_samples") or baseline))
    current = baseline + acquired_total

    resolved = {str(x) for x in (previous.get("resolved_metrics") or []) if str(x)}
    resolved.update(str(x) for x in resolved_metrics if str(x))
    resolved.update(str(x) for x in (research.get("resolved_metrics") or []) if str(x))
    missing = {str(x) for x in (req.get("missing_metrics") or []) if str(x)}
    unresolved = sorted(missing - resolved)
    decision = str(research.get("decision") or "")

    capped_sample = min(current, target) if target else current
    progress_token = json.dumps(
        {
            "sample": capped_sample,
            "resolved_metrics": sorted(resolved),
        },
        sort_keys=True,
    )
    previous_token = str(previous.get("progress_token") or "")
    stagnant = int(previous.get("stagnant_batches") or 0)
    if previous:
        stagnant = stagnant + 1 if progress_token == previous_token else 0
    else:
        baseline_token = json.dumps(
            {
                "sample": min(baseline, target) if target else baseline,
                "resolved_metrics": [],
            },
            sort_keys=True,
        )
        stagnant = 1 if progress_token == baseline_token else 0

    sample_done = current >= target if target else True
    metrics_done = not unresolved
    if _decision_is_terminal(decision):
        status = EA.SATISFIED
        reason = f"research reached terminal decision {decision.upper()}"
    elif sample_done and metrics_done:
        status = EA.SATISFIED
        reason = "request stopping conditions satisfied"
    elif stagnant >= max(1, int(plateau_batches)):
        status = EA.PLATEAUED
        reason = f"no measurable evidence-state change for {stagnant} batches"
    else:
        status = EA.OPEN
        reason = "continue targeted evidence acquisition"

    history = list(previous.get("history") or [])
    history.append({
        "batch_id": batch_id,
        "samples_acquired": acquired,
        "sample_count": current,
        "research_decision": decision,
        "unresolved_metrics": unresolved,
        "recorded_at": _now(),
    })
    history = history[-50:]
    if batch_id:
        seen.append(batch_id)

    progress = {
        "request_id": request_id,
        "status": status,
        "reason": reason,
        "evidence_origin": str(req.get("evidence_origin") or ""),
        "samples_baseline": baseline,
        "samples_acquired": acquired_total,
        "sample_count": current,
        "target_samples": target,
        "sample_deficit": max(0, target - current),
        "last_batch_yield": acquired,
        "batches_completed": int(previous.get("batches_completed") or 0) + 1,
        "stagnant_batches": stagnant,
        "resolved_metrics": sorted(resolved),
        "unresolved_metrics": unresolved,
        "last_research_decision": decision,
        "progress_token": progress_token,
        "seen_batch_ids": seen,
        "history": history,
        "next_action": (
            "CLOSE_REQUEST" if status == EA.SATISFIED else
            "REPLAN_RESEARCH_QUESTION" if status == EA.PLATEAUED else
            "ACQUIRE_MORE_EVIDENCE"
        ),
        "updated_at": _now(),
    }
    request_rows[request_id] = progress
    store["requests"] = request_rows
    store["schema_version"] = SCHEMA_VERSION
    _write(target_path, store)
    _update_request_file(request_id, status, progress, request_path=request_path)
    return progress


def mark_historical_source_exhausted(
    request: Mapping[str, Any] | None,
    *,
    reason: str = "historical_backlog_caught_up",
    eligible_sessions: int | None = None,
    processed_sessions: int | None = None,
    path: str | Path | None = None,
    request_path: str | Path | None = None,
) -> dict[str, Any]:
    """Close an OPEN historical request when no settleable source rows remain.

    This is not fake evidence progress. It records that the currently available
    historical source cannot satisfy the request and hands control back to the
    research planner to reformulate the question instead of replaying forever or
    leaving the supervisor silently idle.
    """
    req = dict(request or {})
    request_id = str(req.get("request_id") or "")
    if not request_id:
        return {}
    allowed = {str(x).upper() for x in (req.get("allowed_lanes") or [])}
    if "HISTORICAL_REPLAY" not in allowed:
        return {
            "request_id": request_id,
            "status": EA.BLOCKED,
            "reason": "HISTORICAL_REPLAY_NOT_ALLOWED_FOR_REQUEST",
        }

    target_path = Path(path) if path is not None else DEFAULT_PROGRESS_PATH
    store = _read(target_path)
    request_rows = dict(store.get("requests") or {})
    previous = dict(request_rows.get(request_id) or {})
    baseline = max(0, int(req.get("current_samples") or 0))
    acquired = max(0, int(previous.get("samples_acquired") or 0))
    target = max(baseline, int(req.get("target_samples") or baseline))
    current = baseline + acquired
    resolved = sorted({str(x) for x in (previous.get("resolved_metrics") or []) if str(x)})
    missing = {str(x) for x in (req.get("missing_metrics") or []) if str(x)}
    unresolved = sorted(missing - set(resolved))
    message = (
        "no unprocessed fully-settleable historical sessions remain; "
        "research question must be replanned"
    )
    if reason and reason != "historical_backlog_caught_up":
        message = f"{reason}: {message}"

    progress = {
        **previous,
        "request_id": request_id,
        "status": EA.PLATEAUED,
        "reason": message,
        "evidence_origin": str(req.get("evidence_origin") or ""),
        "samples_baseline": baseline,
        "samples_acquired": acquired,
        "sample_count": current,
        "target_samples": target,
        "sample_deficit": max(0, target - current),
        "resolved_metrics": resolved,
        "unresolved_metrics": unresolved,
        "source_exhausted": True,
        "source_exhaustion_reason": str(reason or "historical_backlog_caught_up"),
        "eligible_sessions": None if eligible_sessions is None else int(eligible_sessions),
        "processed_sessions": None if processed_sessions is None else int(processed_sessions),
        "next_action": "REPLAN_RESEARCH_QUESTION",
        "updated_at": _now(),
    }
    request_rows[request_id] = progress
    store["requests"] = request_rows
    store["schema_version"] = SCHEMA_VERSION
    _write(target_path, store)
    _update_request_file(request_id, EA.PLATEAUED, progress, request_path=request_path)
    return progress


def load_progress(
    request_id: str,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_PROGRESS_PATH
    return dict((_read(target).get("requests") or {}).get(str(request_id)) or {})
