"""Durable evidence-acquisition directives for autonomous QuantTerm research.

The Research Director must never answer RETEST_WITH_MORE_DATA without saying what
evidence is missing, how much is missing, which lane is allowed to satisfy it,
and when the request is complete. Historical replay can satisfy research/history
requests, but it can never satisfy a FORWARD_PAPER requirement.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
OPEN = "OPEN"
SATISFIED = "SATISFIED"
PLATEAUED = "PLATEAUED"
BLOCKED = "BLOCKED"
_TERMINAL_STATUSES = frozenset({SATISFIED, PLATEAUED, BLOCKED})

DEFAULT_REQUEST_PATH = logs_path("research", "evidence_request.json")

_REQUIRED_RESEARCH_METRICS = (
    "deflated_sharpe",
    "reality_check_p",
    "walk_forward_ok",
    "fdr_significant",
    "benchmark_available",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


@dataclass(frozen=True)
class EvidenceRequest:
    request_id: str
    session_date: str
    strategy_id: str
    gap_kind: str
    objective: str
    evidence_origin: str
    allowed_lanes: tuple[str, ...]
    current_samples: int
    target_samples: int
    sample_deficit: int
    missing_metrics: tuple[str, ...]
    acquisition_tasks: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    priority: float
    thesis_hash: str = ""
    status: str = OPEN
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def missing_required_metrics(context: Mapping[str, Any] | None) -> tuple[str, ...]:
    ctx = dict(context or {})
    raw = dict(ctx.get("raw") or {})
    missing: list[str] = []
    for key in _REQUIRED_RESEARCH_METRICS:
        if key not in raw or raw.get(key) is None:
            missing.append(key)
    return tuple(missing)


def _allowed_lanes(origin: str, gap_kind: str) -> tuple[str, ...]:
    origin = str(origin or "").upper()
    if origin == "FORWARD_PAPER":
        return ("FORWARD_PAPER",)
    if origin in {"HISTORICAL_REPLAY", "RESEARCH_VALIDATION"}:
        return ("HISTORICAL_REPLAY",)
    if gap_kind in {
        "data_insufficiency",
        "missing_universe_history",
        "missing_corporate_actions",
        "missing_benchmark",
        "unsupported_runtime_family",
    }:
        return ("DATA_REFRESH",)
    return ("HISTORICAL_REPLAY",)


def _tasks(sample_deficit: int, missing: Sequence[str], allowed_lanes: Sequence[str]) -> tuple[str, ...]:
    tasks: list[str] = []
    if sample_deficit > 0:
        lane = str((list(allowed_lanes) or ["RESEARCH"])[0]).upper()
        tasks.append(f"ACQUIRE_{lane}_SAMPLES")
    mapping = {
        "deflated_sharpe": "COMPUTE_DEFLATED_SHARPE",
        "reality_check_p": "RUN_REALITY_CHECK",
        "walk_forward_ok": "RUN_WALK_FORWARD_VALIDATION",
        "fdr_significant": "RUN_FDR_CONTROL",
        "benchmark_available": "RESTORE_BENCHMARK_DATA",
    }
    for metric in missing:
        task = mapping.get(str(metric), f"ACQUIRE_{str(metric).upper()}")
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        tasks.append("ACQUIRE_REQUIRED_EVIDENCE")
    return tuple(tasks)


def build_request(
    *,
    session_date: str,
    strategy_id: str,
    gap_kind: str,
    diagnosis: str,
    evidence_origin: str,
    current_samples: int = 0,
    target_samples: int = 30,
    missing_metrics: Sequence[str] = (),
    priority: float = 0.0,
    thesis_hash: str = "",
) -> EvidenceRequest:
    current = max(0, int(current_samples or 0))
    target = max(current, int(target_samples or 0))
    deficit = max(0, target - current)
    missing = tuple(sorted({str(x) for x in missing_metrics if str(x)}))
    lanes = _allowed_lanes(evidence_origin, gap_kind)
    tasks = _tasks(deficit, missing, lanes)
    objective = str(diagnosis or "").strip() or (
        f"Close evidence gap {gap_kind} for {strategy_id or 'system'}"
    )
    stable = {
        "strategy_id": str(strategy_id or ""),
        "gap_kind": str(gap_kind or ""),
        "evidence_origin": str(evidence_origin or ""),
        "target_samples": target,
        "missing_metrics": list(missing),
        "thesis_hash": str(thesis_hash or ""),
    }
    request_id = "evreq_" + hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    stop = [f"sample_size>={target}" if deficit > 0 else "sample_target_already_met"]
    stop.extend(f"metric_available:{metric}" for metric in missing)
    return EvidenceRequest(
        request_id=request_id,
        session_date=str(session_date or "")[:10],
        strategy_id=str(strategy_id or ""),
        gap_kind=str(gap_kind or ""),
        objective=objective,
        evidence_origin=str(evidence_origin or "RESEARCH_VALIDATION").upper(),
        allowed_lanes=tuple(lanes),
        current_samples=current,
        target_samples=target,
        sample_deficit=deficit,
        missing_metrics=missing,
        acquisition_tasks=tasks,
        stop_conditions=tuple(stop),
        priority=round(float(priority or 0.0), 6),
        thesis_hash=str(thesis_hash or ""),
        status=OPEN,
        created_at=_now(),
    )


def save_request(request: EvidenceRequest | Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_REQUEST_PATH
    payload = request.as_dict() if isinstance(request, EvidenceRequest) else dict(request)
    payload["schema_version"] = SCHEMA_VERSION

    # A deterministic evidence request is one investigation, not a recurring
    # scheduler trigger. Once that exact request reaches a terminal state, a
    # later diagnostic pass must not resurrect it merely by rebuilding the same
    # OPEN request. Reopening would erase the stopping decision while retaining
    # the same request_id, so the bounded replan idempotency key would already
    # be spent and the supervisor could sit healthy but permanently idle.
    #
    # A materially different question (including a changed thesis, gap, target,
    # required metrics, strategy, or evidence origin) hashes to a different
    # request_id and is therefore free to open normally. Explicit terminal-state
    # updates to the same request are also still allowed.
    existing = load_request(target)
    existing_status = str(existing.get("status") or "").upper()
    incoming_status = str(payload.get("status") or OPEN).upper()
    same_request = bool(existing.get("request_id")) and (
        str(existing.get("request_id") or "") == str(payload.get("request_id") or "")
    )
    if same_request and existing_status in _TERMINAL_STATUSES and incoming_status == OPEN:
        return existing

    _atomic_json(target, payload)
    return payload


def load_request(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_REQUEST_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def open_request_for_lane(lane: str, *, path: str | Path | None = None) -> dict[str, Any]:
    payload = load_request(path)
    if str(payload.get("status") or "").upper() != OPEN:
        return {}
    allowed = {str(x).upper() for x in (payload.get("allowed_lanes") or [])}
    return payload if str(lane or "").upper() in allowed else {}


def request_from_gap(
    gap,
    *,
    session_date: str,
    context: Mapping[str, Any] | None = None,
    thesis_hash: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    ctx = dict(context or {})
    origin = str(
        ctx.get("evidence_origin")
        or getattr(gap, "evidence_origin", "")
        or "RESEARCH_VALIDATION"
    ).upper()
    current = int(
        ctx.get("n_trades")
        if ctx.get("n_trades") is not None
        else getattr(gap, "current_samples", 0)
        or 0
    )
    target = int(getattr(gap, "target_samples", 0) or 0)
    if target <= 0:
        target = max(30, current)
    missing = (
        missing_required_metrics(ctx)
        if ctx and origin == "RESEARCH_VALIDATION"
        else ()
    )
    req = build_request(
        session_date=session_date,
        strategy_id=str(getattr(gap, "strategy_id", "") or ""),
        gap_kind=str(getattr(gap, "kind", "") or ""),
        diagnosis=str(getattr(gap, "diagnosis", "") or ""),
        evidence_origin=origin,
        current_samples=current,
        target_samples=target,
        missing_metrics=missing,
        priority=float(getattr(gap, "priority", 0.0) or 0.0),
        thesis_hash=thesis_hash,
    )
    return save_request(req, path=path)
