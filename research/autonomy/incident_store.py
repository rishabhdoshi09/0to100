"""Durable, deduplicated operational incident dossiers.

Dialogue is an append-only audit stream; repeated identical failures should not
bury the operator in duplicate chatter. This store keeps one current dossier per
incident identity and increments its occurrence count while preserving first/last
seen, job/progress context, recovery action and resource-governor truth.

It is diagnostic only. It never schedules work, changes capabilities or grants
execution authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from research.autonomy import supervisor_state as ST

SCHEMA_VERSION = 1
STATUS_OPEN = "OPEN"
STATUS_RECOVERED = "RECOVERED"


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def _identity(code: str, job_type: str, idempotency_key: str) -> str:
    raw = "|".join((str(code or ""), str(job_type or ""), str(idempotency_key or "")))
    return "inc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _job_context(job: Any) -> dict[str, Any]:
    if job is None:
        return {}
    return {
        "job_id": str(getattr(job, "job_id", "") or ""),
        "job_type": str(getattr(job, "job_type", "") or ""),
        "idempotency_key": str(getattr(job, "idempotency_key", "") or ""),
        "status": str(getattr(job, "status", "") or ""),
        "attempt": int(getattr(job, "attempt", 0) or 0),
        "critical": bool(getattr(job, "critical", False)),
        "scheduled_for": float(getattr(job, "scheduled_for", 0.0) or 0.0),
        "started_at": float(getattr(job, "started_at", 0.0) or 0.0),
        "result_summary": str(getattr(job, "result_summary", "") or ""),
        "error_code": str(getattr(job, "error_code", "") or ""),
        "error_message": str(getattr(job, "error_message", "") or ""),
        "input_snapshot_id": str(getattr(job, "input_snapshot_id", "") or ""),
        "output_snapshot_id": str(getattr(job, "output_snapshot_id", "") or ""),
    }


def _recovery_action(code: str, job: Mapping[str, Any]) -> str:
    key = str(code or "").upper()
    job_type = str(job.get("job_type") or "")
    if key == "CRITICAL_OVERDUE":
        return "Run the newest durable critical intent; remain degraded until it completes or is explicitly superseded."
    if key in {"DATA_REFRESH_WORKER_ERROR", "DATA_REFRESH_WORKER_STUCK"}:
        return "Restart the canonical data-refresh worker through the durable retry policy; keep snapshot freshness fail-closed."
    if key in {"HANDLER_EXCEPTION", "SUPERVISOR_TICK_EXCEPTION"}:
        return "Retry through the durable backoff budget; escalate to permanent failure after the configured attempt ceiling."
    if key == "HISTORICAL_SCHEDULER_ERROR":
        return "Keep forward/current-market lanes independent; retry historical scheduling without advancing its durable cursor."
    if job_type:
        return f"Follow the durable retry/blocking policy for {job_type}; do not mark success until its authoritative completion state is persisted."
    return "Preserve fail-closed capabilities, diagnose the recorded blocker, and retry only through the canonical supervisor path."


def _progress_context(job: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "stage": "",
        "summary": str(job.get("result_summary") or ""),
        "last_error": str(job.get("error_message") or ""),
    }
    if str(job.get("job_type") or "") == "data_refresh":
        try:
            from research.autonomy.data_refresh_parallel import _progress_payload
            p = dict(_progress_payload() or {})
        except Exception:
            p = {}
        if p:
            out.update({
                "stage": str(p.get("stage") or ""),
                "progress_current": int(p.get("progress_current") or 0),
                "progress_total": int(p.get("progress_total") or 0),
                "percent_complete": p.get("percent_complete"),
                "symbols_per_sec": p.get("symbols_per_sec"),
                "last_progress_epoch": p.get("last_progress_epoch"),
                "target_session": str(p.get("target_session") or ""),
                "snapshot_id": str(p.get("snapshot_id") or ""),
                "activation_status": str(p.get("activation_status") or ""),
                "fetch_failures": int(p.get("fetch_failures") or 0),
            })
    return out


class IncidentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        payload = _load(self.path)
        return {
            "schema_version": SCHEMA_VERSION,
            "incidents": dict(payload.get("incidents") or {}),
        }

    def upsert(
        self,
        *,
        code: str,
        message: str,
        job: Any = None,
        activity_truth: Mapping[str, Any] | None = None,
        resource_governor: Mapping[str, Any] | None = None,
        active_failures: list[str] | tuple[str, ...] | set[str] = (),
    ) -> dict[str, Any]:
        store = self.load()
        rows = dict(store.get("incidents") or {})
        job_ctx = _job_context(job)
        incident_id = _identity(
            code,
            str(job_ctx.get("job_type") or ""),
            str(job_ctx.get("idempotency_key") or ""),
        )
        previous = dict(rows.get(incident_id) or {})
        now = ST._now_ist_iso()
        occurrence = int(previous.get("occurrence_count") or 0) + 1
        progress = _progress_context(job_ctx)
        material_signature = hashlib.sha256(
            json.dumps(
                {
                    "message": str(message or ""),
                    "job_status": job_ctx.get("status"),
                    "attempt": job_ctx.get("attempt"),
                    "stage": progress.get("stage"),
                    "progress_current": progress.get("progress_current"),
                    "progress_total": progress.get("progress_total"),
                    "percent_complete": progress.get("percent_complete"),
                    "last_error": progress.get("last_error"),
                    "resource_decision": (resource_governor or {}).get("decision"),
                    "active_failures": sorted(str(x) for x in active_failures),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        changed = bool(
            not previous
            or str(previous.get("material_signature") or "") != material_signature
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "incident_id": incident_id,
            "status": STATUS_OPEN,
            "code": str(code or ""),
            "message": str(message or ""),
            "first_seen_at": str(previous.get("first_seen_at") or now),
            "last_seen_at": now,
            "occurrence_count": occurrence,
            "material_change_count": int(previous.get("material_change_count") or 0) + (1 if changed else 0),
            "material_signature": material_signature,
            "materially_changed": changed,
            "job": job_ctx,
            "progress": progress,
            "activity_truth": dict(activity_truth or {}),
            "resource_governor": dict(resource_governor or {}),
            "active_failures": sorted(str(x) for x in active_failures),
            "recovery_action": _recovery_action(code, job_ctx),
            "live_money_authority_changed": False,
        }
        rows[incident_id] = row
        _atomic(self.path, {"schema_version": SCHEMA_VERSION, "incidents": rows})
        return row

    def recover(self, incident_id: str, *, note: str = "") -> dict[str, Any]:
        store = self.load()
        rows = dict(store.get("incidents") or {})
        row = dict(rows.get(str(incident_id)) or {})
        if not row:
            return {}
        row["status"] = STATUS_RECOVERED
        row["recovered_at"] = ST._now_ist_iso()
        row["recovery_note"] = str(note or "")
        row["materially_changed"] = True
        rows[str(incident_id)] = row
        _atomic(self.path, {"schema_version": SCHEMA_VERSION, "incidents": rows})
        return row

    def recover_for_job(self, job: Any, *, note: str = "") -> list[dict[str, Any]]:
        """Close open dossiers tied to the exact durable job identity."""
        job_ctx = _job_context(job)
        job_id = str(job_ctx.get("job_id") or "")
        job_type = str(job_ctx.get("job_type") or "")
        key = str(job_ctx.get("idempotency_key") or "")
        if not (job_id or (job_type and key)):
            return []

        store = self.load()
        rows = dict(store.get("incidents") or {})
        recovered: list[dict[str, Any]] = []
        now = ST._now_ist_iso()
        changed = False
        for incident_id, raw in list(rows.items()):
            row = dict(raw or {})
            if str(row.get("status") or "") != STATUS_OPEN:
                continue
            linked = dict(row.get("job") or {})
            same = bool(
                (job_id and str(linked.get("job_id") or "") == job_id)
                or (
                    job_type
                    and key
                    and str(linked.get("job_type") or "") == job_type
                    and str(linked.get("idempotency_key") or "") == key
                )
            )
            if not same:
                continue
            row["status"] = STATUS_RECOVERED
            row["recovered_at"] = now
            row["recovery_note"] = str(note or "authoritative job completed successfully")
            row["materially_changed"] = True
            rows[incident_id] = row
            recovered.append(row)
            changed = True

        if changed:
            _atomic(self.path, {"schema_version": SCHEMA_VERSION, "incidents": rows})
        return recovered

    def recent(self, limit: int = 20, *, open_only: bool = False) -> list[dict[str, Any]]:
        rows = list(self.load().get("incidents", {}).values())
        if open_only:
            rows = [row for row in rows if str(row.get("status") or "") == STATUS_OPEN]
        rows.sort(key=lambda row: str(row.get("last_seen_at") or ""), reverse=True)
        return rows[: max(0, int(limit))]
