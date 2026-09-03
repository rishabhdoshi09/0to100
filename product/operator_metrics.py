"""Scoped operator-intervention metrics. Never rewrite historical rows."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from product import readiness as RDY

ROOT = Path(__file__).resolve().parents[1]
OPS_DB = ROOT / "logs" / "market_ops" / "jobs.db"
AUTO_DB = ROOT / "logs" / "autonomy" / "jobs.db"
RUNTIME_PATH = ROOT / "logs" / "autonomy" / "runtime.json"

AUTOMATED_BY = frozenset(
    {"pipeline", "bootstrap", "autonomy", "autonomous_loop", "market_scan", "desk_pipeline"}
)
HUMAN_BY = frozenset({"terminal", "user", "product_bootstrap"})
NECESSARY_HUMAN_KINDS = frozenset({"KITE_LOGIN", "AUTH_HEALTH", "auth_health"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_started_s() -> float | None:
    try:
        import json

        payload = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        for key in ("started_at_s", "started_epoch", "started_at"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and value > 1_000_000:
                return float(value)
            if isinstance(value, str) and value:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _classify(requested_by: str, kind: str) -> str:
    by = str(requested_by or "")
    job = str(kind or "")
    if job in NECESSARY_HUMAN_KINDS or "kite" in job.lower():
        return "NECESSARY_HUMAN_ACTION"
    if by in HUMAN_BY:
        if "override" in job.lower():
            return "MANUAL_OVERRIDE"
        return "AVOIDABLE_HUMAN_ACTION"
    if "recover" in job.lower() or "retry" in job.lower():
        return "AUTOMATED_RECOVERY"
    if by in AUTOMATED_BY or not by:
        return "AUTOMATED_JOB"
    return "AUTOMATED_JOB"


def _ops_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if OPS_DB.exists():
        con = sqlite3.connect(str(OPS_DB))
        con.row_factory = sqlite3.Row
        for row in con.execute(
            "SELECT kind, requested_by, requested_at, status FROM operations"
        ):
            rows.append(
                {
                    "source": "market_ops",
                    "kind": row["kind"],
                    "requested_by": row["requested_by"],
                    "at": float(row["requested_at"] or 0),
                    "status": row["status"],
                    "class": _classify(row["requested_by"], row["kind"]),
                }
            )
        con.close()
    if AUTO_DB.exists():
        con = sqlite3.connect(str(AUTO_DB))
        con.row_factory = sqlite3.Row
        cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()}
        at_col = "created_at" if "created_at" in cols else (
            "requested_at" if "requested_at" in cols else "enqueued_at"
        )
        if at_col in cols:
            q = f"SELECT job_type, status, {at_col} AS at FROM jobs"
        else:
            q = "SELECT job_type, status, 0 AS at FROM jobs"
        for row in con.execute(q):
            at = row["at"]
            try:
                at_f = float(at or 0)
            except (TypeError, ValueError):
                try:
                    at_f = datetime.fromisoformat(str(at).replace("Z", "+00:00")).timestamp()
                except Exception:
                    at_f = 0.0
            rows.append(
                {
                    "source": "autonomy",
                    "kind": row["job_type"],
                    "requested_by": "autonomy",
                    "at": at_f,
                    "status": row["status"],
                    "class": _classify("autonomy", row["job_type"]),
                }
            )
        con.close()
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    for row in rows:
        key = str(row.get("class") or "AUTOMATED_JOB")
        by_class[key] = by_class.get(key, 0) + 1
    automated = (
        by_class.get("AUTOMATED_JOB", 0)
        + by_class.get("AUTOMATED_RECOVERY", 0)
    )
    human = (
        by_class.get("NECESSARY_HUMAN_ACTION", 0)
        + by_class.get("AVOIDABLE_HUMAN_ACTION", 0)
        + by_class.get("MANUAL_OVERRIDE", 0)
    )
    total = automated + human
    necessary = by_class.get("NECESSARY_HUMAN_ACTION", 0)
    soak_denom = automated + by_class.get("AVOIDABLE_HUMAN_ACTION", 0) + by_class.get(
        "MANUAL_OVERRIDE", 0
    )
    return {
        "automated_jobs": automated,
        "human_required_actions": human,
        "necessary_human_actions": necessary,
        "avoidable_human_actions": by_class.get("AVOIDABLE_HUMAN_ACTION", 0),
        "manual_overrides": by_class.get("MANUAL_OVERRIDE", 0),
        "automated_recoveries": by_class.get("AUTOMATED_RECOVERY", 0),
        "classes": by_class,
        "automation_rate": (automated / total) if total else 1.0,
        "no_kite_soak_automation_rate": (automated / soak_denom) if soak_denom else 1.0,
        "n": len(rows),
    }


def build_operator_metrics(*, session: str = "") -> dict[str, Any]:
    rows = _ops_rows()
    now = time.time()
    run_started = _run_started_s()
    session_start = None
    if session:
        try:
            session_start = datetime.fromisoformat(str(session)[:10]).replace(
                tzinfo=timezone.utc
            ).timestamp()
        except Exception:
            session_start = None
    current_run = [r for r in rows if run_started and r["at"] >= run_started]
    current_session = [r for r in rows if session_start and r["at"] >= session_start]
    last_7d = [r for r in rows if r["at"] >= now - 7 * 86400]
    broker = RDY.broker_live()
    kite_needed = not bool(broker.get("ready"))
    return {
        "schema_version": 2,
        "generated_at": _now(),
        "do_not_rewrite_history": True,
        "scopes": {
            "CURRENT_AUTONOMOUS_RUN": {
                **_summarize(current_run),
                "scope": "CURRENT_AUTONOMOUS_RUN",
                "since": run_started,
            },
            "CURRENT_SESSION": {
                **_summarize(current_session),
                "scope": "CURRENT_SESSION",
                "session": session,
            },
            "LAST_7_DAYS": {**_summarize(last_7d), "scope": "LAST_7_DAYS"},
            "ALL_TIME": {**_summarize(rows), "scope": "ALL_TIME"},
        },
        # Compact fields for existing desk consumers; scoped to CURRENT RUN when known.
        **(
            _summarize(current_run)
            if current_run
            else _summarize(current_session) if current_session else _summarize(last_7d)
        ),
        "necessary_human": ["Kite authentication"] if kite_needed else [],
        "kite_needed_for_paper_entry_only": kite_needed,
    }
