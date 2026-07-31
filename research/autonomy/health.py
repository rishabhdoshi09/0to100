"""
🩺 Capability matrix + read-only status snapshot.

There is no single green/red flag. A subsystem failure reduces exactly the capabilities that depend
on it and nothing else — a non-critical failure never halts the organisation. The status snapshot is
strictly read-only: the retail UI calls it to see what the organisation is doing WITHOUT ever starting
the supervisor.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

# failure codes
AUTH_MISSING = "auth_missing"
AUTH_EXPIRED = "auth_expired"
PROVIDER_UNAVAILABLE = "provider_unavailable"
SNAPSHOT_STALE = "snapshot_stale"
NEWS_UNAVAILABLE = "news_unavailable"
CA_INCOMPLETE = "corporate_actions_incomplete"
LIVE_FEED_STALE = "live_feed_stale"
EVENT_STORE_FAILURE = "event_store_failure"
RISK_GOVERNOR_UNHEALTHY = "risk_governor_unhealthy"
UNRECONCILED = "unreconciled_state"
UNIVERSE_INCOMPLETE = "universe_history_incomplete"
LEARNING_FAILED = "learning_failed"
OWNER_PAUSED = "owner_paused"

# capability levels
ALLOWED = "allowed"
LIMITED = "limited"
BLOCKED = "blocked"
READ_ONLY = "read_only"

# which failures block / limit NEW paper entries
_ENTRY_BLOCK = {AUTH_MISSING, AUTH_EXPIRED, PROVIDER_UNAVAILABLE, SNAPSHOT_STALE,
                EVENT_STORE_FAILURE, RISK_GOVERNOR_UNHEALTHY, UNRECONCILED, OWNER_PAUSED}
_ENTRY_LIMIT = {CA_INCOMPLETE, LIVE_FEED_STALE}
# which failures limit existing-position management (exits are almost never fully blocked)
_EXIT_LIMIT = {SNAPSHOT_STALE, LIVE_FEED_STALE, RISK_GOVERNOR_UNHEALTHY, UNRECONCILED, EVENT_STORE_FAILURE}
# which failures block / limit research
_RESEARCH_BLOCK = {EVENT_STORE_FAILURE}
_RESEARCH_LIMIT = {NEWS_UNAVAILABLE, CA_INCOMPLETE, UNIVERSE_INCOMPLETE, LEARNING_FAILED}


def capabilities(active_failures) -> dict:
    """Most-restrictive-wins capability matrix. Returns a plain dict the UI can render."""
    f = set(active_failures or ())
    new_entries = BLOCKED if (f & _ENTRY_BLOCK) else (LIMITED if (f & _ENTRY_LIMIT) else ALLOWED)
    exits = LIMITED if (f & _EXIT_LIMIT) else ALLOWED           # never fully blocked unless HALTED
    research = BLOCKED if (f & _RESEARCH_BLOCK) else (LIMITED if (f & _RESEARCH_LIMIT) else ALLOWED)
    ui = READ_ONLY if (EVENT_STORE_FAILURE in f) else ALLOWED
    notes = []
    if AUTH_MISSING in f:
        notes.append("Zerodha login required — new entries paused; safe exits continue.")
    if AUTH_EXPIRED in f:
        notes.append("Zerodha session expired — re-login required; historical research remains available.")
    if PROVIDER_UNAVAILABLE in f:
        notes.append("Market-data provider unavailable — new entries paused until current data is trustworthy.")
    if SNAPSHOT_STALE in f:
        notes.append("Market data stale — new entries paused; positions managed if prices are trustworthy.")
    if NEWS_UNAVAILABLE in f:
        notes.append("News feed down — trading unaffected; news-dependent studies paused.")
    if CA_INCOMPLETE in f:
        notes.append("Corporate-action coverage incomplete — affected historical strategies/tests paused.")
    if LIVE_FEED_STALE in f:
        notes.append("Live prices stale for some symbols — those entries blocked; risk-reducing exits continue.")
    if EVENT_STORE_FAILURE in f:
        notes.append("Record store unavailable — new mutations blocked; UI is read-only.")
    if RISK_GOVERNOR_UNHEALTHY in f:
        notes.append("Safety checks unhealthy — new entries blocked; risk reduction only.")
    if UNRECONCILED in f:
        notes.append("Records need a reconciliation pass before new risk.")
    if UNIVERSE_INCOMPLETE in f:
        notes.append("Point-in-time universe history is incomplete — PIT-dependent research remains blocked.")
    if LEARNING_FAILED in f:
        notes.append("Latest learning cycle failed — existing approved paper state continues; new promotion is blocked.")
    if OWNER_PAUSED in f:
        notes.append("Owner paused new paper entries; position management continues.")
    return {"new_paper_entries": new_entries, "existing_exits": exits, "research": research,
            "ui": ui, "active_failures": sorted(f), "notes": notes}


def _fresh(payload: dict, *, max_age_s: float = 90.0) -> bool:
    heartbeat = str(payload.get("heartbeat_ist") or "")
    if not heartbeat:
        return False
    try:
        stamped = datetime.fromisoformat(heartbeat)
        now = datetime.now(tz=stamped.tzinfo) if stamped.tzinfo else datetime.now()
        age = now - stamped
        return timedelta(0) <= age <= timedelta(seconds=max_age_s)
    except Exception:
        return False


def read_status(*, state_path, jobs_db=None, dialogue_path=None) -> dict:
    """Read-only status for the UI. Never starts the supervisor; tolerates missing files.

    ``status.json`` remains the durable state/capability snapshot. ``runtime.json`` is a deliberately
    tiny liveness pulse written by the console driver, so a long scan or data refresh does not make a
    healthy process look offline merely because the durable status snapshot is temporarily unchanged.
    """
    state_path = Path(state_path)
    out = {"supervisor_running": False, "state": "UNKNOWN", "explanation": "", "updated_ist": "",
           "heartbeat_ist": "", "snapshot_id": "", "recent_transitions": [], "jobs": {},
           "recent_dialogue": [], "new_risk_permitted": False, "positions_manageable": True,
           "active_failures": [], "owner_state": {}, "scheduler_of_record": "", "last_cycle": {},
           "scheduler_owner_pid": None, "active_job": {}}
    durable: dict = {}
    try:
        durable = json.loads(state_path.read_text(encoding="utf-8"))
        out.update({"state": durable.get("state", "UNKNOWN"),
                    "explanation": durable.get("explanation", ""),
                    "updated_ist": durable.get("updated_ist", ""),
                    "snapshot_id": durable.get("snapshot_id", ""),
                    "heartbeat_ist": durable.get("heartbeat_ist", durable.get("updated_ist", "")),
                    "new_risk_permitted": bool(durable.get("new_risk_permitted", False)),
                    "positions_manageable": bool(durable.get("positions_manageable", True)),
                    "active_failures": list(durable.get("active_failures", [])),
                    "owner_state": dict(durable.get("owner_state", {})),
                    "scheduler_of_record": str(durable.get("scheduler_of_record", "")),
                    "last_cycle": dict(durable.get("last_cycle", {}) or {}),
                    "scheduler_owner_pid": durable.get("scheduler_owner_pid"),
                    "recent_transitions": list(durable.get("history", []))[-6:]})
    except Exception:
        durable = {}

    runtime_path = state_path.parent / "runtime.json"
    runtime: dict = {}
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        runtime = {}

    # New console versions make runtime.json authoritative for process liveness. Older deployments
    # without that file retain the previous status.json freshness behaviour.
    if runtime:
        runtime_fresh = _fresh(runtime)
        out["supervisor_running"] = bool(runtime.get("process_running", False)) and runtime_fresh
        if runtime.get("heartbeat_ist"):
            out["heartbeat_ist"] = str(runtime.get("heartbeat_ist"))
        if runtime.get("scheduler_owner_pid") is not None:
            out["scheduler_owner_pid"] = runtime.get("scheduler_owner_pid")
        out["active_job"] = dict(runtime.get("active_job", {}) or {})
    else:
        out["supervisor_running"] = bool(durable.get("process_running", True)) and _fresh(durable)

    if jobs_db is not None:
        try:
            from research.autonomy.job_store import JobStore
            js = JobStore(jobs_db)
            counts: dict = {}
            for j in js.list(limit=500):
                counts[j.status] = counts.get(j.status, 0) + 1
            out["jobs"] = counts
            js.close()
        except Exception:
            pass
    if dialogue_path is not None:
        try:
            from research.autonomy.dialogue import DialogueLog
            out["recent_dialogue"] = [
                {"type": r.get("record_type"), "producer": r.get("producer"),
                 "claim": r.get("claim"), "decision": r.get("decision"), "id": r.get("record_id")}
                for r in DialogueLog(dialogue_path).recent(8)]
        except Exception:
            pass
    return out
