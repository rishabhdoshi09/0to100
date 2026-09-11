"""Priority-aware desk data pipeline using the market-ops isolated lanes.

The desk still respects dependency order: official prices before the market
scan, then long-term/news/research work. Secondary work is deliberately
serialized for resource control, but it must never starve critical recovery.
If DATA_PREPARE or MARKET_SCAN becomes due while a secondary lane is active,
the recovery job may be queued into its own isolated lane immediately.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from operations.market_ops import (
    DATA_PREPARE,
    DUE_DILIGENCE_ACQUIRE,
    FNO_FRESH_S,
    FNO_REFRESH,
    LANES,
    LONG_TERM_FRESH_S,
    LONG_TERM_REFRESH,
    LONG_TERM_SCAN,
    MARKET_SCAN,
    NEWS_FRESH_S,
    NEWS_REFRESH,
    SCAN_FRESH_S,
    _stale,
)
from operations.store import BLOCKED, FAILED, PENDING, RUNNING, SUCCEEDED, OperationStore
from core.runtime_paths import logs_dir, logs_path

RETRY_AFTER_FAIL_S = 10 * 60
SNAPSHOT_STALE_S = 90.0
SNAPSHOT_UNKNOWN = "UNKNOWN"
SNAPSHOT_STALE = "STALE"
SNAPSHOT_CURRENT = "CURRENT"


def _snapshot_path() -> Path:
    raw = os.environ.get("QT_DESK_PIPELINE_SNAPSHOT")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[1] / "logs" / "product" / "desk_pipeline.json"


# Dependency/viewing order: Home → Scanner/Recos technical → funds → Reports → research.
DESK_STEPS: tuple[dict[str, str], ...] = (
    {
        "id": "prices",
        "title": "Official prices",
        "page": "Home",
        "why": "Download bhavcopy history so charts and the market scan have bars.",
    },
    {
        "id": "scan",
        "title": "Market scan",
        "page": "Home",
        "why": "One whole-market scan for Home, Scanner, Recommendations, and long-term.",
    },
    {
        "id": "long_term",
        "title": "Long-term / funds",
        "page": "Recommendations",
        "why": "Optional Screener refresh for Wealth Builders. Scan Now already overlays cached funds.",
    },
    {
        "id": "news",
        "title": "Market reports",
        "page": "Market Reports",
        "why": "Street pulse and news for Market Reports.",
    },
    {
        "id": "investigate",
        "title": "Investigate acquire",
        "page": "Stock Intelligence",
        "why": "Download missing or stale evidence for shortlisted names. Failed providers cool down before retrying.",
    },
)

PIPELINE_KINDS = frozenset(
    {
        DATA_PREPARE,
        FNO_REFRESH,
        MARKET_SCAN,
        LONG_TERM_REFRESH,
        LONG_TERM_SCAN,
        NEWS_REFRESH,
        DUE_DILIGENCE_ACQUIRE,
    }
)

# These restore the desk's authoritative market state. They may leapfrog an
# already-running secondary lane, but never overlap another recovery operation.
CRITICAL_RECOVERY_KINDS = frozenset({DATA_PREPARE, MARKET_SCAN})


def _root():
    from operations import market_ops as MO

    return MO.ROOT


def prices_kind_due() -> str | None:
    """DATA_PREPARE if history is thin/stale, else FNO_REFRESH if its file is stale."""
    try:
        from data.bhavcopy_runtime import official_history_freshness

        freshness = official_history_freshness(load_cache=True)
    except Exception:
        freshness = {"current": False, "ready": False, "sessions": 0}
    if not freshness.get("current"):
        return DATA_PREPARE
    if _stale(logs_dir() / "product" / "fno_universe.json", FNO_FRESH_S):
        return FNO_REFRESH
    return None


def scan_is_fresh() -> bool:
    path = logs_dir() / "product" / "latest_momentum_scan.json"
    try:
        from data.bhavcopy_runtime import official_history_freshness

        freshness = official_history_freshness(load_cache=True)
        if not freshness.get("current"):
            return False
        expected = str(freshness.get("expected_latest_completed_session") or "")
    except Exception:
        expected = ""
    try:
        from product.scan_store import load_scan, scan_artifact_is_fresh

        payload = load_scan(path)
        if not payload:
            return False
        as_of = str(payload.get("as_of_session") or payload.get("history_latest_date") or "")[:10]
        if expected and (not as_of or as_of < expected):
            return False
        if expected and as_of >= expected:
            # Session identity is authoritative. Worker restarts must not force a
            # rescan solely because wall-clock age crossed SCAN_FRESH_S.
            return True
        if payload.get("scanned_at"):
            return bool(scan_artifact_is_fresh(path, max_age_s=SCAN_FRESH_S))
    except Exception:
        return False
    return not _stale(path, SCAN_FRESH_S)


def long_term_is_fresh() -> bool:
    return not _stale(logs_dir() / "product" / "latest_long_term_scan.json", LONG_TERM_FRESH_S)


def news_is_fresh() -> bool:
    return not _stale(logs_dir() / "news_curator.sqlite3", NEWS_FRESH_S)


def acquire_freshness() -> dict[str, Any]:
    """Dataset-level research truth. A recent attempt alone is never fresh."""
    try:
        from product.due_diligence.freshness import research_freshness

        return dict(research_freshness() or {})
    except Exception as exc:
        return {
            "fresh": False,
            "retry_due": False,
            "state": "CHECK_FAILED",
            "reason": f"Research freshness check failed: {type(exc).__name__}: {exc}"[:240],
            "unresolved_symbols": [],
            "unresolved_datasets": [],
            "next_retry_at": None,
        }


def acquire_is_fresh() -> bool:
    """Compatibility bool for callers that only need current/not-current."""
    return bool(acquire_freshness().get("fresh"))


def _fresh_s(step_id: str) -> float:
    if step_id == "prices":
        return FNO_FRESH_S
    if step_id == "scan":
        return SCAN_FRESH_S
    if step_id == "long_term":
        return LONG_TERM_FRESH_S
    if step_id == "news":
        return NEWS_FRESH_S
    return 0.0


def _recently_succeeded(store: OperationStore, kinds: set[str], max_age_s: float) -> bool:
    for kind in kinds:
        latest = store.latest(kind)
        if not latest or str(latest.get("status") or "") != SUCCEEDED:
            continue
        try:
            age = time.time() - float(latest.get("updated_at") or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= age < max_age_s:
            return True
    return False


def _kind_for_step(
    step_id: str,
    store: OperationStore | None = None,
    *,
    research: dict[str, Any] | None = None,
) -> str | None:
    if step_id == "prices":
        kind = prices_kind_due()
    elif step_id == "scan":
        kind = None if scan_is_fresh() else MARKET_SCAN
    elif step_id == "long_term":
        kind = None if long_term_is_fresh() else LONG_TERM_REFRESH
    elif step_id == "news":
        kind = None if news_is_fresh() else NEWS_REFRESH
    elif step_id == "investigate":
        state = research if research is not None else acquire_freshness()
        kind = DUE_DILIGENCE_ACQUIRE if (not state.get("fresh") and state.get("retry_due")) else None
    else:
        kind = None
    if (
        kind
        and store is not None
        and step_id != "investigate"
        and _recently_succeeded(store, _kinds_for_id(step_id), _fresh_s(step_id))
    ):
        return None
    return kind


def _recently_failed(store: OperationStore, kind: str) -> bool:
    latest = store.latest(kind)
    if not latest or str(latest.get("status") or "") not in {FAILED, BLOCKED}:
        return False
    try:
        age = time.time() - float(latest.get("updated_at") or 0)
    except (TypeError, ValueError):
        return False
    return 0 <= age < RETRY_AFTER_FAIL_S


def _pipeline_active_items(store: OperationStore) -> list[dict[str, Any]]:
    return [
        item for item in store.active()
        if str(item.get("kind") or "") in PIPELINE_KINDS
    ]


def _primary_active(store: OperationStore) -> dict[str, Any] | None:
    active = _pipeline_active_items(store)
    if not active:
        return None
    for spec in DESK_STEPS:
        kinds = _kinds_for_id(spec["id"])
        match = next((item for item in active if str(item.get("kind") or "") in kinds), None)
        if match:
            return match
    return active[0]


def _pipeline_active(store: OperationStore) -> dict[str, Any] | None:
    """Compatibility helper: return the highest-priority active pipeline item."""
    return _primary_active(store)


def _step_from_kind(kind: str) -> dict[str, str] | None:
    if kind in {DATA_PREPARE, FNO_REFRESH}:
        return dict(DESK_STEPS[0])
    if kind == MARKET_SCAN:
        return dict(DESK_STEPS[1])
    if kind in {LONG_TERM_REFRESH, LONG_TERM_SCAN}:
        return dict(DESK_STEPS[2])
    if kind == NEWS_REFRESH:
        return dict(DESK_STEPS[3])
    if kind == DUE_DILIGENCE_ACQUIRE:
        return dict(DESK_STEPS[4])
    return None


def persist_desk_pipeline_snapshot(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Worker-side write. GET never computes; it only reads this file."""
    target = path or _snapshot_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(payload)
    blob["persisted_at"] = datetime.now(timezone.utc).isoformat()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, default=str), encoding="utf-8")
    tmp.replace(target)
    return target


def _missing_status(*, reason: str) -> dict[str, Any]:
    steps = [
        {**spec, "kind": None, "state": "unknown", "latest_status": None}
        for spec in DESK_STEPS
    ]
    return {
        "sequential": True,
        "critical_recovery_parallel": True,
        "queued_kind": None,
        "queued_created": False,
        "current": None,
        "steps": steps,
        "message": reason,
        "page": "",
        "scan_reused": False,
        "operations": [],
        "active_kind": None,
        "active_kinds": [],
        "research_freshness": None,
        "status_source": "missing",
        "freshness": SNAPSHOT_UNKNOWN,
        "generated_at": None,
        "age_seconds": None,
    }


def load_desk_pipeline_status(path: Path | None = None) -> dict[str, Any]:
    """GET path: read the persisted snapshot only. Never inspect coverage."""
    target = path or _snapshot_path()
    if not target.exists():
        return _missing_status(reason="Desk pipeline status has not been published yet. Workers write this file.")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _missing_status(reason="Persisted desk-pipeline snapshot is unreadable.")
    if not isinstance(payload, dict):
        return _missing_status(reason="Persisted desk-pipeline snapshot is not an object.")
    stamp = str(payload.get("persisted_at") or payload.get("generated_at") or "")
    age = None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = max(0.0, time.time() - parsed.timestamp())
    except (TypeError, ValueError):
        age = None
    out = dict(payload)
    out["age_seconds"] = None if age is None else round(age, 3)
    out["status_source"] = "persisted"
    if age is None or age > SNAPSHOT_STALE_S:
        out["freshness"] = SNAPSHOT_STALE if age is not None else SNAPSHOT_UNKNOWN
        note = "Persisted desk status is stale; workers will refresh it. This GET did not recompute coverage."
        prior = str(out.get("message") or "").strip()
        out["message"] = f"{prior} {note}".strip() if prior else note
    else:
        out["freshness"] = SNAPSHOT_CURRENT
    return out


def describe_desk_pipeline(
    store: OperationStore | None = None,
    *,
    persist: bool = False,
    research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Worker/compute snapshot. GET /api/desk-pipeline uses load_desk_pipeline_status."""
    ops = store or OperationStore()
    snapshot_research = research if research is not None else acquire_freshness()
    return _snapshot(
        ops,
        queued_kind=None,
        queued_op=None,
        created=False,
        research=snapshot_research,
        persist=persist,
    )


def refresh_desk_pipeline_snapshot(store: OperationStore | None = None) -> dict[str, Any]:
    """Worker cycle: compute once, persist, return the same object."""
    return describe_desk_pipeline(store, persist=True)


def _next_due_step(
    store: OperationStore,
    *,
    research: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None, str | None]:
    """Return (step, kind, halted). Failed non-price steps cool down and are skipped."""
    for step in DESK_STEPS:
        kind = _kind_for_step(step["id"], store, research=research)
        if not kind:
            continue
        if _recently_failed(store, kind):
            if step["id"] == "prices":
                return step, None, "prices"
            continue
        return step, kind, None
    return None, None, None


def advance_desk_pipeline(
    store: OperationStore | None = None,
    *,
    requested_by: str = "desk_pipeline",
) -> dict[str, Any]:
    """Queue the next due step, letting critical recovery bypass secondary work."""
    ops = store or OperationStore()
    research = acquire_freshness()
    step, kind, halted = _next_due_step(ops, research=research)
    active_items = _pipeline_active_items(ops)
    primary_active = _primary_active(ops)
    active_kinds = {str(item.get("kind") or "") for item in active_items}

    if halted:
        return _snapshot(
            ops,
            queued_kind=None,
            queued_op=primary_active,
            created=False,
            halted=halted,
            research=research,
            persist=True,
        )

    if active_items:
        # market_ops owns isolated lanes. A secondary provider/download must not
        # starve market truth recovery, but DATA_PREPARE and MARKET_SCAN still
        # serialize against each other to preserve their dependency ordering.
        recovery_active = bool(active_kinds & CRITICAL_RECOVERY_KINDS)
        if kind in CRITICAL_RECOVERY_KINDS and not recovery_active:
            item, created = ops.enqueue(
                kind,
                lane=LANES[kind],
                requested_by=requested_by,
            )
            return _snapshot(
                ops,
                queued_kind=kind,
                queued_op=item,
                created=created,
                research=research,
                persist=True,
            )
        return _snapshot(
            ops,
            queued_kind=None,
            queued_op=primary_active,
            created=False,
            research=research,
            persist=True,
        )

    if kind:
        item, created = ops.enqueue(
            kind,
            lane=LANES[kind],
            requested_by=requested_by,
        )
        return _snapshot(
            ops,
            queued_kind=kind,
            queued_op=item,
            created=created,
            research=research,
            persist=True,
        )

    return _snapshot(
        ops,
        queued_kind=None,
        queued_op=None,
        created=False,
        research=research,
        persist=True,
    )


def _snapshot(
    store: OperationStore,
    *,
    queued_kind: str | None,
    queued_op: dict[str, Any] | None,
    created: bool,
    halted: str | None = None,
    research: dict[str, Any] | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    active_items = _pipeline_active_items(store)
    active_kinds = [str(item.get("kind") or "") for item in active_items]
    primary_active = _primary_active(store)
    active_kind = str((primary_active or {}).get("kind") or "")
    if research is None:
        research = acquire_freshness()
    research_cooling = bool(not research.get("fresh") and not research.get("retry_due"))
    seen_due = False
    steps: list[dict[str, Any]] = []
    for spec in DESK_STEPS:
        kind = _kind_for_step(spec["id"], store, research=research)
        if spec["id"] == "investigate":
            latest = store.latest(DUE_DILIGENCE_ACQUIRE)
        else:
            latest = store.latest(kind) if kind else None
        if spec["id"] == "long_term" and latest is None:
            latest = store.latest(LONG_TERM_SCAN)
        matching_active = next(
            (
                item for item in active_items
                if str(item.get("kind") or "") in _kinds_for_id(spec["id"])
            ),
            None,
        )
        state = "ready"
        if spec["id"] == "investigate" and research_cooling:
            state = "waiting"
            seen_due = True
        elif kind is None and matching_active is None:
            state = "ready"
        elif matching_active:
            state = "running" if str(matching_active.get("status") or "") == RUNNING else "queued"
            seen_due = True
        elif queued_kind and queued_kind in _kinds_for_id(spec["id"]):
            state = "queued" if created or str((queued_op or {}).get("status") or "") == PENDING else "running"
            seen_due = True
        elif kind:
            failed = _recently_failed(store, kind)
            if failed:
                state = "failed" if spec["id"] == "prices" else "skipped_failed"
                if spec["id"] == "prices":
                    seen_due = True
            else:
                state = "waiting"
                seen_due = True
        row: dict[str, Any] = {
            **spec,
            "kind": kind,
            "state": state,
            "latest_status": (latest or {}).get("status"),
        }
        if spec["id"] == "investigate":
            row["freshness_state"] = research.get("state")
            row["unresolved_symbols"] = len(list(research.get("unresolved_symbols") or []))
            row["next_retry_at"] = research.get("next_retry_at")
            row["freshness_reason"] = research.get("reason")
        steps.append(row)

    current = None
    for row in steps:
        if row["state"] in {"running", "queued"}:
            current = {
                "id": row["id"],
                "title": row["title"],
                "kind": row["kind"],
                "status": row["state"],
                "why": row["why"],
                "page": row["page"],
            }
            break

    if halted == "prices":
        message = "Official prices failed recently — wait before retrying. Later desk steps stay paused."
    elif current:
        if len(active_items) > 1:
            message = f"{current['title']} now; {len(active_items)} isolated lanes active. {current['why']}"
        else:
            message = f"{current['title']} now: {current['why']}"
    elif research_cooling:
        count = len(list(research.get("unresolved_symbols") or []))
        next_retry = str(research.get("next_retry_at") or "provider cooldown")
        message = (
            f"Research evidence is still incomplete for {count} shortlisted name(s). "
            f"Recent provider attempts are cooling down; next retry {next_retry}."
        )
    elif all(row["state"] == "ready" for row in steps):
        message = "Desk data is current. Home, Recommendations and Market Reports read saved files."
    else:
        message = "Desk preparation will continue in dependency order; critical recovery may bypass secondary lanes."

    queued_step = _step_from_kind(queued_kind) if queued_kind else None
    operations = []
    if queued_op:
        operations.append(
            {
                "kind": queued_op.get("kind") or queued_kind,
                "operation_id": queued_op.get("operation_id"),
                "status": queued_op.get("status"),
                "created": created,
            }
        )
    payload = {
        "sequential": True,
        "critical_recovery_parallel": True,
        "queued_kind": queued_kind,
        "queued_created": created,
        "current": current,
        "steps": steps,
        "message": message,
        "page": (queued_step or current or {}).get("page") if (queued_step or current) else "",
        "scan_reused": scan_is_fresh(),
        "operations": operations,
        "active_kind": active_kind or None,
        "active_kinds": sorted(set(active_kinds)),
        "research_freshness": research,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_source": "computed",
        "freshness": SNAPSHOT_CURRENT,
        "age_seconds": 0.0,
    }
    if persist:
        to_write = dict(payload)
        to_write["research_freshness"] = _compact_research(research)
        persist_desk_pipeline_snapshot(to_write)
    return payload


def _compact_research(research: dict[str, Any] | None) -> dict[str, Any] | None:
    if not research:
        return research
    unresolved = list(research.get("unresolved_symbols") or [])
    return {
        "fresh": research.get("fresh"),
        "retry_due": research.get("retry_due"),
        "state": research.get("state"),
        "reason": research.get("reason"),
        "unresolved_symbols": unresolved[:20],
        "unresolved_count": len(unresolved),
        "next_retry_at": research.get("next_retry_at"),
        "checked_at": research.get("checked_at"),
    }


def _kinds_for_id(step_id: str) -> set[str]:
    if step_id == "prices":
        return {DATA_PREPARE, FNO_REFRESH}
    if step_id == "scan":
        return {MARKET_SCAN}
    if step_id == "long_term":
        return {LONG_TERM_REFRESH, LONG_TERM_SCAN}
    if step_id == "news":
        return {NEWS_REFRESH}
    if step_id == "investigate":
        return {DUE_DILIGENCE_ACQUIRE}
    return set()
