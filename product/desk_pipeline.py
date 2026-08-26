"""Sequential desk data pipeline — one download at a time, in viewing order.

Home needs official prices then the whole-market scan. Recommendations need
long-term funds next. Market Reports need news last. Automation never starts
the next step while a pipeline job is pending or running. User-clicked Scan Now
still uses the existing MARKET_SCAN control and is not a second scan engine.
"""
from __future__ import annotations

import time
from typing import Any

from operations.market_ops import (
    DATA_PREPARE,
    DUE_DILIGENCE_ACQUIRE,
    DUE_DILIGENCE_FRESH_S,
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

RETRY_AFTER_FAIL_S = 10 * 60

# Viewing order: Home → Scanner/Recos technical → Recos/funds → Market Reports.
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
        "why": "One whole-market scan for Home, Scanner and recommendation setups.",
    },
    {
        "id": "long_term",
        "title": "Long-term / funds",
        "page": "Recommendations",
        "why": "Fundamentals for Best Among and Wealth Builders.",
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
        "why": "Download filings and fundamentals for shortlisted names, then Investigate reads the files.",
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


def _root():
    from operations import market_ops as MO

    return MO.ROOT


def prices_kind_due() -> str | None:
    """DATA_PREPARE if history is thin, else FNO_REFRESH if that file is stale."""
    try:
        from data.bhavcopy_runtime import status as history_status

        history = history_status(load_cache=True)
    except Exception:
        history = {"ready": False, "sessions": 0}
    if not history.get("ready") or int(history.get("sessions", 0) or 0) < 60:
        return DATA_PREPARE
    if _stale(_root() / "logs" / "product" / "fno_universe.json", FNO_FRESH_S):
        return FNO_REFRESH
    return None


def scan_is_fresh() -> bool:
    path = _root() / "logs" / "product" / "latest_momentum_scan.json"
    try:
        from product.scan_store import load_scan, scan_artifact_is_fresh

        payload = load_scan(path)
        if not payload:
            return False
        if payload.get("scanned_at"):
            return bool(scan_artifact_is_fresh(path, max_age_s=SCAN_FRESH_S))
    except Exception:
        return False
    return not _stale(path, SCAN_FRESH_S)


def long_term_is_fresh() -> bool:
    return not _stale(_root() / "logs" / "product" / "latest_long_term_scan.json", LONG_TERM_FRESH_S)


def news_is_fresh() -> bool:
    return not _stale(_root() / "logs" / "news_curator.sqlite3", NEWS_FRESH_S)


def acquire_is_fresh() -> bool:
    try:
        from product.due_diligence.acquire import acquire_is_fresh as facts_fresh

        return bool(facts_fresh())
    except Exception:
        return True


def _fresh_s(step_id: str) -> float:
    if step_id == "prices":
        return FNO_FRESH_S
    if step_id == "scan":
        return SCAN_FRESH_S
    if step_id == "long_term":
        return LONG_TERM_FRESH_S
    if step_id == "news":
        return NEWS_FRESH_S
    if step_id == "investigate":
        return DUE_DILIGENCE_FRESH_S
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


def _kind_for_step(step_id: str, store: OperationStore | None = None) -> str | None:
    if step_id == "prices":
        kind = prices_kind_due()
    elif step_id == "scan":
        kind = None if scan_is_fresh() else MARKET_SCAN
    elif step_id == "long_term":
        kind = None if long_term_is_fresh() else LONG_TERM_REFRESH
    elif step_id == "news":
        kind = None if news_is_fresh() else NEWS_REFRESH
    elif step_id == "investigate":
        kind = None if acquire_is_fresh() else DUE_DILIGENCE_ACQUIRE
    else:
        kind = None
    if kind and store is not None and _recently_succeeded(store, _kinds_for_id(step_id), _fresh_s(step_id)):
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


def _pipeline_active(store: OperationStore) -> dict[str, Any] | None:
    for item in store.active():
        if str(item.get("kind") or "") in PIPELINE_KINDS:
            return item
    return None


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


def describe_desk_pipeline(store: OperationStore | None = None) -> dict[str, Any]:
    """Read-only snapshot. Does not enqueue."""
    ops = store or OperationStore()
    return _snapshot(ops, queued_kind=None, queued_op=None, created=False)


def advance_desk_pipeline(
    store: OperationStore | None = None,
    *,
    requested_by: str = "desk_pipeline",
) -> dict[str, Any]:
    """Enqueue at most the next due step. Skip fresh artifacts. Never invent data."""
    ops = store or OperationStore()
    active = _pipeline_active(ops)
    if active:
        return _snapshot(ops, queued_kind=None, queued_op=active, created=False)

    for step in DESK_STEPS:
        kind = _kind_for_step(step["id"], ops)
        if not kind:
            continue
        if _recently_failed(ops, kind):
            if step["id"] == "prices":
                return _snapshot(ops, queued_kind=None, queued_op=None, created=False, halted="prices")
            continue
        item, created = ops.enqueue(
            kind,
            lane=LANES[kind],
            requested_by=requested_by,
        )
        return _snapshot(ops, queued_kind=kind, queued_op=item, created=created)

    return _snapshot(ops, queued_kind=None, queued_op=None, created=False)


def _snapshot(
    store: OperationStore,
    *,
    queued_kind: str | None,
    queued_op: dict[str, Any] | None,
    created: bool,
    halted: str | None = None,
) -> dict[str, Any]:
    active = _pipeline_active(store)
    active_kind = str((active or {}).get("kind") or "")
    seen_due = False
    steps: list[dict[str, Any]] = []
    for spec in DESK_STEPS:
        kind = _kind_for_step(spec["id"], store)
        latest = store.latest(kind) if kind else None
        if spec["id"] == "long_term" and latest is None:
            latest = store.latest(LONG_TERM_SCAN)
        state = "ready"
        if kind is None:
            state = "ready"
        elif active and str(active.get("kind") or "") in _kinds_for_id(spec["id"]):
            state = "running" if str(active.get("status") or "") == RUNNING else "queued"
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
            elif seen_due or (active and not seen_due):
                state = "waiting"
                seen_due = True
            else:
                state = "waiting"
                seen_due = True
        steps.append(
            {
                **spec,
                "kind": kind,
                "state": state,
                "latest_status": (latest or {}).get("status"),
            }
        )

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
        message = f"{current['title']} now: {current['why']}"
    elif all(row["state"] == "ready" for row in steps):
        message = "Desk data is current. Home, Recommendations and Market Reports read saved files."
    else:
        message = "Desk preparation will continue one step at a time."

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
    return {
        "sequential": True,
        "queued_kind": queued_kind,
        "queued_created": created,
        "current": current,
        "steps": steps,
        "message": message,
        "page": (queued_step or current or {}).get("page") if (queued_step or current) else "",
        "scan_reused": scan_is_fresh(),
        "operations": operations,
        "active_kind": active_kind or None,
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
