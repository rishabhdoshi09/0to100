"""Separate system-health lanes. Never one misleading green light."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _lane(
    key: str,
    label: str,
    status: str,
    *,
    as_of: str = "",
    detail: str = "",
) -> dict[str, Any]:
    state = str(status or "UNKNOWN").upper()
    if state not in {"HEALTHY", "STALE", "MISSING", "BROKEN", "UNKNOWN", "WAITING"}:
        state = "UNKNOWN"
    return {
        "key": key,
        "label": label,
        "status": state,
        "as_of": as_of or "",
        "detail": detail or "",
    }


def _auth_lane(autonomy: Mapping[str, Any]) -> dict[str, Any]:
    state = str(autonomy.get("state") or "")
    failures = [str(x) for x in (autonomy.get("active_failures") or [])]
    auth_fail = state == "AUTH_REQUIRED" or any("auth" in f.lower() for f in failures)
    if auth_fail:
        return _lane(
            "zerodha_auth", "Zerodha authentication", "BROKEN",
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            detail=autonomy.get("plain_state") or "Kite login required",
        )
    running = bool(autonomy.get("running"))
    if running and state not in {"", "UNKNOWN", "STOPPED", "OFFLINE"}:
        return _lane(
            "zerodha_auth", "Zerodha authentication", "HEALTHY",
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            detail=str(autonomy.get("plain_state") or state or "Session present"),
        )
    return _lane(
        "zerodha_auth", "Zerodha authentication", "UNKNOWN",
        as_of=str(autonomy.get("heartbeat_ist") or ""),
        detail=str(autonomy.get("plain_state") or "Autonomy supervisor is not running — auth not verified"),
    )


def _coverage_lane(scan: Mapping[str, Any]) -> dict[str, Any]:
    cov = dict(scan.get("coverage") or {})
    requested = int(cov.get("requested") or scan.get("requested_universe") or 0)
    checked = int(cov.get("checked") or scan.get("universe_size") or 0)
    if not scan.get("available") and not requested:
        return _lane("scan_freshness", "Scan freshness", "MISSING", detail="No saved whole-market scan")
    state = str(scan.get("coverage_state") or cov.get("state") or "")
    status = "HEALTHY" if scan.get("available") else "WAITING"
    if state in {"PARTIAL", "THIN"}:
        status = "STALE"
    return _lane(
        "scan_freshness", "Scan freshness", status,
        as_of=str(scan.get("scanned_at") or ""),
        detail=(
            f"requested {requested:,} · checked {checked:,} · "
            f"qualified {int(cov.get('qualified') or 0):,}"
        ),
    )


def _settlement_status(autonomy: Mapping[str, Any]) -> str:
    status = str(autonomy.get("learning_status") or "").strip().upper()
    if not status:
        return "WAITING"
    if any(token in status for token in ("WAIT", "YET", "NONE", "UNKNOWN", "NO_EOD", "INSUFFICIENT")):
        return "WAITING"
    if any(token in status for token in ("ACTIVE", "SETTLED", "COMPLETE", "LEARNING")):
        return "HEALTHY"
    return "WAITING"


def build_system_health_contract(
    *,
    scan: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    news: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
    autonomy: Mapping[str, Any] | None = None,
    recommendations_available: bool | None = None,
    market_report_as_of: str = "",
    product_wired: bool | None = None,
    fundamental_coverage_pct: float | None = None,
    filings_as_of: str = "",
    paper: Mapping[str, Any] | None = None,
    recommendations_workspace: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scan = dict(scan or {})
    data = dict(data or {})
    news = dict(news or {})
    operations = dict(operations or {})
    autonomy = dict(autonomy or {})
    paper = dict(paper or {})
    bhav = dict(data.get("bhavcopy") or {})
    recos_ok = bool(recommendations_available) if recommendations_available is not None else bool(
        scan.get("available")
    )

    lanes = [
        _auth_lane(autonomy),
        _lane(
            "instrument_universe",
            "Instrument universe",
            "HEALTHY" if int(scan.get("requested_universe") or scan.get("universe_size") or 0) else (
                "HEALTHY" if data.get("ready") else "MISSING"
            ),
            as_of=str(bhav.get("latest_date") or ""),
            detail=f"scan universe {int(scan.get('universe_size') or 0):,}",
        ),
        _lane(
            "daily_ohlcv",
            "Daily OHLCV coverage",
            "HEALTHY" if bhav.get("ready") else "MISSING",
            as_of=str(bhav.get("latest_date") or ""),
            detail=f"{int(bhav.get('sessions') or 0)} sessions · {int(bhav.get('symbols') or 0)} symbols",
        ),
        _lane(
            "fundamental_coverage",
            "Fundamental coverage",
            "UNKNOWN" if fundamental_coverage_pct is None else (
                "HEALTHY" if fundamental_coverage_pct >= 50 else "STALE" if fundamental_coverage_pct > 0 else "MISSING"
            ),
            detail=(
                "Per-name coverage lives on Company Intelligence. This lane is not a blended green light."
                if fundamental_coverage_pct is None else
                f"Latest overlay coverage {fundamental_coverage_pct:.0f}%"
            ),
        ),
        _lane(
            "filings_freshness",
            "Filings freshness",
            "UNKNOWN" if not filings_as_of else "HEALTHY",
            as_of=filings_as_of,
            detail="Filings appear after due-diligence acquire; missing stays missing.",
        ),
        _lane(
            "news_freshness",
            "News freshness",
            "HEALTHY" if news.get("available") else "MISSING",
            as_of=str((news.get("latest_refresh") or {}).get("finished_at") or ""),
            detail=f"{int((news.get('stats') or {}).get('total') or 0)} articles on file",
        ),
        _lane(
            "market_report_freshness",
            "Market report freshness",
            "HEALTHY" if market_report_as_of else "MISSING",
            as_of=market_report_as_of,
            detail="Today's pulse file, or missing — never invented",
        ),
        _coverage_lane(scan),
        _lane(
            "recommendations_freshness",
            "Recommendations freshness",
            "HEALTHY" if recos_ok else "MISSING",
            as_of=str(scan.get("scanned_at") or ""),
            detail="Recommendations read the saved scan; they do not rescore on open",
        ),
        _lane(
            "operations_worker",
            "Operations worker",
            "HEALTHY" if operations.get("running") else "BROKEN",
            as_of=str(operations.get("heartbeat") or ""),
            detail=f"pid {operations.get('worker_pid') or '—'}",
        ),
        _lane(
            "research_worker",
            "Research worker",
            "HEALTHY" if autonomy.get("running") or autonomy.get("process_running") else "WAITING",
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            detail=str(autonomy.get("learning_status") or autonomy.get("plain_state") or ""),
        ),
        _lane(
            "paper_outcome_settlement",
            "Paper outcome settlement",
            _settlement_status(autonomy),
            detail=str(autonomy.get("learning_status") or "No settlement claim without a status"),
        ),
        _lane(
            "backtest_registry",
            "Backtest registry",
            "WAITING",
            detail="Production ensemble parity is UNVERIFIED. Related scanner calibration is not parity.",
        ),
        _lane(
            "frontend_api_contract",
            "Frontend / API contract",
            "HEALTHY" if product_wired else "UNKNOWN" if product_wired is None else "BROKEN",
            detail="wired=true means routes exist; it is not data freshness",
        ),
    ]

    exec_payload = dict(execution or {})
    if not exec_payload:
        try:
            from product.paper_autopilot import execution_health
            exec_payload = execution_health(
                autonomy=autonomy,
                paper=paper,
                workspace=recommendations_workspace,
            )
        except Exception:
            exec_payload = {}
    exec_lanes = dict(exec_payload.get("lanes") or {})
    why = dict(exec_payload.get("why_no_trade") or {})
    scheduler_status = str(exec_lanes.get("autonomy_scheduler") or (
        "HEALTHY" if autonomy.get("running") else "WAITING"
    ))
    paper_exec_status = str(exec_lanes.get("paper_execution") or "UNKNOWN")
    lanes.extend([
        _lane(
            "scanner",
            "Scanner",
            str(exec_lanes.get("scanner") or ("HEALTHY" if scan.get("available") else "MISSING")),
            as_of=str(scan.get("scanned_at") or ""),
            detail="Saved whole-market scan — not a green autonomy badge",
        ),
        _lane(
            "recommendations",
            "Recommendations",
            str(exec_lanes.get("recommendations") or ("HEALTHY" if recos_ok else "MISSING")),
            as_of=str(scan.get("scanned_at") or ""),
            detail="Desk file from the last scan. Empty high-conviction is a valid day.",
        ),
        _lane(
            "selection_authority",
            "Selection authority",
            str(exec_lanes.get("selection_authority") or "WAITING"),
            as_of=str(why.get("as_of") or ""),
            detail=str(why.get("headline") or "No autopilot cycle recorded"),
        ),
        _lane(
            "autonomy_scheduler",
            "Autonomy scheduler",
            scheduler_status,
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            detail=(
                f"pid {autonomy.get('scheduler_owner_pid') or '—'} · "
                f"{'fresh heartbeat' if autonomy.get('running') else 'not running'}"
            ),
        ),
        _lane(
            "paper_execution",
            "Paper execution",
            paper_exec_status,
            as_of=str(why.get("as_of") or ""),
            detail=str(
                exec_payload.get("paper_execution_detail")
                or why.get("headline")
                or "Paper execution is independent of the autonomy badge"
            ),
        ),
        _lane(
            "exit_supervisor",
            "Exit supervisor",
            str(exec_lanes.get("exit_supervisor") or "WAITING"),
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            detail="Stop/target management requires a live scheduler process",
        ),
    ])
    counts = {"HEALTHY": 0, "STALE": 0, "MISSING": 0, "BROKEN": 0, "UNKNOWN": 0, "WAITING": 0}
    for lane in lanes:
        counts[str(lane["status"])] = counts.get(str(lane["status"]), 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collapsed_status": None,
        "note": (
            "Lanes are independent. A healthy worker does not make stale news healthy. "
            "A green autonomy badge does not mean paper execution is healthy."
        ),
        "counts": counts,
        "lanes": lanes,
        "why_no_trade": why,
    }
