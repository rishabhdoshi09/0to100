"""Separate system-health lanes. Never one misleading green light."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


# Lane states. BLOCKED and FAILED are deliberately distinct from MISSING:
# "we have nothing" and "the acquisition that would have given us something was
# refused or errored" are different facts and the operator needs both.
LANE_STATES = frozenset({
    "HEALTHY",
    "STALE",
    "PARTIAL",
    "MISSING",
    "BLOCKED",
    "FAILED",
    "BROKEN",
    "WAITING",
    "UNKNOWN",
})

# Terminal durable-operation statuses that mean the acquisition did not deliver.
_OP_BLOCKED = "BLOCKED"
_OP_FAILED = "FAILED"
_OP_SUCCEEDED = "SUCCEEDED"
_OP_RUNNING = {"RUNNING", "PENDING"}


IST = timezone(timedelta(hours=5, minutes=30))

# product.readiness.broker_status is the authority on whether the desk can talk
# to Zerodha. These are its states, mapped onto lane states once, here.
_BROKER_STATE_TO_LANE = {
    "READY": "HEALTHY",
    "LOGIN_REQUIRED": "BROKEN",
    "SNAPSHOT_REQUIRED": "PARTIAL",
    "UNAVAILABLE": "FAILED",
    "CONFIG_REQUIRED": "BLOCKED",
    "NOT_READY": "UNKNOWN",
    "UNKNOWN": "UNKNOWN",
}


def as_of_utc(value: Any, *, naive_tz: timezone = timezone.utc) -> str:
    """Normalise a lane timestamp to timezone-aware UTC ISO-8601.

    Lanes used to emit three different shapes in one payload: epoch seconds
    from the operations store, UTC ISO from the workers, and a naive
    Asia/Kolkata string from the autonomy heartbeat. Comparing them, or showing
    them side by side, is wrong in both directions — a naive IST stamp read as
    UTC is 5h30m in the future. Everything leaves here as UTC with an explicit
    offset; the frontend localises for display.

    ``naive_tz`` says how to read a value that carries no offset. Callers that
    know their source is IST pass IST; the default assumes UTC.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=naive_tz)
        return moment.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    text = str(value).strip()
    if not text:
        return ""
    # Epoch seconds arrive as strings from the operations store.
    try:
        return datetime.fromtimestamp(float(text), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    candidate = text.replace("Z", "+00:00")
    for parse in (datetime.fromisoformat,):
        try:
            moment = parse(candidate)
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=naive_tz)
        return moment.astimezone(timezone.utc).isoformat()
    # Unparseable: surface it rather than inventing a time.
    return text


# Phrases that describe a lane needing attention. A lane whose own detail says
# one of these things may not also claim to be HEALTHY: the operator reads the
# dot, not the sentence, so a green dot over "running with reduced capability"
# is worse than no dot at all.
_DEGRADED_PHRASES = (
    "reduced capability",
    "not ready",
    "not available",
    "unavailable",
    "login required",
    "login is required",
    "is unavailable",
    "requires attention",
    "needs attention",
    "attention required",
    "paused",
    "blocked",
    "degraded",
    "stopped",
    "halted",
    "failed",
    "failure",
    "error",
    "cannot",
    "could not",
    "is not running",
    "not verified",
    "missing",
    "expired",
)

# Negations that make a degraded phrase benign: "0 errors" is not an error.
_BENIGN_PREFIXES = ("no ", "0 ", "zero ", "without ", "free of ")


def detail_contradicts_healthy(detail: str) -> str:
    """Return the phrase that makes this detail incompatible with HEALTHY.

    Empty string means the detail and a green dot can honestly coexist.
    """
    text = str(detail or "").strip().lower()
    if not text:
        return ""
    for phrase in _DEGRADED_PHRASES:
        start = text.find(phrase)
        while start != -1:
            prefix = text[max(0, start - 12):start]
            if not any(prefix.endswith(b) for b in _BENIGN_PREFIXES):
                return phrase
            start = text.find(phrase, start + 1)
    return ""


def _lane(
    key: str,
    label: str,
    status: str,
    *,
    as_of: Any = "",
    detail: str = "",
    as_of_naive_tz: timezone = timezone.utc,
) -> dict[str, Any]:
    state = str(status or "UNKNOWN").upper()
    if state not in LANE_STATES:
        state = "UNKNOWN"
    text = detail or ""
    contradiction = ""
    if state == "HEALTHY":
        contradiction = detail_contradicts_healthy(text)
        if contradiction:
            # We do not know this lane is healthy — its own detail says
            # otherwise — so it reports UNKNOWN rather than a green dot.
            state = "UNKNOWN"
    lane = {
        "key": key,
        "label": label,
        "status": state,
        "as_of": as_of_utc(as_of, naive_tz=as_of_naive_tz),
        "detail": text,
    }
    if contradiction:
        lane["status_demoted_from"] = "HEALTHY"
        lane["status_demoted_because"] = (
            f"the lane's own detail reports {contradiction!r}"
        )
    return lane


def _op_status(operation: Mapping[str, Any] | None) -> str:
    return str((operation or {}).get("status") or "").upper()


def _op_reason(operation: Mapping[str, Any] | None) -> str:
    op = operation or {}
    for key in ("error_code", "error_message", "message", "blocked_reason"):
        text = str(op.get(key) or "").strip()
        if text:
            return text
    return ""


def _news_lane(news: Mapping[str, Any]) -> dict[str, Any]:
    """News health is the acquisition outcome, not the presence of a store.

    The store and its source-health rows exist from first boot, so a lane keyed
    on "a file/table is reachable" reported HEALTHY while the durable
    NEWS_REFRESH operation was BLOCKED with zero articles on file. Health is the
    business capability: did the last refresh deliver usable articles.
    """
    latest = dict(news.get("latest_refresh") or {})
    stats = dict(news.get("stats") or {})
    try:
        articles = int(stats.get("total") or 0)
    except (TypeError, ValueError):
        articles = 0
    as_of = str(latest.get("finished_at") or "")
    status = _op_status(latest)
    reason = _op_reason(latest)
    detail = f"{articles} articles on file"

    if status == _OP_BLOCKED:
        return _lane(
            "news_freshness", "News freshness", "BLOCKED", as_of=as_of,
            detail=f"Last refresh blocked: {reason or 'no source returned usable data'} · {detail}",
        )
    if status == _OP_FAILED:
        return _lane(
            "news_freshness", "News freshness", "FAILED", as_of=as_of,
            detail=f"Last refresh failed: {reason or 'unknown error'} · {detail}",
        )
    if not articles:
        if status in _OP_RUNNING:
            return _lane(
                "news_freshness", "News freshness", "WAITING", as_of=as_of,
                detail="A news refresh is running; nothing on file yet",
            )
        return _lane(
            "news_freshness", "News freshness", "MISSING", as_of=as_of,
            detail="No articles on file",
        )
    if status and status != _OP_SUCCEEDED and status not in _OP_RUNNING:
        return _lane(
            "news_freshness", "News freshness", "STALE", as_of=as_of,
            detail=f"{detail}; last refresh ended {status}",
        )
    return _lane("news_freshness", "News freshness", "HEALTHY", as_of=as_of, detail=detail)


def _recommendations_lane(
    scan: Mapping[str, Any],
    recommendations_available: bool | None,
) -> dict[str, Any]:
    """Recommendations are a projection of a scan; without one they are MISSING.

    A projection file on disk is not evidence. The lane used to read HEALTHY
    from the file's existence while ``scan.scanned_at`` was empty, contradicting
    the scan lane in the same payload.
    """
    scanned_at = str(scan.get("scanned_at") or "")
    projection_ok = (
        bool(recommendations_available)
        if recommendations_available is not None
        else bool(scan.get("available"))
    )
    detail = "Recommendations read the saved scan; they do not rescore on open"

    if not scanned_at:
        return _lane(
            "recommendations_freshness", "Recommendations freshness", "MISSING",
            detail="No authoritative scan on file — nothing to project from",
        )
    if not projection_ok:
        return _lane(
            "recommendations_freshness", "Recommendations freshness", "WAITING",
            as_of=scanned_at,
            detail="A scan is on file; the recommendation projection is not built yet",
        )
    return _lane(
        "recommendations_freshness", "Recommendations freshness", "HEALTHY",
        as_of=scanned_at, detail=detail,
    )


def _auth_lane(autonomy: Mapping[str, Any]) -> dict[str, Any]:
    # Both halves of this lane come from ONE authoritative state. The previous
    # version read supervisor liveness for the status and borrowed the
    # supervisor's plain_state for the detail, so a DEGRADED supervisor
    # produced a green auth dot over the sentence "running with reduced
    # capability" — and neither half was about authentication at all.
    broker = dict(autonomy.get("broker") or {})
    broker_state = str(broker.get("state") or "").upper()
    if broker_state:
        status = _BROKER_STATE_TO_LANE.get(broker_state, "UNKNOWN")
        return _lane(
            "zerodha_auth", "Zerodha authentication", status,
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            as_of_naive_tz=IST,
            detail=str(broker.get("detail") or broker_state),
        )

    # No broker projection available: say so rather than inferring auth health
    # from something that does not measure it.
    state = str(autonomy.get("state") or "")
    failures = [str(x) for x in (autonomy.get("active_failures") or [])]
    if state == "AUTH_REQUIRED" or any("auth" in f.lower() for f in failures):
        return _lane(
            "zerodha_auth", "Zerodha authentication", "BROKEN",
            as_of=str(autonomy.get("heartbeat_ist") or ""),
            as_of_naive_tz=IST,
            detail=str(autonomy.get("plain_state") or "Kite login required"),
        )
    return _lane(
        "zerodha_auth", "Zerodha authentication", "UNKNOWN",
        as_of=str(autonomy.get("heartbeat_ist") or ""),
        as_of_naive_tz=IST,
        detail="Broker readiness has not been probed — authentication is not verified.",
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
        _news_lane(news),
        _lane(
            "market_report_freshness",
            "Market report freshness",
            "HEALTHY" if market_report_as_of else "MISSING",
            as_of=market_report_as_of,
            detail="Today's pulse file, or missing — never invented",
        ),
        _coverage_lane(scan),
        _recommendations_lane(scan, recommendations_available),
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
            as_of_naive_tz=IST,
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
            str(
                exec_lanes.get("recommendations")
                # Same rule as the freshness lane: a projection without a scan
                # behind it is MISSING, never HEALTHY.
                or _recommendations_lane(scan, recommendations_available)["status"]
            ),
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
            as_of_naive_tz=IST,
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
            as_of_naive_tz=IST,
            detail="Stop/target management requires a live scheduler process",
        ),
    ])
    counts = {state: 0 for state in sorted(LANE_STATES)}
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
