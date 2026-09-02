"""Home Backend Control Plane — projection over existing authorities.

This is not a second health engine, scheduler, job store, or operations model.
It only explains what Home already knows and offers already-registered safe actions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

def _action(control: str = "", *, label: str, kind: str = "control", instruction: str = "") -> dict[str, str]:
    return {
        "id": control or kind,
        "control": control,
        "label": label,
        "kind": kind,
        "instruction": instruction,
    }

SAFE_CONTROLS = frozenset({
    "REFRESH_DATA_NOW",
    "RUN_SCAN_NOW",
    "PAUSE_NEW_PAPER_ENTRIES",
    "RESUME_NEW_PAPER_ENTRIES",
    "RUN_CYCLE_NOW",
    "VERIFY_FORWARD_SOAK",
    "CHECK_SYSTEM",
    "SIMULATE_PAST_DECISIONS",
    "OBSERVE_ONLY_TODAY",
    "CLEAR_OBSERVE_ONLY",
})

FORBIDDEN_CONTROLS = frozenset({
    "KILL_PID",
    "RUN_SHELL",
    "DELETE_QUEUE",
    "WIPE_JOBS",
    "UNLOCK_LIVE_MONEY",
    "LIVE_BUY",
    "LIVE_SELL",
    "BROKER_BUY",
    "BROKER_SELL",
    "PROMOTE_STRATEGY",
    "DISABLE_RISK",
    "DD_BYPASS",
    "CHASE_BYPASS",
})

SECRET_KEYS = frozenset({
    "token", "access_token", "refresh_token", "api_secret", "api_key",
    "secret", "password", "authorization", "kite_access_token",
    "request_token", "session_token", "bearer",
})

LANE_PAGES = {
    "data": ("Research Data", "Open Research Data"),
    "zerodha": ("Automation", "Open System Health"),
    "automation": ("Automation", "Open System Health"),
    "paper_bot": ("Portfolio", "Open My Holdings"),
    "learning": ("Learning", "Open Learning"),
}

_JOB_PLAIN = {
    "data_refresh": "Updating market data",
    "DATA_PREPARE": "Updating market data",
    "market_scan": "Market scan",
    "MARKET_SCAN": "Market scan",
    "paper_cycle": "Paper decision",
    "PAPER_CYCLE": "Paper decision",
    "outcome_resolution": "End-of-day settlement",
    "learning_cycle": "Learning journal",
    "research_cycle": "Research cycle",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _omit_empty(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if value == "" or value == [] or value == {}:
            continue
        if isinstance(value, bool) or value == 0 or value == 0.0:
            out[key] = value
            continue
        out[key] = value
    return out


def _safe_action(control: str = "", *, label: str, kind: str = "control", instruction: str = "") -> dict[str, str] | None:
    if kind == "control" and control and control not in SAFE_CONTROLS:
        return None
    if control in FORBIDDEN_CONTROLS:
        return None
    return _action(control, label=label, kind=kind, instruction=instruction)


def _scrub_technical(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _as_dict(raw).items():
        lowered = str(key).lower()
        if lowered in SECRET_KEYS or any(part in lowered for part in ("token", "secret", "password", "authorization")):
            continue
        if isinstance(value, Mapping):
            nested = _scrub_technical(value)
            if nested:
                out[key] = nested
            continue
        if value is None or value == "" or value == [] or value == {}:
            continue
        out[key] = value
    return out


def _op_kind(ops: Mapping[str, Any], kind: str, bucket: str = "active") -> dict[str, Any]:
    for row in _as_list(ops.get(bucket)):
        item = _as_dict(row)
        if str(item.get("kind") or "") == kind:
            return item
    return {}


def _recent_status(ops: Mapping[str, Any], kind: str, status: str) -> dict[str, Any]:
    for row in _as_list(ops.get("recent")):
        item = _as_dict(row)
        if str(item.get("kind") or "") == kind and str(item.get("status") or "") == status:
            return item
    return {}


def _plain_job(name: Any) -> str:
    text = str(name or "").strip()
    return _JOB_PLAIN.get(text, text.replace("_", " ").title() if text else "")


def _fmt_when(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text and len(text) >= 16:
        return text[11:16]
    if " " in text:
        clock = text.split(" ", 1)[1]
        return clock[:5] if len(clock) >= 5 else clock
    return text


def _fmt_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:10]


def _progress_from_op(op: Mapping[str, Any]) -> dict[str, Any] | None:
    current = op.get("current") if op.get("current") is not None else op.get("progress_current")
    total = op.get("total") if op.get("total") is not None else op.get("progress_total")
    if current is None and total is None and not op.get("stage") and not op.get("status"):
        return None
    return _omit_empty({
        "kind": op.get("kind"),
        "label": _plain_job(op.get("kind")),
        "current": current,
        "total": total,
        "status": op.get("status"),
        "stage": op.get("stage"),
        "message": op.get("message") or op.get("current_message"),
    })


def _position_rows(opens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in opens:
        pos = _as_dict(raw)
        symbol = str(pos.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        rows.append(_omit_empty({
            "symbol": symbol,
            "entry": pos.get("entry_price") if pos.get("entry_price") is not None else pos.get("entry"),
            "status": pos.get("status") or "Open",
            "stop": pos.get("stop_price") if pos.get("stop_price") is not None else pos.get("stop"),
            "target": pos.get("target_price") if pos.get("target_price") is not None else pos.get("target"),
            "risk_used": pos.get("risk_amount") if pos.get("risk_amount") is not None else pos.get("risk_used"),
        }))
    return rows


def _lane(
    lane_id: str,
    *,
    label: str,
    status: str,
    status_code: str,
    summary: str,
    detail: str,
    what: str,
    meaning: str = "",
    waiting_for: str = "",
    current: str = "",
    next_step: str = "",
    last_success_at: Any = None,
    last_failure_at: Any = None,
    last_failure_reason: str = "",
    progress: Mapping[str, Any] | None = None,
    current_job: str = "",
    current_job_id: Any = None,
    current_job_started_at: Any = None,
    next_check_at: Any = None,
    freshness: str = "",
    source: str = "",
    dependencies: Sequence[str] | None = None,
    needs_user: bool = False,
    recovering: bool = False,
    degraded: bool = False,
    primary_action: Mapping[str, Any] | None = None,
    secondary_actions: Sequence[Mapping[str, Any]] | None = None,
    technical: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    page, page_label = LANE_PAGES[lane_id]
    row = {
        "id": lane_id,
        "label": label,
        "status": status,
        "status_code": status_code,
        "summary": summary,
        "detail": detail,
        "what": what,
        "meaning": meaning,
        "waiting_for": waiting_for,
        "current": current,
        "next": next_step,
        "last_success_at": last_success_at,
        "last_failure_at": last_failure_at,
        "last_failure_reason": last_failure_reason,
        "progress": dict(progress) if progress else None,
        "current_job": current_job,
        "current_job_id": current_job_id,
        "current_job_started_at": current_job_started_at,
        "next_check_at": next_check_at,
        "freshness": freshness,
        "source": source,
        "dependencies": list(dependencies or []),
        "needs_user": needs_user,
        "recovering": recovering,
        "degraded": degraded,
        "primary_action": dict(primary_action) if primary_action else None,
        "secondary_actions": [dict(a) for a in (secondary_actions or []) if a],
        "full_details_page": page,
        "full_details_label": page_label,
        "technical": _scrub_technical(technical),
        "live_locked": True,
    }
    if extra:
        row.update(extra)
    return _omit_empty(row)


def build_system_lanes(
    *,
    auto: Mapping[str, Any],
    data_d: Mapping[str, Any],
    paper_d: Mapping[str, Any],
    why_d: Mapping[str, Any],
    soak_d: Mapping[str, Any],
    ops: Mapping[str, Any],
    freshness: Mapping[str, Any],
    verify: Mapping[str, Any],
    kite_ok: bool,
    data_ready: bool,
    history_current: bool,
    preparing: bool,
    data_failed: bool,
    scan_failed: bool,
    scan_ok: bool,
    paper_enabled: bool,
    live_locked: bool,
    taken: Sequence[Mapping[str, Any]],
    opens: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    valid_no_trade: bool,
    cycle_reasons: Sequence[str],
    last_decision: str,
    why_plain: str,
    now_line: str,
    next_line: str,
    learning_simple: str,
    n_real: int,
) -> dict[str, Any]:
    """Project the five Home system lanes from already-loaded authorities."""
    data_lane = _data_lane(
        auto=auto, data_d=data_d, ops=ops, freshness=freshness,
        data_ready=data_ready, history_current=history_current,
        preparing=preparing, data_failed=data_failed, scan_ok=scan_ok,
        next_line=next_line,
    )
    zerodha_lane = _zerodha_lane(auto=auto, kite_ok=kite_ok)
    automation_lane = _automation_lane(
        auto=auto, ops=ops, freshness=freshness,
        history_current=history_current, preparing=preparing,
        scan_failed=scan_failed, scan_ok=scan_ok, data_ready=data_ready,
        now_line=now_line, next_line=next_line,
    )
    paper_lane = _paper_lane(
        auto=auto, paper_d=paper_d, why_d=why_d,
        paper_enabled=paper_enabled, live_locked=live_locked,
        taken=taken, opens=opens, closed=closed,
        valid_no_trade=valid_no_trade, cycle_reasons=cycle_reasons,
        last_decision=last_decision, why_plain=why_plain, next_line=next_line,
    )
    learning_lane = _learning_lane(
        soak_d=soak_d, verify=verify, learning_simple=learning_simple, n_real=n_real,
    )
    system = {
        "data": data_lane,
        "zerodha": zerodha_lane,
        "automation": automation_lane,
        "paper_bot": paper_lane,
        "learning": learning_lane,
    }
    return system


def build_check_system(system: Mapping[str, Any], *, live_locked: bool = True) -> dict[str, Any]:
    """Read-only snapshot of the same Home lanes. Not a second health source."""
    labels = {
        "data": "Data",
        "zerodha": "Zerodha",
        "automation": "Automation",
        "paper_bot": "Paper Bot",
        "learning": "Learning",
    }
    rows = []
    for key, label in labels.items():
        lane = _as_dict(system.get(key))
        status = str(lane.get("status") or "Waiting")
        if key == "automation" and status == "Working":
            shown = "Running"
        elif key == "learning" and status == "Working":
            shown = "Collecting"
        else:
            shown = status
        rows.append({
            "id": key,
            "label": label,
            "status": shown,
            "detail": lane.get("summary") or lane.get("detail") or "",
        })
    rows.append({
        "id": "live_money",
        "label": "Live Money",
        "status": "Locked" if live_locked else "Must stay locked",
        "detail": "Paper only. No live buy button.",
    })
    return {
        "read_only": True,
        "source": "home_os.system",
        "lanes": rows,
        "action": _safe_action("CHECK_SYSTEM", label="Check system", kind="refresh"),
    }


def _data_lane(
    *,
    auto: Mapping[str, Any],
    data_d: Mapping[str, Any],
    ops: Mapping[str, Any],
    freshness: Mapping[str, Any],
    data_ready: bool,
    history_current: bool,
    preparing: bool,
    data_failed: bool,
    scan_ok: bool,
    next_line: str,
) -> dict[str, Any]:
    bhav = _as_dict(data_d.get("bhavcopy"))
    snapshot = _as_dict(data_d.get("snapshot"))
    prepare = _op_kind(ops, "DATA_PREPARE")
    scan_op = _op_kind(ops, "MARKET_SCAN")
    failed = _recent_status(ops, "DATA_PREPARE", "FAILED")
    succeeded = _recent_status(ops, "DATA_PREPARE", "SUCCEEDED")
    available = freshness.get("available_session") or bhav.get("latest_date") or ""
    expected = freshness.get("expected_latest_completed_session") or ""
    reason = str(freshness.get("reason_code") or "")
    symbols = bhav.get("symbols")
    sessions = bhav.get("sessions")
    source = str(bhav.get("source") or snapshot.get("source") or "")
    active_job = _as_dict(auto.get("active_job"))
    refresh_bg = bool(auto.get("data_refresh_background"))
    job_is_data = str(active_job.get("job_type") or "") in {"data_refresh", "DATA_PREPARE"} or bool(prepare)
    recovering = bool(data_failed and (prepare or refresh_bg or job_is_data))

    waiting_for = ""
    dependencies: list[str] = []
    if not history_current and expected:
        waiting_for = f"Official session {expected}"
        dependencies.append("official_nse_bhavcopy")
    if scan_op and not history_current:
        waiting_for = waiting_for or "Official market data before the scan"
        dependencies.append("MARKET_SCAN")
    if reason == "HISTORY_STALE":
        waiting_for = waiting_for or "Later official bars"
        dependencies.append("later_official_bars")

    current = ""
    next_step = "Run the market scan" if history_current or data_ready else "Finish official prices, then scan"
    if prepare or (job_is_data and not data_ready):
        current = "Checking the latest market snapshot"
        if prepare.get("message") or prepare.get("stage"):
            current = str(prepare.get("message") or prepare.get("stage"))
    elif data_ready and history_current:
        current = "Market data is ready."
        next_step = next_line or "Market scan"
    elif not data_ready:
        current = "Getting the latest market data"

    status = "Waiting"
    status_code = "WAITING"
    summary = "Official NSE prices"
    meaning = "QuantTerm uses official NSE history for scans and paper learning."
    if recovering:
        status, status_code = "Working", "RECOVERING"
        summary = "QuantTerm is trying to recover market data."
        meaning = "A data job failed, and the same data lane is running again."
    elif data_failed and not data_ready and not prepare:
        status, status_code = "Problem", "FAILED"
        summary = "Market data stopped progressing."
        meaning = "Official prices did not finish. One Retry uses the same data lane."
        current = current or "No data job is making progress"
    elif prepare or (preparing and not history_current) or (refresh_bg and not data_ready):
        status, status_code = "Working", "WORKING"
        summary = "Updating today's prices."
        meaning = "Nothing needed from you. The delayed market scan will start afterward."
    elif not history_current and (available or expected or reason):
        status, status_code = "Waiting", "WAITING_DEPENDENCY"
        summary = "Today's first scan is waiting for market data."
        meaning = "QuantTerm is still working on official prices. This is a normal dependency."
    elif data_ready and history_current:
        status, status_code = "Ready", "READY"
        if symbols:
            summary = f"{int(symbols):,} stocks have usable history.".replace(",", ",")
            if int(symbols) >= 2300:
                summary = "2,300+ stocks have usable history."
        else:
            summary = "Official NSE prices"
        meaning = "You do not need to do anything."
        if available:
            current = current or f"Last updated: {_fmt_date(available)}"
    elif not data_ready:
        status, status_code = "Waiting", "WAITING"
        summary = "Official prices are not ready yet."
        meaning = "QuantTerm will use them as soon as the official file is current."

    primary = None
    secondary: list[dict[str, str]] = []
    if status == "Problem":
        action = _safe_action("REFRESH_DATA_NOW", label="Retry")
        if action:
            primary = action
    elif status == "Ready":
        action = _safe_action("REFRESH_DATA_NOW", label="Refresh")
        if action:
            secondary.append(action)
    # Working / Waiting: no duplicate refresh.

    last_fail_reason = str(failed.get("error") or failed.get("message") or failed.get("last_error") or "")
    if data_failed and not last_fail_reason:
        last_fail_reason = "DATA_PREPARE failed"

    technical = {
        "snapshot_id": snapshot.get("snapshot_id") or auto.get("snapshot_id"),
        "latest_trading_date": available,
        "expected_session": expected,
        "stale_sessions": freshness.get("stale_sessions"),
        "reason_code": reason or None,
        "source": source or None,
        "sessions": sessions,
        "symbols": symbols,
        "job_id": prepare.get("operation_id") or active_job.get("job_id") or active_job.get("id"),
        "scheduler_state": auto.get("state") or auto.get("operator_state"),
        "last_progress": prepare.get("updated_at") or prepare.get("message"),
        "dependency": waiting_for or None,
        "scan_waiting": bool(scan_op) and not history_current,
        "deferred_or_next_check": prepare.get("next_check_at") or auto.get("next_check_at"),
        "history_current": history_current,
        "data_ready": data_ready,
        "active_operation": prepare.get("kind") or (active_job.get("job_type") if job_is_data else None),
    }

    return _lane(
        "data",
        label="DATA",
        status=status,
        status_code=status_code,
        summary=summary,
        detail=summary,
        what="Official NSE market data used by scan, paper, and learning.",
        meaning=meaning,
        waiting_for=waiting_for,
        current=current,
        next_step=next_step,
        last_success_at=succeeded.get("finished_at") or succeeded.get("updated_at"),
        last_failure_at=failed.get("finished_at") or failed.get("updated_at"),
        last_failure_reason=last_fail_reason,
        progress=_progress_from_op(prepare) if prepare else None,
        current_job=_plain_job(prepare.get("kind") or (active_job.get("job_type") if job_is_data else "")),
        current_job_id=prepare.get("operation_id") or (active_job.get("job_id") if job_is_data else None),
        current_job_started_at=prepare.get("started_at") or (active_job.get("started_at") if job_is_data else None),
        next_check_at=prepare.get("next_check_at") or (active_job.get("next_check_at") if job_is_data else None),
        freshness="current" if history_current and data_ready else (reason.lower() if reason else "unknown"),
        source=source,
        dependencies=dependencies,
        needs_user=status == "Problem",
        recovering=recovering,
        degraded=status == "Problem",
        primary_action=primary,
        secondary_actions=secondary,
        technical=technical,
    )


def _zerodha_lane(*, auto: Mapping[str, Any], kite_ok: bool) -> dict[str, Any]:
    feed = _as_dict(auto.get("live_feed"))
    heartbeat = str(auto.get("heartbeat_ist") or "")
    last_error = str(feed.get("last_error") or "")
    if last_error and any(part in last_error.lower() for part in ("token", "secret", "password")):
        last_error = "Session error"
    ticking = feed.get("symbols_ticking")
    connected = feed.get("connected")
    primary = None
    if not kite_ok:
        primary = _safe_action(
            "",
            label="Login to Zerodha",
            kind="instruction",
            instruction="Run the same one command again, or python main.py login. Home will resume by itself after login.",
        )
    status = "Ready" if kite_ok else "Needs you"
    status_code = "READY" if kite_ok else "NEEDS_YOU"
    summary = "Connected." if kite_ok else "Zerodha login is needed."
    meaning = (
        "Used for live quotes and market observation."
        if kite_ok
        else "Official NSE data and paper learning can continue, but live observation is waiting."
    )
    current = f"Last checked: {_fmt_when(heartbeat)}" if kite_ok and heartbeat else ("Waiting for Zerodha login" if not kite_ok else "Session ok")
    technical = {
        "auth_health": "ok" if kite_ok else "login_required",
        "session_state": auto.get("state"),
        "last_auth_probe": heartbeat or None,
        "live_feed_heartbeat": feed.get("last_connect_ts") or heartbeat or None,
        "symbols_ticking": ticking,
        "instrument_state": {
            "subscribed": feed.get("subscriptions") if feed.get("subscriptions") is not None else feed.get("subscribed_symbols"),
            "fresh": feed.get("fresh_symbols") if isinstance(feed.get("fresh_symbols"), (int, float)) else (
                len(feed.get("fresh_symbols") or []) if isinstance(feed.get("fresh_symbols"), list) else None
            ),
            "stale": len(feed.get("stale_symbols") or []) if isinstance(feed.get("stale_symbols"), list) else feed.get("stale_symbols"),
        },
        "data_source": "zerodha_kite_observation",
        "known_error": last_error or None,
        "plain_state": auto.get("plain_state"),
    }
    return _lane(
        "zerodha",
        label="ZERODHA",
        status=status,
        status_code=status_code,
        summary=summary,
        detail="WAITING FOR ZERODHA LOGIN" if not kite_ok else "Session ok",
        what="Zerodha is the live-quote and observation connection. It cannot place orders from Home.",
        meaning=meaning,
        waiting_for="" if kite_ok else "Zerodha login",
        current=current,
        next_step="Resume live observation after login" if not kite_ok else "Keep watching live quotes",
        needs_user=not kite_ok,
        primary_action=primary,
        technical=technical,
    )


def _automation_lane(
    *,
    auto: Mapping[str, Any],
    ops: Mapping[str, Any],
    freshness: Mapping[str, Any],
    history_current: bool,
    preparing: bool,
    scan_failed: bool,
    scan_ok: bool,
    data_ready: bool,
    now_line: str,
    next_line: str,
) -> dict[str, Any]:
    running = bool(auto.get("running"))
    state = str(auto.get("state") or "")
    operator_state = str(auto.get("operator_state") or "")
    active_job = _as_dict(auto.get("active_job"))
    job_type = str(active_job.get("job_type") or "")
    current_failures = [str(x) for x in _as_list(auto.get("active_failures"))]
    current_failed_jobs = _as_list(auto.get("current_failed_jobs"))
    current_blocked = _as_list(auto.get("current_blocked_critical_jobs"))
    current_counts = _as_dict(auto.get("current_job_counts"))
    historical_counts = _as_dict(auto.get("historical_job_counts"))
    jobs = _as_dict(auto.get("jobs"))
    prepare = _op_kind(ops, "DATA_PREPARE")
    scan_op = _op_kind(ops, "MARKET_SCAN")
    scan_fail = _recent_status(ops, "MARKET_SCAN", "FAILED")
    refresh_bg = bool(auto.get("data_refresh_background")) or job_type == "data_refresh" or bool(prepare)

    pending = current_counts.get("PENDING")
    blocked = current_counts.get("BLOCKED")
    if pending is None:
        pending = jobs.get("PENDING")
    if blocked is None:
        blocked = jobs.get("BLOCKED")

    current = _plain_job(job_type) or now_line
    next_step = next_line
    after_that = ""
    if job_type in {"data_refresh", "DATA_PREPARE"} or prepare:
        current = "Updating market data"
        next_step = "Market scan"
        after_that = "Recommendations → Paper decision"
    elif job_type in {"market_scan"} or scan_op:
        current = "Market scan"
        next_step = "Recommendations → Paper decision"
    elif refresh_bg:
        current = "Updating market data"
        next_step = "Market scan"

    waiting_for = ""
    status = "Waiting"
    status_code = "WAITING"
    summary = "Autonomy supervisor"
    meaning = "Automation is the orchestration brain. It does not invent a second scanner."
    recovering = False
    degraded = False
    primary = None

    genuine_failure = bool(current_failed_jobs or current_blocked or (
        current_failures and operator_state == "DEGRADED"
    ))
    # Historical ledger B/F totals are audit only — never treat as current failure.
    historical_failed = int(historical_counts.get("FAILED", 0) or 0) + int(historical_counts.get("PERMANENT_FAILED", 0) or 0)

    if genuine_failure and not refresh_bg:
        status, status_code = "Problem", "FAILED"
        summary = "Automation needs attention."
        meaning = "A current-session job has stopped making progress. QuantTerm is not using leftover historical failures."
        if current_failed_jobs or current_blocked:
            recovering = False
        primary = _safe_action("RUN_SCAN_NOW", label="Retry") if scan_failed else None
        if data_failed_from_failures(current_failures) and not primary:
            primary = _safe_action("REFRESH_DATA_NOW", label="Retry")
    elif not running:
        status, status_code = "Waiting", "WAITING"
        summary = "Automation is not running right now."
        meaning = "The supervisor heartbeat is not live. Market Operations may still be available separately."
        waiting_for = "Autonomy supervisor"
    elif refresh_bg and not history_current:
        status, status_code = "Waiting", "DELAYED"
        summary = "Today's scan is delayed."
        meaning = "Market data is still updating. QuantTerm will run the scan as soon as the required data is ready."
        waiting_for = "Market data"
        current = "Updating market data"
        next_step = "Market scan"
    elif scan_op or job_type or refresh_bg or preparing:
        status, status_code = "Working", "WORKING"
        summary = "Running normally."
        meaning = "Nothing needed from you."
    elif running and operator_state in {"HEALTHY", "WORKING", ""}:
        status, status_code = "Working", "WORKING"
        summary = "Running normally."
        meaning = "Nothing needed from you."
        current = current or "Watching the market"
    else:
        status, status_code = "Waiting", "WAITING"
        summary = str(auto.get("plain_state") or "Autonomy supervisor")
        meaning = str(auto.get("plain_state") or meaning)

    if scan_failed and not scan_ok and status != "Working":
        if status != "Problem":
            status, status_code = "Problem", "FAILED"
            summary = "The shared scan needs another try."
            meaning = "One Retry uses the same shared scan. It will not start a second scanner."
        primary = primary or _safe_action("RUN_SCAN_NOW", label="Retry")

    technical = {
        "scheduler_owner_pid": auto.get("scheduler_owner_pid"),
        "state": state or None,
        "operator_state": operator_state or None,
        "heartbeat": auto.get("heartbeat_ist") or None,
        "current_job": job_type or None,
        "job_id": active_job.get("job_id") or active_job.get("id") or scan_op.get("operation_id") or prepare.get("operation_id"),
        "job_type": job_type or None,
        "scheduled_for": active_job.get("scheduled_for"),
        "started_at": active_job.get("started_at") or active_job.get("started_monotonic"),
        "attempt": active_job.get("attempt"),
        "current_dependency": waiting_for or active_job.get("blocked_on") or None,
        "next_job": next_step or None,
        "pending_count": pending,
        "blocked_count": blocked,
        "actual_active_failures": current_failures,
        "current_failed_jobs": [
            _omit_empty({"job_type": _as_dict(j).get("job_type"), "error_code": _as_dict(j).get("error_code")})
            for j in current_failed_jobs[:8]
        ],
        "overdue_important_work": None,
        "deferred_work": "data_refresh" if refresh_bg and not history_current else None,
        "last_recovery": None,
        "next_check": active_job.get("next_check_at") or None,
        "historical_failed_jobs": historical_failed or None,
        "historical_note": (
            "Historical ledger failures are audit-only and are not current health."
            if historical_failed else None
        ),
        "data_refresh_background": refresh_bg,
        "scan_waiting_on_data": bool(not history_current and (refresh_bg or not scan_ok)),
    }

    extra = {"after_that": after_that} if after_that else None
    return _lane(
        "automation",
        label="AUTOMATION",
        status=status,
        status_code=status_code,
        summary=summary,
        detail=summary,
        what="Automation runs the shared day: data → scan → recommendations → paper decision → learning.",
        meaning=meaning,
        waiting_for=waiting_for,
        current=current,
        next_step=next_step,
        last_failure_at=scan_fail.get("finished_at") or scan_fail.get("updated_at"),
        last_failure_reason=str(scan_fail.get("error") or scan_fail.get("message") or ""),
        progress=_progress_from_op(scan_op or prepare) if (scan_op or prepare) else None,
        current_job=_plain_job(job_type or (scan_op.get("kind") or prepare.get("kind"))),
        current_job_id=active_job.get("job_id") or scan_op.get("operation_id") or prepare.get("operation_id"),
        current_job_started_at=active_job.get("started_at") or scan_op.get("started_at") or prepare.get("started_at"),
        next_check_at=active_job.get("next_check_at"),
        dependencies=["market_data"] if waiting_for == "Market data" else [],
        needs_user=status == "Problem" and bool(primary),
        recovering=recovering,
        degraded=degraded or status == "Problem",
        primary_action=primary,
        technical=technical,
        extra=extra,
    )


def data_failed_from_failures(failures: Sequence[str]) -> bool:
    return any("data" in f.lower() or "refresh" in f.lower() for f in failures)


def _paper_lane(
    *,
    auto: Mapping[str, Any],
    paper_d: Mapping[str, Any],
    why_d: Mapping[str, Any],
    paper_enabled: bool,
    live_locked: bool,
    taken: Sequence[Mapping[str, Any]],
    opens: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    valid_no_trade: bool,
    cycle_reasons: Sequence[str],
    last_decision: str,
    why_plain: str,
    next_line: str,
) -> dict[str, Any]:
    last_cycle = _as_dict(paper_d.get("last_cycle") or auto.get("last_cycle"))
    positions = _position_rows(opens)
    if not live_locked:
        status, status_code = "Problem", "LIVE_UNLOCKED"
        summary = "Live money must stay locked."
        meaning = "The paper path is the only money path. Home cannot unlock live trading."
        primary = None
    elif not paper_enabled:
        status, status_code = "Needs you", "PAUSED"
        summary = "Paused. Open positions stay watched."
        meaning = "New entries wait until you resume."
        primary = _safe_action("RESUME_NEW_PAPER_ENTRIES", label="Resume")
    elif taken or valid_no_trade or opens or why_d.get("available"):
        status, status_code = "Ready", "READY"
        summary = "ON"
        meaning = why_plain or last_decision or "Paper bot is watching for a good setup."
        primary = None
    else:
        status, status_code = "Waiting", "WAITING"
        summary = "ON"
        meaning = "Waiting for the next scan before another paper decision."
        primary = None

    secondary: list[dict[str, str]] = []
    if paper_enabled and live_locked and status != "Problem":
        pause = _safe_action("PAUSE_NEW_PAPER_ENTRIES", label="Pause new entries")
        cycle = _safe_action("RUN_CYCLE_NOW", label="Run paper cycle now")
        if pause:
            secondary.append(pause)
        if cycle and status == "Ready":
            secondary.append(cycle)

    current = f"Open positions: {len(opens)}"
    if taken:
        current = f"Open positions: {len(opens)} · Today's entries: {len(taken)}"
    next_step = "Another decision after the next scan"
    if opens:
        next_step = "Watch open paper positions through the day"
    elif next_line:
        next_step = next_line

    technical = {
        "selection_authority": "available" if why_d.get("available") else None,
        "policy_result": list(cycle_reasons)[:8] or None,
        "portfolio_result": {
            "open": len(opens),
            "taken_today": len(taken),
            "exits": len(closed),
            "open_risk": paper_d.get("open_risk"),
        },
        "current_risk": paper_d.get("open_risk"),
        "last_cycle_id": last_cycle.get("cycle_id") or last_cycle.get("id") or last_cycle.get("as_of"),
        "decision_id": last_cycle.get("decision_id") or why_d.get("decision"),
        "machine_reason_codes": list(cycle_reasons)[:12] or why_d.get("reasons"),
        "entries_allowed": paper_enabled,
        "live_locked": True,
    }
    extra = {
        "on": paper_enabled,
        "paused": not paper_enabled,
        "positions": positions,
        "positions_open": len(opens),
        "todays_entries": len(taken),
        "last_decision": last_decision,
        "why": why_plain,
    }
    return _lane(
        "paper_bot",
        label="PAPER BOT",
        status=status,
        status_code=status_code,
        summary=summary,
        detail="Paused" if not paper_enabled else (why_d.get("headline") or last_decision or "Paper bot"),
        what="The paper bot may take simulated trades after a scan. It cannot buy with live money.",
        meaning=meaning,
        waiting_for="" if paper_enabled else "Operator resume",
        current=current,
        next_step=next_step,
        needs_user=status == "Needs you",
        degraded=status == "Problem",
        primary_action=primary,
        secondary_actions=secondary,
        technical=technical,
        extra=extra,
    )


def _learning_lane(
    *,
    soak_d: Mapping[str, Any],
    verify: Mapping[str, Any],
    learning_simple: str,
    n_real: int,
) -> dict[str, Any]:
    status_name = str(soak_d.get("FORWARD_SOAK_STATUS") or "NOT_STARTED")
    insufficient = bool(soak_d.get("insufficient_evidence", True))
    settled = soak_d.get("settled_trades")
    rejected = soak_d.get("rejected_candidates_settled")
    coverage = soak_d.get("execution_adjusted_coverage_pct")
    latest = str(verify.get("generated_at") or "")
    collecting = bool(n_real or status_name in {"COLLECTING", "HEALTHY"})
    status = "Working" if collecting else "Waiting"
    status_code = "COLLECTING" if collecting else "WAITING"
    if n_real == 0:
        summary = "QuantTerm is collecting real experience."
        meaning = "Too early to judge. It will learn automatically as real paper decisions settle."
        current = "Real observations: 0"
    else:
        summary = f"Real observations: {n_real}"
        meaning = "Still collecting evidence." if insufficient else learning_simple
        current = f"Real observations: {n_real}"
        if settled is not None:
            current = f"Real observations: {n_real} · Settled trades: {settled}"
        if coverage is not None:
            current = f"{current} · Execution evidence: {coverage}%"

    secondary = []
    verify_action = _safe_action("VERIFY_FORWARD_SOAK", label="Verify now")
    if verify_action:
        secondary.append(verify_action)

    blockers = soak_d.get("promotion_blockers")
    technical = {
        "gross_expectancy": soak_d.get("gross_expectancy"),
        "execution_adjusted_expectancy": soak_d.get("execution_adjusted_expectancy"),
        "promotion_blockers": blockers,
        "policy_version": soak_d.get("policy_version") or soak_d.get("active_policies"),
        "challenger_status": soak_d.get("challengers_under_evaluation"),
        "forward_soak_status": status_name,
        "real_forward_observations": n_real,
        "settled_taken_trades": settled,
        "rejected_waited_settlements": rejected,
        "execution_adjusted_coverage_pct": coverage,
        "latest_learning_update": latest or None,
        "insufficient_evidence": insufficient,
        "evidence_label": soak_d.get("evidence_label"),
        "note": "Gross-only numbers are never proven alpha. Live money stays locked.",
        "live_locked": True,
    }
    extra = {
        "real_forward_observations": n_real,
        "settled_trades": settled,
        "rejected_candidates_settled": rejected,
        "execution_adjusted_coverage_pct": coverage,
        "insufficient_evidence": insufficient,
        "forward_soak_status": status_name,
    }
    return _lane(
        "learning",
        label="LEARNING",
        status=status,
        status_code=status_code,
        summary=summary,
        detail=learning_simple,
        what="Learning records real paper decisions. It does not unlock live money.",
        meaning=meaning,
        current=current,
        next_step="It will learn automatically as real paper decisions settle.",
        last_success_at=latest or None,
        secondary_actions=secondary,
        technical=technical,
        extra=extra,
    )
