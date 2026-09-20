"""Home operating-system projection.

Reads existing artifacts. Does not scan, recommend, or trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from product.operator_language import explain_opportunity, simple_reason

SCHEMA_VERSION = 3

NORMAL = "NORMAL"
LOGIN_REQUIRED = "LOGIN_REQUIRED"
PREPARING = "PREPARING"
FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
MARKET_CLOSED_COMPLETE = "MARKET_CLOSED_COMPLETE"
NO_TRADE = "NO_TRADE"
PAUSED = "PAUSED"
PROBLEM = "PROBLEM"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _action(control: str = "", *, label: str, kind: str = "control", instruction: str = "") -> dict[str, str]:
    return {
        "id": control or kind,
        "control": control,
        "label": label,
        "kind": kind,
        "instruction": instruction,
    }


def _live_safety_projection() -> dict[str, Any]:
    """Project canonical broker-boundary safety without inventing proof."""
    try:
        from product.live_readiness import evaluate_live_readiness

        result = dict(evaluate_live_readiness() or {})
        verified = result.get("live_lock_verified") is True
        locked = result.get("live_locked") if verified else None
        authorized = result.get("live_execution_authorized") if verified else None
        return {
            "live_locked": locked,
            "live_lock_verified": verified,
            "live_execution_authorized": authorized,
            "live_lock_status": str(result.get("live_lock_status") or ("LOCKED" if locked is True else "UNLOCKED" if locked is False else "UNVERIFIED")),
            "live_lock_reason": str(result.get("live_lock_reason") or ""),
            "live_lock_source": str(result.get("live_lock_source") or "product.live_execution_interlock"),
        }
    except Exception as exc:
        return {
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": f"Canonical live-execution interlock could not be verified: {exc}",
            "live_lock_source": "product.live_execution_interlock",
        }


def _session_phase(now: datetime | None = None) -> str:
    try:
        from zoneinfo import ZoneInfo
        from research.autonomy import schedules as sch
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        return str(sch.session_phase(clock.astimezone(ZoneInfo("Asia/Kolkata"))) or "off_session")
    except Exception:
        return "unknown"


def _load_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from product.paper_status import read_paper_status
        paper = read_paper_status()
        out["paper"] = {
            "enabled": paper.enabled,
            "open_positions": list(paper.open_positions or []),
            "closed_trades": list(paper.closed_trades or []),
            "supervisor_running": paper.supervisor_running,
            "open_risk": paper.open_risk,
            "last_cycle": paper.last_cycle,
        }
    except Exception:
        out["paper"] = {}
    try:
        from product.autonomy_status import read_autonomy_status
        out["autonomy"] = read_autonomy_status()
    except Exception:
        out["autonomy"] = {}
    try:
        from product.autopilot_journal import load_journal, why_no_trade
        out["why"] = why_no_trade()
        out["journal"] = load_journal()
    except Exception:
        out["why"] = {}
        out["journal"] = {}
    try:
        from product.forward_soak import scoreboard, load_latest_verification
        out["soak"] = scoreboard()
        out["soak_verify"] = load_latest_verification()
    except Exception:
        out["soak"] = {}
        out["soak_verify"] = {}
    try:
        from product.scan_store import default_scan_path
        from product.forward_soak import _read_json, _scan_payload
        out["scan"] = _scan_payload().get("payload") or _read_json(default_scan_path())
    except Exception:
        out["scan"] = {}
    try:
        from product.forward_soak import _reco_payload
        out["reco"] = _reco_payload().get("payload") or {}
    except Exception:
        out["reco"] = {}
    try:
        from operations.store import OperationStore
        from terminal_api import OPS_DB
        store = OperationStore(OPS_DB)
        out["operations"] = {"active": store.active(), "recent": store.recent(limit=12)}
    except Exception:
        out["operations"] = {"active": [], "recent": []}
    try:
        from data.bhavcopy_runtime import official_history_freshness
        out["history_freshness"] = official_history_freshness(require_store=False)
    except Exception:
        out["history_freshness"] = {}
    return out


_AUTH_STATES = frozenset({"AUTH_REQUIRED", "TOKEN_MISSING", "SESSION_EXPIRED"})
_AUTH_REASON_CODES = frozenset({
    "auth_health",
    "token_missing",
    "session_expired",
    "auth_missing",
    "auth_expired",
})


def broker_session_usable(auto: Mapping[str, Any] | None) -> bool:
    """True only with evidence of a usable broker session. Empty snapshots fail closed."""
    auto = _as_dict(auto)
    if not auto:
        return False
    broker = _as_dict(auto.get("broker"))
    if broker:
        return bool(broker.get("ready") or broker.get("live_data_ready"))
    state = str(auto.get("state") or "").strip().upper()
    if not state or state in _AUTH_STATES or state == "UNKNOWN":
        return False
    if auto.get("available") is False:
        return False
    if auto.get("kite_connected") is False:
        return False
    failures = [str(x).lower() for x in (auto.get("active_failures") or [])]
    if any("auth" in item or "token" in item for item in failures):
        return False
    if str(auto.get("reason_code") or "").lower() in _AUTH_REASON_CODES:
        return False
    explanation = str(auto.get("explanation") or "").lower()
    if "login is required" in explanation or "zerodha login" in explanation:
        return False
    notes = [str(n).lower() for n in (auto.get("capability_notes") or [])]
    if any("zerodha login" in n or "re-login" in n or "session expired" in n for n in notes):
        return False
    feed = _as_dict(auto.get("live_feed"))
    if str(feed.get("status") or "").upper() in _AUTH_STATES:
        return False
    feed_error = str(feed.get("last_error") or "").lower()
    if feed.get("connected") is False and any(
        token in feed_error for token in ("credential", "access token", "login", "auth")
    ):
        return False
    return True


def _canonical_best_trade_rows(*, history_current: bool) -> list[dict[str, Any]]:
    """Read only the canonical production-thesis discovery shortlist for Home.

    Home's primary trade list must never mix scanner rankings, committee rejects,
    WAIT rows or extended names into the same "opportunities" bucket. The
    Decision Simulation gate already owns the current best-trade projection and
    is tied to the fresh saved scan + production thesis, so Home consumes that
    exact read-only output.
    """
    if not history_current:
        return []
    try:
        from product.decision_simulation_gate import status as decision_simulation_status

        gate = dict(decision_simulation_status() or {})
    except Exception:
        return []
    if not gate.get("scan_fresh") or not gate.get("discovery_ready"):
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in gate.get("best_trades") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue
        discovery = str(row.get("discovery_decision") or "").upper().strip()
        committee = str(row.get("decision") or row.get("committee_decision") or "").upper().strip()
        if discovery and discovery != "ENTER_NOW":
            continue
        if committee and committee != "BUY":
            continue
        # ENTER_NOW is the production selection seam's actionable state. Give
        # the Home language layer the matching canonical decision so it cannot
        # fall back to scanner-only labels such as "research candidate".
        if discovery == "ENTER_NOW" and not committee:
            row["decision"] = "BUY"
        out.append(row)
        seen.add(symbol)
        if len(out) >= 3:
            break
    return out


def build_home_os(
    *,
    dashboard: Mapping[str, Any] | None = None,
    paper: Mapping[str, Any] | None = None,
    autonomy: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    why: Mapping[str, Any] | None = None,
    soak: Mapping[str, Any] | None = None,
    soak_verify: Mapping[str, Any] | None = None,
    scan: Mapping[str, Any] | None = None,
    reco: Mapping[str, Any] | None = None,
    journal: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
    radar: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    recovered: Sequence[str] | None = None,
) -> dict[str, Any]:
    loaded = {} if any(x is not None for x in (dashboard, paper, why, soak)) else _load_defaults()
    dash = _as_dict(dashboard)
    paper_d = _as_dict(paper if paper is not None else dash.get("paper") or loaded.get("paper"))
    auto = _as_dict(autonomy if autonomy is not None else dash.get("autonomy") or loaded.get("autonomy"))
    data_d = _as_dict(data if data is not None else dash.get("data") or {})
    why_d = _as_dict(why if why is not None else loaded.get("why"))
    soak_d = _as_dict(soak if soak is not None else loaded.get("soak"))
    verify = _as_dict(soak_verify if soak_verify is not None else loaded.get("soak_verify"))
    scan_d = _as_dict(scan if scan is not None else dash.get("scan") or loaded.get("scan"))
    reco_d = _as_dict(reco if reco is not None else loaded.get("reco"))
    journal_d = _as_dict(journal if journal is not None else loaded.get("journal"))
    ops = _as_dict(operations if operations is not None else dash.get("operations") or loaded.get("operations"))
    radar_d = _as_dict(radar)

    kite_ok = broker_session_usable(auto)
    paper_enabled = paper_d.get("enabled", True) is not False
    owner_state = _as_dict(auto.get("owner_state"))
    observe_date = str(owner_state.get("observe_only_date") or "")[:10]
    today_ist = ""
    try:
        from zoneinfo import ZoneInfo
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        today_ist = clock.astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()
    except Exception:
        today_ist = datetime.now(timezone.utc).date().isoformat()
    observe_only = bool(observe_date and observe_date == today_ist)
    bhav = dict(data_d.get("bhavcopy") or {})
    freshness = _history_freshness(data_d, loaded.get("history_freshness"), now)
    from product.data_readiness import project_official_data_readiness
    data_truth = project_official_data_readiness(
        freshness=freshness,
        data=data_d,
        bhav=bhav,
    )
    history_current = bool(data_truth["history_current"])
    known_reason = str(freshness.get("reason_code") or bhav.get("reason_code") or "")
    has_session = bool(
        bhav.get("latest_date")
        or freshness.get("available_session")
        or freshness.get("expected_latest_completed_session")
        or (known_reason and known_reason != "HISTORY_UNKNOWN")
    )
    # Official HISTORY_CURRENT is the DATA lane. radar_home_workspace calls
    # without a data= payload, so empty data_d/bhav must not default to not-ready.
    # The projector equals history_current; pickle/store "ready" is a different plane.
    data_ready = bool(data_truth["data_ready"])
    scan_ok = bool(scan_d.get("records") or scan_d.get("available") or scan_d.get("scanned_at"))
    if has_session and not history_current:
        scan_ok = False
    reco_ok = bool(reco_d.get("categories") is not None or reco_d.get("schema_version"))
    latest = dict(journal_d.get("latest") or why_d)
    taken = list(why_d.get("taken") or latest.get("taken") or [])
    rejections = list(why_d.get("rejections") or latest.get("rejections") or [])
    waits = list(why_d.get("waits") or latest.get("waits") or [])
    cycle_reasons = [str(x) for x in (latest.get("cycle_reasons") or why_d.get("reasons") or []) if x]
    valid_no_trade = bool(why_d.get("available")) and not taken and (bool(rejections) or bool(waits) or bool(cycle_reasons))
    opens = list(paper_d.get("open_positions") or [])
    closed = list(paper_d.get("closed_trades") or [])
    active_ops = [o for o in list(ops.get("active") or []) if isinstance(o, Mapping)]
    active_kinds = {str(o.get("kind") or "") for o in active_ops}
    preparing = bool(active_kinds & {"DATA_PREPARE", "MARKET_SCAN"})
    data_failed = any(str(o.get("kind")) == "DATA_PREPARE" and str(o.get("status")) == "FAILED" for o in list(ops.get("recent") or []))
    scan_failed = any(str(o.get("kind")) == "MARKET_SCAN" and str(o.get("status")) == "FAILED" for o in list(ops.get("recent") or []))
    phase = _session_phase(now)
    market_closed = phase in {"eod", "off_session", "postmarket"}
    verify_lanes = {
        str(name): str(status or "").upper()
        for name, status in dict(verify.get("lanes") or {}).items()
    }
    has_verification = bool(verify_lanes)
    core_lane_ok = {
        "SCAN": {"PASS"},
        "RECOMMENDATIONS": {"PASS"},
        "SELECTION": {"PASS"},
        "AUTOPILOT": {"PASS"},
        "PAPER EXECUTION": {"PASS", "NO_ELIGIBLE_TRADE"},
        "LIVE MONEY": {"LOCKED"},
    }
    eod_done = bool(has_verification) and all(
        verify_lanes.get(name) in allowed
        for name, allowed in core_lane_ok.items()
    )
    verified_no_trade = bool(
        eod_done and verify_lanes.get("PAPER EXECUTION") == "NO_ELIGIBLE_TRADE"
    )

    live_safety = _live_safety_projection()
    live_locked = live_safety.get("live_locked")
    live_lock_verified = live_safety.get("live_lock_verified") is True

    primary_action = None
    secondary: list[dict[str, str]] = []
    state = NORMAL
    headline = "Everything is running automatically."
    subtext = "QuantTerm is scanning, paper trading and learning. You can leave it running."
    now_line = "Watching the market"
    next_line = "Next automatic paper decision after the scan"

    broker_login_required = not kite_ok
    broker_action_required = bool(
        broker_login_required and paper_enabled and not observe_only and not market_closed
    )
    broker_action = _action(
        label="Login to Zerodha",
        kind="instruction",
        instruction=(
            "Run python main.py login to enable broker-live quotes and broker-dependent paper entry. "
            "Official data, scanning, research, replay, settlement and learning continue without it."
        ),
    )

    if not live_lock_verified:
        state = PROBLEM
        headline = "Live-money safety could not be verified."
        subtext = live_safety.get("live_lock_reason") or "The canonical broker-boundary interlock is unverified."
        now_line = "Live execution safety unverified"
        next_line = "Verify the canonical broker-boundary lock before accepting the system"
    elif live_locked is not True:
        state = PROBLEM
        headline = "Live money must stay locked."
        subtext = "The canonical broker boundary is not locked. The paper path is the only accepted money path."
        now_line = "Paper bot only"
        next_line = "Restore the canonical live-execution lock"
    elif data_failed and not data_ready:
        state = FAILED_RECOVERABLE
        headline = "Market data needs another try."
        subtext = "Official prices did not finish. One Retry uses the same data lane."
        primary_action = _action("REFRESH_DATA_NOW", label="Retry")
        secondary = [_action("RUN_SCAN_NOW", label="Scan now")]
        now_line = "Data refresh failed"
        next_line = "Retry official prices, then scan"
    elif scan_failed and not scan_ok:
        state = FAILED_RECOVERABLE
        headline = "The market scan needs another try."
        subtext = "One Retry uses the same shared scan. It will not start a second scanner."
        primary_action = _action("RUN_SCAN_NOW", label="Retry")
        now_line = "Scan failed"
        next_line = "Shared market scan, then recommendations"
    elif (has_session and not history_current) or preparing or (not data_ready and not scan_ok):
        state = PREPARING
        if has_session and not history_current:
            headline = "Today's latest market data is still being prepared."
            subtext = "Getting the latest market data before scanning."
            now_line = "Preparing official data"
            next_line = "Shared market scan after current prices"
        elif preparing and history_current:
            headline = "Official prices are current. QuantTerm is finishing the next automatic step."
            subtext = "The DATA lane is ready. A background job is still running and will show its own progress."
            now_line = "Market scan running" if any(str(o.get("kind")) == "MARKET_SCAN" for o in active_ops) else "Finishing the official data store"
            next_line = "Recommendations and paper decision"
        else:
            headline = "QuantTerm is getting today's market ready."
            subtext = "No extra click is needed. Progress is the desk pipeline you already have."
            now_line = "Preparing official data" if not data_ready else "Market scan running"
            next_line = "Recommendations and paper decision"
        if any(str(o.get("kind")) == "MARKET_SCAN" for o in active_ops) and history_current:
            op = next(o for o in active_ops if str(o.get("kind")) == "MARKET_SCAN")
            cur = op.get("current") or op.get("progress_current")
            tot = op.get("total") or op.get("progress_total")
            if cur and tot:
                now_line = f"Market scan running · {cur} / {tot}"
        elif any(str(o.get("kind")) == "DATA_PREPARE" for o in active_ops):
            now_line = "Preparing official data" if not history_current else "Finishing the official data store"
    elif not paper_enabled:
        state = PAUSED
        headline = "The paper bot is paused."
        subtext = "Open positions stay watched. New entries wait until you resume."
        primary_action = _action("RESUME_NEW_PAPER_ENTRIES", label="Resume paper entries")
        secondary = [_action("RUN_SCAN_NOW", label="Scan now")]
        now_line = "Watching open paper positions" if opens else "Paper entries paused"
        next_line = "Resume when you want new paper trades"
    elif market_closed and (has_verification or valid_no_trade or taken or closed):
        settle_job_active = bool(active_kinds & {"OUTCOME_RESOLUTION", "outcome_resolution"})
        pending_settle = (
            verify_lanes.get("FORWARD SETTLEMENT") == "PENDING"
            and not closed
            and (phase == "eod" or settle_job_active)
        )
        if has_verification and not eod_done:
            unfinished = [
                f"{name} {verify_lanes.get(name) or 'UNKNOWN'}"
                for name, allowed in core_lane_ok.items()
                if verify_lanes.get(name) not in allowed
            ]
            state = NORMAL
            headline = "Market is closed. Today's QuantTerm workflow is not complete yet."
            subtext = (
                "Automation is still reconciling required day evidence"
                + (f": {', '.join(unfinished[:4])}." if unfinished else ".")
                + " Leave QuantTerm running."
            )
            now_line = "Market closed · day workflow incomplete"
            next_line = "Finish required scan, selection and paper verification"
        elif pending_settle and not verified_no_trade:
            state = NORMAL
            headline = "Today's market is closed. Settlement is still finishing."
            subtext = "Leave QuantTerm running. End-of-day work is automatic."
            now_line = "End-of-day settlement"
            next_line = "Learning journal and forward evidence"
        else:
            no_trade_for_day = verified_no_trade if has_verification else valid_no_trade
            state = NO_TRADE if no_trade_for_day and not taken else MARKET_CLOSED_COMPLETE
            if no_trade_for_day and not taken:
                headline = "No trade today — QuantTerm did not find a setup worth taking."
                subtext = "The paper-decision lane is complete. Longer-horizon evidence can continue automatically."
            else:
                headline = "Today's market work is complete."
                subtext = "Required scan, selection and paper-decision lanes are verified for the day."
            now_line = "Market closed"
            next_line = "Tomorrow's official data, then scan"
    elif valid_no_trade:
        state = NO_TRADE
        headline = "No trade today — nothing was good enough."
        subtext = simple_reason(cycle_reasons[0] if cycle_reasons else why_d.get("decision"), fallback=why_d.get("headline") or "")
        now_line = "Watching for a better setup" if phase not in {"eod", "off_session"} else "Market work recorded"
        next_line = "Next scan, then another paper decision"
        secondary = [_action("PAUSE_NEW_PAPER_ENTRIES", label="Pause paper entries")]
    else:
        state = NORMAL
        if taken:
            names = ", ".join(str(t.get("symbol") or "") for t in taken[:4] if t.get("symbol"))
            headline = "Everything is running automatically."
            subtext = f"Paper bot took {len(taken)} name(s){': ' + names if names else ''}."
            now_line = "Watching open paper positions" if opens else "Paper decision recorded"
        elif opens:
            now_line = "Watching open paper positions"
            next_line = "End-of-day settlement after market close"
        elif preparing:
            now_line = "Market scan running"
        secondary = [
            _action("RUN_SCAN_NOW", label="Scan now"),
            _action("PAUSE_NEW_PAPER_ENTRIES", label="Pause paper entries"),
            _action("OBSERVE_ONLY_TODAY", label="Observe only today"),
        ]

    if observe_only and state in {NORMAL, NO_TRADE, MARKET_CLOSED_COMPLETE}:
        if "nothing was good enough" not in headline.lower() and "did not find" not in headline.lower():
            subtext = (
                "Observe only today. Paper decisions and learning continue. "
                "Live money stays locked. You are not participating."
            )
        else:
            subtext = f"{subtext} Observe only today — paper still records the day."

    if state == NORMAL and not primary_action:
        primary_action = None

    progress = None
    for op in active_ops:
        if str(op.get("kind")) == "MARKET_SCAN":
            progress = {
                "kind": "MARKET_SCAN",
                "label": "Market scan",
                "current": op.get("current") or op.get("done"),
                "total": op.get("total"),
                "status": op.get("status"),
            }

    best_rows = _canonical_best_trade_rows(history_current=history_current)
    opportunities = [explain_opportunity(row) for row in best_rows]

    n_real = int(soak_d.get("real_forward_observations") or 0)
    learning_simple = (
        f"{n_real} real observations. Still too early to judge."
        if n_real and soak_d.get("insufficient_evidence", True)
        else ("QuantTerm is collecting real trading experience." if n_real == 0 else f"{n_real} real observations.")
    )
    last_decision = why_d.get("headline") or latest.get("summary") or "No paper decision yet."
    why_plain = simple_reason(
        (cycle_reasons[0] if cycle_reasons else why_d.get("decision")),
        fallback=last_decision,
    )

    from product.backend_control_plane import build_check_system, build_system_lanes

    system = build_system_lanes(
        auto=auto,
        data_d=data_d,
        paper_d=paper_d,
        why_d=why_d,
        soak_d=soak_d,
        ops=ops,
        freshness=freshness,
        verify=verify,
        kite_ok=kite_ok,
        data_ready=data_ready,
        history_current=history_current,
        preparing=preparing,
        data_failed=data_failed,
        scan_failed=scan_failed,
        scan_ok=scan_ok,
        paper_enabled=paper_enabled,
        live_locked=live_locked,
        live_lock_verified=live_lock_verified,
        taken=taken,
        opens=opens,
        closed=closed,
        valid_no_trade=valid_no_trade,
        cycle_reasons=cycle_reasons,
        last_decision=last_decision,
        why_plain=why_plain,
        now_line=now_line,
        next_line=next_line,
        learning_simple=learning_simple,
        n_real=n_real,
    )
    if broker_action_required:
        zerodha = dict(system.get("zerodha") or {})
        zerodha.update({
            "status": "Needs you",
            "status_code": "LOGIN_REQUIRED",
            "summary": "Login required for broker-dependent paper entry.",
            "detail": "Research and official-data autonomy continue without Zerodha.",
            "meaning": "Log in to enable broker-live quotes and paper entry; this is not a system failure.",
            "needs_user": True,
            "optional_capability": False,
            "blocks_autonomy": False,
            "blocks_live_money": False,
            "primary_action": broker_action,
        })
        system["zerodha"] = zerodha
    check_system = build_check_system(
        system,
        live_locked=live_locked,
        live_lock_verified=live_lock_verified,
        live_lock_reason=str(live_safety.get("live_lock_reason") or ""),
        live_lock_source=str(live_safety.get("live_lock_source") or "product.live_execution_interlock"),
    )

    runtime: dict[str, Any] = {}
    try:
        from product.runtime_lifecycle import inspect_runtime
        runtime = inspect_runtime(api_serving=True)
    except Exception as exc:
        runtime = {
            "lifecycle": "FAILED",
            "reason": str(exc)[:240],
            "reasons": [str(exc)[:240]],
            "components": [],
        }
    lifecycle = str(runtime.get("lifecycle") or "")
    if lifecycle == "FAILED":
        state = FAILED_RECOVERABLE
        headline = "A required backend process is down."
        subtext = str(runtime.get("reason") or "The desk will not hide a dead backend.")
        now_line = f"FAILED · {runtime.get('reason') or 'see logs/stack'}"
        next_line = "Supervisor restarts the failed process when recovery is safe"
        primary_action = _action("CHECK_SYSTEM", label="Check system", kind="refresh")
    elif lifecycle == "RECOVERING":
        headline = "QuantTerm is recovering a failed process."
        subtext = str(runtime.get("reason") or "A worker heartbeat is stale and the supervisor is restarting it.")
        now_line = f"RECOVERING · {now_line}"
        next_line = "Confirm health after the restart"
    elif lifecycle == "DEGRADED" and state in {NORMAL, PREPARING}:
        now_line = f"DEGRADED · {now_line}"
        if runtime.get("reason") and state == NORMAL:
            subtext = str(runtime.get("reason"))

    activity = _activity(scan_d, why_d, latest, ops, verify, taken, recovered=list(recovered or []))
    yesterday = _yesterday(verify, soak_d, why_d, scan_ok, reco_ok, live_safety)
    readiness: dict[str, Any] = {}
    try:
        from product.readiness import inspect_readiness
        readiness = inspect_readiness()
    except Exception:
        readiness = {}

    required_attention: list[dict[str, Any]] = []
    if state in {LOGIN_REQUIRED, FAILED_RECOVERABLE, PAUSED, PROBLEM}:
        required_attention.append({
            "id": str((primary_action or {}).get("id") or state),
            "label": str((primary_action or {}).get("label") or headline),
            "reason": subtext,
            "action": primary_action,
        })
    if broker_action_required and state not in {FAILED_RECOVERABLE, PAUSED, PROBLEM}:
        state = LOGIN_REQUIRED
        headline = "Zerodha login is needed for broker-dependent paper entry."
        subtext = (
            "This is the only expected human step. Official data, scans, research, replay, "
            "settlement and learning continue automatically."
        )
        now_line = "Non-broker autonomy is running"
        next_line = "Broker paper entry resumes after login"
        primary_action = broker_action
        required_attention = [{
            "id": "BROKER_LOGIN_REQUIRED",
            "label": "Login to Zerodha",
            "reason": subtext,
            "action": broker_action,
        }]

    optional_attention: list[dict[str, Any]] = []
    if broker_login_required and not broker_action_required:
        optional_attention.append({
            "id": "BROKER_LOGIN_OPTIONAL",
            "label": "Connect Zerodha",
            "reason": (
                "Optional right now: enables broker-live quotes and broker-dependent paper entry. "
                "Official data, scan, research, shadow tracking, settlement and learning continue without it."
            ),
            "action": broker_action,
        })
    need_me = bool(required_attention)

    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "headline": headline,
        "subtext": subtext,
        "need_me": need_me,
        "attention": {
            "required_now": required_attention,
            "optional": optional_attention,
            "count": len(required_attention),
            "optional_count": len(optional_attention),
            "headline": "Action required" if required_attention else "No action required",
        },
        "primary_action": primary_action,
        "secondary_actions": (secondary + [_action("SIMULATE_PAST_DECISIONS", label="Simulate past decisions")])[:4],
        "simulate_action": _action("SIMULATE_PAST_DECISIONS", label="Simulate past decisions"),
        "past_decisions": _past_decisions(live_safety),
        "now": now_line,
        "next": next_line,
        "progress": progress,
        "today": {
            "market_open": not market_closed,
            "market_phase": phase,
            "market_mood": str((dash.get("market") or {}).get("health") or radar_d.get("market_health") or ""),
            "scan_age": str(scan_d.get("scanned_at") or radar_d.get("scan_scanned_at") or ""),
            "data_fresh": bool(data_ready and history_current),
            "expected_session": freshness.get("expected_latest_completed_session") or "",
            "available_session": freshness.get("available_session") or bhav.get("latest_date") or "",
            "stale_sessions": freshness.get("stale_sessions"),
            "history_reason_code": freshness.get("reason_code") or "",
            "last_automatic_action": now_line,
            "next_automatic_action": next_line,
        },
        "opportunities": opportunities[:8],
        "observe_only": observe_only,
        "observe_only_date": observe_date if observe_only else "",
        "paper_bot": {
            "on": bool(paper_enabled),
            "paused": not paper_enabled,
            "positions_open": len(opens),
            "todays_entries": len(taken),
            "exits": len(closed),
            "risk_used": paper_d.get("open_risk"),
            "last_decision": last_decision,
            "why": why_plain,
            "why_technical": cycle_reasons or why_d.get("reasons") or [],
            "positions": list(system.get("paper_bot", {}).get("positions") or []),
        },
        "learning": {
            "simple": learning_simple,
            "real_forward_n": n_real,
            "execution_adjusted_coverage_pct": soak_d.get("execution_adjusted_coverage_pct"),
            "insufficient_evidence": bool(soak_d.get("insufficient_evidence", True)),
            "forward_soak_status": soak_d.get("FORWARD_SOAK_STATUS") or "NOT_STARTED",
            "promotion_blockers": soak_d.get("promotion_blockers"),
            "live_locked": live_locked,
            "live_lock_verified": live_lock_verified,
        },
        "system": system,
        "check_system": check_system,
        "recent_activity": activity,
        "yesterday": yesterday,
        "recovered": list(recovered or []),
        **live_safety,
        "broker": {
            "status": (
                "LOGIN_REQUIRED" if broker_action_required
                else "OPTIONAL_LOGIN" if broker_login_required
                else "READY"
            ),
            "login_required": broker_login_required,
            "requires_operator_now": broker_action_required,
            "blocks_autonomy": False,
            "detail": (
                "Login is required for broker-dependent paper entry; non-broker autonomy continues."
                if broker_action_required
                else "Broker-live quotes and broker-dependent paper entry are unavailable until login; autonomous research work continues."
                if broker_login_required
                else "Broker session is usable."
            ),
        },
        "readiness": readiness,
        "runtime": runtime,
        "history_freshness": {
            "current": history_current,
            "expected_latest_completed_session": freshness.get("expected_latest_completed_session") or "",
            "available_session": freshness.get("available_session") or bhav.get("latest_date") or "",
            "stale_sessions": freshness.get("stale_sessions"),
            "reason_code": freshness.get("reason_code") or "",
        },
        "verify": {
            "lanes": (verify.get("lanes") or {}),
            "soak_status": verify.get("soak_status") or soak_d.get("FORWARD_SOAK_STATUS"),
            "generated_at": verify.get("generated_at") or "",
        },
        "four_questions": {
            "what": "Home is QuantTerm's operating system for one market day.",
            "found": headline,
            "meaning": subtext,
            "action": (primary_action or {}).get("label") if need_me else "Nothing. Leave it running.",
        },
    }


def _history_freshness(
    data_d: Mapping[str, Any],
    loaded: Any,
    now: datetime | None,
) -> dict[str, Any]:
    bhav = dict(data_d.get("bhavcopy") or {})
    if isinstance(data_d.get("history_freshness"), Mapping):
        return dict(data_d.get("history_freshness") or {})
    if bhav.get("reason_code") or "current" in bhav or bhav.get("expected_latest_completed_session"):
        current_flag = bhav.get("current")
        reason = str(bhav.get("reason_code") or "")
        if current_flag is None:
            current_flag = reason == "HISTORY_CURRENT"
        return {
            "current": bool(current_flag),
            "expected_latest_completed_session": bhav.get("expected_latest_completed_session") or "",
            "available_session": bhav.get("available_session") or bhav.get("latest_date") or "",
            "stale_sessions": bhav.get("stale_sessions"),
            "reason_code": reason,
        }
    if bhav.get("latest_date"):
        try:
            from data.bhavcopy_runtime import official_history_freshness
            return official_history_freshness(bhav, now=now, load_cache=False, require_store=False)
        except Exception:
            return {"current": False, "available_session": bhav.get("latest_date"), "reason_code": "HISTORY_PROBE_FAILED"}
    if isinstance(loaded, Mapping) and (
        "current" in loaded or loaded.get("reason_code") or loaded.get("expected_latest_completed_session")
    ):
        return dict(loaded)
    if isinstance(loaded, Mapping) and loaded.get("history_freshness"):
        return dict(loaded.get("history_freshness") or {})
    return {"current": False, "reason_code": "HISTORY_UNKNOWN"}


def _activity(
    scan: Mapping[str, Any],
    why: Mapping[str, Any],
    latest: Mapping[str, Any],
    ops: Mapping[str, Any],
    verify: Mapping[str, Any],
    taken: Sequence[Mapping[str, Any]],
    *,
    recovered: Sequence[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    scan_at = str(scan.get("scanned_at") or "")
    if scan_at:
        records_n = len(list(scan.get("records") or []))
        try:
            scanned_n = int(scan.get("scanned") or scan.get("universe_size") or 0)
        except (TypeError, ValueError):
            scanned_n = 0
        n = scanned_n or records_n
        rows.append({"at": scan_at, "text": f"Market scan completed" + (f" · {n} names checked" if n else "")})
    if why.get("available"):
        stamp = str(latest.get("recorded_at") or latest.get("as_of") or why.get("as_of") or "")
        if taken:
            names = ", ".join(str(t.get("symbol") or "") for t in taken[:4] if t.get("symbol"))
            rows.append({"at": stamp, "text": f"Paper position opened · {names}".strip(" ·")})
        else:
            rows.append({"at": stamp, "text": f"No paper trade — {simple_reason((why.get('reasons') or ['NO_TRADE'])[0])}"})
    for op in list(ops.get("recent") or [])[:6]:
        if not isinstance(op, Mapping):
            continue
        kind = str(op.get("kind") or "")
        status = str(op.get("status") or "")
        when = str(op.get("finished_at") or op.get("requested_at") or "")
        if kind and status:
            rows.append({"at": when, "text": f"{kind.replace('_', ' ').title()} {status.lower()}"})
    lanes = dict(verify.get("lanes") or {})
    if lanes.get("FORWARD SETTLEMENT") == "PASS":
        rows.append({"at": str(verify.get("generated_at") or ""), "text": "End-of-day settlement complete"})
    if lanes.get("LEARNING INGESTION") == "PASS":
        rows.append({"at": str(verify.get("generated_at") or ""), "text": "Learning journal updated"})
    for item in recovered:
        rows.append({"at": "", "text": f"Recovered {item} — one owner, no duplicate work"})
    rows.sort(key=lambda r: r.get("at") or "", reverse=True)
    return rows[:12]


def _yesterday(
    verify: Mapping[str, Any],
    soak: Mapping[str, Any],
    why: Mapping[str, Any],
    scan_ok: bool,
    reco_ok: bool,
    live_safety: Mapping[str, Any],
) -> dict[str, Any]:
    lanes = dict(verify.get("lanes") or {})
    return {
        "scan": scan_ok or lanes.get("SCAN") == "PASS",
        "paper_decisions": bool(why.get("available")) or lanes.get("SELECTION") == "PASS",
        "settlement": lanes.get("FORWARD SETTLEMENT") in {"PASS", "PENDING"},
        "settlement_pending": lanes.get("FORWARD SETTLEMENT") == "PENDING",
        "learning": lanes.get("LEARNING INGESTION") in {"PASS", "PENDING"},
        "forward_evidence": lanes.get("FORWARD SETTLEMENT") == "PASS" or bool(soak.get("real_forward_observations")),
        "live_locked": live_safety.get("live_locked"),
        "live_lock_verified": live_safety.get("live_lock_verified") is True,
    }


def _past_decisions(live_safety: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from product.decision_simulator import load_latest
        report = load_latest()
    except Exception:
        report = {}
    safety = {
        "live_locked": live_safety.get("live_locked"),
        "live_lock_verified": live_safety.get("live_lock_verified") is True,
    }
    if not report:
        return {"available": False, "provenance": "BACKTEST", **safety}
    return {
        "available": True,
        "provenance": report.get("provenance") or "BACKTEST",
        "status": report.get("status"),
        "run_id": report.get("run_id"),
        "engine": report.get("engine"),
        "period_start": report.get("period_start"),
        "period_end": report.get("period_end"),
        "trading_sessions": report.get("trading_sessions"),
        "sessions_done": report.get("sessions_done"),
        "sessions_total": report.get("sessions_total"),
        "universe_observations": report.get("universe_observations"),
        "stocks_evaluated": report.get("stocks_evaluated"),
        "decisions_tested": report.get("decisions_tested") or 0,
        "would_take": report.get("would_take") or report.get("BUY"),
        "rejected": report.get("rejected"),
        "BUY": report.get("BUY"),
        "WAIT": report.get("WAIT"),
        "AVOID": report.get("AVOID"),
        "REJECT": report.get("REJECT"),
        "correct_rejections": report.get("correct_rejections"),
        "missed_winners": report.get("missed_winners"),
        "avoided_losers": report.get("avoided_losers"),
        "good_waits": report.get("good_waits"),
        "outcomes_matured": report.get("outcomes_matured"),
        "open_unresolved": report.get("open_unresolved"),
        "filters_helped": report.get("filters_helped") or [],
        "filters_hurt": report.get("filters_hurt") or [],
        "simple": report.get("simple") or "",
        "note": report.get("note") or "",
        "PIT_STRONG": report.get("PIT_STRONG"),
        "PIT_PARTIAL": report.get("PIT_PARTIAL"),
        "PIT_MARKET_ONLY": report.get("PIT_MARKET_ONLY"),
        "scorecards_note": "Method/family scorecards stay inspect-only until sample floors.",
        **safety,
        "not_promotion_evidence": True,
    }
