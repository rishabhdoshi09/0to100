"""Bounded EOD settlement for the autonomy supervisor.

The normal PAPER intelligence cycle is intentionally rich: it can evaluate strategies,
build evidence cards and (when the snapshot changes) run in-sample evidence. That work
must never sit inside OUTCOME_RESOLUTION. EOD settlement only needs to:

1. mark already-open PAPER positions against the completed official session,
2. persist/learn any closes,
3. settle frozen BUY/WAIT/AVOID decisions from official bhavcopy.

This installer replaces only the production OUTCOME_RESOLUTION handler and Deps method.
It does not create a second scheduler or trading path and never enables live money.
"""
from __future__ import annotations

import os
from importlib import import_module
from typing import Any

_INSTALLED = False


def _trade_row(trade: Any) -> dict[str, Any]:
    if hasattr(trade, "as_dict"):
        try:
            return dict(trade.as_dict())
        except Exception:
            return {}
    try:
        return dict(trade)
    except Exception:
        return {}


# Calendar days scanned back for sessions an outage skipped (~30 NSE sessions,
# beyond any max_holding_days the paper book uses).
_CATCH_UP_MAX_DAYS = 45


def catch_up_missed_sessions(book, bars_fn, day: str, *, session_required=None) -> dict[str, Any]:
    """Mark every completed session an open position missed, oldest first.

    The EOD settlement marks one session. When the host is down (owner's Mac
    16-18 Nov 2026: two sessions) the skipped sessions were never checked, so a
    stop or target touched only on a missed day was silently survived — P&L a
    real exchange-side GTT would not have allowed. Each missed weekday is read
    with the same exact-session official bars (``bars_fn``); a holiday returns
    no bars and is skipped. ``PaperBook.mark`` refuses out-of-order sessions.
    """
    from datetime import date as _date, timedelta

    opens = list((getattr(book, "open", {}) or {}).values())
    out: dict[str, Any] = {"sessions": [], "closed": []}
    if not opens or not callable(bars_fn):
        return out
    starts = []
    for pos in opens:
        start = str(getattr(pos, "last_marked_session", "") or "")[:10] or str(getattr(pos, "entry_date", "") or "")[:10]
        if len(start) == 10:
            starts.append(start)
    if session_required is not None and len(starts) != len(opens):
        return {**out, "error": "Open paper position has no usable session cursor"}
    if not starts:
        return out
    try:
        end = _date.fromisoformat(str(day)[:10])
        start = (min(_date.fromisoformat(cursor) for cursor in starts)
                 if session_required is not None else _date.fromisoformat(min(starts)))
        floor = end - timedelta(days=_CATCH_UP_MAX_DAYS)
        if session_required is not None and start < floor:
            return {**out, "error": "Paper mark cursor predates the bounded catch-up window"}
        cursor = max(start, floor) + timedelta(days=1)
    except ValueError:
        return {**out, "error": "Invalid paper session cursor"} if session_required is not None else out
    while cursor < end:
        session = cursor.isoformat()
        try:
            expected = session_required(cursor) if session_required is not None else cursor.weekday() < 5
            if expected:
                bars = dict(bars_fn(session) or {})
                if not bars and session_required is not None:
                    raise ValueError("Official archive unavailable for a verified NSE session")
                if bars:
                    out["closed"].extend(book.mark(bars, session, allow_entry_session=False))
                    out["sessions"].append(session)
        except Exception as exc:
            out["error"] = f"{session}: {type(exc).__name__}: {exc}"[:240]
            break
        cursor += timedelta(days=1)
    return out


def reconcile_bars_held(book, bars_fn, *, max_days: int = _CATCH_UP_MAX_DAYS,
                        session_required=None) -> list[dict[str, Any]]:
    """Make ``bars_held`` mean completed sessions since entry, not marks applied.

    A position that missed marks before the catch-up existed keeps a short
    count forever (RPEL, entered 2026-11-11: bars_held 3 after 5 sessions,
    because 16-17 Nov were never marked), so MAX_HOLD fires late. The count is
    rebuilt from the same exact-session official bars: a weekday counts when
    the symbol traded that day, from the day after entry through the last
    marked session. Returns the corrections made.
    """
    from datetime import date as _date, timedelta

    cache: dict[str, dict] = {}

    def traded(day: str, symbol: str) -> bool:
        if day not in cache:
            cache[day] = dict(bars_fn(day) or {})
        return symbol in cache[day]

    changes: list[dict[str, Any]] = []
    for pos in list((getattr(book, "open", {}) or {}).values()):
        entry = str(getattr(pos, "entry_date", "") or "")[:10]
        last = str(getattr(pos, "last_marked_session", "") or "")[:10]
        if len(entry) != 10 or len(last) != 10 or last <= entry:
            continue
        try:
            day, end = _date.fromisoformat(entry) + timedelta(days=1), _date.fromisoformat(last)
        except ValueError:
            continue
        if (end - day).days > max_days:
            continue
        sessions = 0
        while day <= end:
            expected = session_required(day) if session_required is not None else day.weekday() < 5
            if expected and traded(day.isoformat(), str(pos.symbol)):
                sessions += 1
            day += timedelta(days=1)
        if sessions and sessions != int(getattr(pos, "bars_held", 0) or 0):
            changes.append({"symbol": pos.symbol, "from": int(pos.bars_held or 0), "to": sessions})
            pos.bars_held = sessions
    return changes


def settle_paper_session(brain, session_date: str, *, session_required=None) -> dict[str, Any]:
    """Mark the existing PAPER book once for one official session.

    No strategy evaluation, recommendation generation or historical backtest is run here.
    ``PaperBook.mark`` is session-idempotent, so retries/restarts cannot age a position twice.
    Entry-session full-day OHLC is explicitly disabled in this production EOD lane.
    """
    day = str(session_date or "")[:10]
    book = getattr(brain, "intel_book", None)
    open_count = len(getattr(book, "open", {}) or {}) if book is not None else 0
    recovery_error = str(getattr(brain, "intel_book_recovery_error", "") or "")
    if recovery_error:
        return {
            "status": "PAPER_BOOK_RECOVERY_FAILED", "as_of_date": day,
            "positions_closed": [], "outcomes_recorded": [],
            "open_before": open_count, "open_after": open_count, "bars": 0,
            "warnings": [recovery_error[:240]], "live_locked": True,
        }
    if not day:
        return {
            "status": "NO_SESSION",
            "as_of_date": "",
            "positions_closed": [],
            "outcomes_recorded": [],
            "open_before": open_count,
            "open_after": open_count,
            "bars": 0,
            "warnings": ["No completed session was supplied."],
            "live_locked": True,
        }

    lock = getattr(brain, "_intel_lock", None)
    acquired = True
    if lock is not None:
        try:
            acquired = bool(lock.acquire(blocking=False))
        except TypeError:
            acquired = bool(lock.acquire(False))
    if not acquired:
        return {
            "status": "SKIPPED_LOCKED",
            "as_of_date": day,
            "positions_closed": [],
            "outcomes_recorded": [],
            "open_before": open_count,
            "open_after": open_count,
            "bars": 0,
            "warnings": ["Paper mutation lock is busy; supervisor will retry."],
            "live_locked": True,
        }

    try:
        book = brain.intel_book
        open_before = len(getattr(book, "open", {}) or {})
        if session_required is not None and open_before:
            from datetime import date

            try:
                if not session_required(date.fromisoformat(day)):
                    raise ValueError("The supplied date is not an NSE trading session")
            except ValueError as exc:
                return {"status": "BARS_UNAVAILABLE", "as_of_date": day,
                        "positions_closed": [], "outcomes_recorded": [],
                        "open_before": open_before, "open_after": open_before, "bars": 0,
                        "warnings": [str(exc)], "live_locked": True}
        bars: dict[str, Any] = {}
        bars_fn = getattr(brain, "bars_fn", None)
        if callable(bars_fn):
            try:
                bars = dict(bars_fn(day) or {})
            except Exception as exc:
                return {
                    "status": "BARS_UNAVAILABLE",
                    "as_of_date": day,
                    "positions_closed": [],
                    "outcomes_recorded": [],
                    "open_before": open_before,
                    "open_after": open_before,
                    "bars": 0,
                    "warnings": [f"Official paper bars unavailable: {type(exc).__name__}: {exc}"[:240]],
                    "live_locked": True,
                }
        if open_before and not bars:
            return {
                "status": "BARS_UNAVAILABLE",
                "as_of_date": day,
                "positions_closed": [],
                "outcomes_recorded": [],
                "open_before": open_before,
                "open_after": open_before,
                "bars": 0,
                "warnings": ["Open PAPER positions exist but the completed-session bar set is empty."],
                "live_locked": True,
            }

        catch_up = catch_up_missed_sessions(book, bars_fn, day, session_required=session_required)
        closed = list(catch_up.get("closed") or [])
        settlement_error = str(catch_up.get("error") or "")
        if bars and not settlement_error:
            try:
                closed.extend(book.mark(bars, day, allow_entry_session=False))
            except Exception as exc:
                settlement_error = f"{day}: {type(exc).__name__}: {exc}"[:240]
        bars_held_fixes: list[dict[str, Any]] = []
        if callable(bars_fn) and not settlement_error:
            try:
                bars_held_fixes = reconcile_bars_held(book, bars_fn, session_required=session_required)
            except Exception:
                bars_held_fixes = []
        rows = [_trade_row(trade) for trade in closed]
        first_closed = len(book.closed) - len(closed)
        book.pending_eod_audit.extend((day, index) for index in range(first_closed, len(book.closed)))

        # Persist book state first. Audit/event projection is useful, but it must never
        # be able to roll back a valid paper close if a decoder is temporarily degraded.
        book_saved = False
        try:
            if brain._save_intel_book() is False:
                raise RuntimeError("paper book save returned failure")
            book_saved = True
        except Exception as exc:
            save_error = f"Paper book save failed: {type(exc).__name__}: {exc}"[:240]
            settlement_error = f"{settlement_error}; {save_error}"[:240] if settlement_error else save_error

        warnings: list[str] = []
        recorded: list[str] = []
        if book.pending_eod_audit and book_saved:
            try:
                from research.intelligence import decoder_registry as REG
                from research.intelligence.runtime import events as EV

                store = brain.event_store
                for audit_day, index in book.pending_eod_audit:
                    row = _trade_row(book.closed[index])
                    cycle_id = f"eod-settlement:{audit_day}"
                    for rec in REG.decode("outcome", row, ctx={"split": "forward"}):
                        store.append(rec)
                    for rec in REG.decode("execution", row):
                        store.append(rec)
                    EV.emit(
                        store,
                        cycle_id,
                        EV.PAPER_POSITION_CLOSED,
                        strategy_id=str(row.get("strategy_id") or ""),
                        symbol=str(row.get("symbol") or ""),
                        reason=str(row.get("exit_reason") or ""),
                        event_ts=audit_day,
                        summary={
                            "realized_R": round(float(row.get("realized_R") or 0.0), 4),
                            "pnl": round(float(row.get("pnl") or 0.0), 2),
                            "settlement_lane": "EOD_ONLY",
                        },
                    )
                    EV.emit(
                        store,
                        cycle_id,
                        EV.OUTCOME_DECODED,
                        strategy_id=str(row.get("strategy_id") or ""),
                        symbol=str(row.get("symbol") or ""),
                        event_ts=audit_day,
                        summary={
                            "realized_R": round(float(row.get("realized_R") or 0.0), 4),
                            "settlement_lane": "EOD_ONLY",
                        },
                    )
                if hasattr(store, "save"):
                    store.save()
                recorded = [book.closed[index].strategy_id for _, index in book.pending_eod_audit]
                book.pending_eod_audit.clear()
            except Exception as exc:
                projection_error = f"Outcome audit projection degraded: {type(exc).__name__}: {exc}"[:240]
                warnings.append(projection_error)
                settlement_error = settlement_error or projection_error

        try:
            brain.runtime_state.reconcile(book)
            if hasattr(brain.runtime_state, "save"):
                brain.runtime_state.save()
        except Exception as exc:
            warnings.append(f"Runtime reconciliation degraded: {type(exc).__name__}: {exc}"[:240])

        return {
            "status": "SETTLEMENT_INCOMPLETE" if settlement_error else "EOD_SETTLED",
            "error": settlement_error,
            "as_of_date": day,
            "positions_closed": rows,
            "outcomes_recorded": recorded,
            "open_before": open_before,
            "open_after": len(getattr(book, "open", {}) or {}),
            "bars": len(bars),
            "missed_sessions_marked": list(catch_up.get("sessions") or []),
            "bars_held_reconciled": bars_held_fixes,
            "warnings": warnings + ([f"Paper settlement interrupted: {settlement_error}"] if settlement_error else []),
            "settlement_lane": "EOD_ONLY",
            "full_intelligence_cycle": False,
            "historical_backtest": False,
            "live_locked": True,
        }
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _verified_settlement_calendar():
    from core.runtime_paths import logs_path
    from research.intelligence.data import nse_calendar as calendar

    path = logs_path("nse_calendar_manifest.json")
    manifest = calendar.load_manifest(path if path.is_file() else None)
    holidays = calendar.load_holidays()
    years = calendar.covered_years(holidays, manifest)
    special = calendar.documented_special_sessions(manifest)

    def required(day):
        if day.year not in years:
            raise ValueError(f"Unverified NSE calendar year {day.year}")
        return day.isoformat() in special or calendar.is_session(day, holidays)

    return required


def _resolve_outcomes_light(self, session_date: str, capability_failures=()):
    """Deps.resolve_outcomes replacement: PAPER marks + learning only."""
    scheduler = import_module("research.auto_research.scheduler")
    brain = scheduler.get_brain()
    production = brain is getattr(scheduler, "_BRAIN", None)
    if (production and brain.intel_book.open
            and not getattr(brain, "intel_book_recovery_error", "")):
        from research.auto_research.providers import ensure_production_paper_pipeline

        if not ensure_production_paper_pipeline():
            return {"status": "PAPER_PIPELINE_UNAVAILABLE",
                    "warnings": ["Protected paper settlement adapter unavailable."]}
    calendar = _verified_settlement_calendar() if production and brain.intel_book.open else None
    result = settle_paper_session(brain, session_date, session_required=calendar)
    if result.get("status") != "EOD_SETTLED":
        return result

    learning_book = brain.intel_book
    if production:
        try:
            from product.paper_learning_loop import recover_taken_entries

            learning_book = scheduler.load_committed_paper_book(brain.intel_book_path)
            result["recovered_taken_entries"] = recover_taken_entries(
                learning_book, brain.event_store, as_of=str(session_date)[:10])
        except Exception as exc:
            result["status"] = "SETTLEMENT_INCOMPLETE"
            result.setdefault("warnings", []).append(f"Taken-entry recovery failed: {exc}"[:240])
            return result

    try:
        from product.paper_learning_loop import ingest_closed_book

        result["paper_learning"] = ingest_closed_book(learning_book)
    except Exception as exc:
        result["paper_learning"] = {"error": str(exc)[:200]}

    try:
        from product.forward_soak import settle_and_report

        result["forward_soak"] = settle_and_report(str(session_date), book=learning_book)
    except Exception as exc:
        result["forward_soak"] = {"error": str(exc)[:200]}

    try:
        self.telegram.notify_paper_cycle(result, book=brain.intel_book)
    except Exception:
        pass
    return result


def _run_outcome_resolution_light(ctx):
    """Autonomy OUTCOME_RESOLUTION without committee/research/backtest work."""
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import schedules as SCH
    from research.autonomy import jobs as J

    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays() if hasattr(ctx.deps, "holidays") else None
    try:
        scheduled = SCH.scheduled_special_session(getattr(ctx, "job", None))
    except ValueError:
        return J.JobResult(JS.PERMANENT_FAILED, "Invalid special-session settlement date",
                           error_code="INVALID_SPECIAL_SESSION")
    from datetime import date

    special = date.fromisoformat(scheduled) if scheduled else None
    if special:
        session_date = scheduled
        ready = (SCH.verified_special_cash_session(special, holidays) and special <= now.date()
                 and (special < now.date() or now.time() >= SCH.SPECIAL_ARCHIVE_POLL_START)
                 and SCH.special_archive_available(special))
        available = scheduled if ready else ""
    else:
        session_date = SCH.last_completed_session_date(now, holidays) or now.date().isoformat()
        official = J._official_ready(ctx)
        available = str(official.get("available_session") or official.get("latest_date") or "")[:10]
        ready = official.get("current") or (available and available >= session_date)
    if not ready:
        return J.JobResult(
            JS.BLOCKED,
            f"official completed-session bars required before outcome resolution ({available or 'none'} < {session_date})",
            blocked_on=J.DEP_OUTCOME_DATA,
            failures={H.SNAPSHOT_STALE},
        )

    try:
        if hasattr(ctx.deps, "resolve_outcomes"):
            result = ctx.deps.resolve_outcomes(session_date, ctx.active_failures) or {}
        else:
            result = ctx.deps.run_paper_cycle(False) or {}
    except Exception as exc:
        return J.JobResult(
            JS.RETRYABLE_FAILED,
            "paper outcome settlement failed",
            error_code="PAPER_SETTLEMENT_ERROR",
            error_message=str(exc)[:240],
        )

    paper_status = str((result or {}).get("status") or "")
    retryable = {
        "SKIPPED_LOCKED": "PAPER_SETTLEMENT_BUSY",
        "BARS_UNAVAILABLE": "PAPER_BARS_UNAVAILABLE",
        "NO_SESSION": "PAPER_SESSION_UNAVAILABLE",
        "SETTLEMENT_INCOMPLETE": "PAPER_SETTLEMENT_ERROR",
        "PAPER_PIPELINE_UNAVAILABLE": "PAPER_PIPELINE_UNAVAILABLE",
        "PAPER_BOOK_RECOVERY_FAILED": "PAPER_BOOK_RECOVERY_FAILED",
    }
    if paper_status in retryable:
        return J.JobResult(
            JS.RETRYABLE_FAILED,
            "paper outcome settlement not ready",
            error_code=retryable[paper_status],
            error_message="; ".join(str(x) for x in ((result or {}).get("warnings") or []))[:240],
            metadata=result or {},
        )

    # Counterfactual/taken-vs-not-taken settlement is also lightweight. Crucially,
    # do not invoke the legacy full autonomous-loop outcome trigger here: that path
    # reevaluates committees/research and can perform unrelated heavy work.
    try:
        from product.autonomous_loop import settle_official_outcomes

        official_settle = settle_official_outcomes(session_date)
    except Exception as exc:
        official_settle = {"settled": [], "pending": [], "failed": [{"error": str(exc)[:240]}], "n_settled": 0}

    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from product.paper_self_feed import ingest_paper_cycle

            ingest_paper_cycle(result or {}, as_of=session_date, slot="eod")
        except Exception:
            pass

    if isinstance(result, dict):
        result["official_settlement"] = official_settle
    closed = len((result or {}).get("positions_closed", []))
    recorded = len((result or {}).get("outcomes_recorded", []))
    matured = int((official_settle or {}).get("n_settled") or 0)
    # Surfaced-BUY outcomes: this is the handler production runs, so the
    # signal_log settlement must live here (it was only in the replaced one).
    signal_note = J.settle_signal_log(result)
    return J.JobResult(
        JS.SUCCEEDED,
        f"outcomes resolved · {closed} book closes · {recorded} decoded · {matured} official{signal_note}",
        unblocks=(f"{J.DEP_OUTCOMES}:{session_date}",),
        metadata=result or {},
    )


def install_outcome_liveness() -> None:
    """Install the bounded production outcome lane exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return
    from research.autonomy import jobs as J
    from research.autonomy import schedules as SCH

    J.Deps.resolve_outcomes = _resolve_outcomes_light
    J.run_outcome_resolution = _run_outcome_resolution_light
    J.HANDLERS[SCH.OUTCOME_RESOLUTION] = _run_outcome_resolution_light
    _INSTALLED = True
