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


def settle_paper_session(brain, session_date: str) -> dict[str, Any]:
    """Mark the existing PAPER book once for one official session.

    No strategy evaluation, recommendation generation or historical backtest is run here.
    ``PaperBook.mark`` is session-idempotent, so retries/restarts cannot age a position twice.
    """
    day = str(session_date or "")[:10]
    book = getattr(brain, "intel_book", None)
    open_count = len(getattr(book, "open", {}) or {}) if book is not None else 0
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

        closed = list(book.mark(bars, day) if bars else [])
        rows = [_trade_row(trade) for trade in closed]

        # Persist book state first. Audit/event projection is useful, but it must never
        # be able to roll back a valid paper close if a decoder is temporarily degraded.
        try:
            brain._save_intel_book()
        except Exception:
            pass

        warnings: list[str] = []
        if closed:
            try:
                from research.intelligence import decoder_registry as REG
                from research.intelligence.runtime import events as EV

                cycle_id = f"eod-settlement:{day}"
                store = brain.event_store
                for trade, row in zip(closed, rows):
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
                        event_ts=day,
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
                        event_ts=day,
                        summary={
                            "realized_R": round(float(row.get("realized_R") or 0.0), 4),
                            "settlement_lane": "EOD_ONLY",
                        },
                    )
                if hasattr(store, "save"):
                    store.save()
            except Exception as exc:
                warnings.append(f"Outcome audit projection degraded: {type(exc).__name__}: {exc}"[:240])

        try:
            brain.runtime_state.reconcile(book)
            if hasattr(brain.runtime_state, "save"):
                brain.runtime_state.save()
        except Exception as exc:
            warnings.append(f"Runtime reconciliation degraded: {type(exc).__name__}: {exc}"[:240])

        return {
            "status": "EOD_SETTLED",
            "as_of_date": day,
            "positions_closed": rows,
            "outcomes_recorded": [str(row.get("strategy_id") or "") for row in rows if row],
            "open_before": open_before,
            "open_after": len(getattr(book, "open", {}) or {}),
            "bars": len(bars),
            "warnings": warnings,
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


def _resolve_outcomes_light(self, session_date: str, capability_failures=()):
    """Deps.resolve_outcomes replacement: PAPER marks + learning only."""
    from research.auto_research.scheduler import get_brain

    brain = get_brain()
    result = settle_paper_session(brain, session_date)
    if result.get("status") != "EOD_SETTLED":
        return result

    try:
        from product.paper_learning_loop import ingest_closed_book

        result["paper_learning"] = ingest_closed_book(brain.intel_book)
    except Exception as exc:
        result["paper_learning"] = {"error": str(exc)[:200]}

    try:
        from product.forward_soak import settle_and_report

        result["forward_soak"] = settle_and_report(str(session_date), book=brain.intel_book)
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
    session_date = SCH.last_completed_session_date(now, holidays) or now.date().isoformat()
    official = J._official_ready(ctx)
    available = str(official.get("available_session") or official.get("latest_date") or "")[:10]
    if not official.get("current") and (not available or available < session_date):
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
    if paper_status in {"SKIPPED_LOCKED", "BARS_UNAVAILABLE", "NO_SESSION"}:
        return J.JobResult(
            JS.RETRYABLE_FAILED,
            "paper outcome settlement not ready",
            error_code=(
                "PAPER_SETTLEMENT_BUSY" if paper_status == "SKIPPED_LOCKED"
                else "PAPER_BARS_UNAVAILABLE" if paper_status == "BARS_UNAVAILABLE"
                else "PAPER_SESSION_UNAVAILABLE"
            ),
            error_message="; ".join(str(x) for x in ((result or {}).get("warnings") or []))[:240],
            metadata=result or {},
        )

    # Counterfactual/taken-vs-not-taken settlement is also lightweight. Crucially,
    # do NOT call autonomous_loop.advance_loop(trigger='outcome_resolution'): that path
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
    return J.JobResult(
        JS.SUCCEEDED,
        f"outcomes resolved · {closed} book closes · {recorded} decoded · {matured} official",
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
