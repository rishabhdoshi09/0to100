"""Operational fast-path guards for the autonomy job registry.

The paper cycle is an execution/position-management job. It must not monopolise the
single supervisor worker with research-scale in-sample computation when entries are
closed and there are no positions to manage.

This module installs a narrow wrapper around the canonical paper-cycle handler. It
never fabricates a result, never bypasses position management, and fails closed: when
open-position state cannot be read, the original handler runs unchanged.
"""
from __future__ import annotations

from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST


def _open_position_count() -> int | None:
    """Return the canonical intelligence-book open count, or ``None`` if unknown."""
    try:
        from research.auto_research.scheduler import get_brain

        book = get_brain().intel_book
        opened = getattr(book, "open", None)
        if opened is None:
            return None
        return len(opened)
    except Exception:
        return None


def guarded_paper_cycle(ctx, original_handler=None):
    """Avoid research-scale work for a management-only empty paper cycle.

    The original paper-cycle handler remains authoritative whenever entries are open,
    positions exist, or position state cannot be verified.
    """
    original = original_handler or JOBS.run_paper_cycle
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays()
    entries_ok, reason, phase = JOBS._entry_reason(now, holidays, ctx)
    open_positions = _open_position_count()

    if not entries_ok and open_positions == 0:
        reason = reason or "NEW_ENTRIES_BLOCKED"
        return JOBS.JobResult(
            JS.SUCCEEDED,
            f"paper cycle: no-op · {phase} · no open positions · {reason}",
            state_hint=ST.OBSERVING,
            new_entries_allowed=False,
            metadata={
                "eligibility": "NO_POSITIONS_TO_MANAGE",
                "entry_block_reason": reason,
                "session_phase": phase,
                "open_positions": 0,
                "fast_path": True,
            },
        )

    return original(ctx)


def install_operational_guards() -> None:
    """Install idempotent production guards into the canonical handler registry."""
    current = JOBS.HANDLERS.get(SCH.PAPER_CYCLE)
    if current is None or getattr(current, "_quantterm_operational_guard", False):
        return

    def wrapped(ctx):
        return guarded_paper_cycle(ctx, original_handler=current)

    wrapped._quantterm_operational_guard = True
    wrapped.__name__ = getattr(current, "__name__", "run_paper_cycle")
    wrapped.__doc__ = getattr(current, "__doc__", None)
    JOBS.HANDLERS[SCH.PAPER_CYCLE] = wrapped
