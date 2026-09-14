"""Canonical paper-cycle outcome composition.

The legacy intelligence runtime is still valuable for position management and
research evidence, but new paper entries are owned by the recommendation
selection authority. The management pass therefore runs with entries disabled
on purpose. Its mechanical ``BLOCKED_SAFETY`` eligibility must never overwrite
the later recommendation/PaperBook outcome.

This installer follows QuantTerm's existing runtime-installer pattern. It does
not add a second execution path: it composes the two existing phases and keeps
``run_reco_paper_cycle -> brain.intel_book`` as the sole new-entry authority.
"""
from __future__ import annotations

from typing import Any, Mapping


PAPER_EXECUTION_FAILED = "PAPER_EXECUTION_FAILED"
EXECUTION_INCONSISTENT = "EXECUTION_INCONSISTENT"

_INSTALLED = False


def merge_cycle_truth(
    management: Mapping[str, Any] | None,
    reco: Mapping[str, Any] | None,
    *,
    entries_allowed: bool,
    entry_block_reason: str,
    session_phase: str,
) -> dict[str, Any]:
    """Merge management evidence with the canonical recommendation outcome.

    ``TRADED`` is accepted only with persisted ``positions_opened`` evidence.
    A genuine recommendation safety/data/entry-window block remains blocked.
    """
    result = dict(management or {})
    recommendation = dict(reco or {})

    management_eligibility = str(result.get("eligibility") or "")
    management_block_reason = str(result.get("entry_block_reason") or "")
    result["management_eligibility"] = management_eligibility
    result["management_entry_block_reason"] = management_block_reason

    management_opened = list(result.get("positions_opened") or [])
    reco_opened = list(recommendation.get("positions_opened") or [])
    opened = management_opened + reco_opened
    result["positions_opened"] = opened

    result["reco_autopilot"] = {
        "taken": recommendation.get("taken") or [],
        "rejections": recommendation.get("rejections") or [],
        "waits": recommendation.get("waits") or [],
        "final_decision": recommendation.get("final_decision"),
        "eligibility": recommendation.get("eligibility"),
        "cycle_reasons": recommendation.get("cycle_reasons") or [],
        "summary": recommendation.get("summary") or "",
        "entries_allowed": recommendation.get("entries_allowed", bool(entries_allowed)),
        "entry_block_reason": recommendation.get("entry_block_reason", entry_block_reason),
        "cycle_id": recommendation.get("cycle_id"),
        "source": recommendation.get("source"),
    }

    canonical = str(recommendation.get("eligibility") or "").strip().upper()
    if canonical == "TRADED":
        if reco_opened:
            result["eligibility"] = "TRADED"
        else:
            result["eligibility"] = EXECUTION_INCONSISTENT
            result["execution_truth_error"] = "TRADED_WITHOUT_PERSISTED_POSITION"
    elif canonical:
        result["eligibility"] = canonical

    result["new_entries_allowed"] = bool(
        recommendation.get("entries_allowed", entries_allowed)
    )
    result["entry_block_reason"] = str(
        recommendation.get("entry_block_reason", entry_block_reason) or ""
    )
    result["session_phase"] = str(
        recommendation.get("session_phase", session_phase) or session_phase
    )
    return result


def install_paper_cycle_truth() -> None:
    """Install the canonical two-phase paper-cycle composer exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from research.autonomy import jobs as JOBS

    if getattr(JOBS.Deps.run_paper_cycle, "_quantterm_paper_cycle_truth", False):
        _INSTALLED = True
        return

    def run_paper_cycle(
        self,
        entries_allowed: bool,
        entry_block_reason="",
        session_phase="intraday",
        capability_failures=(),
    ):
        from research.auto_research.scheduler import get_brain

        live = self.live_feed
        brain = get_brain()

        management = brain.run_intelligence_cycle_day(
            new_entries_allowed=False,
            entry_block_reason=(
                "RECO_SELECTION_AUTHORITY" if entries_allowed else entry_block_reason
            ),
            session_phase=session_phase,
            capability_failures=capability_failures,
            fresh_live_symbols=(live.fresh_symbols() if live is not None else ()),
        )
        result = dict(management or {})
        failure_message = ""
        failure_cause: Exception | None = None

        try:
            from product.paper_autopilot import run_reco_paper_cycle

            try:
                paper_on = bool(brain.is_paper_auto_enabled())
            except Exception:
                paper_on = True

            reco = run_reco_paper_cycle(
                book=brain.intel_book,
                as_of=str(result.get("as_of_date") or ""),
                entries_allowed=bool(entries_allowed),
                entry_block_reason=entry_block_reason,
                session_phase=session_phase,
                paper_enabled=paper_on,
            )
            result = merge_cycle_truth(
                result,
                reco,
                entries_allowed=bool(entries_allowed),
                entry_block_reason=str(entry_block_reason or ""),
                session_phase=str(session_phase or ""),
            )

            brain.state.last_intel_cycle = dict(result)
            try:
                brain._save_intel_book()
            except Exception:
                pass

            if str(result.get("eligibility") or "").upper() == EXECUTION_INCONSISTENT:
                failure_message = (
                    f"{EXECUTION_INCONSISTENT}: "
                    f"{result.get('execution_truth_error') or 'canonical paper execution integrity failure'}"
                )
        except Exception as exc:
            # Phase 1 deliberately runs with entries disabled. If the canonical
            # paper executor crashes, inheriting phase 1's BLOCKED_SAFETY would
            # turn an execution failure into a fake safety/no-trade outcome.
            result.setdefault("reco_autopilot", {})
            result["reco_autopilot"]["error"] = str(exc)[:300]
            result["reco_autopilot"]["eligibility"] = PAPER_EXECUTION_FAILED
            result["management_eligibility"] = str(result.get("eligibility") or "")
            result["management_entry_block_reason"] = str(
                result.get("entry_block_reason") or ""
            )
            result["eligibility"] = PAPER_EXECUTION_FAILED
            result["execution_truth_error"] = (
                f"{type(exc).__name__}: canonical paper executor failed"
            )
            result["entry_block_reason"] = PAPER_EXECUTION_FAILED
            result["new_entries_allowed"] = False
            result["session_phase"] = str(session_phase or "")

            # Persist the failure projection before re-raising so the UI and the
            # durable job ledger agree that execution failed rather than showing
            # the management-only BLOCKED_SAFETY state.
            brain.state.last_intel_cycle = dict(result)
            try:
                brain._save_intel_book()
            except Exception:
                pass

            failure_message = f"{PAPER_EXECUTION_FAILED}: {type(exc).__name__}: {exc}"
            failure_cause = exc

        # Notification receives the same final truth projected by the brain.
        self.telegram.notify_paper_cycle(result, book=brain.intel_book)

        # Do not let the durable PAPER_CYCLE job record SUCCEEDED for an executor
        # crash or a TRADED-without-fill integrity violation. The existing job
        # wrapper converts this raised error into RETRYABLE_FAILED/CYCLE_ERROR.
        if failure_message:
            failure = RuntimeError(failure_message)
            if failure_cause is not None:
                raise failure from failure_cause
            raise failure

        return result

    run_paper_cycle._quantterm_paper_cycle_truth = True
    JOBS.Deps.run_paper_cycle = run_paper_cycle
    _INSTALLED = True
