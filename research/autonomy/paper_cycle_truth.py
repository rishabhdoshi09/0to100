"""Canonical paper-cycle outcome composition.

The legacy intelligence runtime is still valuable for position management and
research evidence, but new paper entries are owned by the recommendation
selection authority.  The management pass therefore runs with entries disabled
on purpose.  Its mechanical ``BLOCKED_SAFETY`` eligibility must never overwrite
the later recommendation/PaperBook outcome.

This installer follows QuantTerm's existing runtime-installer pattern.  It does
not add a second execution path: it composes the two existing phases and keeps
``run_reco_paper_cycle -> brain.intel_book`` as the sole new-entry authority.
"""
from __future__ import annotations

from typing import Any, Mapping

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
            # Fail closed: an execution label without a persisted PaperBook fill
            # is an integrity error, never a successful trade.
            result["eligibility"] = "EXECUTION_INCONSISTENT"
            result["execution_truth_error"] = "TRADED_WITHOUT_PERSISTED_POSITION"
    elif canonical:
        result["eligibility"] = canonical

    # Top-level entry semantics describe the canonical recommendation phase.
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

        # Phase 1 manages exits/evidence only.  RECO_SELECTION_AUTHORITY is an
        # intentional internal block, not the final paper-entry outcome.
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
            # Supervisor status reads brain.state.last_intel_cycle.  Persist the
            # combined truth so the UI cannot keep showing the management-only
            # BLOCKED_SAFETY after the recommendation phase completed.
            brain.state.last_intel_cycle = dict(result)
            try:
                brain._save_intel_book()
            except Exception:
                pass
        except Exception as exc:
            result.setdefault("reco_autopilot", {})
            result["reco_autopilot"]["error"] = str(exc)[:300]

        # Notification must see the same final result the durable job records.
        self.telegram.notify_paper_cycle(result, book=brain.intel_book)
        return result

    run_paper_cycle._quantterm_paper_cycle_truth = True
    JOBS.Deps.run_paper_cycle = run_paper_cycle
    _INSTALLED = True
