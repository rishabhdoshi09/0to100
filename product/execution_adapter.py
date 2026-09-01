"""Adapter boundary: decision brain vs execution venue.

Paper is the training ground. Live money stays fail-closed. Strategy intelligence
must never live inside a broker adapter — the same SelectionDecision feeds either
venue later.
"""
from __future__ import annotations

from typing import Any


class LiveMoneyLocked(RuntimeError):
    """Live adapter refused the decision. Paper path is unaffected."""


class PaperExecutionAdapter:
    """Submit a selection decision to the paper book / pipeline."""

    venue = "paper"

    def submit(self, decision: Any, *, book, as_of: str, snapshot_id: str):
        from product.paper_autopilot import execute_paper_decision
        return execute_paper_decision(decision, book=book, as_of=as_of, snapshot_id=snapshot_id)


class LiveExecutionAdapter:
    """Present so the architecture is real. Always refuses in this task."""

    venue = "live"

    def submit(self, decision: Any, **_kwargs):
        raise LiveMoneyLocked(
            "Live money is fail-closed. The same decision object may feed a live "
            "adapter later only after the readiness contract is met and an owner enables it."
        )


def default_adapter(*, live: bool = False):
    if live:
        return LiveExecutionAdapter()
    return PaperExecutionAdapter()
