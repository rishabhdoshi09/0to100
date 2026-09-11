"""Adapter boundary: decision brain vs execution venue.

Paper is the training ground. Live money stays fail-closed. Strategy intelligence
must never live inside a broker adapter — the same SelectionDecision feeds either
venue later.
"""
from __future__ import annotations

from typing import Any

from product.live_execution_interlock import (
    LiveExecutionBlocked,
    assert_live_execution_allowed,
)


# Backwards-compatible exception name used by existing callers/tests.  It is an
# alias to the same canonical broker-boundary failure class so there is one
# safety contract rather than two independent locks.
LiveMoneyLocked = LiveExecutionBlocked


class PaperExecutionAdapter:
    """Submit a selection decision to the paper book / pipeline."""

    venue = "paper"

    def submit(self, decision: Any, *, book, as_of: str, snapshot_id: str):
        from product.paper_autopilot import execute_paper_decision
        return execute_paper_decision(decision, book=book, as_of=as_of, snapshot_id=snapshot_id)


class LiveExecutionAdapter:
    """Live adapter consumes the same canonical live-execution interlock."""

    venue = "live"

    def submit(self, decision: Any, **_kwargs):
        assert_live_execution_allowed("live_execution_adapter.submit")
        # No live venue implementation is intentionally reachable in the current
        # paper/shadow production contract.  If authorization is implemented in
        # the future, this adapter must be wired to the certified OMS, never a
        # direct broker escape hatch.
        raise LiveExecutionBlocked(
            "Live authorization existed but no certified OMS venue is configured; blocked."
        )


def default_adapter(*, live: bool = False):
    if live:
        return LiveExecutionAdapter()
    return PaperExecutionAdapter()
