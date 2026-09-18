"""A broken recommendations load must surface as a failure, never as a quiet
"nothing was good enough today" cycle.

product.paper_autopilot.run_reco_paper_cycle used to wrap its call to
load_recommendations() in a bare `except Exception: payload = {}`.
load_recommendations() itself already fails closed to None for every normal
case (missing file, corrupt JSON, schema mismatch) -- it never raises for
those. So the only thing that broad except could actually catch was a real
code-level failure (a broken import, an unexpected bug), and it converted
that into an empty card list, which the rest of the cycle then reports as
eligibility=NO_ELIGIBLE_TRADE / "nothing was good enough today".

Both real callers of run_reco_paper_cycle (research/autonomy/jobs.py and
research/autonomy/paper_cycle_truth.py) are specifically built to catch an
exception from this function and classify it as a system/execution failure
instead of a no-trade outcome -- paper_cycle_truth.py even says so in a
comment: "If the canonical paper executor crashes, inheriting phase 1's
BLOCKED_SAFETY would turn an execution failure into a fake safety/no-trade
outcome." The inner swallow defeated that design before the exception could
ever reach them.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from product.paper_autopilot import run_reco_paper_cycle


def _now():
    return datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def test_broken_recommendations_load_raises_instead_of_faking_no_trade(monkeypatch):
    import product.recommendations_store as store

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated recommendations provider failure")

    monkeypatch.setattr(store, "load_recommendations", _boom)

    with pytest.raises(RuntimeError, match="simulated recommendations provider failure"):
        run_reco_paper_cycle(
            book=SimpleNamespace(),
            workspace=None,
            cards=None,
            as_of="2026-09-01",
            now=_now(),
            entries_allowed=True,
            paper_enabled=True,
            persist_journal=False,
        )


def test_missing_recommendations_file_is_still_a_normal_no_trade_cycle(monkeypatch):
    """The normal "no file yet" case must keep working exactly as before:
    load_recommendations() returning None is not an error."""
    import product.recommendations_store as store
    from research.auto_research.paper_book import PaperBook

    monkeypatch.setattr(store, "load_recommendations", lambda *a, **k: None)

    cycle = run_reco_paper_cycle(
        book=PaperBook(capital=100_000.0),
        workspace=None,
        cards=None,
        as_of="2026-09-01",
        now=_now(),
        entries_allowed=True,
        paper_enabled=True,
        persist_journal=False,
    )
    assert cycle["eligibility"] == "NO_ELIGIBLE_TRADE"
    assert cycle["candidates_seen"] == 0
