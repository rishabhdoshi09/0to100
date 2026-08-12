"""Phase A / A4 — CycleContext research seam tests (no behaviour change)."""
from __future__ import annotations

from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence.runtime.research_seams import (
    ChallengerEvidenceView,
    HorizonView,
    MarketStructureView,
    NetworkRiskView,
)
from research.intelligence.runtime import modes as MODES


def test_cyclecontext_seams_default_none_and_cycle_id_stable():
    ctx = CycleContext(as_of_date="d010", mode=MODES.PAPER_AUTO, data_ok=True)
    assert ctx.market_structure is None
    assert ctx.network_risk is None
    assert ctx.horizon_view is None
    assert ctx.challenger_evidence is None
    cid = ctx.cycle_id()

    # Attaching research seams must NOT change cycle identity.
    ctx.market_structure = MarketStructureView(
        as_of="d010", method="hierarchical", cluster_id_by_symbol={"AAA": 0}
    )
    ctx.network_risk = NetworkRiskView(as_of="d010", contagion_score=0.1)
    ctx.horizon_view = HorizonView(as_of="d010", best_supported_horizon="5d")
    ctx.challenger_evidence = ChallengerEvidenceView(
        as_of="d010", verdict="KEEP_INCUMBENT", role="research_signal_model"
    )
    assert ctx.cycle_id() == cid


def test_existing_constructor_kwargs_still_work():
    ctx = CycleContext(
        as_of_date="d026",
        mode=MODES.PAPER_AUTO,
        data_ok=True,
        data_snapshot_id="snap1",
        forward_eligible=False,
        new_entries_allowed=False,
        entry_block_reason="test",
    )
    assert ctx.data_snapshot_id == "snap1"
    assert ctx.forward_eligible is False
    assert ctx.market_structure is None
