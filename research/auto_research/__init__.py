"""
🧠 Autonomous Research Brain — QuantTerm thinking for itself.

Runs headless on a schedule and, with no human intervention, does the whole loop: read the
data situation → generate ideas → reason through each one in the open (a transparent
chain-of-thought, written to an append-only research thread) → reject the weak ones →
track what is decaying or improving across cycles → and PROPOSE improvements.

It auto-advances research all the way to ONE gate — `AWAITING_USER_APPROVAL` — and stops
there. It never approves, never activates paper, never touches live, never places an order.
That single stop is the deliberate human seatbelt: by the project's own safety directives,
real-money autonomy requires a person's approval.
"""
from research.auto_research.thread import ResearchThread, ThreadEntry
from research.auto_research.loop import run_cycle, CycleReport, Proposal, canonical_readiness
from research.auto_research.learning import LearningLedger, LearningEvent
from research.auto_research.knowledge import Knowledge, FamilyKnowledge
from research.auto_research.growth import calibrate, Calibration
from research.auto_research import costs
from research.auto_research.paper_book import PaperBook, PaperPosition, ClosedTrade
from research.auto_research.paper_autonomy import PaperAutonomyManager, PaperStrategy
from research.auto_research.scheduler import AutoResearchBrain, BrainState, get_brain

__all__ = [
    "ResearchThread", "ThreadEntry", "run_cycle", "CycleReport", "Proposal",
    "canonical_readiness", "LearningLedger", "LearningEvent",
    "Knowledge", "FamilyKnowledge", "calibrate", "Calibration",
    "PaperBook", "PaperPosition", "ClosedTrade",
    "PaperAutonomyManager", "PaperStrategy",
    "AutoResearchBrain", "BrainState", "get_brain",
]
