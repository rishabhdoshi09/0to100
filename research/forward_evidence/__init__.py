"""Forward trading evidence — PAPER_FORWARD lane on existing PAPER_AUTO infrastructure.

Does NOT invent a second OMS, PaperBook, risk engine, or scientific memory.
Does NOT enable LIVE / LIMITED_LIVE broker submission.
"""
from __future__ import annotations

from research.forward_evidence.sources import (
    HISTORICAL_BACKTEST,
    PAPER_FORWARD,
    LIMITED_LIVE,
    LIVE,
    EVIDENCE_SOURCES,
)
from research.forward_evidence.maturity import (
    BACKTEST_ONLY,
    FORWARD_PAPER_ACCUMULATING,
    FORWARD_PAPER_SUFFICIENT,
    LIMITED_LIVE_ACCUMULATING,
    LIVE_EVIDENCE_ACCUMULATING,
    LIVE_VALIDATED,
    classify_maturity,
)
from research.forward_evidence.service import (
    ensure_armed,
    system_status,
    plain_operating_guide,
)

__all__ = [
    "HISTORICAL_BACKTEST",
    "PAPER_FORWARD",
    "LIMITED_LIVE",
    "LIVE",
    "EVIDENCE_SOURCES",
    "BACKTEST_ONLY",
    "FORWARD_PAPER_ACCUMULATING",
    "FORWARD_PAPER_SUFFICIENT",
    "LIMITED_LIVE_ACCUMULATING",
    "LIVE_EVIDENCE_ACCUMULATING",
    "LIVE_VALIDATED",
    "classify_maturity",
    "ensure_armed",
    "system_status",
    "plain_operating_guide",
]
