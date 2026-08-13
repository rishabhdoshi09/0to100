"""Forward-evidence maturity labels (presentation + policy). Promotion still needs
objective gates in core.evidence_levels — thresholds alone never auto-promote."""
from __future__ import annotations

BACKTEST_ONLY = "BACKTEST_ONLY"
FORWARD_PAPER_ACCUMULATING = "FORWARD_PAPER_ACCUMULATING"
FORWARD_PAPER_SUFFICIENT = "FORWARD_PAPER_SUFFICIENT"
LIMITED_LIVE_ACCUMULATING = "LIMITED_LIVE_ACCUMULATING"
LIVE_EVIDENCE_ACCUMULATING = "LIVE_EVIDENCE_ACCUMULATING"
LIVE_VALIDATED = "LIVE_VALIDATED"

# Soft observational thresholds (reporting only — not automatic promotion)
_PAPER_ACCUM_N = 10
_PAPER_SUFFICIENT_N = 40  # aligns with autonomy paper_proven floor; E4 still needs ≥300


def classify_maturity(
    *,
    historical_n: int = 0,
    paper_n: int = 0,
    limited_live_n: int = 0,
    live_n: int = 0,
    paper_expectancy_r: float | None = None,
    live_expectancy_r: float | None = None,
) -> str:
    """Classify observational maturity. Never grants live authority."""
    if live_n >= _PAPER_SUFFICIENT_N and (live_expectancy_r or 0) > 0:
        return LIVE_VALIDATED
    if live_n > 0:
        return LIVE_EVIDENCE_ACCUMULATING
    if limited_live_n > 0:
        return LIMITED_LIVE_ACCUMULATING
    if paper_n >= _PAPER_SUFFICIENT_N:
        return FORWARD_PAPER_SUFFICIENT
    if paper_n >= _PAPER_ACCUM_N or paper_n > 0:
        return FORWARD_PAPER_ACCUMULATING
    if historical_n > 0:
        return BACKTEST_ONLY
    return BACKTEST_ONLY


def plain_label(maturity: str) -> str:
    return {
        BACKTEST_ONLY: "Only past-data tests so far",
        FORWARD_PAPER_ACCUMULATING: "Collecting simulated forward trades",
        FORWARD_PAPER_SUFFICIENT: "Enough paper trades to review carefully",
        LIMITED_LIVE_ACCUMULATING: "Limited real-money evidence accumulating",
        LIVE_EVIDENCE_ACCUMULATING: "Real-money evidence accumulating",
        LIVE_VALIDATED: "Real-money evidence looks stable (still owner-gated)",
    }.get(maturity, maturity)
