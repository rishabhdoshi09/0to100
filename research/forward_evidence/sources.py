"""Evidence source labels — never silently merge PAPER and LIVE as identical quality."""
from __future__ import annotations

HISTORICAL_BACKTEST = "HISTORICAL_BACKTEST"
PAPER_FORWARD = "PAPER_FORWARD"
LIMITED_LIVE = "LIMITED_LIVE"
LIVE = "LIVE"

EVIDENCE_SOURCES = (
    HISTORICAL_BACKTEST,
    PAPER_FORWARD,
    LIMITED_LIVE,
    LIVE,
)

# Map existing OutcomeObservation.split values → canonical evidence_source
_SPLIT_MAP = {
    "in_sample": HISTORICAL_BACKTEST,
    "out_of_sample": HISTORICAL_BACKTEST,
    "forward": PAPER_FORWARD,
    "paper": PAPER_FORWARD,
    "limited_live": LIMITED_LIVE,
    "live": LIVE,
}


def from_split(split: str) -> str:
    return _SPLIT_MAP.get(str(split or "").strip().lower(), PAPER_FORWARD)


def assert_known(source: str) -> str:
    s = str(source or "").strip().upper()
    if s not in EVIDENCE_SOURCES:
        raise ValueError(f"unknown evidence_source {source!r}; expected one of {EVIDENCE_SOURCES}")
    return s
