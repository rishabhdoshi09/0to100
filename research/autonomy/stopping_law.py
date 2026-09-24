"""Plateau/stopping law over immutable historical-replay acquisition evidence.

This module is research-only.  It consumes append-only REALIZED acquisition
records and cannot confer forward-paper or live-money authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"
REALIZED_EVENT = "REALIZED"


@dataclass(frozen=True)
class StoppingDecision:
    stop: bool
    reason: str
    observations: int
    recent_yield: int
    prior_yield: int
    zero_yield_streak: int


def _eligible(records: Iterable[Mapping[str, Any]], request_id: str) -> list[Mapping[str, Any]]:
    wanted = str(request_id or "").strip()
    if not wanted:
        raise ValueError("request_id is required")
    rows: list[Mapping[str, Any]] = []
    for row in records:
        if str(row.get("event") or "").upper() != REALIZED_EVENT:
            continue
        if str(row.get("evidence_origin") or "").upper() != EVIDENCE_ORIGIN:
            raise ValueError("stopping law refuses non-historical realized evidence")
        if str(row.get("request_id") or "") != wanted:
            continue
        if not str(row.get("record_fingerprint") or "").strip():
            raise ValueError("realized evidence missing immutable record fingerprint")
        rows.append(row)
    return rows


def evaluate_realized_gain(
    records: Iterable[Mapping[str, Any]],
    *,
    request_id: str,
    min_observations: int = 8,
    window: int = 4,
    max_zero_yield_streak: int = 4,
) -> StoppingDecision:
    """Return a conservative plateau decision from realized historical yield.

    Stops only after enough immutable observations and either (a) a sustained
    zero-yield streak or (b) the recent window produces no more eligible
    samples than the preceding window.  This intentionally cannot inspect
    outcomes, PnL, or forward evidence, preventing outcome-driven replay
    acquisition and historical/forward evidence leakage.
    """
    if min_observations < 2 or window < 1 or max_zero_yield_streak < 1:
        raise ValueError("invalid stopping-law bounds")
    rows = _eligible(records, request_id)
    yields = [max(0, int(row.get("eligible_samples") or 0)) for row in rows]
    n = len(yields)
    zero_streak = 0
    for value in reversed(yields):
        if value != 0:
            break
        zero_streak += 1
    recent = sum(yields[-window:]) if yields else 0
    prior = sum(yields[-2 * window:-window]) if n > window else 0
    if n < min_observations:
        return StoppingDecision(False, "INSUFFICIENT_REALIZED_EVIDENCE", n, recent, prior, zero_streak)
    if zero_streak >= max_zero_yield_streak:
        return StoppingDecision(True, "SUSTAINED_ZERO_INFORMATION_GAIN", n, recent, prior, zero_streak)
    if n >= 2 * window and recent <= prior:
        return StoppingDecision(True, "REALIZED_INFORMATION_GAIN_PLATEAU", n, recent, prior, zero_streak)
    return StoppingDecision(False, "CONTINUE_INFORMATION_ACQUISITION", n, recent, prior, zero_streak)
