"""Deterministic stopping laws for historical active-learning replay.

This module never promotes historical evidence to forward evidence and never
changes execution authority. It only decides whether another HISTORICAL_REPLAY
batch is justified by remaining ex-ante coverage gaps.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

MIN_KNOWN_COVERAGE = 3
MAX_UNKNOWN_SHARE = 0.50


def coverage_gap(coverage: Mapping[str, int], regimes: Sequence[str]) -> dict[str, int]:
    """Return how many additional PIT observations each observed regime needs."""
    clean = {str(k): max(0, int(v)) for k, v in dict(coverage or {}).items()}
    return {
        str(regime): max(0, MIN_KNOWN_COVERAGE - clean.get(str(regime), 0))
        for regime in sorted({str(x) for x in regimes if str(x)})
    }


def assess_replay_need(
    *,
    coverage: Mapping[str, int],
    remaining_regimes: Sequence[str],
    remaining_unknown: int = 0,
    remaining_total: int | None = None,
) -> dict[str, Any]:
    """Fail closed on missing context; stop only when replay has no useful gap.

    The law is intentionally outcome-blind. It may use only point-in-time regime
    labels and durable cursor counts, never future P&L, winners, or trade results.
    """
    remaining = [str(x) for x in remaining_regimes if str(x)]
    counts = Counter(remaining)
    total = max(0, int(remaining_total if remaining_total is not None else len(remaining) + remaining_unknown))
    unknown = max(0, int(remaining_unknown))
    unknown_share = (unknown / total) if total else 0.0
    gaps = coverage_gap(coverage, counts.keys())
    useful = {
        regime: min(int(counts.get(regime, 0)), int(gap))
        for regime, gap in gaps.items()
        if gap > 0 and counts.get(regime, 0) > 0
    }
    expected_gain = sum(useful.values())

    if total == 0:
        decision, reason = "STOP", "historical_backlog_caught_up"
    elif unknown_share > MAX_UNKNOWN_SHARE:
        decision, reason = "CONTINUE", "pit_context_too_incomplete_to_prove_plateau"
    elif expected_gain > 0:
        decision, reason = "CONTINUE", "coverage_gap_remains"
    else:
        decision, reason = "STOP", "coverage_plateau"

    return {
        "decision": decision,
        "reason": reason,
        "expected_information_gain": int(expected_gain),
        "coverage_gap": gaps,
        "remaining_by_regime": dict(sorted(counts.items())),
        "remaining_unknown": unknown,
        "remaining_total": total,
        "unknown_share": round(float(unknown_share), 6),
        "outcome_blind": True,
        "evidence_lane": "HISTORICAL_REPLAY",
        "live_money_unchanged": True,
    }
