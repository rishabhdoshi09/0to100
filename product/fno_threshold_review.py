"""Recommendation-only threshold review for the F&O min-score entry gate.

This module NEVER changes a gate. product.fno_historical_walkforward's
``min_score_to_take`` (default 60.0) is the threshold that decides which
historical candidates would have been "taken" -- this module mines the
per-score-bucket outcome breakdown product.fno_historical_loop already
collects (classification_counts_by_score_bucket: CORRECT_REJECTION /
MISSED_WINNER / AVOIDED_LOSER / RAN_AWAY_WITHOUT_ENTRY / GOOD_WAIT / FLAT,
split by whether the candidate would have been taken) to answer one
question: "does the evidence below the current threshold look meaningfully
different from the evidence above it?"

The answer is always one of three RECOMMENDATION labels, never an
instruction:
  INSUFFICIENT_SAMPLE          -- too few graded candidates in a bucket to
                                   say anything (min 30, matching the desk's
                                   standing evidence floor).
  KEEP_CURRENT_THRESHOLD       -- enough sample, but the below-threshold
                                   bucket does not look meaningfully better
                                   than the current taken population.
  CONSIDER_LOWERING_THRESHOLD  -- enough sample, and the rejected
                                   candidates just below today's threshold
                                   show a materially higher "would-be winner"
                                   rate than the false-positive ("would-be
                                   loser") rate, with no matching rise in
                                   losers -- worth a human look, nothing more.

Nothing here writes to product.fno_historical_walkforward's
min_score_to_take or any other gate. A caller that wants to act on a
recommendation does so by hand, exactly like every other
recommend-don't-autonomously-change gate on this desk (see
product.decision_journal's calibration_report for the equity-side
precedent this mirrors).
"""
from __future__ import annotations

from typing import Any, Mapping

from product.fno_historical_walkforward import evaluate_point_in_time_candidate

#: Matches product.fno_historical_walkforward.evaluate_point_in_time_candidate's
#: own default -- this module reviews the ACTUAL threshold in force, never a
#: guess. A caller running a different threshold passes it explicitly.
DEFAULT_MIN_SCORE_TO_TAKE = float(
    (evaluate_point_in_time_candidate.__kwdefaults__ or {}).get("min_score_to_take", 60.0)
)

MIN_SAMPLE = 30
#: A "would-be winner" rate this many points above the "would-be loser" rate
#: (both measured on the same below-threshold population) is the bar for
#: even a CONSIDER recommendation -- deliberately conservative so a handful
#: of lucky rejects never reads as a signal.
MATERIAL_GAP_PP = 15.0

_WINNER_LABELS = ("MISSED_WINNER",)
_LOSER_LABELS = ("AVOIDED_LOSER",)


def _rate(counts: Mapping[str, int], labels: tuple[str, ...], total: int) -> float:
    if total <= 0:
        return 0.0
    hits = sum(int(counts.get(label) or 0) for label in labels)
    return hits / total * 100.0


def _bucket_floor(bucket: str) -> float | None:
    if bucket.startswith("BELOW_"):
        return -1.0
    if bucket.endswith("_PLUS"):
        try:
            return float(bucket.split("_")[0])
        except ValueError:
            return None
    try:
        return float(bucket.split("_")[0])
    except ValueError:
        return None


def review_min_score_threshold(
    *,
    min_score_to_take: float = DEFAULT_MIN_SCORE_TO_TAKE,
    by_bucket: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None,
) -> dict[str, Any]:
    """One recommendation for the min-score gate, from real graded history.

    ``by_bucket`` defaults to product.fno_historical_loop.status()'s
    classification_counts_by_score_bucket when not supplied directly (tests
    and offline analysis can pass a fixture instead of touching disk).
    """
    if by_bucket is None:
        from product.fno_historical_loop import status as historical_status
        by_bucket = dict(historical_status().get("classification_counts_by_score_bucket") or {})

    below_counts: dict[str, int] = {}
    below_total = 0
    above_counts: dict[str, int] = {}
    above_total = 0
    for bucket, taken_map in (by_bucket or {}).items():
        floor = _bucket_floor(bucket)
        if floor is None:
            continue
        # "not_taken" rows in a bucket at/above the threshold, or "taken" rows
        # below it, should not exist given evaluate_point_in_time_candidate's
        # own taken = score >= min_score_to_take rule -- read only the row
        # that actually matches this threshold's below/above split.
        not_taken = dict((taken_map or {}).get("not_taken") or {})
        taken = dict((taken_map or {}).get("taken") or {})
        if floor < min_score_to_take:
            for label, n in not_taken.items():
                below_counts[label] = below_counts.get(label, 0) + int(n)
                below_total += int(n)
        else:
            for label, n in taken.items():
                above_counts[label] = above_counts.get(label, 0) + int(n)
                above_total += int(n)

    result: dict[str, Any] = {
        "min_score_to_take": min_score_to_take,
        "below_threshold_sample": below_total,
        "above_threshold_sample": above_total,
        "below_threshold_would_be_winner_rate_pct": round(_rate(below_counts, _WINNER_LABELS, below_total), 2),
        "below_threshold_would_be_loser_rate_pct": round(_rate(below_counts, _LOSER_LABELS, below_total), 2),
        "above_threshold_winner_rate_pct": round(_rate(above_counts, _WINNER_LABELS, above_total), 2),
        "above_threshold_loser_rate_pct": round(_rate(above_counts, _LOSER_LABELS, above_total), 2),
        "min_sample": MIN_SAMPLE,
        "material_gap_pp": MATERIAL_GAP_PP,
        "autonomous_change_applied": False,
        "action_required": "NONE -- recommendation only; a human decides whether to change the gate.",
    }

    if below_total < MIN_SAMPLE:
        result["recommendation"] = "INSUFFICIENT_SAMPLE"
        result["reason"] = (
            f"Only {below_total} graded historical candidate(s) fell below the "
            f"current threshold ({min_score_to_take}); {MIN_SAMPLE} needed before "
            "this gate's evidence means anything."
        )
        return result

    gap = result["below_threshold_would_be_winner_rate_pct"] - result["below_threshold_would_be_loser_rate_pct"]
    if gap >= MATERIAL_GAP_PP:
        result["recommendation"] = "CONSIDER_LOWERING_THRESHOLD"
        result["reason"] = (
            f"Below-threshold candidates would-be-won {result['below_threshold_would_be_winner_rate_pct']:.1f}% "
            f"of the time vs would-be-lost {result['below_threshold_would_be_loser_rate_pct']:.1f}% -- a "
            f"{gap:.1f} point gap on {below_total} graded candidates. Worth a human look; the gate is unchanged."
        )
    else:
        result["recommendation"] = "KEEP_CURRENT_THRESHOLD"
        result["reason"] = (
            f"Below-threshold candidates do not show a material would-be-winner "
            f"edge over would-be-losers (gap {gap:.1f}pp on {below_total} graded candidates)."
        )
    return result
