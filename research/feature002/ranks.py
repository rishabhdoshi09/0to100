"""Frozen FEATURE-002 shadow ranks. Computed only inside one candidate set."""
from __future__ import annotations

from typing import Any, Callable, Iterable


def _key_rs(row: dict[str, Any]) -> tuple:
    pct = row.get("rs_percentile")
    score = row.get("rs_score")
    return (
        pct is None,
        -(float(pct) if pct is not None else 0.0),
        score is None,
        -(float(score) if score is not None else 0.0),
        str(row.get("symbol") or ""),
    )


def _key_trend(row: dict[str, Any]) -> tuple:
    n = row.get("n_structure_passed")
    pa = row.get("pct_above_sma200")
    sp = row.get("ma_spread_50_200_pct")
    return (
        n is None,
        -(int(n) if n is not None else 0),
        pa is None,
        -(float(pa) if pa is not None else 0.0),
        sp is None,
        -(float(sp) if sp is not None else 0.0),
        str(row.get("symbol") or ""),
    )


def assign_competition_ranks(rows: list[dict[str, Any]], key_fn: Callable) -> list[int]:
    """1 = best. Stable competition ranking (ties share min rank)."""
    order = sorted(range(len(rows)), key=lambda i: key_fn(rows[i]))
    ranks = [0] * len(rows)
    prev_key = None
    prev_rank = 0
    for pos, i in enumerate(order, start=1):
        k = key_fn(rows[i])
        if prev_key is not None and k == prev_key:
            ranks[i] = prev_rank
        else:
            ranks[i] = pos
            prev_rank = pos
            prev_key = k
    return ranks


def percentile_from_ranks(ranks: Iterable[int], n: int) -> list[float]:
    """Higher is better. 100 = best rank in the set. Average-rank ties already in ranks."""
    if n <= 1:
        return [50.0 for _ in ranks]
    return [100.0 * (n - r) / (n - 1) for r in ranks]


def apply_shadow_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mutates copies only — caller must pass detached row dicts."""
    n = len(rows)
    for i, row in enumerate(rows, start=1):
        row["production_rank"] = i
    r1 = assign_competition_ranks(rows, _key_rs)
    r2 = assign_competition_ranks(rows, _key_trend)
    p1 = percentile_from_ranks(r1, n)
    p2 = percentile_from_ranks(r2, n)
    for row, a, b, pa, pb in zip(rows, r1, r2, p1, p2):
        row["rs_rank"] = a
        row["trend_rank"] = b
        row["rs_pctl_in_set"] = pa
        row["trend_pctl_in_set"] = pb
        rs_ok = row.get("rs_percentile") is not None
        tr_ok = row.get("n_structure_passed") is not None
        if rs_ok or tr_ok:
            row["r3_score"] = (0.67 * pa) + (0.33 * pb)
        else:
            row["r3_score"] = None
    r3 = assign_competition_ranks(
        rows,
        lambda r: (
            r.get("r3_score") is None,
            -(float(r["r3_score"]) if r.get("r3_score") is not None else 0.0),
            _key_rs(r),
            _key_trend(r),
        ),
    )
    for row, c in zip(rows, r3):
        row["combined_shadow_rank"] = c
        row["shadow_rank_version"] = "feature-002.ranks.v1"
    return rows
