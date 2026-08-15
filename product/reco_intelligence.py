"""Evidence helpers that make recommendation buckets smarter — not a UI feature.

Uses the same measured languages the rest of QuantTerm already trusts:
  - conservative EV (Wilson lower-bound) when outcomes exist
  - walk-forward combo_edge demotion when EV is thin
  - light confluence / grade tie-breaks (never invent edge)

Fail-open: missing stats never invent picks and never blank a bucket.
Additive only — safe for clients that ignore unknown card fields.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# Signals that count as confluence evidence (not every tag is equal).
_CONFLUENCE = frozenset({
    "MOMENTUM", "GOLDEN_CROSS", "PRE_BREAKOUT", "BREAKOUT_52W", "BREAKOUT_RES",
    "DOUBLE_BOTTOM", "ACCUMULATION", "VOL_SQUEEZE", "DELIVERY_SPIKE",
    "VCP", "FLAT_BASE",
})


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def tag_rows_ev(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    try:
        from scan.ev_engine import tag_ev
        tag_ev(out)
    except Exception:
        pass
    return out


def measured_edge_ok(row: Mapping[str, Any]) -> bool:
    """Demote-only: negative conservative EV or negative combo edge → False.

    No claim (thin sample) → True. Never invents a veto from missing data.
    """
    if row.get("ev_lb_pct") is not None:
        return _f(row.get("ev_lb_pct")) >= 0.0
    try:
        from scan.signal_backtest import combo_edge
        sigs = [str(s) for s in (row.get("signals") or []) if s]
        if not sigs:
            return True
        edge = combo_edge(sigs)
        if edge is None:
            return True
        return float(edge) >= 0.0
    except Exception:
        return True


def confluence_count(row: Mapping[str, Any]) -> int:
    sigs = {str(s).upper() for s in (row.get("signals") or [])}
    return len(sigs & _CONFLUENCE)


def grade_rank(row: Mapping[str, Any]) -> int:
    return {"A": 2, "B": 1}.get(str(row.get("breakout_grade") or "").upper(), 0)


def rank_key(row: Mapping[str, Any]) -> tuple:
    """EV-first when claimed; then score; then grade / confluence tie-breaks.

    Compatible with ``ev_rank_key`` ordering: any real EV claim still outranks
    score-only rows. Extra tuple slots are additive tie-breaks only.
    """
    try:
        from scan.ev_engine import ev_rank_key
        base = ev_rank_key(dict(row))
    except Exception:
        base = (
            0,
            0.0,
            _f(row.get("conviction_rank") or row.get("score") or row.get("combined_score")),
        )
    return (
        *base,
        grade_rank(row),
        confluence_count(row),
    )
