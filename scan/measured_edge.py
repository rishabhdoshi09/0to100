"""Apply full-universe signal-backtest evidence onto live scan results.

This is the learning bridge: historical expectancy on the whole bhav universe
ranks and demotes today's setups. Never invents edge; returns None when the
report is missing, truncated, or too thin to trust.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping

from scan.signal_backtest import (
    combo_edge,
    load_report,
    report_is_actionable,
    universe_evidence_note,
)

_EDGE_VETO_R = -0.05


def apply_measured_edge(
    rows: Iterable[Any],
    *,
    veto_r: float = _EDGE_VETO_R,
) -> int:
    """Mutate scan rows (StockSignal objects or dict records) with ``edge_r``.

    Returns how many rows received a measured edge. BUY/STRONG BUY with a proven
    loser combo (≤ veto_r) are demoted to WATCH. Rows are re-sorted by
    (verdict tier, score + 40·edge).
    """
    rep = load_report()
    if not report_is_actionable(rep):
        return 0

    note = universe_evidence_note(rep)
    materialised: list[Any] = list(rows)
    tagged = 0

    def _get(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, Mapping):
            return row.get(key, default)
        return getattr(row, key, default)

    def _set(row: Any, key: str, value: Any) -> None:
        if isinstance(row, MutableMapping):
            row[key] = value
        else:
            setattr(row, key, value)

    for row in materialised:
        keys = [str(k) for k in (_get(row, "signals") or []) if k]
        if not keys:
            continue
        edge = combo_edge(keys)
        if edge is None:
            continue
        _set(row, "edge_r", float(edge))
        tagged += 1
        verdict = str(_get(row, "verdict") or "")
        if float(edge) <= float(veto_r) and verdict in {"STRONG BUY", "BUY"}:
            _set(row, "verdict", "WATCH")
            reasons = list(_get(row, "reasons") or [])
            msg = (
                f"Measured LOSER edge {float(edge):+.2f}R on {note} — "
                "demoted until evidence improves"
            )
            if msg not in reasons:
                reasons.insert(0, msg)
            _set(row, "reasons", reasons)

    vrank = {"STRONG BUY": 2, "BUY": 1}

    def _sort_key(row: Any) -> tuple:
        edge = float(_get(row, "edge_r", 0) or 0)
        score = float(_get(row, "score", 0) or 0)
        return (vrank.get(str(_get(row, "verdict") or ""), 0), score + edge * 40)

    try:
        materialised.sort(key=_sort_key, reverse=True)
        if isinstance(rows, list):
            rows[:] = materialised
    except Exception:
        pass
    return tagged
