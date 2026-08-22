"""Rebalance calendars and next-session execution. No same-close fills."""
from __future__ import annotations

from datetime import date


def month_ends(sessions: list[date]) -> list[date]:
    last: dict[tuple[int, int], date] = {}
    for d in sessions:
        last[(d.year, d.month)] = d
    return [last[k] for k in sorted(last)]


def quarter_ends(ends: list[date]) -> list[date]:
    return [d for d in ends if d.month in (3, 6, 9, 12)]


def every_other(ends: list[date]) -> list[date]:
    return list(ends[::2])


def every_n_sessions(sessions: list[date], n: int, start: date) -> list[date]:
    out: list[date] = []
    i = 0
    while i < len(sessions):
        if sessions[i] >= start:
            out.append(sessions[i])
            i += n
        else:
            i += 1
    return out


def next_session(sessions: list[date], t: date, index: dict[date, int] | None = None) -> date | None:
    """First official session strictly after t. None if none exists."""
    if index is not None:
        i = index.get(t)
        if i is None:
            later = [d for d in sessions if d > t]
            return later[0] if later else None
        return sessions[i + 1] if i + 1 < len(sessions) else None
    later = [d for d in sessions if d > t]
    return later[0] if later else None


def holding_window(
    sessions: list[date],
    rebalance: date,
    next_rebalance: date | None,
    index: dict[date, int] | None = None,
) -> tuple[date, date] | None:
    """Rank at rebalance close; enter next open; exit next-rebalance next open."""
    if next_rebalance is None:
        return None
    entry = next_session(sessions, rebalance, index)
    exit_d = next_session(sessions, next_rebalance, index)
    if entry is None or exit_d is None:
        return None
    if entry <= rebalance or exit_d <= next_rebalance:
        return None
    if exit_d <= entry:
        return None
    return entry, exit_d


def one_way_turnover(prev: list[str], picks: list[str], n: int) -> float:
    if not prev:
        return 1.0
    added = len(set(picks) - set(prev))
    removed = len(set(prev) - set(picks))
    return (added + removed) / (2.0 * max(int(n), 1))


def cost_fraction(one_way: float, rt_cost_pct: float) -> float:
    """rt_cost_pct is percent points (0.32 = 0.32%), not a fraction."""
    return float(one_way) * float(rt_cost_pct) / 100.0
