"""Point-in-time active-learning curriculum for historical replay.

This first curriculum stage targets ex-ante market-regime scarcity only. It
never ranks sessions by their future trade outcome, so selection cannot peek at
the answer it is trying to learn. Adaptive batches remain HISTORICAL_REPLAY
research evidence and are not promotion/forward evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HistoricalMarketState:
    as_of: str
    available: bool
    regime: str = "UNAVAILABLE"
    volatility: str = "UNAVAILABLE"
    close: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    atr_ratio: float | None = None
    history_rows: int = 0
    source: str = "official_nse_index_cache"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _series(frame, name: str):
    for key in (name, name.lower(), name.upper(), name.title()):
        if key in getattr(frame, "columns", []):
            return np.asarray(frame[key], dtype=float)
    return None


def classify_session(
    as_of: str,
    *,
    index_fn: Callable[[str, str], Any] | None = None,
    min_history: int = 200,
) -> HistoricalMarketState:
    """Classify a past session using only official index rows available at T."""
    loader = index_fn
    if loader is None:
        from data.index_store import get_index_ohlcv_as_of_cached as loader
    try:
        frame = loader("^NSEI", str(as_of)[:10])
    except Exception as exc:
        return HistoricalMarketState(
            str(as_of)[:10], False, reason=f"index load failed: {type(exc).__name__}"
        )
    if frame is None or getattr(frame, "empty", True):
        return HistoricalMarketState(str(as_of)[:10], False, reason="official index history unavailable")
    rows = int(len(frame))
    if rows < int(min_history):
        return HistoricalMarketState(
            str(as_of)[:10], False, history_rows=rows,
            reason=f"need {int(min_history)} PIT index rows; have {rows}",
        )
    close = _series(frame, "Close")
    high = _series(frame, "High")
    low = _series(frame, "Low")
    if close is None or close.size < int(min_history):
        return HistoricalMarketState(str(as_of)[:10], False, history_rows=rows, reason="close history unavailable")
    if high is None:
        high = close.copy()
    if low is None:
        low = close.copy()
    if not np.all(np.isfinite(close[-200:])):
        return HistoricalMarketState(str(as_of)[:10], False, history_rows=rows, reason="non-finite index history")

    px = float(close[-1])
    sma50 = float(np.mean(close[-50:]))
    sma200 = float(np.mean(close[-200:]))
    if px > sma50 > sma200:
        regime = "BULL_TREND"
    elif px < sma50 < sma200:
        regime = "BEAR"
    elif px > sma200 and px < sma50:
        regime = "DISTRIBUTION"
    elif px < sma200 and px > sma50:
        regime = "RECOVERY"
    else:
        regime = "CHOPPY"

    tr = []
    for i in range(max(1, len(close) - 35), len(close)):
        tr.append(max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        ))
    atr_ratio = None
    volatility = "UNAVAILABLE"
    if len(tr) >= 20:
        recent = float(np.mean(tr[-5:]))
        older = float(np.mean(tr[-20:-5]))
        if older > 0:
            atr_ratio = recent / older
            if atr_ratio > 1.20:
                volatility = "EXPANDING"
            elif atr_ratio < 0.85:
                volatility = "CONTRACTING"
            else:
                volatility = "STABLE"

    return HistoricalMarketState(
        as_of=str(as_of)[:10],
        available=True,
        regime=regime,
        volatility=volatility,
        close=round(px, 6),
        sma50=round(sma50, 6),
        sma200=round(sma200, 6),
        atr_ratio=None if atr_ratio is None else round(float(atr_ratio), 6),
        history_rows=rows,
    )


def select_regime_balanced_sessions(
    eligible_sessions: Sequence[str],
    *,
    processed_sessions: Sequence[str] = (),
    batch_size: int = 8,
    classifier: Callable[[str], HistoricalMarketState] | None = None,
) -> dict[str, Any]:
    """Choose unprocessed sessions that reduce PIT regime-coverage imbalance.

    If PIT regime metadata is unavailable, selection falls back to the earliest
    unprocessed sessions. This selector does not inspect future trade outcomes.
    """
    classify = classifier or (lambda day: classify_session(day))
    eligible = [str(x)[:10] for x in eligible_sessions]
    eligible_set = set(eligible)
    processed = [str(x)[:10] for x in processed_sessions if str(x)[:10] in eligible_set]
    processed_set = set(processed)
    remaining = [day for day in eligible if day not in processed_set]
    want = max(1, int(batch_size))
    if not remaining:
        return {
            "sessions": [],
            "selection_policy": "ACTIVE_REGIME_COVERAGE",
            "reason": "historical_backlog_caught_up",
            "coverage_before": {},
            "coverage_after": {},
            "session_states": {},
        }

    state_cache: dict[str, HistoricalMarketState] = {}
    def state(day: str) -> HistoricalMarketState:
        if day not in state_cache:
            try:
                state_cache[day] = classify(day)
            except Exception as exc:
                state_cache[day] = HistoricalMarketState(
                    day, False, reason=f"classification failed: {type(exc).__name__}"
                )
        return state_cache[day]

    coverage = Counter(
        s.regime for day in processed if (s := state(day)).available
    )
    buckets: dict[str, list[str]] = defaultdict(list)
    unknown: list[str] = []
    for day in remaining:
        s = state(day)
        if s.available:
            buckets[s.regime].append(day)
        else:
            unknown.append(day)

    selected: list[str] = []
    coverage_before = dict(sorted(coverage.items()))
    # Greedy balancing: after each pick, increment that regime before choosing
    # again. This is deterministic and prevents one rare regime from consuming
    # the entire batch merely because its initial count was lowest.
    while len(selected) < want and any(buckets.values()):
        available_regimes = [r for r, q in buckets.items() if q]
        regime = min(available_regimes, key=lambda r: (coverage.get(r, 0), r))
        day = buckets[regime].pop(0)
        selected.append(day)
        coverage[regime] += 1

    # Truthful fallback for dates whose PIT index context is unavailable.
    for day in unknown:
        if len(selected) >= want:
            break
        selected.append(day)

    states = {day: state(day).as_dict() for day in selected}
    known_selected = sum(1 for day in selected if state(day).available)
    policy = "ACTIVE_REGIME_COVERAGE" if known_selected else "DURABLE_CURSOR"
    return {
        "sessions": selected,
        "selection_policy": policy,
        "selection_objective": "reduce point-in-time regime coverage imbalance",
        "outcome_blind_selection": True,
        "coverage_before": coverage_before,
        "coverage_after": dict(sorted(coverage.items())),
        "session_states": states,
        "known_regime_sessions_selected": known_selected,
        "unknown_regime_sessions_selected": len(selected) - known_selected,
    }
