"""One rulebook for "did this trade work".

Until 2026-11-15 every evidence source judged trades by its own rules:

    walk-forward backtest   10 sessions, breakeven trail at +2%, gap ignored
    forward outcome tracker 15 sessions, no trail
    decision journal         5 sessions, close-to-close
    paper book              20 days, no trail, gap-aware exits

So the backtest, the forward record and the paper book measured three
different strategies, and the evidence gate compared numbers that were not
comparable. Every source now reads its horizon and trade-management rules
from here, and resolves trades with :func:`simulate_trade`.

Env overrides keep their historical names so an operator's existing setting
still wins: QT_EVAL_HORIZON (all), QT_OUTCOME_HORIZON / QT_DECISION_HORIZON
(legacy, per source), QT_BT_BREAKEVEN_PCT (trail trigger).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


def _int_env(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


#: Sessions a trade is given to reach its stop or target before it is marked
#: to the close. 10 is the walk-forward backtest's horizon, the source the
#: scanner's calibration and evidence gate are built on.
HORIZON_SESSIONS = _int_env("QT_EVAL_HORIZON", default=10)
OUTCOME_HORIZON_SESSIONS = _int_env("QT_OUTCOME_HORIZON", "QT_EVAL_HORIZON", default=HORIZON_SESSIONS)
DECISION_HORIZON_SESSIONS = _int_env("QT_DECISION_HORIZON", "QT_EVAL_HORIZON", default=HORIZON_SESSIONS)
#: Once a session's high reaches entry * (1 + this/100) the stop moves to entry.
BREAKEVEN_TRIGGER_PCT = _float_env("QT_BT_BREAKEVEN_PCT", 2.0)


def spec_tag(horizon: int | None = None) -> str:
    """Identity of the rulebook a settled outcome was measured under.

    Every settled row carries it, so outcomes measured under different rules
    (the pre-2026-11-15 5-session close-to-close counterfactuals, say) can be
    told apart and never pooled into one learning score.
    """
    return f"sim:h{int(horizon or OUTCOME_HORIZON_SESSIONS)}:be{BREAKEVEN_TRIGGER_PCT:g}:gap"


@dataclass(frozen=True)
class TradeResult:
    """One simulated trade. ``r`` is net of nothing: costs are applied by the caller."""

    outcome: str            # WIN | LOSS | SCRATCH | FLAT | NO_FILL
    r: float                # (exit - fill) / (fill - stop); 0.0 for NO_FILL
    fill_price: float       # 0.0 for NO_FILL
    exit_price: float       # 0.0 for NO_FILL
    entry_offset: int       # bar index of the fill, -1 for NO_FILL
    exit_offset: int        # bar index of the exit, -1 for NO_FILL

    @property
    def filled(self) -> bool:
        return self.outcome != "NO_FILL"

    @property
    def worked(self) -> int:
        """Forward-tracker convention: 1 profit, 0 loss or scratch, -1 never filled."""
        if not self.filled:
            return -1
        return 1 if self.r > 0 else 0


def simulate_trade(
    entry: float,
    stop: float,
    target: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    opens: Sequence[float] | None = None,
    be_pct: float | None = None,
) -> TradeResult:
    """Resolve one long breakout trade over the bars after the signal.

    A buy-stop at ``entry`` fills on the first bar whose high reaches it, at
    the open when the bar gaps above entry. The stop is checked before the
    target on every bar (conservative when both are inside one bar). With
    opens, a gap through the stop exits at the open (a loss worse than 1R) and
    a gap through the target exits at the open. Once a high reaches the
    breakeven trigger the stop moves to the fill price, so a faded winner is a
    SCRATCH. A trade still open after the last bar is FLAT, marked to that
    close. R is measured on the risk actually taken: fill minus stop.
    """
    if be_pct is None:
        be_pct = BREAKEVEN_TRIGGER_PCT
    entry, stop, target = float(entry), float(stop), float(target)
    if not (entry > 0 and stop > 0 and entry > stop):
        return TradeResult("NO_FILL", 0.0, 0.0, 0.0, -1, -1)
    n = min(len(highs), len(lows), len(closes))
    fill = 0.0
    fill_i = -1
    cur_stop = stop
    for i in range(n):
        h, low, c = float(highs[i]), float(lows[i]), float(closes[i])
        o = float(opens[i]) if opens is not None and i < len(opens) else None
        if fill_i < 0:
            if h < entry:
                continue
            fill_i = i
            fill = max(entry, o) if o is not None and o > 0 else entry
            if fill <= stop:
                # Opened through the stop and above entry is impossible; a
                # malformed bar must not invent a trade.
                return TradeResult("NO_FILL", 0.0, 0.0, 0.0, -1, -1)
            be_level = fill * (1 + be_pct / 100) if be_pct > 0 else None
            gap_bar = False      # the fill bar's open is the fill, not a gap exit
        else:
            gap_bar = o is not None and o > 0
        risk = fill - stop
        if gap_bar and o <= cur_stop:
            return _closed(fill, o, cur_stop, risk, fill_i, i)
        if gap_bar and o >= target:
            return TradeResult("WIN", (o - fill) / risk, fill, o, fill_i, i)
        if low <= cur_stop:
            return _closed(fill, cur_stop, cur_stop, risk, fill_i, i)
        if h >= target:
            return TradeResult("WIN", (target - fill) / risk, fill, target, fill_i, i)
        if be_level is not None and cur_stop < fill and h >= be_level:
            cur_stop = fill
    if fill_i < 0:
        return TradeResult("NO_FILL", 0.0, 0.0, 0.0, -1, -1)
    last = float(closes[n - 1])
    return TradeResult("FLAT", (last - fill) / (fill - stop), fill, last, fill_i, n - 1)


def _closed(fill: float, exit_px: float, stop_level: float, risk: float, fill_i: int, i: int) -> TradeResult:
    r = (exit_px - fill) / risk
    if stop_level >= fill and exit_px >= fill:
        return TradeResult("SCRATCH", r, fill, exit_px, fill_i, i)
    return TradeResult("LOSS" if r < 0 else "SCRATCH", r, fill, exit_px, fill_i, i)
