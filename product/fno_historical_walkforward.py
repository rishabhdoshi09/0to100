"""Point-in-time walk-forward counterfactual simulation for F&O underlying
candidates, using only real historical equity OHLC.

WHAT THIS MODULE DOES AND DOES NOT PROVE
-----------------------------------------
No historical NSE option-chain (strike, premium, OI, IV, Greeks) data
source exists anywhere in this repository, live or historical, and none is
fabricated here. This module grades ONLY the underlying directional call
-- never an option contract. Every row it produces carries
``option_evidence_status = UNDERLYING_ONLY_COUNTERFACTUAL`` for exactly
that reason; nothing here may ever claim FULL_OPTION_EVIDENCE or
PARTIAL_OPTION_EVIDENCE.

The breakout/breakdown screen in :func:`evaluate_point_in_time_candidate`
is a deliberately simple, self-contained rule. It does NOT reproduce
``fo_setup.py``'s live scoring formula (which depends on live futures-OI
and relative-strength feeds this module has no point-in-time source for).
It exists only to generate a chronologically honest stream of candidates
to grade, and must never be mistaken for the production scanner.

EVIDENCE WEIGHT, NOT JUST EVIDENCE CLASS
-----------------------------------------
Every outcome this module settles is recorded with
``evidence_class=COUNTERFACTUAL``. product.conditional_evidence
.ranking_evidence() -- the only thing product.decision_ranking.rank() ever
reads -- explicitly refuses any evidence_class outside
{PAPER_FORWARD, REAL_FORWARD} (see product.evidence_class.MARKET_EVIDENCE).
That is not a limitation of this module; it is the desk-wide rule that a
hypothesis about a trade nobody actually took must never carry the same
weight as one that was. Historical simulation here establishes evidence
cells (count, expectancy_R, Wilson lower bound) that report what would
have happened -- visible for research and for "why this trade" reporting
-- but only a REAL settled PAPER_FORWARD trade (product.fno_evidence) can
move F&O ranking. tests/test_fno_walk_forward_learning.py proves this
boundary holds for the walk-forward path specifically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from product.conditional_evidence import EvidenceUpdate, record_outcome
from product.counterfactual_learning import classify_forward
from product.decision_chain import Outcome
from product.evidence_class import COUNTERFACTUAL
from product.fno_evidence import fno_context_key

UNDERLYING_ONLY_COUNTERFACTUAL = "UNDERLYING_ONLY_COUNTERFACTUAL"

_LONG = "LONG"
_SHORT = "SHORT"


@dataclass(frozen=True)
class FnoWalkForwardCandidate:
    symbol: str
    as_of: str
    direction: str
    entry: float
    stop: float
    target: float
    score: float
    taken: bool
    setup: dict[str, Any] = field(default_factory=dict)
    option_evidence_status: str = UNDERLYING_ONLY_COUNTERFACTUAL

    @property
    def context_key(self) -> str:
        return fno_context_key(self.setup)


def evaluate_point_in_time_candidate(
    symbol: str,
    as_of: date,
    history: Any,
    *,
    lookback_days: int = 20,
    breakout_buffer_pct: float = 0.5,
    volume_multiple: float = 1.5,
    min_score_to_take: float = 60.0,
) -> FnoWalkForwardCandidate | None:
    """A stock-only, point-in-time breakout/breakdown screen.

    ``history`` must be a pandas-like DataFrame indexed by session date with
    High/Low/Close/Volume columns, sorted ascending. This function
    defensively re-filters to rows on or before ``as_of`` itself -- a
    caller passing extra future rows by mistake can never leak them into
    the candidate produced for this date.
    """
    import pandas as pd

    as_of_ts = pd.Timestamp(as_of)
    window = history[history.index <= as_of_ts]
    if len(window) < lookback_days + 2:
        return None
    if window.index[-1] != as_of_ts:
        # as_of has no session in this data -- never guess a nearby one.
        return None
    prior = window.iloc[-(lookback_days + 1):-1]
    today = window.iloc[-1]
    prior_high = float(prior["High"].max())
    prior_low = float(prior["Low"].min())
    avg_vol = float(prior["Volume"].mean())
    today_close = float(today["Close"])
    today_vol = float(today["Volume"])
    atr = float((prior["High"] - prior["Low"]).mean())
    if today_close <= 0 or atr <= 0:
        return None

    breakout_up = today_close > prior_high * (1 + breakout_buffer_pct / 100.0)
    breakout_down = today_close < prior_low * (1 - breakout_buffer_pct / 100.0)
    volume_confirmed = avg_vol > 0 and today_vol >= avg_vol * volume_multiple
    if not (breakout_up or breakout_down) or not volume_confirmed:
        return None

    direction = _LONG if breakout_up else _SHORT
    reference = prior_high if breakout_up else prior_low
    breakout_pct = abs(today_close - reference) / today_close * 100.0
    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0
    score = min(100.0, 40.0 + min(30.0, vol_ratio * 10.0) + min(30.0, breakout_pct * 10.0))

    if direction == _LONG:
        entry, stop = today_close, today_close - 2.0 * atr
        target = entry + 2.0 * (entry - stop)
    else:
        entry, stop = today_close, today_close + 2.0 * atr
        target = entry - 2.0 * (stop - entry)

    setup = {
        "direction": direction,
        "score": round(score, 1),
        "atr_pct": round(atr / today_close * 100.0, 3),
        "breakout_distance_pct": round(breakout_pct, 3),
        # Sector/regime context is not knowable point-in-time from a single
        # symbol's OHLC alone; bucket neutrally rather than invent a signal.
        "components": {"nifty_alignment": 0.0, "sector_strength": 0.0},
    }
    return FnoWalkForwardCandidate(
        symbol=symbol,
        as_of=as_of.isoformat(),
        direction=direction,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        score=setup["score"],
        taken=setup["score"] >= min_score_to_take,
        setup=setup,
    )


def classify_walk_forward_outcome(
    candidate: FnoWalkForwardCandidate, *, forward_close: float
) -> str:
    """Real forward-bar grading. ``forward_close`` must come from a real
    session strictly after ``candidate.as_of`` -- classify_forward itself
    is symmetric in direction, so LONG/SHORT are normalised to one sign
    convention (positive = the move the candidate wanted) before reuse."""
    entry = candidate.entry
    if candidate.direction == _LONG:
        forward_return_pct = (forward_close - entry) / entry * 100.0
    else:
        forward_return_pct = (entry - forward_close) / entry * 100.0
    return classify_forward(
        entry=entry,
        stop=candidate.stop if candidate.direction == _LONG else entry - abs(candidate.stop - entry),
        target=candidate.target if candidate.direction == _LONG else entry + abs(candidate.target - entry),
        forward_return_pct=forward_return_pct,
    )


def record_walk_forward_outcome(
    candidate: FnoWalkForwardCandidate,
    *,
    forward_close: float,
    resolved_at: str,
    path: str | None = None,
) -> EvidenceUpdate | None:
    """Fold one graded historical candidate into the COUNTERFACTUAL-class
    evidence cell for its context. Never usable by product.decision_ranking
    .rank() to move a ranking score -- see the module docstring -- but
    inspectable (count, expectancy_R, Wilson lower bound) as a research
    hypothesis about this setup/regime/sector combination.
    """
    key = candidate.context_key
    if not key:
        return None
    risk = abs(candidate.entry - candidate.stop)
    if risk <= 0:
        return None
    if candidate.direction == _LONG:
        realized_R = (forward_close - candidate.entry) / risk
    else:
        realized_R = (candidate.entry - forward_close) / risk
    seed = f"WF-{candidate.symbol}-{candidate.as_of}-{candidate.direction}"
    outcome = Outcome(
        position_id=seed,
        paper_order_id=seed,
        paper_intent_id=seed,
        decision_id=seed,
        symbol=candidate.symbol,
        exit_reason="WALK_FORWARD_HORIZON",
        realized_R=realized_R,
        entry_price=candidate.entry,
        exit_price=forward_close,
        entry_session=candidate.as_of,
        exit_session=str(resolved_at)[:10],
        evidence_class=COUNTERFACTUAL,
        resolved_at=resolved_at,
    )
    return record_outcome(
        outcome, context_key=key, evidence_class=COUNTERFACTUAL, path=path
    )
