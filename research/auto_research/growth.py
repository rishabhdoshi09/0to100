"""
🌱 Growth — how the system actually gets smarter (backtest → forward test → learn).

The single most important lesson in quant research: a good BACKTEST is not an edge. An edge
is a backtest result that KEEPS WORKING out-of-sample. So the system grows up by doing both,
automatically, and comparing them:

    1. BACKTEST (in-sample)   — discovery evaluates a candidate on history. Survivors only.
    2. FORWARD TEST (out-of-sample) — survivors are deployed to PAPER and traded each day on
       data the backtest never saw.
    3. CALIBRATE — compare the forward (paper) edge against the backtest edge:
         • forward holds up      → CONFIRMED   (trust rises, keep trading, favour the family)
         • forward much weaker    → DECAYED     (trust falls)
         • forward edge vanished  → OVERFIT     (retire it — the backtest lied)
    4. REMEMBER — fold the verdict into persistent Knowledge; tomorrow's search is biased by
       what forward-tests well. The child remembers and updates its priors.

`calibrate()` is pure and deterministic. The daily orchestration lives on the brain
(`AutoResearchBrain.grow_one_day`). Everything stays PAPER-only and live-locked.
"""
from __future__ import annotations

from dataclasses import dataclass

# a strategy needs at least this many out-of-sample paper trades before we judge its forward
MIN_FORWARD_TRADES = 20

CONFIRMED = "CONFIRMED"
WEAKER_POSITIVE = "WEAKER_POSITIVE"
DECAYED = "DECAYED"
OVERFIT = "OVERFIT"
FORWARD_PENDING = "FORWARD_PENDING"


@dataclass
class Calibration:
    strategy_id: str
    family: str
    backtest_R: float
    forward_R: float
    n_forward: int
    verdict: str
    keep: bool                    # should it keep trading in paper?
    note: str

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


def calibrate(strategy_id: str, family: str, backtest_R: float, forward_R: float,
              n_forward: int, *, min_forward: int = MIN_FORWARD_TRADES,
              confirm_frac: float = 0.7, decay_frac: float = 0.3,
              forward_lower_R: float | None = None) -> Calibration:
    """Compare a strategy's forward (paper) edge to its backtest edge and return a verdict.

    Thresholds are expressed as fractions of the backtested edge, so a strategy that keeps
    ≥70% of its backtested R forward is CONFIRMED, one that keeps <30% has DECAYED, and one
    whose forward edge is non-positive is OVERFIT (the backtest did not generalise).

    Noise-awareness: when `forward_lower_R` (a conservative lower estimate, e.g. mean − 1 SE)
    is supplied, the OVERFIT test uses IT rather than the point mean — so a couple of lucky
    trades can't rescue a strategy whose edge isn't statistically distinguishable from zero.
    """
    if n_forward < min_forward:
        return Calibration(strategy_id, family, backtest_R, forward_R, n_forward,
                           FORWARD_PENDING, keep=True,
                           note=f"forward test still running ({n_forward}/{min_forward} trades)")
    edge_floor = forward_lower_R if forward_lower_R is not None else forward_R
    if edge_floor <= 0:
        lb = "" if forward_lower_R is None else f" (lower est. {forward_lower_R:+.2f}R)"
        return Calibration(strategy_id, family, backtest_R, forward_R, n_forward, OVERFIT,
                           keep=False,
                           note=f"backtest said {backtest_R:+.2f}R but forward is "
                                f"{forward_R:+.2f}R{lb} — not distinguishable from zero "
                                "out-of-sample (overfit).")
    # positive forward edge — how much of the backtest did it keep?
    kept = (forward_R / backtest_R) if backtest_R > 0 else 1.0
    if kept >= confirm_frac:
        return Calibration(strategy_id, family, backtest_R, forward_R, n_forward, CONFIRMED,
                           keep=True,
                           note=f"forward {forward_R:+.2f}R held up vs backtest "
                                f"{backtest_R:+.2f}R ({kept:.0%} kept) — confirmed.")
    if kept < decay_frac:
        return Calibration(strategy_id, family, backtest_R, forward_R, n_forward, DECAYED,
                           keep=False,
                           note=f"forward {forward_R:+.2f}R is only {kept:.0%} of backtest "
                                f"{backtest_R:+.2f}R — edge decaying, standing it down.")
    return Calibration(strategy_id, family, backtest_R, forward_R, n_forward, WEAKER_POSITIVE,
                       keep=True,
                       note=f"forward {forward_R:+.2f}R is weaker than backtest "
                            f"{backtest_R:+.2f}R ({kept:.0%}) but still positive — keep watching.")
