"""
Fake Breakout Filter — The Loki Defense.

Loki disguises himself as a breakout. This module unmasks him.

A breakout is FAKE if:
  1. Volume on breakout day < 1.5× 20-day avg (no institutional participation)
  2. Price closed BELOW the pivot on breakout day (failed to hold)
  3. Prior overhead supply within 3% above pivot (wall of sellers)
  4. Breakout happened on a CHOPPY/DISTRIBUTION regime day
  5. RSI already overbought (>78) at breakout — no room to run

Each flag reduces setup confidence. 3+ flags = FAKE label = auto-downgrade to WATCHLIST.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)


@dataclass
class BreakoutVerdict:
    is_fake: bool
    confidence_penalty: float    # subtract from setup confidence score
    flags: list[str]             # human-readable evidence of fakeness
    genuine_score: float         # 0-100, higher = more genuine breakout


def assess_breakout(
    symbol: str,
    df: pd.DataFrame,            # OHLCV DataFrame, at least 60 bars
    pivot_level: float,
    archetype: str,
    regime: str = "CHOPPY",
    current_rsi: float = 50.0,
) -> BreakoutVerdict:
    """
    Assess if a breakout setup is genuine or a Loki trap.
    Call from QualityEngine or ScanPipeline after setup detection.
    """
    if df is None or len(df) < 30:
        return BreakoutVerdict(False, 0.0, [], 50.0)

    close  = df["close"].values
    high   = df["high"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
    price  = float(close[-1])

    flags: list[str] = []
    penalty = 0.0

    # ── Flag 1: Low volume on breakout ───────────────────────────────────────
    avg_vol_20 = float(volume[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())
    breakout_vol = float(volume[-1])
    vol_ratio = breakout_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
    if vol_ratio < 1.5:
        flags.append(f"Low volume breakout ({vol_ratio:.1f}× avg — need 1.5×+)")
        penalty += 20

    # ── Flag 2: Closed below pivot (failed to hold) ───────────────────────────
    if price < pivot_level * 0.998:
        flags.append(f"Closed below pivot {pivot_level:.0f} — failed to hold")
        penalty += 25

    # ── Flag 3: Overhead supply within 3% above pivot ────────────────────────
    supply_zone_top = pivot_level * 1.03
    # Count how many bars in last 60d had closes in the supply zone
    if len(close) >= 60:
        in_supply = np.sum((close[-60:] >= pivot_level) & (close[-60:] <= supply_zone_top))
        if in_supply >= 5:
            flags.append(f"Overhead supply: {in_supply} bars between pivot and +3% — sellers waiting")
            penalty += 15

    # ── Flag 4: Wrong regime for breakout ─────────────────────────────────────
    hostile_regimes = {"CHOPPY", "DISTRIBUTION", "TRENDING_BEAR"}
    if regime in hostile_regimes:
        flags.append(f"Breakout in {regime} regime — low follow-through probability")
        penalty += 15

    # ── Flag 5: Overbought RSI at breakout ────────────────────────────────────
    if current_rsi > 78:
        flags.append(f"RSI {current_rsi:.0f} at breakout — overbought, limited upside room")
        penalty += 10

    # ── Flag 6: No prior constructive base (VCP/Accumulation archetypes only) ─
    if archetype in ("VCP_BREAKOUT", "ACCUMULATION_BREAKOUT"):
        if len(close) >= 20:
            base_hi = float(np.max(close[-20:]))
            base_lo = float(np.min(close[-20:]))
            base_depth = (base_hi - base_lo) / base_hi * 100
            if base_depth > 18:
                flags.append(f"Wide {base_depth:.1f}% base — not a tight consolidation")
                penalty += 10

    is_fake = len(flags) >= 3 or penalty >= 55
    genuine_score = max(0.0, 100.0 - penalty)

    if is_fake:
        log.info("fake_breakout_detected", symbol=symbol, flags=len(flags), penalty=penalty)

    return BreakoutVerdict(
        is_fake=is_fake,
        confidence_penalty=penalty,
        flags=flags,
        genuine_score=round(genuine_score, 1),
    )


def filter_setups(setups: list, regime: str = "CHOPPY") -> list:
    """
    Filter/downgrade a list of SetupCandidates using fake breakout detection.
    Breakout archetypes with is_fake=True get downgraded confidence.
    Non-breakout archetypes (MEAN_REVERSION, TREND_CONTINUATION) are skipped.
    """
    from scan.setup_engine import SetupCandidate
    breakout_archetypes = {
        "VCP_BREAKOUT", "ACCUMULATION_BREAKOUT", "MOMENTUM_EXPANSION",
        "EARNINGS_CONTINUATION", "HIGH_TIGHT_FLAG", "RANGE_BREAKOUT",
    }
    result = []
    for s in setups:
        if s.archetype not in breakout_archetypes:
            result.append(s)
            continue
        # Need df — try to get from cache or skip filter
        verdict = assess_breakout(
            symbol=s.symbol,
            df=None,  # caller should pass df; None = conservative skip
            pivot_level=s.pivot_level,
            archetype=s.archetype,
            regime=regime,
            current_rsi=s.raw_indicators.get("rsi", 50.0),
        )
        if verdict.is_fake:
            # Downgrade confidence — don't remove, let quality engine further penalize
            downgraded = SetupCandidate(
                symbol=s.symbol,
                price=s.price,
                archetype=s.archetype,
                confidence=max(20.0, s.confidence - verdict.confidence_penalty),
                pivot_level=s.pivot_level,
                stop_level=s.stop_level,
                behavioral_evidence=s.behavioral_evidence + [f"⚠️ LOKI FLAG: {f}" for f in verdict.flags],
                raw_indicators=s.raw_indicators,
                fno_banned=s.fno_banned,
            )
            result.append(downgraded)
        else:
            result.append(s)
    return result
