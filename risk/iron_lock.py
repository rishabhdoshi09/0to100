"""
IronLock — Deterministic pre-trade hard gates.

These rules run BEFORE the LLM and AFTER the LLM. Nothing bypasses them.
If any rule fails, the trade is blocked — no exceptions, no fallbacks.

Rules:
  1. Regime confidence must be >= 40 (not flying blind)
  2. Today's drawdown must be < 5% (not in freefall)
  3. FNO-banned symbols blocked (can't build fresh position)
  4. Max 3 consecutive losses this session (tilt protection)
  5. No new position in same sector as existing open position (concentration)
  6. Kill switch check (emergency stop respected)
  7. Max open positions not exceeded
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from logger import get_logger

log = get_logger(__name__)


@dataclass
class LockVerdict:
    approved: bool
    rule_failed: Optional[str]   # None if approved
    detail: str


class IronLock:
    """
    Stateless deterministic gate. Call IronLock.check() before every trade.
    All inputs come from RegimeState, SetupCandidate, and PortfolioState-like dicts.
    """

    @staticmethod
    def check(
        regime_confidence: float,
        today_drawdown_pct: float,
        fno_banned: bool,
        consecutive_losses: int,
        open_positions: list[dict],   # list of {symbol, sector}
        candidate_sector: Optional[str],
        kill_switch_active: bool,
        max_open_positions: int = 5,
    ) -> LockVerdict:
        """
        Returns LockVerdict. approved=False means HARD BLOCK — do not trade.
        All rules checked in priority order. First failure returns immediately.
        """

        # Rule 0: Kill switch
        if kill_switch_active:
            log.warning("ironlock_kill_switch_active")
            return LockVerdict(False, "KILL_SWITCH", "Emergency stop is active. No new trades.")

        # Rule 1: Regime confidence
        if regime_confidence < 40:
            log.warning("ironlock_regime_confidence_low", confidence=regime_confidence)
            return LockVerdict(
                False, "REGIME_CONFIDENCE",
                f"Regime confidence {regime_confidence:.0f}% < 40% — market classification unreliable. Stand down."
            )

        # Rule 2: Daily drawdown
        if today_drawdown_pct >= 5.0:
            log.warning("ironlock_daily_drawdown_breached", drawdown=today_drawdown_pct)
            return LockVerdict(
                False, "DAILY_DRAWDOWN",
                f"Down {today_drawdown_pct:.1f}% today. 5% daily loss limit reached. No new trades."
            )

        # Rule 3: F&O ban
        if fno_banned:
            log.warning("ironlock_fno_banned")
            return LockVerdict(
                False, "FNO_BAN",
                "Symbol is in NSE F&O ban list. Cannot build fresh position."
            )

        # Rule 4: Consecutive losses (tilt protection)
        if consecutive_losses >= 3:
            log.warning("ironlock_consecutive_losses", count=consecutive_losses)
            return LockVerdict(
                False, "CONSECUTIVE_LOSSES",
                f"{consecutive_losses} consecutive losses this session. Step back — this is tilt territory."
            )

        # Rule 5: Sector concentration
        if candidate_sector:
            existing_sectors = [p.get("sector", "") for p in open_positions]
            if candidate_sector in existing_sectors:
                log.warning("ironlock_sector_concentration", sector=candidate_sector)
                return LockVerdict(
                    False, "SECTOR_CONCENTRATION",
                    f"Already holding a {candidate_sector} position. No double sector exposure."
                )

        # Rule 6: Max open positions
        if len(open_positions) >= max_open_positions:
            log.warning("ironlock_max_positions", count=len(open_positions), max=max_open_positions)
            return LockVerdict(
                False, "MAX_POSITIONS",
                f"At maximum {max_open_positions} open positions. Close one before opening another."
            )

        log.info("ironlock_approved", open_positions=len(open_positions))
        return LockVerdict(True, None, "All gates cleared.")
