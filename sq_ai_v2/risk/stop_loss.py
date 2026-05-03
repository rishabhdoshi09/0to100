"""
Stop-loss manager.

Three stop types per position:
  1. ATR trailing stop  — ratchets upward as price rises.
  2. Hard percentage stop — absolute maximum loss allowed.
  3. Time stop — exit if held for > max_days without profit.

Usage:
    mgr = StopLossManager()
    mgr.open(symbol, entry_price, entry_date, atr)
    should_exit, reason = mgr.check(symbol, current_price, current_date)
    mgr.update_trailing(symbol, current_price, atr)  # call every bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional, Tuple

from loguru import logger


@dataclass
class StopRecord:
    symbol: str
    entry_price: float
    entry_date: date
    atr: float
    hard_stop: float       # absolute price level (loss stop)
    trailing_stop: float   # ratcheted up as price moves in our favour
    max_days: int = 10     # time stop
    peak_price: float = 0.0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price


class StopLossManager:
    """
    Manages active stops for all open positions.
    """

    _ATR_MULTIPLIER_HARD = 2.0      # hard stop = entry - 2 × ATR
    _ATR_MULTIPLIER_TRAIL = 2.5     # trailing stop starts at entry - 2.5 × ATR
    _HARD_STOP_PCT = 0.05           # max 5% loss regardless of ATR

    def __init__(self) -> None:
        self._stops: Dict[str, StopRecord] = {}

    # ── Position lifecycle ────────────────────────────────────────────────

    def open(
        self,
        symbol: str,
        entry_price: float,
        entry_date: date,
        atr: float,
        max_days: int = 10,
    ) -> None:
        hard_stop = max(
            entry_price - self._ATR_MULTIPLIER_HARD * atr,
            entry_price * (1 - self._HARD_STOP_PCT),
        )
        trailing_stop = entry_price - self._ATR_MULTIPLIER_TRAIL * atr

        self._stops[symbol] = StopRecord(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            atr=atr,
            hard_stop=hard_stop,
            trailing_stop=trailing_stop,
            max_days=max_days,
            peak_price=entry_price,
        )
        logger.debug(
            f"Stop opened: {symbol} entry={entry_price:.2f} "
            f"hard={hard_stop:.2f} trail={trailing_stop:.2f}"
        )

    def close(self, symbol: str) -> None:
        self._stops.pop(symbol, None)

    # ── Update trailing stop ──────────────────────────────────────────────

    def update_trailing(
        self, symbol: str, current_price: float, current_atr: float
    ) -> None:
        """Ratchet trailing stop upward as price rises."""
        if symbol not in self._stops:
            return
        rec = self._stops[symbol]
        if current_price > rec.peak_price:
            rec.peak_price = current_price
            new_trail = current_price - self._ATR_MULTIPLIER_TRAIL * current_atr
            rec.trailing_stop = max(rec.trailing_stop, new_trail)

    # ── Check all stops ───────────────────────────────────────────────────

    def check(
        self,
        symbol: str,
        current_price: float,
        current_date: date,
    ) -> Tuple[bool, str]:
        """
        Returns (should_exit, reason).
        Call every bar for each open position.
        """
        if symbol not in self._stops:
            return False, "no_stop"

        rec = self._stops[symbol]

        # 1. Hard stop
        if current_price <= rec.hard_stop:
            return True, f"hard_stop price={current_price:.2f} stop={rec.hard_stop:.2f}"

        # 2. Trailing stop
        if current_price <= rec.trailing_stop:
            return True, f"trailing_stop price={current_price:.2f} stop={rec.trailing_stop:.2f}"

        # 3. Time stop
        days_held = (current_date - rec.entry_date).days
        if days_held >= rec.max_days:
            unrealised_pct = (current_price - rec.entry_price) / rec.entry_price
            if unrealised_pct <= 0:
                return True, f"time_stop days={days_held} pnl_pct={unrealised_pct:.2%}"

        return False, "ok"

    def check_all(
        self,
        prices: Dict[str, float],
        current_date: date,
    ) -> Dict[str, str]:
        """Check all open positions; return {symbol: reason} for exits."""
        exits = {}
        for sym, rec in list(self._stops.items()):
            price = prices.get(sym, rec.entry_price)
            should_exit, reason = self.check(sym, price, current_date)
            if should_exit:
                exits[sym] = reason
        return exits

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_stop_levels(self, symbol: str) -> Optional[Dict]:
        if symbol not in self._stops:
            return None
        r = self._stops[symbol]
        return {
            "hard_stop": r.hard_stop,
            "trailing_stop": r.trailing_stop,
            "peak_price": r.peak_price,
            "entry_price": r.entry_price,
        }

    def all_stops(self) -> Dict[str, Dict]:
        return {sym: self.get_stop_levels(sym) for sym in self._stops}
