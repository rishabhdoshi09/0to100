"""
Risk manager — the non-negotiable gatekeeper between signals and execution.

Responsibilities:
  1. Per-trade position size enforcement.
  2. Max capital exposure (portfolio level).
  3. Max open positions.
  4. Daily loss limit (triggers kill switch).
  5. Duplicate position guard.
  6. Manual kill switch (thread-safe).

DESIGN PRINCIPLE:
  If ANY check fails → trade is BLOCKED.
  Risk manager never modifies signals — it only approves or rejects.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date
from typing import Dict

from config import settings
from llm.signal_validator import TradingSignal
from logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    adjusted_size: float  # actual quantity to trade (shares)
    signal: TradingSignal


class RiskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kill_switch_active: bool = False
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = date.today()

    # ── Kill Switch ────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "manual") -> None:
        with self._lock:
            self._kill_switch_active = True
        log.critical("KILL_SWITCH_ACTIVATED", reason=reason)

    def deactivate_kill_switch(self) -> None:
        with self._lock:
            self._kill_switch_active = False
        log.warning("kill_switch_deactivated")

    def is_kill_switch_active(self) -> bool:
        with self._lock:
            return self._kill_switch_active

    # ── Daily PnL Tracking ─────────────────────────────────────────────────

    def record_pnl(self, pnl: float) -> None:
        """Call after each trade fill with realized PnL."""
        with self._lock:
            today = date.today()
            if today != self._daily_pnl_date:
                # New trading day — reset
                self._daily_pnl = 0.0
                self._daily_pnl_date = today
            self._daily_pnl += pnl

    def get_daily_pnl(self) -> float:
        with self._lock:
            return self._daily_pnl

    # ── Trade Approval ─────────────────────────────────────────────────────

    def evaluate(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        open_positions: Dict[str, float],  # symbol → market value
        last_price: float,
    ) -> RiskDecision:
        """
        Evaluate whether the trade represented by *signal* is allowed.

        Returns RiskDecision with approved=False and reason if blocked.
        """
        # ── 0. Kill switch ────────────────────────────────────────────────
        if self.is_kill_switch_active():
            return self._block(signal, "kill_switch_active", 0.0)

        # ── 1. Daily loss limit ────────────────────────────────────────────
        daily_loss_threshold = -portfolio_value * settings.max_daily_loss_pct
        daily_pnl = self.get_daily_pnl()
        if daily_pnl < daily_loss_threshold:
            self.activate_kill_switch("daily_loss_limit_breached")
            return self._block(signal, f"daily_loss_limit_breached:{daily_pnl:.0f}", 0.0)

        # ── 2. HOLD signals never reach execution ──────────────────────────
        if signal.action == "HOLD" or signal.rejected:
            return self._block(signal, "hold_or_rejected", 0.0)

        # ── 3. SELL fast-path — exits always approved if position exists ──────
        if signal.action == "SELL":
            pos_value = open_positions.get(signal.symbol, 0.0)
            if pos_value <= 0:
                return self._block(signal, "no_position_to_sell", 0.0)
            if last_price <= 0:
                return self._block(signal, "invalid_price:zero", 0.0)
            quantity = max(1, int(pos_value / last_price))
            log.info("risk_approved_sell", symbol=signal.symbol, quantity=quantity)
            return RiskDecision(
                approved=True,
                reason="sell_approved",
                adjusted_size=quantity,
                signal=signal,
            )

        # ── 4. BUY checks only below this line ────────────────────────────

        # ── 5. Max open positions ──────────────────────────────────────────
        current_open = len(open_positions)
        if current_open >= settings.max_open_positions:
            return self._block(signal, f"max_open_positions:{current_open}", 0.0)

        # ── 6. Duplicate position guard ────────────────────────────────────
        if signal.symbol in open_positions:
            return self._block(signal, f"already_long:{signal.symbol}", 0.0)

        # ── 7. Max capital exposure check ─────────────────────────────────
        total_exposure = sum(open_positions.values())
        if total_exposure / max(portfolio_value, 1) >= settings.max_capital_exposure:
            return self._block(signal, f"max_exposure_breached:{total_exposure:.0f}", 0.0)

        # ── 8. Compute BUY size (shares) ───────────────────────────────────
        max_trade_value = portfolio_value * settings.max_position_size_pct
        desired_trade_value = portfolio_value * min(
            signal.position_size, settings.max_position_size_pct
        )
        trade_value = min(desired_trade_value, max_trade_value)

        if last_price <= 0:
            return self._block(signal, "invalid_price:zero", 0.0)

        quantity = int(trade_value / last_price)
        if quantity < 1:
            return self._block(signal, f"quantity_too_small:{quantity}", 0.0)

        log.info(
            "risk_approved",
            symbol=signal.symbol,
            action=signal.action,
            quantity=quantity,
            trade_value=trade_value,
            confidence=signal.confidence,
        )
        return RiskDecision(
            approved=True,
            reason="all_checks_passed",
            adjusted_size=quantity,
            signal=signal,
        )

    def get_risk_limits_dict(self) -> Dict:
        return {
            "max_capital_exposure": settings.max_capital_exposure,
            "max_position_size_pct": settings.max_position_size_pct,
            "max_open_positions": settings.max_open_positions,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "min_signal_confidence": settings.min_signal_confidence,
            "kill_switch_active": self.is_kill_switch_active(),
            "daily_pnl": self.get_daily_pnl(),
        }

    @staticmethod
    def _block(signal: TradingSignal, reason: str, size: float) -> RiskDecision:
        log.warning("risk_blocked", symbol=signal.symbol, reason=reason)
        return RiskDecision(
            approved=False,
            reason=reason,
            adjusted_size=size,
            signal=signal,
        )
