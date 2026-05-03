"""Risk manager – position sizing, kill-switch, exposure caps.

Implements:
  • Kelly fraction (capped at 0.25)
  • Volatility targeting (per-symbol annualised vol → fraction of equity)
  • 4 % daily loss kill-switch
  • 50 % gross exposure cap
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RiskConfig:
    daily_loss_limit_pct: float = 4.0
    max_exposure_pct: float = 50.0
    target_vol_pct: float = 15.0
    max_position_pct: float = 10.0
    kelly_cap: float = 0.25
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    time_stop_bars: int = 20

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            daily_loss_limit_pct=float(os.environ.get("SQ_DAILY_LOSS_LIMIT_PCT", 4.0)),
            max_exposure_pct=float(os.environ.get("SQ_MAX_EXPOSURE_PCT", 50.0)),
        )


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig.from_env()

    # ---------------------------------------------------------- kill switch
    def kill_switch_triggered(self, equity_today: float, equity_open: float,
                              gross_exposure: float, equity: float) -> tuple[bool, str]:
        daily_pnl_pct = (equity_today - equity_open) / equity_open * 100 if equity_open else 0.0
        if daily_pnl_pct <= -self.cfg.daily_loss_limit_pct:
            return True, f"daily-loss {daily_pnl_pct:.2f}% ≤ -{self.cfg.daily_loss_limit_pct}%"
        if equity > 0 and gross_exposure / equity * 100 >= self.cfg.max_exposure_pct:
            return True, f"exposure {gross_exposure/equity*100:.1f}% ≥ {self.cfg.max_exposure_pct}%"
        return False, ""

    # ---------------------------------------------------------- sizing
    def position_size(self, equity: float, price: float, atr_value: float,
                      confidence: float, win_rate: float = 0.55,
                      win_loss_ratio: float = 1.5) -> int:
        """Return integer share count.

        Uses *the smaller of*:
          • Kelly: f* = W - (1-W)/R   (capped at ``kelly_cap``)
          • Volatility-target: target_vol / realised_vol
        Then caps by ``max_position_pct`` of equity.
        """
        if equity <= 0 or price <= 0 or atr_value <= 0:
            return 0
        kelly = max(0.0, win_rate - (1 - win_rate) / max(win_loss_ratio, 0.01))
        kelly = min(kelly, self.cfg.kelly_cap)

        # treat ATR as a daily-vol proxy in price units → convert to %
        ann_vol = (atr_value / price) * (252 ** 0.5)
        vol_target = self.cfg.target_vol_pct / 100.0
        vol_frac = vol_target / max(ann_vol, 0.01)

        frac = min(kelly, vol_frac, self.cfg.max_position_pct / 100.0) * max(confidence, 0.0)
        notional = equity * frac
        qty = int(notional // price)
        return max(qty, 0)

    # ---------------------------------------------------------- stops
    def stop_and_target(self, entry: float, atr_value: float,
                        side: int = 1) -> tuple[float, float]:
        """Return (stop, target).  side=+1 long, -1 short."""
        stop = entry - side * self.cfg.atr_stop_mult * atr_value
        tgt = entry + side * self.cfg.atr_target_mult * atr_value
        return float(stop), float(tgt)
