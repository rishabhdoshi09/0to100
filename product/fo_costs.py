"""Versioned, deterministic cost model for NSE long-option paper evidence.

Rates mirror the published Zerodha resident-individual / NSE option schedule
verified in September 2026. The model covers normal market buy+sell execution
of long options; exercised/physically-settled options are intentionally outside
this paper lane.

Every rate can be overridden explicitly through environment variables so a
brokerage/tax revision does not require changing paper-ledger semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import os
from typing import Callable


def _pct(value: float) -> float:
    return float(value) / 100.0


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _round_rupee_half_up(value: float) -> float:
    return float(Decimal(str(max(0.0, value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class NseLongOptionCostSchedule:
    brokerage_per_order: float = 20.0
    transaction_charge_pct: float = 0.03553
    stt_sell_pct: float = 0.15
    stamp_buy_pct: float = 0.003
    sebi_charge_per_crore: float = 10.0
    gst_pct: float = 18.0
    effective_from: str = "2026-04-01"
    verified_as_of: str = "2026-09-26"

    @classmethod
    def from_env(cls) -> "NseLongOptionCostSchedule":
        return cls(
            brokerage_per_order=_env_float("QT_ZERODHA_OPTION_BROKERAGE_PER_ORDER", 20.0),
            transaction_charge_pct=_env_float("QT_NSE_OPTION_TRANSACTION_CHARGE_PCT", 0.03553),
            stt_sell_pct=_env_float("QT_FO_OPTION_STT_SELL_PCT", 0.15),
            stamp_buy_pct=_env_float("QT_FO_OPTION_STAMP_BUY_PCT", 0.003),
            sebi_charge_per_crore=_env_float("QT_SEBI_TURNOVER_CHARGE_PER_CRORE", 10.0),
            gst_pct=_env_float("QT_GST_PCT", 18.0),
        )

    @property
    def name(self) -> str:
        return (
            "ZERODHA_NSE_LONG_OPTIONS_2026_04"
            f":B{self.brokerage_per_order:g}"
            f":TXN{self.transaction_charge_pct:g}%"
            f":STT{self.stt_sell_pct:g}%"
            f":STAMP{self.stamp_buy_pct:g}%"
            f":SEBI{self.sebi_charge_per_crore:g}/CR"
            f":GST{self.gst_pct:g}%"
        )

    def round_trip_cost(self, entry_price: float, exit_price: float, quantity: int) -> float:
        """Return estimated charges for one normal long-option buy then sell.

        STT follows the published sell-side option-premium rule and is rounded
        to the nearest rupee using half-up arithmetic. Exchange, SEBI, stamp and
        GST are kept at paise precision in the final aggregate.
        """
        entry = float(entry_price)
        exit_ = float(exit_price)
        qty = int(quantity)
        if entry <= 0 or exit_ <= 0 or qty <= 0:
            return 0.0

        buy_turnover = entry * qty
        sell_turnover = exit_ * qty
        total_turnover = buy_turnover + sell_turnover

        brokerage = self.brokerage_per_order * 2.0
        exchange = total_turnover * _pct(self.transaction_charge_pct)
        sebi = total_turnover * (self.sebi_charge_per_crore / 10_000_000.0)
        gst = (brokerage + exchange + sebi) * _pct(self.gst_pct)
        stt = _round_rupee_half_up(sell_turnover * _pct(self.stt_sell_pct))
        stamp = buy_turnover * _pct(self.stamp_buy_pct)

        return round(brokerage + exchange + sebi + gst + stt + stamp, 2)


def zerodha_nse_option_cost_model_from_env() -> tuple[Callable[[float, float, int], float], str]:
    schedule = NseLongOptionCostSchedule.from_env()
    return schedule.round_trip_cost, schedule.name
