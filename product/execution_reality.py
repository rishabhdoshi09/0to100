"""Canonical Execution Reality Engine (shadow / analytics-only).

Paper and backtest *gross* results assume a perfect fill at the intended
price. This engine produces a second, labelled
``execution_adjusted_result`` that applies India cash-equity costs and
fill-quality effects *without overwriting the gross evidence*.

Non-negotiable:
- Real paper order creation is NOT modified by this engine.
- Missing microstructure stays missing (estimated vs measured).
- Live money remains fail-closed elsewhere.

Reuse: wraps ``execution.cost_model.zerodha_charges`` so brokerage/STT/
exchange/SEBI/GST/stamp/DP stay one formula, not a second invented model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from execution.cost_model import zerodha_charges

ENGINE_VERSION = "1"
SCHEMA_VERSION = 1  # previous: none. Additive analytics artifact.
SHADOW_MODE = True  # must stay True until an explicit promotion contract

# Conservative India cash-equity *estimates* used only when the caller
# does not supply a measured value. Each is labelled estimated.
DEFAULT_SPREAD_BPS = 5.0  # half-spread paid on entry+exit if unknown
DEFAULT_SLIPPAGE_BPS = 0.0
DEFAULT_PARTICIPATION_CAP = 0.10  # 10% of bar volume
# Statutory rates live in execution.cost_model (Zerodha CNC). Do not fork them.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModeledField:
    """One modeled cost/fill effect with full provenance."""

    name: str
    value: float | None
    unit: str
    source: str
    formula: str
    assumptions: str
    timestamp: str
    confidence: str  # high | medium | low | none
    measured: bool
    estimated: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(
    name: str,
    value: float | None,
    *,
    unit: str,
    source: str,
    formula: str,
    assumptions: str,
    confidence: str,
    measured: bool,
    notes: str = "",
    timestamp: str | None = None,
) -> ModeledField:
    return ModeledField(
        name=name,
        value=value,
        unit=unit,
        source=source,
        formula=formula,
        assumptions=assumptions,
        timestamp=timestamp or _now_iso(),
        confidence=confidence,
        measured=bool(measured),
        estimated=not measured,
        notes=notes,
    )


def _missing(name: str, *, reason: str, unit: str = "inr") -> ModeledField:
    return _field(
        name,
        None,
        unit=unit,
        source="unavailable",
        formula="none",
        assumptions=reason,
        confidence="none",
        measured=False,
        notes="missing stays missing — not treated as zero in evidence",
    )


@dataclass
class FillQuality:
    fill_status: str  # PERFECT | PARTIAL | NO_FILL | DELAYED | GAP_THROUGH_STOP | GAP_THROUGH_ENTRY | CIRCUIT
    intended_qty: float
    filled_qty: float
    intended_price: float
    fill_price: float | None
    reason: str
    fields: list[ModeledField] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_status": self.fill_status,
            "intended_qty": self.intended_qty,
            "filled_qty": self.filled_qty,
            "intended_price": self.intended_price,
            "fill_price": self.fill_price,
            "reason": self.reason,
            "fields": [f.to_dict() for f in self.fields],
        }


@dataclass
class ExecutionRealityResult:
    engine_version: str
    schema_version: int
    shadow_mode: bool
    affects_paper_orders: bool
    gross_result: dict[str, Any]
    execution_adjusted_result: dict[str, Any]
    charges: list[ModeledField]
    fill: FillQuality
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "shadow_mode": self.shadow_mode,
            "affects_paper_orders": self.affects_paper_orders,
            "gross_result": dict(self.gross_result),
            "execution_adjusted_result": dict(self.execution_adjusted_result),
            "charges": [c.to_dict() for c in self.charges],
            "fill": self.fill.to_dict(),
            "warnings": list(self.warnings),
        }


class ExecutionRealityEngine:
    """Shadow analytics. Does not open, close, or resize paper positions."""

    def __init__(self, *, shadow_mode: bool = True) -> None:
        self.shadow_mode = bool(shadow_mode)
        self.affects_paper_orders = False  # hard: analytics only

    def analyze_round_trip(
        self,
        *,
        side: str = "BUY",
        qty: float,
        entry_price: float,
        exit_price: float,
        bid: float | None = None,
        ask: float | None = None,
        slippage_bps: float | None = None,
        bar_volume: float | None = None,
        participation_cap: float = DEFAULT_PARTICIPATION_CAP,
        open_price: float | None = None,
        stop_price: float | None = None,
        circuit_hit: bool = False,
        delayed_fill: bool = False,
        no_fill: bool = False,
        measured_charges: Mapping[str, float] | None = None,
        as_of: str | None = None,
    ) -> ExecutionRealityResult:
        ts = as_of or _now_iso()
        warnings: list[str] = []
        intended_qty = float(qty)
        intended_entry = float(entry_price)
        intended_exit = float(exit_price)
        side_u = str(side or "BUY").upper()

        fill = self._fill_quality(
            side=side_u,
            qty=intended_qty,
            entry_price=intended_entry,
            exit_price=intended_exit,
            bid=bid,
            ask=ask,
            slippage_bps=slippage_bps,
            bar_volume=bar_volume,
            participation_cap=participation_cap,
            open_price=open_price,
            stop_price=stop_price,
            circuit_hit=circuit_hit,
            delayed_fill=delayed_fill,
            no_fill=no_fill,
            timestamp=ts,
            warnings=warnings,
        )

        filled = float(fill.filled_qty)
        fill_entry = float(fill.fill_price) if fill.fill_price is not None else intended_entry
        # Exit still uses intended_exit unless gap-through-stop replaced it.
        fill_exit = intended_exit
        for f in fill.fields:
            if f.name == "gap_through_stop" and f.value is not None:
                fill_exit = float(f.value)

        gross_pnl = (fill_exit - intended_entry) * intended_qty
        if side_u == "SELL":
            gross_pnl = (intended_entry - fill_exit) * intended_qty

        # Gross ALWAYS uses intended prices/qty — never overwritten by fill
        # quality. Adjusted uses filled qty and fill prices.
        gross = {
            "qty": intended_qty,
            "entry_price": intended_entry,
            "exit_price": intended_exit,
            "pnl": round(gross_pnl, 6),
            "source": "intended_prices",
            "notes": "raw gross evidence; never overwritten by execution reality",
        }

        if filled <= 0 or fill.fill_status in ("NO_FILL", "CIRCUIT"):
            adj_pnl = 0.0
            charges: list[ModeledField] = []
            buy_to = 0.0
            sell_to = 0.0
        else:
            buy_to = filled * fill_entry
            sell_to = filled * fill_exit
            charges = self._charge_fields(
                fill_entry=fill_entry,
                fill_exit=fill_exit,
                filled_qty=filled,
                measured=measured_charges,
                timestamp=ts,
            )
            total_charges = sum(float(c.value or 0.0) for c in charges if c.value is not None)
            adj_gross = (fill_exit - fill_entry) * filled
            if side_u == "SELL":
                adj_gross = (fill_entry - fill_exit) * filled
            adj_pnl = adj_gross - total_charges

        adjusted = {
            "qty": filled,
            "entry_price": fill_entry if filled else None,
            "exit_price": fill_exit if filled else None,
            "pnl": round(adj_pnl, 6),
            "buy_turnover": round(buy_to, 6),
            "sell_turnover": round(sell_to, 6),
            "charges_total": round(sum(float(c.value or 0.0) for c in charges if c.value is not None), 6),
            "shadow_mode": self.shadow_mode,
            "notes": "analytics only; does not change paper order fills",
        }

        return ExecutionRealityResult(
            engine_version=ENGINE_VERSION,
            schema_version=SCHEMA_VERSION,
            shadow_mode=self.shadow_mode,
            affects_paper_orders=False,
            gross_result=gross,
            execution_adjusted_result=adjusted,
            charges=charges,
            fill=fill,
            warnings=warnings,
        )

    def _fill_quality(
        self,
        *,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        bid: float | None,
        ask: float | None,
        slippage_bps: float | None,
        bar_volume: float | None,
        participation_cap: float,
        open_price: float | None,
        stop_price: float | None,
        circuit_hit: bool,
        delayed_fill: bool,
        no_fill: bool,
        timestamp: str,
        warnings: list[str],
    ) -> FillQuality:
        fields: list[ModeledField] = []
        filled = qty
        fill_price = entry_price
        status = "PERFECT"
        reason = "intended_price"

        if no_fill:
            return FillQuality(
                fill_status="NO_FILL",
                intended_qty=qty,
                filled_qty=0.0,
                intended_price=entry_price,
                fill_price=None,
                reason="NO_FILL",
                fields=[
                    _field(
                        "no_fill",
                        0.0,
                        unit="qty",
                        source="scenario",
                        formula="filled_qty=0",
                        assumptions="order did not fill (scenario or liquidity)",
                        confidence="high" if no_fill else "low",
                        measured=True,
                        timestamp=timestamp,
                    )
                ],
            )

        if circuit_hit:
            warnings.append("circuit_limit: fill blocked")
            return FillQuality(
                fill_status="CIRCUIT",
                intended_qty=qty,
                filled_qty=0.0,
                intended_price=entry_price,
                fill_price=None,
                reason="CIRCUIT_LIMIT",
                fields=[
                    _field(
                        "circuit_limit",
                        1.0,
                        unit="flag",
                        source="scenario",
                        formula="circuit_hit → no fill",
                        assumptions="India cash circuit; no invented auction fill",
                        confidence="high",
                        measured=True,
                        timestamp=timestamp,
                    )
                ],
            )

        # Liquidity / volume participation
        if bar_volume is not None:
            cap_qty = float(bar_volume) * float(participation_cap)
            fields.append(
                _field(
                    "volume_participation",
                    cap_qty,
                    unit="qty",
                    source="bar_volume",
                    formula="min(qty, bar_volume * participation_cap)",
                    assumptions=f"participation_cap={participation_cap}; no L2 book",
                    confidence="medium",
                    measured=True,
                    timestamp=timestamp,
                )
            )
            if cap_qty <= 0:
                warnings.append("illiquid: bar volume cannot support a fill")
                return FillQuality(
                    fill_status="NO_FILL",
                    intended_qty=qty,
                    filled_qty=0.0,
                    intended_price=entry_price,
                    fill_price=None,
                    reason="ILLIQUID",
                    fields=fields,
                )
            if qty > cap_qty:
                filled = cap_qty
                status = "PARTIAL"
                reason = "VOLUME_PARTICIPATION"
                fields.append(
                    _field(
                        "partial_fill",
                        filled,
                        unit="qty",
                        source="bar_volume",
                        formula="filled = bar_volume * participation_cap",
                        assumptions="no L2; conservative participation cap",
                        confidence="medium",
                        measured=True,
                        timestamp=timestamp,
                    )
                )
        else:
            fields.append(
                _missing(
                    "volume_participation",
                    reason="bar_volume not supplied; liquidity not invented",
                    unit="qty",
                )
            )

        # Spread: measured when bid/ask present, else estimated default
        spread_cost = 0.0
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2.0
            half = (float(ask) - float(bid)) / 2.0
            if side == "BUY":
                fill_price = float(ask)
            else:
                fill_price = float(bid)
            spread_cost = abs(fill_price - mid)
            fields.append(
                _field(
                    "bid_ask_spread",
                    spread_cost,
                    unit="inr_per_share",
                    source="quote",
                    formula="BUY→ask, SELL→bid vs mid",
                    assumptions="top of book only; no depth",
                    confidence="high",
                    measured=True,
                    timestamp=timestamp,
                )
            )
            if status == "PERFECT":
                status = "SPREAD"
                reason = "PAID_SPREAD"
        else:
            fields.append(
                _field(
                    "bid_ask_spread",
                    entry_price * DEFAULT_SPREAD_BPS / 1e4,
                    unit="inr_per_share",
                    source="default_estimate",
                    formula=f"price * {DEFAULT_SPREAD_BPS} bps",
                    assumptions="no bid/ask supplied; default half-spread estimate, labelled estimated",
                    confidence="low",
                    measured=False,
                    timestamp=timestamp,
                    notes="do not invent microstructure; this is a labelled estimate",
                )
            )
            # Default spread is recorded but NOT applied to fill_price unless
            # the caller asked for estimated spread application via slippage.
            warnings.append("spread estimated; fill_price left at intended unless slippage/quote provided")

        # Slippage (bps of intended price)
        if slippage_bps is not None:
            slip = entry_price * float(slippage_bps) / 1e4
            if side == "BUY":
                fill_price = fill_price + slip
            else:
                fill_price = fill_price - slip
            fields.append(
                _field(
                    "slippage",
                    slip,
                    unit="inr_per_share",
                    source="scenario" if slippage_bps else "none",
                    formula="price * slippage_bps / 1e4",
                    assumptions="adverse; applied on entry",
                    confidence="medium",
                    measured=True,
                    timestamp=timestamp,
                )
            )
            if status in ("PERFECT", "SPREAD"):
                status = "SLIPPAGE"
                reason = "SLIPPAGE"
        else:
            fields.append(
                _missing("slippage", reason="slippage_bps not supplied", unit="inr_per_share")
            )

        # Gap through stop (long): open below stop → exit at open, not stop
        if stop_price is not None and open_price is not None and side == "BUY":
            if float(open_price) < float(stop_price):
                fields.append(
                    _field(
                        "gap_through_stop",
                        float(open_price),
                        unit="inr",
                        source="ohlc_open",
                        formula="if open < stop: exit=open (gap-through)",
                        assumptions="daily bar; no overnight insurance",
                        confidence="high",
                        measured=True,
                        timestamp=timestamp,
                    )
                )
                status = "GAP_THROUGH_STOP"
                reason = "GAP_THROUGH_STOP"
            else:
                fields.append(
                    _field(
                        "gap_through_stop",
                        0.0,
                        unit="flag",
                        source="ohlc_open",
                        formula="open >= stop → no gap-through",
                        assumptions="daily bar",
                        confidence="high",
                        measured=True,
                        timestamp=timestamp,
                    )
                )
        elif stop_price is not None:
            fields.append(
                _missing("gap_through_stop", reason="open_price not supplied; gap not invented")
            )

        # Gap through entry: open already through intended limit
        if open_price is not None and side == "BUY" and float(open_price) > float(entry_price):
            fields.append(
                _field(
                    "gap_through_entry",
                    float(open_price),
                    unit="inr",
                    source="ohlc_open",
                    formula="open > intended entry → fill at open or no-fill policy",
                    assumptions="daily bar; modelled as worse fill, not a free skip",
                    confidence="high",
                    measured=True,
                    timestamp=timestamp,
                )
            )
            fill_price = float(open_price)
            if status == "PERFECT":
                status = "GAP_THROUGH_ENTRY"
                reason = "GAP_THROUGH_ENTRY"

        if delayed_fill:
            fields.append(
                _field(
                    "delayed_fill",
                    1.0,
                    unit="flag",
                    source="scenario",
                    formula="fill delayed; same price unless other effects",
                    assumptions="no tick tape; delay flagged, price not invented",
                    confidence="medium",
                    measured=True,
                    timestamp=timestamp,
                )
            )
            if status == "PERFECT":
                status = "DELAYED"
                reason = "DELAYED_FILL"

        return FillQuality(
            fill_status=status,
            intended_qty=qty,
            filled_qty=filled,
            intended_price=entry_price,
            fill_price=fill_price,
            reason=reason,
            fields=fields,
        )

    def _charge_fields(
        self,
        *,
        fill_entry: float,
        fill_exit: float,
        filled_qty: float,
        measured: Mapping[str, float] | None,
        timestamp: str,
    ) -> list[ModeledField]:
        """Zerodha CNC breakdown — one canonical India cash formula."""
        qty_int = int(round(float(filled_qty)))
        z = zerodha_charges(float(fill_entry), float(fill_exit), qty_int, "CNC")
        measured = dict(measured or {})
        mapping = [
            ("brokerage", "brokerage", "Zerodha CNC equity delivery = ₹0", "delivery brokerage is free; MIS would be ₹20/leg"),
            ("stt", "stt", "STT 0.1% of buy+sell turnover (CNC)", "both sides for delivery, from execution.cost_model"),
            ("exchange", "txn", "NSE txn ~0.00297% of turnover", "buy+sell"),
            ("sebi", "sebi", "SEBI ₹10 / crore of turnover", "buy+sell"),
            ("stamp_duty", "stamp", "Stamp 0.015% of buy turnover", "buy-side cash equity"),
            ("gst", "gst", "GST 18% on (brokerage + exch + sebi + DP)", "not on STT/stamp"),
            ("dp_charges", "dp", "DP charge on sell (CNC)", "per-scrip sell-side depository"),
        ]
        out: list[ModeledField] = []
        for name, zkey, formula, assumptions in mapping:
            if name in measured:
                out.append(
                    _field(
                        name,
                        float(measured[name]),
                        unit="inr",
                        source="broker_contract_note",
                        formula=formula,
                        assumptions=assumptions,
                        confidence="high",
                        measured=True,
                        timestamp=timestamp,
                    )
                )
            else:
                out.append(
                    _field(
                        name,
                        float(z.get(zkey, 0.0)),
                        unit="inr",
                        source="execution.cost_model.zerodha_charges",
                        formula=formula,
                        assumptions=assumptions + "; estimated from public Zerodha CNC schedule",
                        confidence="high",
                        measured=False,
                        timestamp=timestamp,
                    )
                )
        return out


def annotate_shadow(trade: Mapping[str, Any], result: ExecutionRealityResult) -> dict[str, Any]:
    """Attach analytics without mutating fill/qty on the trade record."""
    out = dict(trade)
    out["execution_reality"] = result.to_dict()
    out["gross_pnl_preserved"] = result.gross_result.get("pnl")
    return out


def shadow_for_paper_fill(
    *,
    qty: float | None,
    entry: float | None,
    target: float | None,
    stop: float | None = None,
) -> dict[str, Any] | None:
    """Hypothetical round-trip analytics for an already-filled paper ticket.

    Does not resize, reprice, or reject the fill. Missing legs stay missing.
    """
    try:
        q = float(qty or 0)
        e = float(entry or 0)
        x = float(target or 0)
    except (TypeError, ValueError):
        return None
    if q <= 0 or e <= 0 or x <= 0:
        return None
    stop_px = None
    try:
        if stop is not None and float(stop) > 0:
            stop_px = float(stop)
    except (TypeError, ValueError):
        stop_px = None
    result = ExecutionRealityEngine(shadow_mode=True).analyze_round_trip(
        qty=q,
        entry_price=e,
        exit_price=x,
        stop_price=stop_px,
    )
    return result.to_dict()
