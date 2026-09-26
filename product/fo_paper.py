"""Lot-aware, long-premium paper ledger for NSE options.

This ledger is intentionally independent from the cash-equity PaperBook.
It models long CE/PE only, never writes options, never calls a broker, and
labels whether statutory/broker costs were configured for evidence use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from typing import Any, Callable, Mapping
from uuid import uuid4


CostModel = Callable[[float, float, int], float]


@dataclass
class FoPaperPosition:
    trade_id: str
    underlying: str
    option_symbol: str
    option_type: str
    context_key: str
    entry_price: float
    stop_price: float
    target_price: float
    lot_size: int
    lots: int
    quantity: int
    opened_at: str
    max_holding_sessions: int
    risk_amount: float
    setup_score: float
    option_score: float
    bars_held: int = 0
    last_mark_session: str = ""
    max_mark: float = 0.0
    min_mark: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FoPaperTrade:
    trade_id: str
    underlying: str
    option_symbol: str
    option_type: str
    context_key: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    quantity: int
    opened_at: str
    settled_at: str
    exit_reason: str
    gross_pnl: float
    costs: float
    net_pnl: float
    net_option_return_pct: float
    mfe_pct: float
    mae_pct: float
    evidence_lane: str = "FORWARD_PAPER"
    settled: bool = True
    cost_model_status: str = "UNCONFIGURED"
    false_breakout: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FoPaperBook:
    """Long-premium options paper book with conservative same-bar exits."""

    def __init__(
        self,
        capital: float = 100_000.0,
        *,
        risk_per_trade_pct: float = 0.01,
        max_premium_pct: float = 0.10,
        max_positions: int = 5,
        max_total_risk_pct: float = 0.05,
        slippage_bps: float = 5.0,
        cost_model: CostModel | None = None,
        cost_model_name: str = "",
    ) -> None:
        self.capital = float(capital)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.max_premium_pct = float(max_premium_pct)
        self.max_positions = int(max_positions)
        self.max_total_risk_pct = float(max_total_risk_pct)
        self.slippage_bps = float(slippage_bps)
        self.cost_model = cost_model
        self.cost_model_name = str(cost_model_name or "")
        self.open: dict[str, FoPaperPosition] = {}
        self.closed: list[FoPaperTrade] = []
        self.refusals: list[tuple[str, str]] = []
        self.realized_pnl = 0.0

    @property
    def fully_costed(self) -> bool:
        return self.cost_model is not None and bool(self.cost_model_name)

    def _entry_fill(self, price: float, ask: float | None) -> float:
        observed = float(ask or 0.0)
        if observed > 0:
            return observed
        return float(price) * (1.0 + self.slippage_bps / 1e4)

    def _exit_fill(self, price: float, bid: float | None) -> float:
        observed = float(bid or 0.0)
        if observed > 0:
            return observed
        return float(price) * (1.0 - self.slippage_bps / 1e4)

    def open_position(
        self,
        *,
        underlying: str,
        option_symbol: str,
        option_type: str,
        entry: float,
        stop: float,
        target: float,
        lot_size: int,
        opened_at: str,
        max_holding_sessions: int,
        context_key: str = "",
        setup_score: float = 0.0,
        option_score: float = 0.0,
        ask: float | None = None,
        requested_lots: int | None = None,
    ) -> FoPaperPosition | None:
        symbol = str(option_symbol or "").strip()
        kind = str(option_type or "").upper()
        if kind not in {"CE", "PE"}:
            self.refusals.append((symbol, "LONG_PREMIUM_CE_PE_ONLY"))
            return None
        if symbol in self.open:
            self.refusals.append((symbol, "ALREADY_OPEN"))
            return None
        clean_underlying = str(underlying or "").upper()
        if any(pos.underlying == clean_underlying for pos in self.open.values()):
            self.refusals.append((symbol, "UNDERLYING_ALREADY_OPEN"))
            return None
        if len(self.open) >= self.max_positions:
            self.refusals.append((symbol, "MAX_POSITIONS"))
            return None
        entry = float(entry)
        stop = float(stop)
        target = float(target)
        lot_size = int(lot_size)
        if entry <= 0 or stop <= 0 or stop >= entry or target <= entry or lot_size <= 0:
            self.refusals.append((symbol, "INVALID_TRADE_PLAN"))
            return None

        fill = self._entry_fill(entry, ask)
        per_unit_risk = fill - stop
        if per_unit_risk <= 0:
            self.refusals.append((symbol, "STOP_NOT_BELOW_EXECUTABLE_ENTRY"))
            return None
        equity = max(0.0, self.capital + self.realized_pnl)
        risk_budget = equity * self.risk_per_trade_pct
        premium_budget = equity * self.max_premium_pct
        total_risk_budget = equity * self.max_total_risk_pct
        open_risk = sum(float(pos.risk_amount) for pos in self.open.values())
        remaining_total_risk = max(0.0, total_risk_budget - open_risk)
        risk_budget = min(risk_budget, remaining_total_risk)
        risk_per_lot = per_unit_risk * lot_size
        premium_per_lot = fill * lot_size
        lots_by_risk = floor(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
        lots_by_premium = floor(premium_budget / premium_per_lot) if premium_per_lot > 0 else 0
        lots = min(lots_by_risk, lots_by_premium)
        if requested_lots is not None:
            lots = min(lots, max(0, int(requested_lots)))
        if lots < 1:
            self.refusals.append((symbol, "RISK_OR_PREMIUM_BUDGET_TOO_SMALL_FOR_ONE_LOT"))
            return None

        qty = lots * lot_size
        pos = FoPaperPosition(
            trade_id=uuid4().hex,
            underlying=clean_underlying,
            option_symbol=symbol,
            option_type=kind,
            context_key=str(context_key or ""),
            entry_price=round(fill, 4),
            stop_price=round(stop, 4),
            target_price=round(target, 4),
            lot_size=lot_size,
            lots=lots,
            quantity=qty,
            opened_at=str(opened_at),
            max_holding_sessions=max(1, int(max_holding_sessions)),
            risk_amount=round(per_unit_risk * qty, 2),
            setup_score=float(setup_score),
            option_score=float(option_score),
            last_mark_session=str(opened_at)[:10],
            max_mark=round(fill, 4),
            min_mark=round(fill, 4),
        )
        self.open[symbol] = pos
        return pos

    def mark(
        self,
        quotes: Mapping[str, Mapping[str, Any]],
        *,
        session: str,
    ) -> list[FoPaperTrade]:
        """Mark one session. When stop and target both hit, STOP wins conservatively."""
        settled: list[FoPaperTrade] = []
        for symbol, pos in list(self.open.items()):
            raw = quotes.get(symbol)
            if not isinstance(raw, Mapping):
                continue
            close = float(raw.get("close") or raw.get("last_price") or 0.0)
            if close <= 0:
                continue
            open_px = float(raw.get("open") or close)
            high = float(raw.get("high") or close)
            low = float(raw.get("low") or close)
            bid = float(raw.get("bid") or 0.0)
            clean_session = str(session)[:10]
            if clean_session and clean_session != pos.last_mark_session:
                pos.bars_held += 1
                pos.last_mark_session = clean_session
            pos.max_mark = max(pos.max_mark, high, close)
            pos.min_mark = min(pos.min_mark, low, close)

            exit_price: float | None = None
            reason = ""
            if open_px <= pos.stop_price:
                exit_price, reason = open_px, "GAP_STOP"
            elif open_px >= pos.target_price:
                exit_price, reason = open_px, "GAP_TARGET"
            elif low <= pos.stop_price and high >= pos.target_price:
                exit_price, reason = pos.stop_price, "AMBIGUOUS_BAR_STOP_FIRST"
            elif low <= pos.stop_price:
                exit_price, reason = pos.stop_price, "STOP"
            elif high >= pos.target_price:
                exit_price, reason = pos.target_price, "TARGET"
            elif pos.bars_held >= pos.max_holding_sessions:
                exit_price, reason = close, "MAX_HOLD"

            if exit_price is not None:
                # A closing quote bid is valid for a current/mark-to-market MAX_HOLD exit.
                # For historical STOP/TARGET/GAP triggers it is from the end-of-bar snapshot,
                # not the instant the trigger fired; using it would introduce impossible fills.
                execution_bid = bid if reason == "MAX_HOLD" else None
                settled.append(self._close(pos, exit_price, reason, str(session), bid=execution_bid))
        return settled

    def _close(
        self,
        pos: FoPaperPosition,
        exit_price: float,
        reason: str,
        settled_at: str,
        *,
        bid: float | None = None,
    ) -> FoPaperTrade:
        fill = self._exit_fill(exit_price, bid)
        gross = (fill - pos.entry_price) * pos.quantity
        costs = 0.0
        if self.cost_model is not None:
            costs = max(0.0, float(self.cost_model(pos.entry_price, fill, pos.quantity)))
        net = gross - costs
        entry_notional = pos.entry_price * pos.quantity
        ret = net / entry_notional * 100.0 if entry_notional > 0 else 0.0
        mfe = (pos.max_mark - pos.entry_price) / pos.entry_price * 100.0
        mae = (pos.min_mark - pos.entry_price) / pos.entry_price * 100.0
        trade = FoPaperTrade(
            trade_id=pos.trade_id,
            underlying=pos.underlying,
            option_symbol=pos.option_symbol,
            option_type=pos.option_type,
            context_key=pos.context_key,
            entry_price=pos.entry_price,
            exit_price=round(fill, 4),
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            quantity=pos.quantity,
            opened_at=pos.opened_at,
            settled_at=settled_at,
            exit_reason=reason,
            gross_pnl=round(gross, 2),
            costs=round(costs, 2),
            net_pnl=round(net, 2),
            net_option_return_pct=round(ret, 4),
            mfe_pct=round(mfe, 4),
            mae_pct=round(mae, 4),
            cost_model_status=(
                f"CONFIGURED:{self.cost_model_name}" if self.fully_costed
                else "UNCONFIGURED_GROSS_ONLY"
            ),
        )
        self.realized_pnl += net
        self.closed.append(trade)
        del self.open[pos.option_symbol]
        return trade

    def evidence_rows(self) -> list[dict[str, Any]]:
        """Only fully-costed closed trades are eligible for net production evidence."""
        rows = []
        for trade in self.closed:
            row = trade.as_dict()
            row["production_evidence_eligible"] = self.fully_costed
            rows.append(row)
        return rows
