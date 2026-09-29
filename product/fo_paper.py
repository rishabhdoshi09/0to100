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

ENTRY_MINUTE_PENDING = "PENDING"
ENTRY_MINUTE_CLEAR = "CLEAR_NO_TRIGGER"
ENTRY_MINUTE_FULLY_OBSERVED = "FULLY_OBSERVED"
ENTRY_MINUTE_AMBIGUOUS = "AMBIGUOUS_BOUNDARY_TOUCH"
ENTRY_MINUTE_UNAVAILABLE = "UNAVAILABLE"
ENTRY_MINUTE_EVIDENCE_OK = frozenset({
    ENTRY_MINUTE_CLEAR,
    ENTRY_MINUTE_FULLY_OBSERVED,
})


def entry_minute_path_complete(status: str) -> bool:
    return str(status or "").upper() in ENTRY_MINUTE_EVIDENCE_OK


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
    instrument_token: int = 0
    horizon: str = ""
    exit_policy: str = "SESSION_HOLD"
    strike: float = 0.0
    expiry: str = ""
    entry_iv_pct: float = 0.0
    entry_underlying_spot: float = 0.0
    trailing_stop_price: float = 0.0
    entry_minute_status: str = ENTRY_MINUTE_PENDING
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
    entry_minute_status: str = ENTRY_MINUTE_PENDING
    exit_observation_complete: bool = True
    path_observation_complete: bool = False
    path_observation_reason: str = ""
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
        max_daily_premium_pct: float | None = None,
        premium_deployed_today: float = 0.0,
        max_positions: int = 5,
        max_total_risk_pct: float = 0.05,
        slippage_bps: float = 5.0,
        trail_activation_r: float = 1.0,
        trail_distance_r: float = 1.0,
        cost_model: CostModel | None = None,
        cost_model_name: str = "",
    ) -> None:
        self.capital = float(capital)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.max_premium_pct = float(max_premium_pct)
        self.max_daily_premium_pct = float(
            max_premium_pct if max_daily_premium_pct is None else max_daily_premium_pct
        )
        self.premium_deployed_today = max(0.0, float(premium_deployed_today))
        self.max_positions = int(max_positions)
        self.max_total_risk_pct = float(max_total_risk_pct)
        self.slippage_bps = float(slippage_bps)
        self.trail_activation_r = max(0.0, float(trail_activation_r))
        self.trail_distance_r = max(0.0, float(trail_distance_r))
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
        instrument_token: int = 0,
        horizon: str = "",
        exit_policy: str = "SESSION_HOLD",
        strike: float = 0.0,
        expiry: str = "",
        entry_iv_pct: float = 0.0,
        entry_underlying_spot: float = 0.0,
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
        if target <= fill:
            self.refusals.append((symbol, "TARGET_NOT_ABOVE_EXECUTABLE_ENTRY"))
            return None
        per_unit_risk = fill - stop
        if per_unit_risk <= 0:
            self.refusals.append((symbol, "STOP_NOT_BELOW_EXECUTABLE_ENTRY"))
            return None
        equity = max(0.0, self.capital + self.realized_pnl)
        risk_budget = equity * self.risk_per_trade_pct
        per_trade_premium_budget = equity * self.max_premium_pct
        daily_premium_budget = equity * self.max_daily_premium_pct
        remaining_daily_premium = max(
            0.0,
            daily_premium_budget - self.premium_deployed_today,
        )
        premium_budget = min(per_trade_premium_budget, remaining_daily_premium)
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
            if premium_per_lot > 0 and remaining_daily_premium < premium_per_lot:
                self.refusals.append((symbol, "DAILY_PREMIUM_BUDGET_EXHAUSTED"))
            else:
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
            instrument_token=max(0, int(instrument_token or 0)),
            horizon=str(horizon or "").upper(),
            exit_policy=str(exit_policy or "SESSION_HOLD").upper(),
            strike=max(0.0, float(strike or 0.0)),
            expiry=str(expiry or "")[:10],
            entry_iv_pct=max(0.0, float(entry_iv_pct or 0.0)),
            entry_underlying_spot=max(0.0, float(entry_underlying_spot or 0.0)),
            trailing_stop_price=round(stop, 4),
            entry_minute_status=ENTRY_MINUTE_PENDING,
            last_mark_session=str(opened_at)[:10],
            max_mark=round(fill, 4),
            min_mark=round(fill, 4),
        )
        self.open[symbol] = pos
        self.premium_deployed_today = round(
            self.premium_deployed_today + fill * qty,
            4,
        )
        return pos

    def mark(
        self,
        quotes: Mapping[str, Mapping[str, Any]],
        *,
        session: str,
        advance_session: bool = True,
        observation_complete: bool = True,
        force_eod: bool = False,
        iv_crush_symbols: set[str] | frozenset[str] | None = None,
    ) -> list[FoPaperTrade]:
        """Mark prices; optionally advance holding-session age.

        Historical intraday replay uses advance_session=False so chronological
        STOP/TARGET reconstruction cannot accidentally trigger MAX_HOLD before
        the current supervision instant. Same-bar ambiguity remains STOP-first.
        """
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
            if advance_session and clean_session and clean_session != pos.last_mark_session:
                pos.bars_held += 1
                pos.last_mark_session = clean_session

            # Trail from prior observed path only. The current bar's high cannot
            # tighten a stop that the current bar's low may already have crossed.
            prior_max = max(float(pos.max_mark or 0.0), pos.entry_price)
            risk_unit = max(0.0, pos.entry_price - pos.stop_price)
            effective_stop = max(pos.stop_price, float(pos.trailing_stop_price or 0.0))
            if (
                risk_unit > 0
                and prior_max >= pos.entry_price + self.trail_activation_r * risk_unit
            ):
                effective_stop = max(
                    effective_stop,
                    pos.entry_price,
                    prior_max - self.trail_distance_r * risk_unit,
                )
            trail_active = effective_stop > pos.stop_price + 1e-9

            exit_price: float | None = None
            reason = ""
            if open_px <= effective_stop:
                exit_price = open_px
                reason = "GAP_TRAIL_STOP" if trail_active else "GAP_STOP"
            elif open_px >= pos.target_price:
                exit_price, reason = open_px, "GAP_TARGET"
            elif low <= effective_stop and high >= pos.target_price:
                exit_price = effective_stop
                reason = (
                    "AMBIGUOUS_BAR_TRAIL_STOP_FIRST"
                    if trail_active else "AMBIGUOUS_BAR_STOP_FIRST"
                )
            elif low <= effective_stop:
                exit_price = effective_stop
                reason = "TRAIL_STOP" if trail_active else "STOP"
            elif high >= pos.target_price:
                exit_price, reason = pos.target_price, "TARGET"
            elif symbol in (iv_crush_symbols or ()):
                exit_price, reason = close, "IV_CRUSH"
            elif bool(force_eod) and str(pos.exit_policy or "").upper() == "EOD":
                exit_price, reason = close, "EOD"
            elif pos.bars_held >= pos.max_holding_sessions:
                exit_price, reason = close, "MAX_HOLD"

            # Excursion evidence must stop at the exit boundary. Do not credit a
            # later/unknown-order bar high as MFE after a stop, or a later bar
            # extension above a target after the position has already exited.
            if exit_price is None:
                pos.max_mark = max(pos.max_mark, high, close)
                pos.min_mark = min(pos.min_mark, low, close)
            elif reason in {
                "GAP_STOP", "GAP_TRAIL_STOP", "STOP", "TRAIL_STOP",
                "AMBIGUOUS_BAR_STOP_FIRST", "AMBIGUOUS_BAR_TRAIL_STOP_FIRST",
            }:
                pos.max_mark = max(pos.max_mark, open_px)
                pos.min_mark = min(pos.min_mark, float(exit_price), open_px)
            elif reason in {"GAP_TARGET", "TARGET"}:
                pos.max_mark = max(pos.max_mark, float(exit_price), open_px)
                pos.min_mark = min(pos.min_mark, low, open_px)
            else:
                # MAX_HOLD / EOD / IV_CRUSH occur after the observed interval
                # survived without stop/target, so its range is legitimate.
                pos.max_mark = max(pos.max_mark, high, close)
                pos.min_mark = min(pos.min_mark, low, close)

            if exit_price is None:
                if risk_unit > 0 and pos.max_mark >= pos.entry_price + self.trail_activation_r * risk_unit:
                    pos.trailing_stop_price = round(max(
                        pos.stop_price,
                        pos.entry_price,
                        pos.max_mark - self.trail_distance_r * risk_unit,
                    ), 4)
                else:
                    pos.trailing_stop_price = round(max(
                        pos.stop_price,
                        float(pos.trailing_stop_price or 0.0),
                    ), 4)

            if exit_price is not None:
                # Current policy exits use observed bid when available. Historical
                # STOP/TARGET/TRAIL triggers remain bar-priced to avoid using a
                # later quote for an earlier event.
                execution_bid = bid if reason in {"MAX_HOLD", "EOD", "IV_CRUSH"} else None
                settled.append(
                    self._close(
                        pos,
                        exit_price,
                        reason,
                        str(session),
                        bid=execution_bid,
                        observation_complete=observation_complete,
                    )
                )
        return settled

    def _close(
        self,
        pos: FoPaperPosition,
        exit_price: float,
        reason: str,
        settled_at: str,
        *,
        bid: float | None = None,
        observation_complete: bool = True,
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
        entry_complete = entry_minute_path_complete(pos.entry_minute_status)
        path_complete = bool(entry_complete and observation_complete)
        if not entry_complete:
            path_reason = f"ENTRY_MINUTE_{pos.entry_minute_status or ENTRY_MINUTE_PENDING}"
        elif not observation_complete:
            path_reason = "EXIT_PARTIAL_INTERVAL_UNOBSERVED"
        else:
            path_reason = ""

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
            entry_minute_status=str(pos.entry_minute_status or ENTRY_MINUTE_PENDING),
            exit_observation_complete=bool(observation_complete),
            path_observation_complete=path_complete,
            path_observation_reason=path_reason,
        )
        self.realized_pnl += net
        self.closed.append(trade)
        del self.open[pos.option_symbol]
        return trade

    def evidence_rows(self) -> list[dict[str, Any]]:
        """Only fully-costed, path-observable trades are production evidence."""
        rows = []
        for trade in self.closed:
            row = trade.as_dict()
            row["production_evidence_eligible"] = bool(
                self.fully_costed and trade.path_observation_complete
            )
            if not self.fully_costed:
                row["evidence_exclusion_reason"] = "COST_MODEL_UNCONFIGURED"
            elif not trade.path_observation_complete:
                row["evidence_exclusion_reason"] = (
                    trade.path_observation_reason
                    or f"ENTRY_MINUTE_{trade.entry_minute_status or ENTRY_MINUTE_PENDING}"
                )
            rows.append(row)
        return rows
