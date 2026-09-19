"""Paper-only simulated ledger with durable risk/execution semantics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from research.intelligence.runtime.position_sizing import size_long_cash


@dataclass
class PaperPosition:
    strategy_id: str
    symbol: str
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    entry_date: str
    max_holding_days: int
    risk_amount: float
    requested_risk_pct: float = 0.0
    approved_risk_pct: float = 0.0
    bars_held: int = 0
    last_marked_session: str = ""
    decision_id: str = ""
    paper_intent_id: str = ""
    context_key: str = ""
    sector: str = ""

    @property
    def r_unit(self) -> float:
        return max(1e-9, self.entry_price - self.stop_price)

    def as_dict(self):
        return asdict(self)


@dataclass
class ClosedTrade:
    strategy_id: str
    symbol: str
    entry_price: float
    exit_price: float
    stop_price: float
    qty: int
    entry_date: str
    exit_date: str
    exit_reason: str
    realized_R: float
    pnl: float
    decision_id: str = ""
    paper_intent_id: str = ""
    context_key: str = ""

    def as_dict(self):
        return asdict(self)


def _cost_model_name(model) -> str:
    if model is None:
        return ""
    module = str(getattr(model, "__module__", "") or "")
    name = str(getattr(model, "__name__", "") or "")
    if module == "research.auto_research.costs" and name == "india_cash_costs":
        return "india_cash_costs"
    return ""


def _resolve_cost_model(name: str):
    if not name:
        return None
    if name == "india_cash_costs":
        from research.auto_research.costs import india_cash_costs
        return india_cash_costs
    raise ValueError("unsupported paper cost model")


def _validated_risk_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("risk_config must be an object")
    risk = float(raw["risk_per_trade_pct"])
    pos = float(raw["max_position_pct"])
    total = float(raw["max_total_risk_pct"])
    maximum = int(raw["max_positions"])
    slip = float(raw["slippage_bps"])
    if not all(isfinite(v) for v in (risk, pos, total, slip)):
        raise ValueError("non-finite risk contract")
    if not (0 < risk <= 1 and 0 < pos <= 1 and 0 < total <= 1):
        raise ValueError("invalid risk fractions")
    if maximum <= 0 or slip < 0:
        raise ValueError("invalid execution contract")
    model_name = str(raw.get("cost_model") or "")
    model = _resolve_cost_model(model_name)
    return {
        "risk_per_trade_pct": risk,
        "max_position_pct": pos,
        "max_total_risk_pct": total,
        "max_positions": maximum,
        "slippage_bps": slip,
        "cost_model": model,
        "cost_model_name": model_name,
    }


class PaperBook:
    """Simulated long-only cash-equity book. No live-order dependency exists here."""

    def __init__(self, capital: float = 100_000.0, *, risk_per_trade_pct: float = 0.01,
                 max_position_pct: float = 0.10, max_total_risk_pct: float = 0.05,
                 max_positions: int = 5, slippage_bps: float = 0.0, cost_model=None):
        self.capital = float(capital)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.max_positions = max_positions
        self.slippage_bps = float(slippage_bps)
        self.cost_model = cost_model
        self.open: dict[tuple, PaperPosition] = {}
        self.closed: list[ClosedTrade] = []
        self.realized_pnl = 0.0
        self.equity_curve: list[float] = [self.capital]
        self.refusals: list[tuple] = []

    def open_position(self, strategy_id: str, symbol: str, entry: float, stop: float,
                      target: float, date: str, max_holding_days: int, *,
                      risk_pct_of_capital: float | None = None,
                      quantity: int | None = None, decision_id: str = "",
                      paper_intent_id: str = "", context_key: str = "") -> PaperPosition | None:
        key = (strategy_id, symbol)
        if key in self.open:
            self.refusals.append((symbol, "already open for this strategy")); return None
        if len(self.open) >= self.max_positions:
            self.refusals.append((symbol, "max open positions reached")); return None
        requested = self.risk_per_trade_pct * 100.0 if risk_pct_of_capital is None else risk_pct_of_capital
        sizing = size_long_cash(
            capital=self.capital, entry=entry, stop=stop, requested_risk_pct=requested,
            max_risk_fraction=self.risk_per_trade_pct, max_position_fraction=self.max_position_pct,
            slippage_bps=self.slippage_bps, requested_quantity=quantity,
        )
        if not sizing.ok:
            self.refusals.append((symbol, _sizing_refusal(sizing.reason_code))); return None
        if self.open_risk() + sizing.risk_amount > self.capital * self.max_total_risk_pct + 1e-6:
            self.refusals.append((symbol, "total open risk cap (5%) reached")); return None
        pos = PaperPosition(
            strategy_id=strategy_id, symbol=symbol, entry_price=sizing.effective_entry,
            stop_price=float(stop), target_price=float(target), qty=sizing.quantity,
            entry_date=date, max_holding_days=max_holding_days, risk_amount=sizing.risk_amount,
            requested_risk_pct=sizing.requested_risk_pct, approved_risk_pct=sizing.actual_risk_pct,
            decision_id=str(decision_id or ""), paper_intent_id=str(paper_intent_id or ""),
            context_key=str(context_key or ""),
        )
        self.open[key] = pos
        return pos

    def open_intent(self, intent, *, date: str) -> PaperPosition | None:
        q = int(getattr(intent, "required_quantity", 0) or 0)
        return self.open_position(
            intent.strategy_id, intent.symbol, float(intent.intended_entry), float(intent.stop_price),
            float(intent.target_price), date, int(intent.holding_horizon_days),
            risk_pct_of_capital=float(intent.intended_risk_pct), quantity=q if q > 0 else None,
            decision_id=str(getattr(intent, "decision_id", "") or ""),
            paper_intent_id=str(getattr(intent, "record_id", "") or ""),
            context_key=str(getattr(intent, "context_key", "") or ""),
        )

    def open_risk(self) -> float:
        return sum(p.qty * p.r_unit for p in self.open.values())

    def mark(self, bars: dict, date: str, *, allow_entry_session: bool = True) -> list[ClosedTrade]:
        closed_now = []
        session = str(date or "")[:10]
        marked_any = False
        for key, pos in list(self.open.items()):
            if not session or (not allow_entry_session and str(pos.entry_date or "")[:10] >= session):
                continue
            if str(pos.last_marked_session or "")[:10] == session:
                continue
            bar = bars.get(pos.symbol)
            if bar is None:
                continue
            if len(bar) >= 4:
                op, high, low, close = map(float, bar[:4])
            else:
                op = None
                high, low, close = map(float, bar[:3])
            pos.last_marked_session = session
            pos.bars_held += 1
            marked_any = True
            exit_price = exit_reason = None
            if op is not None and op <= pos.stop_price:
                exit_price, exit_reason = op, "GAP_STOP"
            elif op is not None and op >= pos.target_price:
                exit_price, exit_reason = op, "GAP_TARGET"
            elif low <= pos.stop_price:
                exit_price, exit_reason = pos.stop_price, "STOP"
            elif high >= pos.target_price:
                exit_price, exit_reason = pos.target_price, "TARGET"
            elif pos.bars_held >= pos.max_holding_days:
                exit_price, exit_reason = close, "MAX_HOLD"
            if exit_price is not None:
                closed_now.append(self._close(key, pos, exit_price, exit_reason, session))
        if marked_any:
            self.equity_curve.append(self.equity(bars))
        return closed_now

    def _close(self, key, pos: PaperPosition, exit_price: float, reason: str, date: str) -> ClosedTrade:
        exit_fill = exit_price * (1.0 - self.slippage_bps / 1e4)
        gross = (exit_fill - pos.entry_price) * pos.qty
        cost = 0.0
        if self.cost_model is not None:
            try:
                cost = float(self.cost_model(pos.entry_price, exit_fill, pos.qty))
            except Exception:
                cost = 0.0
        pnl = gross - cost
        realized_R = pnl / (pos.qty * pos.r_unit)
        self.realized_pnl += pnl
        trade = ClosedTrade(
            strategy_id=pos.strategy_id, symbol=pos.symbol, entry_price=pos.entry_price,
            exit_price=exit_fill, stop_price=pos.stop_price, qty=pos.qty, entry_date=pos.entry_date,
            exit_date=date, exit_reason=reason, realized_R=realized_R, pnl=pnl,
            decision_id=pos.decision_id or "", paper_intent_id=pos.paper_intent_id or "",
            context_key=pos.context_key or "",
        )
        self.closed.append(trade)
        del self.open[key]
        return trade

    def equity(self, bars: dict | None = None) -> float:
        eq = self.capital + self.realized_pnl
        if bars:
            for pos in self.open.values():
                bar = bars.get(pos.symbol)
                if bar is not None:
                    close = float(bar[3]) if len(bar) >= 4 else float(bar[2])
                    eq += (close - pos.entry_price) * pos.qty
        return eq

    def stats(self, strategy_id: str | None = None) -> dict:
        trades = [t for t in self.closed if strategy_id is None or t.strategy_id == strategy_id]
        n = len(trades)
        wins = [t for t in trades if t.realized_R > 0]
        losses = [t for t in trades if t.realized_R <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        mean_r = sum(t.realized_R for t in trades) / n if n else 0.0
        pf = gross_win / gross_loss if gross_loss > 1e-9 else (float("inf") if gross_win else 0.0)
        return {"n_trades": n, "win_rate": round(len(wins) / n, 4) if n else 0.0,
                "expectancy_R": round(mean_r, 4), "profit_factor": round(pf, 3) if pf != float("inf") else None,
                "net_pnl": round(sum(t.pnl for t in trades), 2), "max_drawdown_pct": round(self._max_dd(), 4),
                "equity": round(self.equity(), 2),
                "open_positions": len([p for p in self.open.values() if strategy_id is None or p.strategy_id == strategy_id])}

    def r_stats(self, strategy_id: str | None = None) -> dict:
        rs = [t.realized_R for t in self.closed if strategy_id is None or t.strategy_id == strategy_id]
        n = len(rs)
        if not n:
            return {"n": 0, "mean_R": 0.0, "stderr_R": 0.0, "lower_R": 0.0}
        mean = sum(rs) / n
        se = 0.0
        if n > 1:
            var = sum((r - mean) ** 2 for r in rs) / (n - 1)
            se = (var ** 0.5) / (n ** 0.5)
        return {"n": n, "mean_R": round(mean, 4), "stderr_R": round(se, 4), "lower_R": round(mean - se, 4)}

    def _max_dd(self) -> float:
        peak = self.equity_curve[0] if self.equity_curve else self.capital
        mdd = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            if peak > 0:
                mdd = max(mdd, (peak - value) / peak)
        return mdd

    def _risk_config(self) -> dict:
        return {"risk_per_trade_pct": float(self.risk_per_trade_pct),
                "max_position_pct": float(self.max_position_pct),
                "max_total_risk_pct": float(self.max_total_risk_pct),
                "max_positions": int(self.max_positions), "slippage_bps": float(self.slippage_bps),
                "cost_model": _cost_model_name(self.cost_model)}

    def as_dict(self) -> dict:
        return {"capital": self.capital, "realized_pnl": round(self.realized_pnl, 2),
                "equity": round(self.equity(), 2), "n_closed": len(self.closed), "n_open": len(self.open),
                "stats": self.stats(), "equity_curve": [round(v, 2) for v in self.equity_curve[-120:]],
                "risk_config": self._risk_config()}

    def snapshot(self) -> dict:
        return {"capital": self.capital, "realized_pnl": self.realized_pnl,
                "equity_curve": self.equity_curve, "closed": [t.as_dict() for t in self.closed],
                "open": [p.as_dict() for p in self.open.values()], "risk_config": self._risk_config()}

    def restore(self, snap: dict) -> None:
        """Restore atomically. Corrupt/new risk contracts leave the current book untouched."""
        try:
            capital = float(snap.get("capital", self.capital))
            realized = float(snap.get("realized_pnl", 0.0))
            curve = list(snap.get("equity_curve", [capital])) or [capital]
            closed = [ClosedTrade(**t) for t in snap.get("closed", [])]
            opened = {}
            for raw in snap.get("open", []):
                pos = PaperPosition(**raw)
                opened[(pos.strategy_id, pos.symbol)] = pos
            risk = _validated_risk_config(snap["risk_config"]) if "risk_config" in snap else None
        except Exception:
            return
        self.capital = capital
        self.realized_pnl = realized
        self.equity_curve = curve
        self.closed = closed
        self.open = opened
        if risk is not None:
            self.risk_per_trade_pct = risk["risk_per_trade_pct"]
            self.max_position_pct = risk["max_position_pct"]
            self.max_total_risk_pct = risk["max_total_risk_pct"]
            self.max_positions = risk["max_positions"]
            self.slippage_bps = risk["slippage_bps"]
            self.cost_model = risk["cost_model"]


def _sizing_refusal(reason_code: str) -> str:
    return {"INVALID_ENTRY_STOP": "invalid entry/stop (need entry>stop>0)",
            "NON_POSITIVE_RISK": "approved risk percentage must be positive",
            "RISK_BUDGET_TOO_SMALL": "risk unit too wide for approved sizing",
            "POSITION_CAP_TOO_SMALL": "price too high for 10% cap",
            "INVALID_REQUESTED_QUANTITY": "invalid target portfolio quantity",
            "NON_POSITIVE_QUANTITY": "target portfolio quantity must be positive",
            "QUANTITY_EXCEEDS_APPROVED_LIMIT": "target portfolio quantity exceeds house limits",
            "INVALID_NUMERIC_INPUT": "invalid approved risk percentage",
            "NON_FINITE_INPUT": "invalid approved risk percentage"}.get(reason_code, f"position sizing refused: {reason_code}")
