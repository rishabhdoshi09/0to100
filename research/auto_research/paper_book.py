"""
📄 PaperBook — a self-contained simulated ledger for full paper autonomy.

This is where the autonomous brain is allowed to "blow up paper money": it opens and closes
SIMULATED positions, marks them against real price bars, and books realized R. It exists so
the system can trade its own approved strategies hands-off and learn from the outcomes.

Safety by construction:
  • It imports NOTHING from any real-order execution path. There is no code path from here
    to a real order. It can only move numbers in memory.
  • It enforces the house risk invariants (1% maximum risk/trade, 10% per name, 5% total
    open risk, max open positions) and consumes Brain 2's smaller approved risk budgets.

Pure and deterministic: given the same opens and the same price bars, the book is identical.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

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

    @property
    def r_unit(self) -> float:
        return max(1e-9, self.entry_price - self.stop_price)   # long only

    def as_dict(self): return asdict(self)


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
    exit_reason: str            # STOP / TARGET / MAX_HOLD
    realized_R: float
    pnl: float

    def as_dict(self): return asdict(self)


class PaperBook:
    """A simulated long-only book with real risk caps. Long-only mirrors the cash-equity
    reality the rest of QuantTerm assumes (no overnight equity shorts in India)."""

    def __init__(self, capital: float = 100_000.0, *, risk_per_trade_pct: float = 0.01,
                 max_position_pct: float = 0.10, max_total_risk_pct: float = 0.05,
                 max_positions: int = 5, slippage_bps: float = 0.0, cost_model=None):
        self.capital = float(capital)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.max_positions = max_positions
        # frictions — default OFF (frictionless) so direct unit tests stay exact; the paper
        # autonomy manager turns them on with realistic India cash-equity values.
        self.slippage_bps = float(slippage_bps)      # entry+exit slippage, basis points
        self.cost_model = cost_model                 # callable(entry, exit, qty) -> ₹
        self.open: dict[tuple, PaperPosition] = {}
        self.closed: list[ClosedTrade] = []
        self.realized_pnl = 0.0
        self.equity_curve: list[float] = [self.capital]
        self.refusals: list[tuple] = []          # (symbol, reason) — auditable

    # ── sizing + open ────────────────────────────────────────────────────────────
    def open_position(self, strategy_id: str, symbol: str, entry: float, stop: float,
                      target: float, date: str, max_holding_days: int, *,
                      risk_pct_of_capital: float | None = None,
                      quantity: int | None = None) -> PaperPosition | None:
        """Open a simulated long from an approved risk budget and optional exact quantity.

        ``risk_pct_of_capital`` uses percentage points: ``1.0`` means one percent of
        capital and ``0.25`` means a quarter percent. A Target Portfolio may provide an
        exact quantity; the book revalidates it and refuses any quantity above house caps.
        """
        key = (strategy_id, symbol)
        if key in self.open:
            self.refusals.append((symbol, "already open for this strategy")); return None
        if len(self.open) >= self.max_positions:
            self.refusals.append((symbol, "max open positions reached")); return None

        requested_risk_pct = (
            float(self.risk_per_trade_pct) * 100.0
            if risk_pct_of_capital is None else risk_pct_of_capital
        )
        sizing = size_long_cash(
            capital=self.capital,
            entry=entry,
            stop=stop,
            requested_risk_pct=requested_risk_pct,
            max_risk_fraction=self.risk_per_trade_pct,
            max_position_fraction=self.max_position_pct,
            slippage_bps=self.slippage_bps,
            requested_quantity=quantity,
        )
        if not sizing.ok:
            self.refusals.append((symbol, _sizing_refusal(sizing.reason_code)))
            return None

        if self.open_risk() + sizing.risk_amount > self.capital * self.max_total_risk_pct + 1e-6:
            self.refusals.append((symbol, "total open risk cap (5%) reached")); return None

        pos = PaperPosition(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_price=sizing.effective_entry,
            stop_price=float(stop),
            target_price=float(target),
            qty=sizing.quantity,
            entry_date=date,
            max_holding_days=max_holding_days,
            risk_amount=sizing.risk_amount,
            requested_risk_pct=sizing.requested_risk_pct,
            approved_risk_pct=sizing.actual_risk_pct,
        )
        self.open[key] = pos
        return pos

    def open_intent(self, intent, *, date: str) -> PaperPosition | None:
        """Execute an exact Target Portfolio delta in PAPER after revalidation."""
        required_quantity = int(getattr(intent, "required_quantity", 0) or 0)
        return self.open_position(
            intent.strategy_id,
            intent.symbol,
            float(intent.intended_entry),
            float(intent.stop_price),
            float(intent.target_price),
            date,
            int(intent.holding_horizon_days),
            risk_pct_of_capital=float(intent.intended_risk_pct),
            quantity=(required_quantity if required_quantity > 0 else None),
        )

    def open_risk(self) -> float:
        return sum(p.qty * p.r_unit for p in self.open.values())

    # ── mark-to-market: advance one bar for every open position ──────────────────
    def mark(self, bars: dict, date: str) -> list[ClosedTrade]:
        """Advance one trading day. `bars` maps symbol -> (high, low, close) OR
        (open, high, low, close). When an open is given, a GAP THROUGH the stop fills at the
        gap price (worse than the stop) and a gap through the target fills at the gap (better)
        — honest to how NSE actually opens. Otherwise closes STOP-first (conservative), then
        TARGET, then MAX_HOLD. Returns the trades closed on this bar."""
        closed_now: list[ClosedTrade] = []
        for key, pos in list(self.open.items()):
            bar = bars.get(pos.symbol)
            if bar is None:
                continue
            if len(bar) >= 4:
                op, high, low, close = (float(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]))
            else:
                op, high, low, close = (None, float(bar[0]), float(bar[1]), float(bar[2]))
            pos.bars_held += 1
            exit_price = exit_reason = None
            if op is not None and op <= pos.stop_price:      # gap DOWN through stop → worse fill
                exit_price, exit_reason = op, "GAP_STOP"
            elif op is not None and op >= pos.target_price:  # gap UP through target → better fill
                exit_price, exit_reason = op, "GAP_TARGET"
            elif low <= pos.stop_price:                      # stop-first (conservative)
                exit_price, exit_reason = pos.stop_price, "STOP"
            elif high >= pos.target_price:
                exit_price, exit_reason = pos.target_price, "TARGET"
            elif pos.bars_held >= pos.max_holding_days:
                exit_price, exit_reason = close, "MAX_HOLD"
            if exit_price is not None:
                closed_now.append(self._close(key, pos, exit_price, exit_reason, date))
        self.equity_curve.append(self.equity(bars))
        return closed_now

    def _close(self, key, pos: PaperPosition, exit_price: float, reason: str,
               date: str) -> ClosedTrade:
        # exit slippage — a seller gets hit down
        exit_fill = exit_price * (1.0 - self.slippage_bps / 1e4)
        gross = (exit_fill - pos.entry_price) * pos.qty
        cost = 0.0
        if self.cost_model is not None:
            try:
                cost = float(self.cost_model(pos.entry_price, exit_fill, pos.qty))
            except Exception:
                cost = 0.0
        pnl = gross - cost                                   # NET of frictions — honest
        realized_R = pnl / (pos.qty * pos.r_unit)            # net R, comparable to expectancy
        self.realized_pnl += pnl
        t = ClosedTrade(strategy_id=pos.strategy_id, symbol=pos.symbol,
                        entry_price=pos.entry_price, exit_price=exit_fill,
                        stop_price=pos.stop_price, qty=pos.qty, entry_date=pos.entry_date,
                        exit_date=date, exit_reason=reason, realized_R=realized_R, pnl=pnl)
        self.closed.append(t)
        del self.open[key]
        return t

    def equity(self, bars: dict | None = None) -> float:
        """Realized equity plus open mark-to-market against the latest close (if given)."""
        eq = self.capital + self.realized_pnl
        if bars:
            for pos in self.open.values():
                bar = bars.get(pos.symbol)
                if bar is not None:
                    eq += (float(bar[2]) - pos.entry_price) * pos.qty
        return eq

    # ── reporting ────────────────────────────────────────────────────────────────
    def stats(self, strategy_id: str | None = None) -> dict:
        trades = ([t for t in self.closed if t.strategy_id == strategy_id]
                  if strategy_id else self.closed)
        n = len(trades)
        wins = [t for t in trades if t.realized_R > 0]
        losses = [t for t in trades if t.realized_R <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        expectancy_R = (sum(t.realized_R for t in trades) / n) if n else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_win else 0.0)
        return {
            "n_trades": n,
            "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "expectancy_R": round(expectancy_R, 4),
            "profit_factor": (round(pf, 3) if pf != float("inf") else None),
            "net_pnl": round(sum(t.pnl for t in trades), 2),
            "max_drawdown_pct": round(self._max_dd(), 4),
            "equity": round(self.equity(), 2),
            "open_positions": len([p for p in self.open.values()
                                   if strategy_id is None or p.strategy_id == strategy_id]),
        }

    def r_stats(self, strategy_id: str | None = None) -> dict:
        """Mean R, standard error of the mean, and a conservative lower estimate
        (mean − 1·SE) of a strategy's realized R. The lower estimate is what noise-aware
        calibration judges, so a couple of lucky trades can't fake an edge."""
        rs = [t.realized_R for t in self.closed
              if strategy_id is None or t.strategy_id == strategy_id]
        n = len(rs)
        if n == 0:
            return {"n": 0, "mean_R": 0.0, "stderr_R": 0.0, "lower_R": 0.0}
        mean = sum(rs) / n
        if n > 1:
            var = sum((r - mean) ** 2 for r in rs) / (n - 1)
            se = (var ** 0.5) / (n ** 0.5)
        else:
            se = 0.0
        return {"n": n, "mean_R": round(mean, 4), "stderr_R": round(se, 4),
                "lower_R": round(mean - se, 4)}

    def _max_dd(self) -> float:
        peak = self.equity_curve[0] if self.equity_curve else self.capital
        mdd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                mdd = max(mdd, (peak - v) / peak)
        return mdd

    def as_dict(self) -> dict:
        return {"capital": self.capital, "realized_pnl": round(self.realized_pnl, 2),
                "equity": round(self.equity(), 2), "n_closed": len(self.closed),
                "n_open": len(self.open), "stats": self.stats(),
                "equity_curve": [round(v, 2) for v in self.equity_curve[-120:]]}

    # ── persistence (the book remembers its trades across restarts) ──────────────
    def snapshot(self) -> dict:
        return {"capital": self.capital, "realized_pnl": self.realized_pnl,
                "equity_curve": self.equity_curve,
                "closed": [t.as_dict() for t in self.closed],
                "open": [p.as_dict() for p in self.open.values()]}

    def restore(self, snap: dict) -> None:
        try:
            self.capital = float(snap.get("capital", self.capital))
            self.realized_pnl = float(snap.get("realized_pnl", 0.0))
            self.equity_curve = list(snap.get("equity_curve", [self.capital])) or [self.capital]
            self.closed = [ClosedTrade(**t) for t in snap.get("closed", [])]
            self.open = {}
            for p in snap.get("open", []):
                pos = PaperPosition(**p)
                self.open[(pos.strategy_id, pos.symbol)] = pos
        except Exception:
            pass                                    # corrupt snapshot ⇒ keep fresh book


def _sizing_refusal(reason_code: str) -> str:
    return {
        "INVALID_ENTRY_STOP": "invalid entry/stop (need entry>stop>0)",
        "NON_POSITIVE_RISK": "approved risk percentage must be positive",
        "RISK_BUDGET_TOO_SMALL": "risk unit too wide for approved sizing",
        "POSITION_CAP_TOO_SMALL": "price too high for 10% cap",
        "INVALID_REQUESTED_QUANTITY": "invalid target portfolio quantity",
        "NON_POSITIVE_QUANTITY": "target portfolio quantity must be positive",
        "QUANTITY_EXCEEDS_APPROVED_LIMIT": "target portfolio quantity exceeds house limits",
        "INVALID_NUMERIC_INPUT": "invalid approved risk percentage",
        "NON_FINITE_INPUT": "invalid approved risk percentage",
    }.get(reason_code, f"position sizing refused: {reason_code}")
