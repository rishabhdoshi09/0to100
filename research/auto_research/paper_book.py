"""
📄 PaperBook — a self-contained simulated ledger for full paper autonomy.

This is where the autonomous brain is allowed to "blow up paper money": it opens and closes
SIMULATED positions, marks them against real price bars, and books realized R. It exists so
the system can trade its own approved strategies hands-off and learn from the outcomes.

Safety by construction:
  • It imports NOTHING from any real-order execution path. There is no code path from here
    to a real order. It can only move numbers in memory.
  • It still enforces the house risk invariants (1% risk/trade, 10% per name, 5% total open
    risk, max open positions) — "blow up" means bad strategies losing simulated money, not
    reckless sizing that would mask which ideas actually work.

Pure and deterministic: given the same opens and the same price bars, the book is identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


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
                 max_positions: int = 5):
        self.capital = float(capital)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.max_positions = max_positions
        self.open: dict[tuple, PaperPosition] = {}
        self.closed: list[ClosedTrade] = []
        self.realized_pnl = 0.0
        self.equity_curve: list[float] = [self.capital]
        self.refusals: list[tuple] = []          # (symbol, reason) — auditable

    # ── sizing + open ────────────────────────────────────────────────────────────
    def open_position(self, strategy_id: str, symbol: str, entry: float, stop: float,
                      target: float, date: str, max_holding_days: int) -> PaperPosition | None:
        """Open a simulated long. Returns the position, or None if a risk cap refuses it."""
        key = (strategy_id, symbol)
        if key in self.open:
            self.refusals.append((symbol, "already open for this strategy")); return None
        if len(self.open) >= self.max_positions:
            self.refusals.append((symbol, "max open positions reached")); return None
        if not (entry > 0 and stop > 0 and entry > stop):
            self.refusals.append((symbol, "invalid entry/stop (need entry>stop>0)")); return None

        risk_amount = self.capital * self.risk_per_trade_pct
        r_unit = entry - stop
        qty = int(math.floor(risk_amount / r_unit))
        if qty <= 0:
            self.refusals.append((symbol, "risk unit too wide for 1% sizing")); return None
        # 10% concentration cap
        if qty * entry > self.capital * self.max_position_pct:
            qty = int(math.floor(self.capital * self.max_position_pct / entry))
        if qty <= 0:
            self.refusals.append((symbol, "price too high for 10% cap")); return None
        # 5% total open risk cap
        new_risk = qty * r_unit
        if self.open_risk() + new_risk > self.capital * self.max_total_risk_pct + 1e-6:
            self.refusals.append((symbol, "total open risk cap (5%) reached")); return None

        pos = PaperPosition(strategy_id=strategy_id, symbol=symbol, entry_price=entry,
                            stop_price=stop, target_price=target, qty=qty, entry_date=date,
                            max_holding_days=max_holding_days, risk_amount=new_risk)
        self.open[key] = pos
        return pos

    def open_risk(self) -> float:
        return sum(p.qty * p.r_unit for p in self.open.values())

    # ── mark-to-market: advance one bar for every open position ──────────────────
    def mark(self, bars: dict, date: str) -> list[ClosedTrade]:
        """Advance one trading day. `bars` maps symbol -> (high, low, close). A position
        closes STOP-first (conservative: assume the adverse level trades first), then TARGET,
        then MAX_HOLD. Returns the trades closed on this bar."""
        closed_now: list[ClosedTrade] = []
        for key, pos in list(self.open.items()):
            bar = bars.get(pos.symbol)
            if bar is None:
                continue
            high, low, close = float(bar[0]), float(bar[1]), float(bar[2])
            pos.bars_held += 1
            exit_price = exit_reason = None
            if low <= pos.stop_price:                       # stop-first (conservative)
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
        pnl = (exit_price - pos.entry_price) * pos.qty
        realized_R = (exit_price - pos.entry_price) / pos.r_unit
        self.realized_pnl += pnl
        t = ClosedTrade(strategy_id=pos.strategy_id, symbol=pos.symbol,
                        entry_price=pos.entry_price, exit_price=exit_price,
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
                "n_open": len(self.open), "stats": self.stats()}
