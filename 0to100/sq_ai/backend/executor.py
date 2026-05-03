"""Order executor – paper-trading by default, live Kite when SQ_PAPER_TRADING=false."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sq_ai.portfolio.tracker import PortfolioTracker


JOURNAL_PATH = os.environ.get(
    "SQ_JOURNAL_PATH",
    str(Path(__file__).resolve().parents[2] / "journal.md"),
)
JOURNAL_HEADER = (
    "| timestamp | symbol | regime | reasoning | entry_price | "
    "stop_price | target_price |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _journal_append(row: dict[str, Any]) -> None:
    p = Path(JOURNAL_PATH)
    if not p.exists():
        p.write_text(JOURNAL_HEADER)
    safe_reason = (row.get("reasoning") or "").replace("|", "/").replace("\n", " ")[:240]
    line = (
        f"| {row['timestamp']} | {row['symbol']} | {row.get('regime', '')} | "
        f"{safe_reason} | {row['entry_price']:.2f} | "
        f"{row['stop_price']:.2f} | {row['target_price']:.2f} |\n"
    )
    with p.open("a") as f:
        f.write(line)


@dataclass
class Order:
    symbol: str
    side: str           # BUY / SELL
    qty: int
    price: float
    stop: float
    target: float


class Executor:
    def __init__(self, tracker: PortfolioTracker,
                 paper: bool | None = None) -> None:
        self.tracker = tracker
        self.paper = (
            paper
            if paper is not None
            else os.environ.get("SQ_PAPER_TRADING", "true").lower() == "true"
        )
        self._kite = None
        if not self.paper:
            try:
                from kiteconnect import KiteConnect  # noqa: WPS433
                k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
                k.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
                self._kite = k
            except Exception as exc:  # pragma: no cover
                print(f"[Executor] live mode init failed → paper: {exc}")
                self.paper = True

    # -------------------------------------------------------------- BUY
    def buy(self, order: Order, reasoning: str = "",
            regime: int | str = "") -> dict[str, Any]:
        if order.qty <= 0:
            return {"status": "skip", "reason": "qty<=0"}
        broker_id = self._place(order, "BUY")
        # immediately push stop/target as separate broker orders (live only)
        if not self.paper and self._kite is not None:
            self._place_stop_target(order)
        trade_id = self.tracker.open_trade(
            order.symbol, order.price, order.qty, order.stop, order.target
        )
        _journal_append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": order.symbol,
            "regime": regime,
            "reasoning": reasoning,
            "entry_price": order.price,
            "stop_price": order.stop,
            "target_price": order.target,
        })
        return {"status": "ok", "trade_id": trade_id, "broker_id": broker_id,
                "mode": "paper" if self.paper else "live"}

    # -------------------------------------------------------------- SELL / EXIT
    def exit_position(self, trade_id: int, exit_price: float) -> dict[str, Any]:
        broker_id = self._place(
            Order("?", "SELL", 0, exit_price, 0, 0), "SELL"
        ) if not self.paper else "paper-exit"
        pnl = self.tracker.close_trade(trade_id, exit_price)
        return {"status": "ok", "trade_id": trade_id, "pnl": pnl,
                "broker_id": broker_id}

    # -------------------------------------------------------------- check stops/targets
    def check_exits(self, last_prices: dict[str, float]) -> list[dict]:
        events: list[dict] = []
        for pos in self.tracker.open_positions():
            sym = pos["symbol"]
            ltp = last_prices.get(sym)
            if ltp is None:
                continue
            if ltp <= pos["stop_initial"]:
                events.append({"trade_id": pos["id"], "symbol": sym,
                               "reason": "stop", "price": ltp,
                               **self.exit_position(pos["id"], ltp)})
            elif ltp >= pos["target"]:
                events.append({"trade_id": pos["id"], "symbol": sym,
                               "reason": "target", "price": ltp,
                               **self.exit_position(pos["id"], ltp)})
            else:
                entry_dt = datetime.fromisoformat(pos["entry_date"].replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - entry_dt).days
                if age_days >= 20:
                    events.append({"trade_id": pos["id"], "symbol": sym,
                                   "reason": "time", "price": ltp,
                                   **self.exit_position(pos["id"], ltp)})
        return events

    # -------------------------------------------------------------- internals
    def _place(self, order: Order, side: str) -> str:
        if self.paper or self._kite is None:
            return f"paper-{side.lower()}-{order.symbol}-{datetime.now(timezone.utc).timestamp():.0f}"
        try:                                           # pragma: no cover
            return self._kite.place_order(
                tradingsymbol=order.symbol.replace(".NS", ""),
                exchange="NSE",
                transaction_type=side,
                quantity=order.qty,
                order_type="MARKET",
                product="CNC",
                variety="regular",
            )
        except Exception as exc:                       # pragma: no cover
            print(f"[Executor._place] live order failed: {exc}")
            return f"live-failed-{side}"

    def _place_stop_target(self, order: Order) -> None:                # pragma: no cover
        try:
            self._kite.place_order(
                tradingsymbol=order.symbol.replace(".NS", ""),
                exchange="NSE", transaction_type="SELL",
                quantity=order.qty, order_type="SL",
                price=order.stop, trigger_price=order.stop,
                product="CNC", variety="regular",
            )
            self._kite.place_order(
                tradingsymbol=order.symbol.replace(".NS", ""),
                exchange="NSE", transaction_type="SELL",
                quantity=order.qty, order_type="LIMIT",
                price=order.target,
                product="CNC", variety="regular",
            )
        except Exception as exc:
            print(f"[Executor._place_stop_target] {exc}")
