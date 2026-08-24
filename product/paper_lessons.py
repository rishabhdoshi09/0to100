"""Turn closed paper trades into the next research action.

Paper losses do not rewrite BUY lists, ranking, or autopilot. They tell the
operator which symbol and exit type to inspect in Backtest.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

BACKTEST_PURPOSE = (
    "Backtest answers one business question: did this style of trade make or "
    "lose money on past data after costs? Use it after a paper loss, before "
    "you size up, and when the market is closed."
)

BACKTEST_DOES_NOT_CHANGE = (
    "A backtest result does not change today's BUY list, ranking, or paper "
    "autopilot. Rules stay frozen until a separate evidence review says otherwise."
)

PAPER_TO_BACKTEST = (
    "After a paper loss, test that stock in Backtest. If the pattern lost "
    "historically too, do not keep repeating it in size. If it historically "
    "paid, the paper loss is one outcome — still keep risk small."
)


def _get(trade: Any, name: str, default: Any = None) -> Any:
    if isinstance(trade, Mapping):
        return trade.get(name, default)
    return getattr(trade, name, default)


def trade_pnl(trade: Any) -> float:
    raw = _get(trade, "pnl")
    if raw is not None and str(raw) != "":
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    try:
        return float(_get(trade, "realized_R", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_paper_loss(trade: Any) -> bool:
    pnl = trade_pnl(trade)
    if pnl < 0:
        return True
    try:
        return float(_get(trade, "realized_R", 0.0) or 0.0) < 0
    except (TypeError, ValueError):
        return False


def _exit_label(reason: str) -> str:
    key = str(reason or "").strip().upper()
    return {
        "STOP": "stop hit",
        "TARGET": "target hit",
        "MAX_HOLD": "time stop",
        "TIME": "time stop",
        "MANUAL": "manual exit",
    }.get(key, (reason or "exit").replace("_", " ").lower())


def paper_loss_lessons(closed_trades: Iterable[Any], *, limit: int = 5) -> tuple[dict[str, Any], ...]:
    """Most recent losing paper trades, each with a Backtest next step."""
    losses: list[tuple[str, float, Any]] = []
    for trade in closed_trades or ():
        if not is_paper_loss(trade):
            continue
        symbol = str(_get(trade, "symbol", "") or "").upper()
        if not symbol:
            continue
        exit_date = str(_get(trade, "exit_date", "") or "")
        losses.append((exit_date, trade_pnl(trade), trade))
    losses.sort(key=lambda row: (row[0], -row[1]), reverse=True)  # newest first, then largest loss
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, pnl, trade in losses:
        symbol = str(_get(trade, "symbol", "") or "").upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        reason = str(_get(trade, "exit_reason", "") or "")
        r_mult = _get(trade, "realized_R")
        try:
            r_val = float(r_mult) if r_mult is not None else None
        except (TypeError, ValueError):
            r_val = None
        r_bit = f" ({r_val:+.2f}R)" if r_val is not None else ""
        out.append({
            "symbol": symbol,
            "pnl": pnl,
            "realized_R": r_val,
            "exit_reason": reason,
            "entry_date": str(_get(trade, "entry_date", "") or ""),
            "exit_date": str(_get(trade, "exit_date", "") or ""),
            "headline": f"{symbol} paper loss — {_exit_label(reason)}{r_bit}",
            "next_step": (
                f"Open Backtest, keep universe on this stock, and run {symbol}. "
                "Check whether this entry style paid after costs."
            ),
            "does_not_change": BACKTEST_DOES_NOT_CHANGE,
        })
        if len(out) >= max(0, int(limit)):
            break
    return tuple(out)
