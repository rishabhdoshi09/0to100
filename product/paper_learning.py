"""Daily paper-memory overlay: learn from closed simulated trades.

This is how the autonomous bot gets pickier on PAPER. It does not rewrite the
scanner BUY list, FEATURE-002 ranking, or unlock live execution.

Rules (conservative, sample-aware):
  • two consecutive losses on the same name → 5 calendar-day cooldown
  • three or more closed trades with positive mean R → prefer that name among
    the same strategy's other signals
  • a single loss is noise — it is recorded, not banned
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from product.paper_lessons import _get, is_paper_loss, trade_pnl

SCHEMA_VERSION = 1
COOLDOWN_DAYS = 5
CONSECUTIVE_LOSSES = 2
PREFER_MIN_TRADES = 3
DEFAULT_PATH = Path("logs/product/paper_memory.json")

LIVE_STILL_LOCKED = (
    "Live orders stay locked. Paper memory only changes which PAPER names "
    "are skipped or preferred. Real-money automation requires owner approval "
    "after the paper book is proven — the bot cannot open that door itself."
)

PROMOTION_LADDER = (
    "Paper auto is on. Each day the bot folds closed paper trades into memory, "
    "then the next paper cycle skips cooldown names and prefers proven ones. "
    "Brain 1 still needs a larger sample before a family is evidence-qualified. "
    "Live execution stays locked until the owner approves a capital envelope — "
    "the bot cannot open that door."
)

EMPTY_MEMORY: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "as_of": "",
    "symbols": [],
    "cooldown": [],
    "prefer": [],
    "closed_trades": 0,
    "summary": "No closed paper trades yet — nothing to learn.",
    "live_locked": True,
    "disclaimer": LIVE_STILL_LOCKED,
}


def memory_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_PAPER_MEMORY")
    if override:
        return Path(override)
    return DEFAULT_PATH


def _as_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _realized_r(trade: Any) -> float:
    raw = _get(trade, "realized_R")
    try:
        if raw is not None and str(raw) != "":
            return float(raw)
    except (TypeError, ValueError):
        pass
    pnl = trade_pnl(trade)
    return pnl


def build_paper_memory(closed_trades: Iterable[Any], *, as_of: str,
                       cooldown_days: int = COOLDOWN_DAYS) -> dict[str, Any]:
    """Fold closed paper trades into cooldown / prefer lists. Pure; no I/O."""
    as_of_d = _as_date(as_of) or date.today()
    by_symbol: dict[str, list[Any]] = defaultdict(list)
    for trade in closed_trades or ():
        symbol = str(_get(trade, "symbol", "") or "").upper()
        if not symbol:
            continue
        by_symbol[symbol].append(trade)

    symbols: list[dict[str, Any]] = []
    cooldown: list[dict[str, Any]] = []
    prefer: list[str] = []
    for symbol, trades in sorted(by_symbol.items()):
        trades.sort(key=lambda t: str(_get(t, "exit_date", "") or ""))
        n = len(trades)
        rs = [_realized_r(t) for t in trades]
        mean_r = sum(rs) / n if n else 0.0
        losses = sum(1 for t in trades if is_paper_loss(t))
        wins = n - losses
        last = trades[-1]
        last_exit = str(_get(last, "exit_date", "") or "")
        last_reason = str(_get(last, "exit_reason", "") or "")
        last2 = trades[-CONSECUTIVE_LOSSES:]
        consecutive = len(last2) >= CONSECUTIVE_LOSSES and all(is_paper_loss(t) for t in last2)
        until = ""
        if consecutive:
            last_d = _as_date(last_exit) or as_of_d
            until_d = last_d + timedelta(days=int(cooldown_days))
            until = until_d.isoformat()
            if as_of_d <= until_d:
                cooldown.append({
                    "symbol": symbol,
                    "until": until,
                    "reason": f"{CONSECUTIVE_LOSSES} consecutive paper losses — skip new entries until {until}",
                    "last_exit": last_exit,
                    "last_reason": last_reason,
                })
        on_cd = bool(until) and as_of_d <= (_as_date(until) or as_of_d)
        is_prefer = (n >= PREFER_MIN_TRADES and mean_r > 0 and not on_cd)
        if is_prefer:
            prefer.append(symbol)
        symbols.append({
            "symbol": symbol,
            "n": n,
            "wins": wins,
            "losses": losses,
            "mean_R": round(mean_r, 4),
            "last_exit": last_exit,
            "last_reason": last_reason,
            "cooldown_until": until if on_cd else "",
            "prefer": is_prefer,
        })

    n_closed = sum(row["n"] for row in symbols)
    if n_closed == 0:
        summary = "No closed paper trades yet — nothing to learn."
    else:
        summary = (
            f"{n_closed} closed paper trade(s) across {len(symbols)} name(s). "
            f"{len(cooldown)} on cooldown, {len(prefer)} preferred. "
            + LIVE_STILL_LOCKED
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": str(as_of),
        "symbols": symbols,
        "cooldown": cooldown,
        "prefer": prefer,
        "closed_trades": n_closed,
        "summary": summary,
        "live_locked": True,
        "disclaimer": LIVE_STILL_LOCKED,
    }


def save_paper_memory(memory: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = memory_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(memory), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_paper_memory(path: str | Path | None = None) -> dict[str, Any]:
    target = memory_path(path)
    if not target.exists():
        return dict(EMPTY_MEMORY)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            return dict(EMPTY_MEMORY)
        payload.setdefault("cooldown", [])
        payload.setdefault("prefer", [])
        payload.setdefault("symbols", [])
        payload.setdefault("live_locked", True)
        return payload
    except Exception:
        return dict(EMPTY_MEMORY)


def remember_paper_book(closed_trades: Iterable[Any], *, as_of: str,
                        path: str | Path | None = None) -> dict[str, Any]:
    memory = build_paper_memory(closed_trades, as_of=as_of)
    save_paper_memory(memory, path)
    return memory


def public_memory(memory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Safe payload for the terminal API and bash-stack UI. Fail-open."""
    try:
        payload = dict(memory) if memory is not None else load_paper_memory()
    except Exception:
        payload = dict(EMPTY_MEMORY)
    return {
        "available": True,
        "as_of": str(payload.get("as_of") or ""),
        "closed_trades": int(payload.get("closed_trades") or 0),
        "cooldown": list(payload.get("cooldown") or []),
        "prefer": [str(s) for s in (payload.get("prefer") or [])],
        "summary": str(payload.get("summary") or EMPTY_MEMORY["summary"]),
        "live_locked": True,
        "disclaimer": LIVE_STILL_LOCKED,
        "ladder": PROMOTION_LADDER,
    }


def on_cooldown(memory: Mapping[str, Any] | None, symbol: str, *, as_of: str) -> str:
    """Return the cooldown reason if this name is blocked for new paper entries."""
    if not memory:
        return ""
    as_of_d = _as_date(as_of)
    key = str(symbol or "").upper()
    for row in memory.get("cooldown") or []:
        if str(row.get("symbol") or "").upper() != key:
            continue
        until = _as_date(row.get("until"))
        if as_of_d and until and as_of_d <= until:
            return str(row.get("reason") or "PAPER_LESSON_COOLDOWN")
    return ""


def select_paper_signal(signals: Iterable[Mapping[str, Any]], memory: Mapping[str, Any] | None,
                        *, as_of: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Pick the next paper name: skip cooldowns, prefer proven paper winners, else keep order."""
    rows = [dict(s) for s in signals or ()]
    skipped: list[str] = []
    eligible: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        reason = on_cooldown(memory, symbol, as_of=as_of)
        if reason:
            skipped.append(f"{symbol}:{reason}")
            continue
        eligible.append(row)
    if not eligible:
        return None, tuple(skipped)
    prefer = {str(s).upper() for s in (memory or {}).get("prefer") or []}
    if prefer:
        eligible.sort(key=lambda row: (0 if str(row.get("symbol") or "").upper() in prefer else 1))
    return eligible[0], tuple(skipped)
