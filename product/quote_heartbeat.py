"""Live quote heartbeat for the React terminal.

Reuses the existing Kite WebSocket store (`data.live_ticker`) and falls back
to REST `get_live_quotes` when the stream is down. Never invents prices.
Charts / dossiers / US desk stay on EOD — this is for visible LTP only.
"""
from __future__ import annotations

import time
from typing import Any, Iterable


_INDEX_ALIASES = {
    "NIFTY": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTYBANK": "NIFTY BANK",
}


def _clean_symbols(symbols: Iterable[str], *, limit: int = 40) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym or len(sym) > 32 or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= max(1, min(int(limit), 80)):
            break
    return out


def _session_open() -> bool:
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        from research.autonomy import schedules as SCH

        return bool(SCH.market_is_open(now))
    except Exception:
        pass
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)
    except Exception:
        return False


def _from_ticker(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        from data import live_ticker as LT

        streaming = False
        try:
            streaming = bool(LT.watch(symbols))
        except Exception:
            streaming = False
        ticks = LT.get_ticks(symbols) or {}
        status = LT.status() if hasattr(LT, "status") else {}
        out: dict[str, dict[str, Any]] = {}
        for sym, row in ticks.items():
            price = row.get("price")
            if price is None:
                continue
            out[sym] = {
                "symbol": sym,
                "price": float(price),
                "chg_pct": row.get("chg_pct"),
                "volume": row.get("volume"),
                "high": row.get("high"),
                "low": row.get("low"),
                "age_s": row.get("age_s"),
                "source": "kite_ws",
                "streaming": True,
            }
        return out, {"streaming": streaming, **dict(status or {})}
    except Exception as exc:
        return {}, {"streaming": False, "error": str(exc)}


def _from_rest(symbols: list[str]) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    try:
        from data.live_quotes import get_live_quotes

        raw = get_live_quotes(symbols, ttl=5.0) or {}
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    now_ts = time.time()
    for sym, row in raw.items():
        price = row.get("price")
        if price is None:
            continue
        out[str(sym).upper()] = {
            "symbol": str(sym).upper(),
            "price": float(price),
            "chg_pct": row.get("chg_pct"),
            "age_s": 0.0,
            "source": str(row.get("source") or "rest"),
            "streaming": False,
            "fetched_at": now_ts,
        }
    return out


def _from_index_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    want = [s for s in symbols if s in _INDEX_ALIASES or s in {"NIFTY", "BANKNIFTY"}]
    if not want:
        return {}
    try:
        from data.live_quotes import get_index_quotes

        raw = get_index_quotes(["NIFTY", "BANKNIFTY"]) or {}
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for sym in want:
        key = "NIFTY" if sym.startswith("NIFTY") and "BANK" not in sym else "BANKNIFTY"
        if sym in {"BANKNIFTY", "NIFTYBANK"}:
            key = "BANKNIFTY"
        row = raw.get(key) or {}
        if not row.get("price"):
            continue
        out[sym] = {
            "symbol": sym,
            "price": float(row["price"]),
            "chg_pct": row.get("chg_pct"),
            "age_s": 0.0,
            "source": str(row.get("source") or "index"),
            "streaming": False,
        }
    return out


def build_quote_heartbeat(
    symbols: Iterable[str] | None = None,
    *,
    limit: int = 40,
    prefer_stream: bool = True,
) -> dict[str, Any]:
    """Return live LTP snapshot for requested symbols (honest about source/age)."""
    requested = _clean_symbols(symbols or [], limit=limit)
    if not requested:
        requested = ["NIFTY", "BANKNIFTY"]

    session_open = _session_open()
    quotes: dict[str, dict[str, Any]] = {}
    stream_meta: dict[str, Any] = {"streaming": False}

    if prefer_stream and session_open:
        stock_syms = [s for s in requested if s not in _INDEX_ALIASES and s not in {"NIFTY", "BANKNIFTY", "NIFTY50", "NIFTYBANK"}]
        if stock_syms:
            streamed, stream_meta = _from_ticker(stock_syms)
            quotes.update(streamed)

    missing = [s for s in requested if s not in quotes]
    if missing:
        # Indices often come from a separate path.
        quotes.update(_from_index_quotes(missing))
        missing = [s for s in requested if s not in quotes]
    if missing:
        # REST is fine after hours too — returns last traded print, not a fake tick stream.
        rest = _from_rest(missing)
        if not session_open:
            for row in rest.values():
                row["streaming"] = False
        quotes.update(rest)
        missing = [s for s in requested if s not in quotes]

    rows = [quotes[s] for s in requested if s in quotes]
    sources = sorted({str(r.get("source") or "") for r in rows if r.get("source")})
    max_age = max((float(r.get("age_s") or 0) for r in rows), default=None)

    return {
        "available": bool(rows),
        "session_open": session_open,
        "streaming": bool(stream_meta.get("streaming")) and session_open,
        "watching": int(stream_meta.get("watching") or len(rows)),
        "requested": requested,
        "missing": missing,
        "quotes": {r["symbol"]: r for r in rows},
        "rows": rows,
        "sources": sources,
        "max_age_s": max_age,
        "generated_at": time.time(),
        "places_orders": False,
        "live_locked": True,
        "honesty": (
            "Live LTP heartbeat for visible symbols. "
            "Charts, fundamentals, and research reports stay on official EOD history. "
            + (
                "Market session open — Kite stream preferred, REST fallback."
                if session_open
                else "Market closed — no fake ticks; use EOD close for history."
            )
        ),
    }
