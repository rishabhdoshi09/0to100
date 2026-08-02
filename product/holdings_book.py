"""Retail broker holdings book — your demat shares, not paper.

Persists to ``logs/product/holdings.json``. Can be filled by:
  • Zerodha Kite sync (``sync_from_kite``)
  • Manual / paste import (``replace_holdings``)

Holdings keep the broker tradingsymbol exactly (including ``-BE`` T2T series).
Research-universe filters must never hide a share you already own.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "holdings.json"


def holdings_path(path: Path | None = None) -> Path:
    env = os.environ.get("QT_HOLDINGS_FILE", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def research_symbol(tradingsymbol: str) -> str:
    """Map broker series suffixes to the EQ research/bhav ticker when possible."""
    sym = str(tradingsymbol or "").strip().upper()
    for suf in ("-BE", "-BZ", "-BL", "-SM"):
        if sym.endswith(suf):
            return sym[: -len(suf)]
    return sym


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_holding(row: Mapping[str, Any]) -> dict[str, Any] | None:
    sym = str(
        row.get("tradingsymbol")
        or row.get("symbol")
        or row.get("trading_symbol")
        or ""
    ).strip().upper()
    if not sym:
        return None
    qty = _i(row.get("quantity") if row.get("quantity") is not None else row.get("qty"))
    if qty == 0:
        # Allow T1-only rows from Kite
        qty = _i(row.get("t1_quantity"))
    avg = _f(row.get("average_price") if row.get("average_price") is not None else row.get("avg"))
    ltp = _f(
        row.get("last_price")
        if row.get("last_price") is not None
        else row.get("ltp")
        if row.get("ltp") is not None
        else row.get("close")
    )
    invested = _f(row.get("invested"))
    if invested <= 0 and qty and avg:
        invested = round(qty * avg, 2)
    current = _f(row.get("current_value"))
    if current <= 0 and qty and ltp:
        current = round(qty * ltp, 2)
    pnl = row.get("pnl")
    if pnl is None and invested:
        pnl = round(current - invested, 2)
    else:
        pnl = _f(pnl)
    pnl_pct = row.get("pnl_pct")
    if pnl_pct is None and invested:
        pnl_pct = round((pnl / invested) * 100.0, 2) if invested else 0.0
    else:
        pnl_pct = _f(pnl_pct)
    return {
        "tradingsymbol": sym,
        "research_symbol": research_symbol(sym),
        "quantity": qty,
        "t1_quantity": _i(row.get("t1_quantity")),
        "average_price": round(avg, 4),
        "last_price": round(ltp, 4),
        "invested": round(invested, 2),
        "current_value": round(current, 2),
        "pnl": round(_f(pnl), 2),
        "pnl_pct": round(_f(pnl_pct), 2),
        "day_change": _f(row.get("day_change")),
        "day_change_percentage": _f(
            row.get("day_change_percentage")
            if row.get("day_change_percentage") is not None
            else row.get("day_change_pct")
        ),
        "product": str(row.get("product") or "CNC"),
        "exchange": str(row.get("exchange") or "NSE"),
        "isin": str(row.get("isin") or ""),
        "collateral_quantity": _i(row.get("collateral_quantity")),
    }


def empty_book(*, message: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": "",
        "source": "",
        "available": False,
        "holdings": [],
        "summary": {
            "count": 0,
            "invested": 0.0,
            "current_value": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "day_pnl": 0.0,
        },
        "message": message or "No broker holdings saved yet. Sync from Zerodha or import your demat book.",
        "places_orders": False,
    }


def _summarize(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    invested = round(sum(_f(h.get("invested")) for h in holdings), 2)
    current = round(sum(_f(h.get("current_value")) for h in holdings), 2)
    pnl = round(current - invested, 2)
    day = round(sum(_f(h.get("day_change")) * _i(h.get("quantity")) for h in holdings), 2)
    # Prefer broker day_change_percentage when present on rows; else derive.
    if any(h.get("day_change_percentage") for h in holdings):
        # Approximate portfolio day % from rupee day pnl vs prior close value.
        prior = current - day
        day_pct = round((day / prior) * 100.0, 2) if prior else 0.0
    else:
        day_pct = 0.0
    return {
        "count": len(holdings),
        "invested": invested,
        "current_value": current,
        "pnl": pnl,
        "pnl_pct": round((pnl / invested) * 100.0, 2) if invested else 0.0,
        "day_pnl": day,
        "day_pnl_pct": day_pct,
    }


def load_holdings(path: Path | None = None) -> dict[str, Any]:
    p = holdings_path(path)
    if not p.exists():
        return empty_book()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return empty_book(message="Holdings file is unreadable — re-import or sync from Zerodha.")
    holdings = []
    for row in list(raw.get("holdings") or []):
        norm = normalize_holding(row if isinstance(row, Mapping) else {})
        if norm:
            holdings.append(norm)
    if not holdings:
        return empty_book(message=str(raw.get("message") or ""))
    return {
        "schema_version": 1,
        "updated_at": str(raw.get("updated_at") or ""),
        "source": str(raw.get("source") or "file"),
        "available": True,
        "holdings": holdings,
        "summary": _summarize(holdings),
        "message": "",
        "places_orders": False,
    }


def save_holdings(
    holdings: Iterable[Mapping[str, Any]],
    *,
    source: str,
    path: Path | None = None,
) -> dict[str, Any]:
    p = holdings_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in holdings:
        norm = normalize_holding(row)
        if norm and norm["quantity"] != 0:
            rows.append(norm)
    rows.sort(key=lambda r: r["tradingsymbol"])
    payload = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "source": source,
        "available": bool(rows),
        "holdings": rows,
        "summary": _summarize(rows),
        "message": "" if rows else "Import contained no quantity > 0 rows.",
        "places_orders": False,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return payload


def enrich_ltp(book: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing / stale LTP from local bhav when possible (EQ research symbol)."""
    holdings = [dict(h) for h in list(book.get("holdings") or [])]
    if not holdings:
        return dict(book)
    try:
        from data.bhavcopy_store import get_ohlcv
    except Exception:
        get_ohlcv = None  # type: ignore
    for row in holdings:
        if _f(row.get("last_price")) > 0:
            # Still refresh from bhav when source is import and price looks stale? keep broker LTP.
            continue
        if get_ohlcv is None:
            continue
        for candidate in (row.get("research_symbol"), row.get("tradingsymbol")):
            try:
                frame = get_ohlcv(str(candidate))
            except Exception:
                frame = None
            if frame is None or len(frame) == 0:
                continue
            close = _f(frame["close"].iloc[-1])
            if close <= 0:
                continue
            row["last_price"] = round(close, 4)
            qty = _i(row.get("quantity"))
            row["current_value"] = round(qty * close, 2)
            invested = _f(row.get("invested"))
            row["pnl"] = round(row["current_value"] - invested, 2)
            row["pnl_pct"] = round((row["pnl"] / invested) * 100.0, 2) if invested else 0.0
            break
    out = dict(book)
    out["holdings"] = holdings
    out["summary"] = _summarize(holdings)
    out["available"] = bool(holdings)
    return out


def build_holdings_payload(path: Path | None = None) -> dict[str, Any]:
    return enrich_ltp(load_holdings(path))


def sync_from_kite(path: Path | None = None) -> dict[str, Any]:
    """Pull CNC holdings from Zerodha when the session is connected."""
    try:
        from data.kite_client import KiteClient
    except Exception as exc:
        return {
            **empty_book(message=f"Kite client unavailable: {exc}"),
            "synced": False,
        }
    try:
        kc = KiteClient()
    except Exception as exc:
        return {
            **empty_book(message=f"Zerodha not connected: {exc}"),
            "synced": False,
        }
    if not kc.is_connected():
        return {
            **empty_book(message="Zerodha access token missing — connect Kite, then Sync holdings."),
            "synced": False,
        }
    try:
        raw = list(kc.get_holdings() or [])
    except Exception as exc:
        return {
            **empty_book(message=f"Kite holdings fetch failed: {exc}"),
            "synced": False,
        }
    if not raw:
        book = save_holdings([], source="kite", path=path)
        book["synced"] = True
        book["message"] = "Kite connected but returned zero holdings."
        return book
    book = save_holdings(raw, source="kite", path=path)
    book["synced"] = True
    return enrich_ltp(book)


def holdings_symbols(path: Path | None = None) -> list[str]:
    book = load_holdings(path)
    out: list[str] = []
    for row in book.get("holdings") or []:
        sym = str(row.get("tradingsymbol") or "").strip().upper()
        if sym:
            out.append(sym)
        research = str(row.get("research_symbol") or "").strip().upper()
        if research and research not in out:
            out.append(research)
    return out
