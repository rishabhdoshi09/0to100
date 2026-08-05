"""Active Buys book — stocks you are buying / holding by intent.

Sources:
  • manual add (user-authored)
  • Zerodha / demat sync (``sync_from_holdings``) — tracks CNC holdings as active buys

Never invents symbols, prices, or fills. Never places orders.
Persists to ``logs/product/buy_book.json``.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "buy_book.json"
SCHEMA_VERSION = 1


def buy_book_path(path: Path | None = None) -> Path:
    env = os.environ.get("QT_BUY_BOOK_FILE", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_symbol(value: Any) -> str:
    sym = str(value or "").strip().upper()
    # Drop broker series suffixes for research/bhav lookup.
    for suf in ("-BE", "-BZ", "-BL", "-SM"):
        if sym.endswith(suf):
            sym = sym[: -len(suf)]
            break
    return sym


def empty_book() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "items": [],
        "places_orders": False,
        "live_locked": True,
        "honesty": (
            "Active Buys tracks stocks you are buying — manually or synced from Zerodha holdings. "
            "Each row emphasizes technicals (EMA/support/volume) and fundamentals (Screener cache). "
            "Results compare your entry/avg to live LTP or EOD. "
            "Zerodha sync uses demat avg + qty; estimate P&L is not a sell ticket. "
            "Never places orders."
        ),
    }


def load_book(path: Path | None = None) -> dict[str, Any]:
    target = buy_book_path(path)
    if not target.exists():
        return empty_book()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            return empty_book()
        out = empty_book()
        out.update({k: payload.get(k, out.get(k)) for k in ("updated_at", "items")})
        out["schema_version"] = SCHEMA_VERSION
        out["items"] = [normalize_item(row) for row in payload.get("items") or [] if normalize_item(row)]
        return out
    except Exception:
        return empty_book()


def save_book(book: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = buy_book_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = empty_book()
    payload["updated_at"] = _utc_now()
    payload["items"] = [normalize_item(row) for row in (book.get("items") or []) if normalize_item(row)]
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def normalize_item(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    symbol = _clean_symbol(row.get("symbol") or row.get("tradingsymbol"))
    if not symbol or len(symbol) > 32:
        return None
    status = str(row.get("status") or "active").strip().lower()
    if status not in {"active", "closed", "paused"}:
        status = "active"
    item_id = str(row.get("id") or "").strip() or uuid.uuid4().hex[:12]
    qty = _f(row.get("quantity") if row.get("quantity") is not None else row.get("qty"))
    if qty is not None and qty <= 0:
        qty = None
    source = str(row.get("source") or "manual").strip().lower()
    if source not in {"manual", "zerodha", "holdings"}:
        source = "manual"
    demat_pnl = _f(row.get("demat_pnl"))
    demat_pnl_pct = _f(row.get("demat_pnl_pct"))
    return {
        "id": item_id,
        "symbol": symbol,
        "entry_price": _f(
            row.get("entry_price")
            if row.get("entry_price") is not None
            else row.get("average_price")
            if row.get("average_price") is not None
            else row.get("avg")
        ),
        "stop_price": _f(row.get("stop_price") if row.get("stop_price") is not None else row.get("stop")),
        "quantity": qty,
        "notes": str(row.get("notes") or "")[:240],
        "status": status,
        "source": source,
        "demat_pnl": demat_pnl,
        "demat_pnl_pct": demat_pnl_pct,
        "added_at": str(row.get("added_at") or _utc_now()),
        "updated_at": str(row.get("updated_at") or _utc_now()),
    }


def list_active(path: Path | None = None) -> list[dict[str, Any]]:
    book = load_book(path)
    return [row for row in book.get("items") or [] if row.get("status") == "active"]


def add_item(
    symbol: str,
    *,
    entry_price: float | None = None,
    stop_price: float | None = None,
    quantity: float | None = None,
    notes: str = "",
    source: str = "manual",
    demat_pnl: float | None = None,
    demat_pnl_pct: float | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    book = load_book(path)
    sym = _clean_symbol(symbol)
    if not sym:
        raise ValueError("invalid symbol")
    items = list(book.get("items") or [])
    existing = next((r for r in items if r.get("symbol") == sym and r.get("status") == "active"), None)
    now = _utc_now()
    src = str(source or "manual").strip().lower()
    if src not in {"manual", "zerodha", "holdings"}:
        src = "manual"
    if existing:
        existing["entry_price"] = entry_price if entry_price is not None else existing.get("entry_price")
        existing["stop_price"] = stop_price if stop_price is not None else existing.get("stop_price")
        if quantity is not None:
            existing["quantity"] = quantity if quantity > 0 else None
        if notes:
            existing["notes"] = str(notes)[:240]
        # Zerodha sync may upgrade source; never downgrade a zerodha row to manual on empty sync.
        if src in {"zerodha", "holdings"} or not existing.get("source"):
            existing["source"] = src
        if demat_pnl is not None:
            existing["demat_pnl"] = demat_pnl
        if demat_pnl_pct is not None:
            existing["demat_pnl_pct"] = demat_pnl_pct
        existing["updated_at"] = now
        item = existing
    else:
        item = normalize_item(
            {
                "symbol": sym,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "quantity": quantity,
                "notes": notes,
                "status": "active",
                "source": src,
                "demat_pnl": demat_pnl,
                "demat_pnl_pct": demat_pnl_pct,
                "added_at": now,
                "updated_at": now,
            }
        )
        assert item is not None
        items.append(item)
    book["items"] = items
    saved = save_book(book, path)
    saved_item = next(r for r in saved["items"] if r["symbol"] == sym and r["status"] == "active")
    return saved_item


def sync_from_holdings(
    *,
    refresh_kite: bool = True,
    path: Path | None = None,
) -> dict[str, Any]:
    """Pull Zerodha/demat holdings into Active Buys for tracking.

    - ``refresh_kite=True`` calls Kite sync first (when connected)
    - Upserts each CNC holding as an active buy (avg → entry, qty preserved)
    - Closes prior ``zerodha``/``holdings`` rows that are no longer in demat
    - Leaves manually added buys untouched unless the same symbol is held
    """
    from product.holdings_book import build_holdings_payload, research_symbol, sync_from_kite

    if refresh_kite:
        holdings_payload = sync_from_kite()
        if not holdings_payload.get("available") and not holdings_payload.get("synced"):
            # Fall back to last saved book so UI can still track offline demat snapshot.
            holdings_payload = build_holdings_payload()
    else:
        holdings_payload = build_holdings_payload()

    holdings = [
        row
        for row in (holdings_payload.get("holdings") or [])
        if isinstance(row, Mapping)
    ]
    held_symbols: set[str] = set()
    upserted: list[dict[str, Any]] = []
    for row in holdings:
        tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "")
        symbol = research_symbol(tradingsymbol) or _clean_symbol(tradingsymbol)
        qty = _f(row.get("quantity") if row.get("quantity") is not None else row.get("qty"))
        t1 = _f(row.get("t1_quantity"))
        total_qty = (qty or 0) + (t1 or 0)
        if total_qty <= 0 or not symbol:
            continue
        held_symbols.add(symbol)
        avg = _f(row.get("average_price") if row.get("average_price") is not None else row.get("avg"))
        item = add_item(
            symbol,
            entry_price=avg,
            quantity=total_qty,
            notes=str(row.get("notes") or f"Zerodha holding ({tradingsymbol or symbol})"),
            source="zerodha",
            demat_pnl=_f(row.get("pnl")),
            demat_pnl_pct=_f(row.get("pnl_pct")),
            path=path,
        )
        upserted.append(item)

    closed: list[str] = []
    book = load_book(path)
    for row in list(book.get("items") or []):
        if row.get("status") != "active":
            continue
        if str(row.get("source") or "") not in {"zerodha", "holdings"}:
            continue
        if str(row.get("symbol") or "") not in held_symbols:
            set_status(str(row.get("id")), "closed", path=path)
            closed.append(str(row.get("symbol")))

    return {
        "accepted": True,
        "synced_from": str(holdings_payload.get("source") or ("kite" if refresh_kite else "file")),
        "holdings_available": bool(holdings_payload.get("available")),
        "holdings_message": str(holdings_payload.get("message") or ""),
        "upserted": len(upserted),
        "symbols": sorted(held_symbols),
        "closed_stale_zerodha": closed,
        "items": upserted,
        "places_orders": False,
        "honesty": (
            "Zerodha holdings were mapped into Active Buys for tracking. "
            "Entry = demat average price. QuantTerm does not place orders."
        ),
    }


def remove_item(item_id: str, *, path: Path | None = None) -> bool:
    book = load_book(path)
    before = len(book.get("items") or [])
    book["items"] = [r for r in (book.get("items") or []) if r.get("id") != item_id]
    if len(book["items"]) == before:
        # Also allow remove-by-symbol for active rows.
        sym = _clean_symbol(item_id)
        book["items"] = [
            r for r in (book.get("items") or [])
            if not (r.get("symbol") == sym and r.get("status") == "active")
        ]
    if len(book["items"]) == before:
        return False
    save_book(book, path)
    return True


def set_status(item_id: str, status: str, *, path: Path | None = None) -> dict[str, Any] | None:
    book = load_book(path)
    status = str(status or "").strip().lower()
    if status not in {"active", "closed", "paused"}:
        raise ValueError("status must be active|closed|paused")
    for row in book.get("items") or []:
        if row.get("id") == item_id or row.get("symbol") == _clean_symbol(item_id):
            row["status"] = status
            row["updated_at"] = _utc_now()
            save_book(book, path)
            return row
    return None


def symbols(path: Path | None = None) -> list[str]:
    return sorted({str(r["symbol"]) for r in list_active(path)})
