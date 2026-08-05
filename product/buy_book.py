"""Active Buys book — stocks you are buying / holding by intent.

Distinct from:
  • watchlist (candidates)
  • demat holdings (broker-owned)
  • paper portfolio (simulated)

User-authored only. Never invents symbols, prices, or fills.
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
            "Active Buys is your list of stocks you are buying or watching as buys. "
            "Health warnings use official history + live LTP when available. "
            "Not a buy/sell ticket and never places orders."
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
    return {
        "id": item_id,
        "symbol": symbol,
        "entry_price": _f(row.get("entry_price") if row.get("entry_price") is not None else row.get("avg")),
        "stop_price": _f(row.get("stop_price") if row.get("stop_price") is not None else row.get("stop")),
        "notes": str(row.get("notes") or "")[:240],
        "status": status,
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
    notes: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    book = load_book(path)
    sym = _clean_symbol(symbol)
    if not sym:
        raise ValueError("invalid symbol")
    items = list(book.get("items") or [])
    existing = next((r for r in items if r.get("symbol") == sym and r.get("status") == "active"), None)
    now = _utc_now()
    if existing:
        existing["entry_price"] = entry_price if entry_price is not None else existing.get("entry_price")
        existing["stop_price"] = stop_price if stop_price is not None else existing.get("stop_price")
        if notes:
            existing["notes"] = str(notes)[:240]
        existing["updated_at"] = now
        item = existing
    else:
        item = normalize_item(
            {
                "symbol": sym,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "notes": notes,
                "status": "active",
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
