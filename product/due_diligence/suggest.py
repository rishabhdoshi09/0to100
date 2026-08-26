"""Ticker / company-name typeahead for the manual Stock Investigator."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FALLBACK_CSV = ROOT / "data" / "nse_symbols_fallback.csv"


@lru_cache(maxsize=1)
def _fallback_names() -> dict[str, str]:
    out: dict[str, str] = {}
    if not FALLBACK_CSV.exists():
        return out
    try:
        with FALLBACK_CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                symbol = str(row.get("SYMBOL") or "").strip().upper()
                name = str(row.get("NAME") or "").strip()
                if symbol:
                    out[symbol] = name or symbol
    except OSError:
        return out
    return out


def _universe_names() -> dict[str, str]:
    names = dict(_fallback_names())
    try:
        from data.nse_universe import get_nse_universe_with_names
        names.update({str(k).upper(): str(v) for k, v in (get_nse_universe_with_names() or {}).items()})
    except Exception:
        pass
    try:
        from product.scan_store import load_scan
        for row in list((load_scan() or {}).get("records") or []):
            symbol = str(row.get("symbol") or "").upper()
            company = str(row.get("company") or "").strip()
            if symbol and company:
                names[symbol] = company
    except Exception:
        pass
    try:
        from product.long_term_store import load_long_term_scan
        for row in list((load_long_term_scan() or {}).get("records") or []):
            symbol = str(row.get("symbol") or "").upper()
            company = str(row.get("company") or "").strip()
            if symbol and company:
                names[symbol] = company
    except Exception:
        pass
    return names


def suggest_tickers(query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """ICICI → ICICIBANK — ICICI Bank Ltd. Empty query → empty list."""
    needle = str(query or "").strip()
    if len(needle) < 2:
        return []
    q = needle.upper()
    ql = needle.lower()
    scored: list[tuple[int, str, str]] = []
    for symbol, name in _universe_names().items():
        name_u = name.upper()
        name_l = name.lower()
        if q == symbol:
            rank = 0
        elif symbol.startswith(q):
            rank = 1
        elif q in symbol:
            rank = 2
        elif ql in name_l or q in name_u:
            rank = 3
        else:
            continue
        scored.append((rank, symbol, name or symbol))
    scored.sort(key=lambda item: (item[0], item[1]))
    out: list[dict[str, Any]] = []
    seen = set()
    for rank, symbol, name in scored:
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append({
            "symbol": symbol,
            "company": name,
            "label": f"{symbol} — {name}",
            "match": {0: "exact", 1: "prefix", 2: "symbol", 3: "name"}[rank],
        })
        if len(out) >= limit:
            break
    return out
