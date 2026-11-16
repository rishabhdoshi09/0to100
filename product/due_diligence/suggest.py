"""Ticker / company-name typeahead for the manual Stock Investigator."""
from __future__ import annotations

import csv
import threading
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


#: Type-ahead must answer as you type. Rebuilding the index parsed the full
#: saved scan and long-term scan on every keystroke (9 s cold on the owner's
#: Mac, 2026-11-16). The merged index is kept until one of its sources
#: changes, or 10 minutes pass (the universe list has no file to watch).
_INDEX_TTL_S = 600.0
_INDEX_LOCK = threading.Lock()
_INDEX: dict[str, Any] = {"key": None, "at": 0.0, "names": None}


def _index_key() -> tuple:
    from product.projection_cache import file_identity
    paths = []
    try:
        from product.scan_store import DEFAULT_SCAN_PATH
        paths.append(DEFAULT_SCAN_PATH)
    except Exception:
        pass
    try:
        from product.long_term_store import DEFAULT_LONG_TERM_PATH
        paths.append(DEFAULT_LONG_TERM_PATH)
    except Exception:
        pass
    return file_identity(*paths)


def _universe_names() -> dict[str, str]:
    import time as _time

    key = _index_key()
    with _INDEX_LOCK:
        cached = _INDEX["names"]
        if cached is not None and _INDEX["key"] == key and _time.monotonic() - _INDEX["at"] < _INDEX_TTL_S:
            return cached
        names = _build_universe_names()
        _INDEX.update(key=key, at=_time.monotonic(), names=names)
        return names


def warm_suggest_index() -> int:
    """Build the type-ahead index ahead of the first keystroke (API startup)."""
    return len(_universe_names())


def _build_universe_names() -> dict[str, str]:
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
    nifty: set[str] = set()
    try:
        from data.nse_universe import NIFTY50
        nifty = {str(s).upper() for s in NIFTY50}
    except Exception:
        nifty = {"ICICIBANK", "HDFCBANK", "TCS", "INFY", "RELIANCE"}
    scored: list[tuple[int, int, str, str]] = []
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
        scored.append((rank, 0 if symbol in nifty else 1, symbol, name or symbol))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    out: list[dict[str, Any]] = []
    seen = set()
    for rank, _boost, symbol, name in scored:
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
