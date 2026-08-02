"""
Corporate-Action Adjustment — make historical prices continuous & trustworthy.

NSE bhavcopy is UNADJUSTED, so a 1:1 bonus or a 1→5 split reads as a phantom
crash. `core.data_integrity` DETECTS those gaps; this module REMOVES them when
a real ledger is present.

Design:
  • An event is {symbol, ex_date, factor, type}. `factor` = share-count multiple
    on the ex-date (1:1 bonus → 2.0, 1→5 split → 5.0).
  • Prices BEFORE ex-date are divided by the cumulative factor; volumes multiplied.
  • load_events() reads `logs/ca_events.json`. Absent file → {} (no fake data).
  • This module NEVER invents events from price gaps. Operator/vendor ingest only.
  • Adjustment is applied ON READ so the on-disk store stays raw.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_CA_FILE = Path(__file__).resolve().parent.parent / "logs" / "ca_events.json"
_VALID_TYPES = {"split", "bonus", "consolidation", "dividend"}


def _events_path() -> Path:
    override = os.getenv("QT_CA_EVENTS_FILE")
    return Path(override) if override else _CA_FILE


def events_path() -> Path:
    return _events_path()


def _coerce_rows(raw) -> list:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        nested = raw.get("events") or raw.get("rows") or raw.get("data")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
    return []


def load_events(path=None) -> dict:
    """Read the corporate-action table → {SYMBOL: [event, ...]}.

    Returns {} when the file is absent or unreadable — NEVER a fabricated table.
    """
    import pandas as pd

    p = Path(path) if path else _events_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    for row in _coerce_rows(raw):
        try:
            sym = str(row["symbol"]).strip().upper()
            factor = float(row["factor"])
            ex = pd.Timestamp(row["ex_date"])
            typ = str(row.get("type", "split")).lower()
            if not sym or factor <= 0 or factor == 1.0 or pd.isna(ex):
                continue
            if typ not in _VALID_TYPES:
                continue
            out.setdefault(sym, []).append({"ex_date": ex, "factor": factor, "type": typ})
        except Exception:
            continue
    for sym in out:
        out[sym].sort(key=lambda e: e["ex_date"])
    return out


def ledger_status(path=None) -> dict:
    p = Path(path) if path else _events_path()
    events = load_events(p)
    return {
        "available": p.exists(),
        "path": str(p),
        "symbols": len(events),
        "events": sum(len(v) for v in events.values()),
        "research_grade": bool(events),
    }


def validate_event_rows(rows) -> list[dict]:
    """Return cleaned serialisable event rows; drop invalid entries."""
    import pandas as pd

    cleaned: list[dict] = []
    seen: set[tuple] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            sym = str(row.get("symbol", "")).strip().upper()
            factor = float(row.get("factor"))
            ex = pd.Timestamp(row.get("ex_date"))
            typ = str(row.get("type", "split")).lower()
            if not sym or factor <= 0 or factor == 1.0 or pd.isna(ex):
                continue
            if typ not in _VALID_TYPES:
                continue
            key = (sym, str(ex.date()), round(factor, 8), typ)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "symbol": sym,
                "ex_date": str(ex.date()),
                "factor": float(factor),
                "type": typ,
            })
        except Exception:
            continue
    cleaned.sort(key=lambda r: (r["symbol"], r["ex_date"], r["type"]))
    return cleaned


def write_events(rows, path=None, *, source: str = "operator") -> dict:
    """Atomically write a corporate-action ledger. Never invents events."""
    p = Path(path) if path else _events_path()
    cleaned = validate_event_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "events": cleaned,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    try:
        from data.bhavcopy_store import reload_corporate_actions

        reload_corporate_actions()
    except Exception:
        pass
    return ledger_status(p)


def merge_events(rows, path=None, *, source: str = "operator_merge") -> dict:
    """Merge new rows into the existing ledger (dedupe by symbol/ex_date/factor/type)."""
    p = Path(path) if path else _events_path()
    existing_raw: list = []
    if p.exists():
        try:
            existing_raw = _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            existing_raw = []
    return write_events(list(existing_raw) + list(rows or []), path=p, source=source)


def ingest_from_path(source_path, *, dest=None) -> dict:
    """Ingest an operator-supplied CA JSON/CSV into the canonical ledger."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"corporate-action source not found: {src}")
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".csv":
        import csv
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows = _coerce_rows(json.loads(text))
    return merge_events(rows, path=dest, source=f"ingest:{src.name}")


def adjust_frame(df, events, copy: bool = True):
    """Back-adjust one symbol's OHLCV frame for its corporate actions."""
    if df is None or getattr(df, "empty", True) or not events:
        return (df.copy() if copy else df) if df is not None else df
    import numpy as np
    import pandas as pd

    out = df.copy() if copy else df
    idx = pd.DatetimeIndex(out.index)
    divisor = np.ones(len(out), dtype=float)
    for e in events:
        divisor[idx < e["ex_date"]] *= e["factor"]
    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    for c in price_cols:
        out[c] = out[c].to_numpy(dtype=float) / divisor
    if "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(dtype=float) * divisor
    return out


def is_continuous(df, threshold_pct: float = None) -> bool:
    """True when the (adjusted) close series has NO phantom gap."""
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return True
    from core.data_integrity import phantom_gaps, _GAP_PCT

    thr = threshold_pct if threshold_pct is not None else _GAP_PCT
    return len(phantom_gaps(df["close"].to_numpy(dtype=float), thr)) == 0
