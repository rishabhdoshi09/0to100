"""Point-in-time valuation ledger for research (operator / vendor supplied).

Canonical file: ``logs/pit_valuations.json``

Each row MUST carry ``available_ts`` (the date the figures became public).
``BhavDataProvider.valuation`` only returns a row when ``available_ts <= as_of``.
Never map screener ``fetched_at`` into ``available_ts`` — that would leak look-ahead.

This module never invents fundamentals; it only stores and serves operator files.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "pit_valuations.json"
_NUMERIC = (
    "pe", "price_to_sales", "ev_to_sales", "market_cap_cr",
    "sales_growth_pct", "earnings_growth_pct", "pe_percentile_own", "age_days",
)


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("QT_PIT_VALUATIONS_FILE")
    return Path(override) if override else _DEFAULT_PATH


def _coerce_rows(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        nested = raw.get("rows") or raw.get("valuations") or raw.get("data") or []
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
    return []


def validate_rows(rows) -> list[dict]:
    import pandas as pd

    cleaned: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            sym = str(row.get("symbol", "")).strip().upper()
            avail = pd.Timestamp(row.get("available_ts") or row.get("as_of") or row.get("published_at"))
            if not sym or pd.isna(avail):
                continue
            item: dict[str, Any] = {"symbol": sym, "available_ts": str(avail.date())}
            for key in _NUMERIC:
                if row.get(key) in (None, ""):
                    continue
                item[key] = float(row[key])
            cleaned.append(item)
        except Exception:
            continue
    cleaned.sort(key=lambda r: (r["symbol"], r["available_ts"]))
    return cleaned


def write_valuations(rows, path: str | Path | None = None, *, source: str = "operator") -> dict:
    p = ledger_path(path)
    cleaned = validate_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": cleaned,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return ledger_status(p)


def merge_valuations(rows, path: str | Path | None = None, *, source: str = "operator_merge") -> dict:
    p = ledger_path(path)
    existing: list[dict] = []
    if p.exists():
        try:
            existing = _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            existing = []
    return write_valuations(list(existing) + list(rows or []), path=p, source=source)


def ingest_from_path(source_path, *, dest=None) -> dict:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"PIT valuations source not found: {src}")
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".csv":
        import csv
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows = _coerce_rows(json.loads(text))
    cleaned = validate_rows(rows)
    if not cleaned:
        raise ValueError(f"no valid PIT valuation rows in {src}")
    return merge_valuations(cleaned, path=dest, source=f"ingest:{src.name}")


def ledger_status(path: str | Path | None = None) -> dict:
    p = ledger_path(path)
    if not p.exists():
        return {
            "available": False,
            "path": str(p),
            "rows": 0,
            "symbols": 0,
            "research_grade": False,
            "source": "",
            "note": "no PIT valuation ledger on file",
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": True,
            "path": str(p),
            "rows": 0,
            "symbols": 0,
            "research_grade": False,
            "source": "",
            "note": f"PIT valuation ledger unreadable ({exc})",
        }
    rows = validate_rows(_coerce_rows(raw))
    source = ""
    if isinstance(raw, dict):
        source = str(raw.get("source") or "")
    return {
        "available": bool(rows),
        "path": str(p),
        "rows": len(rows),
        "symbols": len({r["symbol"] for r in rows}),
        "research_grade": bool(rows) and not str(source).upper().startswith("SAMPLE"),
        "source": source or "operator",
        "note": "",
        "generated_at": (raw.get("generated_at") if isinstance(raw, dict) else "") or "",
    }


def get_valuation(symbol: str, as_of, path: str | Path | None = None) -> dict | None:
    """Return the latest valuation for ``symbol`` with ``available_ts <= as_of``.

    Never returns a future-published row. Missing ledger → None.
    """
    import pandas as pd

    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    p = ledger_path(path)
    if not p.exists():
        return None
    try:
        rows = validate_rows(_coerce_rows(json.loads(p.read_text(encoding="utf-8"))))
    except Exception:
        return None
    asof = pd.Timestamp(as_of).date()
    best = None
    for row in rows:
        if row["symbol"] != sym:
            continue
        avail = pd.Timestamp(row["available_ts"]).date()
        if avail <= asof and (best is None or avail >= pd.Timestamp(best["available_ts"]).date()):
            best = dict(row)
    return best
