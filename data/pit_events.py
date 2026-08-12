"""Point-in-time corporate event / earnings ledger (official AVAILABLE_AT).

Canonical file: ``logs/pit_events.json``

Each row MUST carry ``available_at`` — when the information became public on the
exchange (broadcast / disseminate time). Never use local scrape ``fetched_at``.

This module never invents events; it stores and serves operator / NSE ingest files.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "pit_events.json"

EVENT_TYPES = (
    "EARNINGS_RESULT",
    "FINANCIAL_RESULT_UPDATE",
    "CORPORATE_ANNOUNCEMENT",
    "OTHER",
)


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("QT_PIT_EVENTS_FILE")
    return Path(override) if override else _DEFAULT_PATH


def _coerce_rows(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        nested = raw.get("rows") or raw.get("events") or raw.get("data") or []
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
    return []


def _parse_available_at(raw: Any) -> str | None:
    """Normalize to ISO date (YYYY-MM-DD). Time-of-day kept in available_at_ts when present."""
    import pandas as pd

    if raw in (None, ""):
        return None
    try:
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            return None
        return str(ts.date())
    except Exception:
        return None


def _parse_available_at_ts(raw: Any) -> str | None:
    import pandas as pd

    if raw in (None, ""):
        return None
    try:
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            return None
        # Prefer timezone-aware UTC ISO when possible
        if ts.tzinfo is None:
            return ts.isoformat()
        return ts.tz_convert("UTC").isoformat()
    except Exception:
        return None


def validate_rows(rows) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        avail = _parse_available_at(
            row.get("available_at")
            or row.get("available_ts")
            or row.get("broadCastDate")
            or row.get("exchdisstime")
            or row.get("an_dt")
            or row.get("published_at")
        )
        if not sym or not avail:
            continue
        et = str(row.get("event_type") or "OTHER").strip().upper()
        if et not in EVENT_TYPES:
            et = "OTHER"
        item: dict[str, Any] = {
            "symbol": sym,
            "available_at": avail,
            "event_type": et,
        }
        ts = _parse_available_at_ts(
            row.get("available_at_ts")
            or row.get("broadCastDate")
            or row.get("exchdisstime")
            or row.get("an_dt")
            or row.get("sort_date")
        )
        if ts:
            item["available_at_ts"] = ts
        for key in (
            "isin", "period", "period_start", "period_end", "relating_to",
            "financial_year", "consolidated", "audited", "headline", "desc",
            "source", "source_url", "seq_id", "event_id", "security_id",
        ):
            if row.get(key) not in (None, ""):
                item[key] = row[key]
        # Stable id for immutability / dedupe
        if not item.get("event_id"):
            blob = json.dumps(
                {k: item.get(k) for k in (
                    "symbol", "available_at", "event_type", "period_end",
                    "consolidated", "seq_id", "headline", "desc",
                )},
                sort_keys=True, default=str,
            )
            item["event_id"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
        cleaned.append(item)
    # Dedupe by event_id keeping first
    seen: set[str] = set()
    out: list[dict] = []
    for row in cleaned:
        eid = row["event_id"]
        if eid in seen:
            continue
        seen.add(eid)
        out.append(row)
    out.sort(key=lambda r: (r["symbol"], r["available_at"], r["event_type"], r["event_id"]))
    return out


def write_events(rows, path: str | Path | None = None, *, source: str = "operator") -> dict:
    p = ledger_path(path)
    cleaned = validate_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "key_requirement": "AVAILABLE_AT",
        "never_uses_fetched_at_as_available_at": True,
        "rows": cleaned,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return ledger_status(p)


def merge_events(rows, path: str | Path | None = None, *, source: str = "operator_merge") -> dict:
    p = ledger_path(path)
    existing: list[dict] = []
    if p.exists():
        try:
            existing = _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            existing = []
    return write_events(list(existing) + list(rows or []), path=p, source=source)


def ingest_from_path(source_path, *, dest=None) -> dict:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"PIT events source not found: {src}")
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".csv":
        import csv
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows = _coerce_rows(json.loads(text))
    cleaned = validate_rows(rows)
    if not cleaned:
        raise ValueError(f"no valid PIT event rows in {src}")
    return merge_events(cleaned, path=dest, source=f"ingest:{src.name}")


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
            "note": "no PIT event ledger on file",
            "by_event_type": {},
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
            "note": f"PIT event ledger unreadable ({exc})",
            "by_event_type": {},
        }
    rows = validate_rows(_coerce_rows(raw))
    source = str(raw.get("source") or "") if isinstance(raw, dict) else ""
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["event_type"]] = by_type.get(r["event_type"], 0) + 1
    dates = [r["available_at"] for r in rows]
    return {
        "available": bool(rows),
        "path": str(p),
        "rows": len(rows),
        "symbols": len({r["symbol"] for r in rows}),
        "research_grade": bool(rows) and not str(source).upper().startswith("SAMPLE"),
        "source": source or "operator",
        "note": "",
        "generated_at": (raw.get("generated_at") if isinstance(raw, dict) else "") or "",
        "by_event_type": by_type,
        "date_range": [min(dates), max(dates)] if dates else [None, None],
        "key_requirement": "AVAILABLE_AT",
    }


def get_events(
    symbol: str | None,
    as_of,
    *,
    path: str | Path | None = None,
    event_type: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Return events with ``available_at <= as_of`` (optional symbol / type / since)."""
    import pandas as pd

    p = ledger_path(path)
    if not p.exists():
        return []
    try:
        rows = validate_rows(_coerce_rows(json.loads(p.read_text(encoding="utf-8"))))
    except Exception:
        return []
    try:
        asof = str(pd.Timestamp(as_of).date())
    except Exception:
        asof = str(as_of).strip()[:10]
    since_d = None
    if since:
        try:
            since_d = str(pd.Timestamp(since).date())
        except Exception:
            since_d = str(since).strip()[:10]
    sym = str(symbol or "").strip().upper() or None
    et = str(event_type or "").strip().upper() or None
    out = []
    for row in rows:
        if sym and row["symbol"] != sym:
            continue
        if et and row["event_type"] != et:
            continue
        avail = row["available_at"]
        if avail > asof:
            continue
        if since_d and avail < since_d:
            continue
        out.append(dict(row))
    return out


def content_hash(path: str | Path | None = None) -> str | None:
    p = ledger_path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()
