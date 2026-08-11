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
  • Only share-count events (split / bonus / consolidation) adjust prices.
    Cash dividends are rejected — they are not a share multiple.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CA_FILE = _ROOT / "logs" / "ca_events.json"
_TODO_FILE = _ROOT / "logs" / "ca_events.todo.csv"
_VERIFY_CACHE = _ROOT / "logs" / "ca_verify_cache.json"
# Share-count adjustments only. Cash dividends must NOT use this factor path.
_VALID_TYPES = {"split", "bonus", "consolidation"}
_REJECTED_TYPES = {"dividend"}
_VERIFY_CACHE_TTL_S = float(os.getenv("QT_CA_VERIFY_CACHE_TTL", "900") or 900)


def _events_path() -> Path:
    override = os.getenv("QT_CA_EVENTS_FILE")
    return Path(override) if override else _CA_FILE


def events_path() -> Path:
    return _events_path()


def todo_path() -> Path:
    override = os.getenv("QT_CA_TODO_FILE")
    return Path(override) if override else _TODO_FILE


def _verify_cache_path() -> Path:
    override = os.getenv("QT_CA_VERIFY_CACHE_FILE")
    return Path(override) if override else _VERIFY_CACHE


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
            if typ in _REJECTED_TYPES or typ not in _VALID_TYPES:
                continue
            out.setdefault(sym, []).append({"ex_date": ex, "factor": factor, "type": typ})
        except Exception:
            continue
    for sym in out:
        out[sym].sort(key=lambda e: e["ex_date"])
    return out


def _count_rejected(path=None) -> dict[str, int]:
    """Count rows dropped (e.g. cash dividend) — honesty for operators."""
    p = Path(path) if path else _events_path()
    if not p.exists():
        return {"dividend": 0, "invalid": 0}
    try:
        rows = _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return {"dividend": 0, "invalid": 0}
    rejected = {"dividend": 0, "invalid": 0}
    for row in rows:
        if not isinstance(row, dict):
            rejected["invalid"] += 1
            continue
        typ = str(row.get("type", "split")).lower()
        if typ in _REJECTED_TYPES:
            rejected["dividend"] += 1
        elif typ not in _VALID_TYPES:
            rejected["invalid"] += 1
    return rejected


def _read_verify_cache() -> dict | None:
    p = _verify_cache_path()
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("checked_at_unix") or 0)
        if age < 0 or age > _VERIFY_CACHE_TTL_S:
            return None
        return payload
    except Exception:
        return None


def _write_verify_cache(payload: dict) -> None:
    p = _verify_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["checked_at_unix"] = time.time()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def refresh_adjustment_verify(*, sample: int = 80) -> dict:
    """Run verify_ca_adjustment and cache the result for product readiness."""
    from core.data_integrity import verify_ca_adjustment

    result = verify_ca_adjustment(sample=int(sample))
    _write_verify_cache(result)
    return result


def ledger_status(path=None, *, verify: bool = False, sample: int = 80) -> dict:
    """Fast ledger facts + optional / cached adjustment verification.

    ``research_grade`` means an operator ledger with ≥1 share-count event exists.
    It does NOT invent events. Product READY still wants ``adjustment_verified``.
    """
    p = Path(path) if path else _events_path()
    events = load_events(p)
    n_events = sum(len(v) for v in events.values())
    rejected = _count_rejected(p)
    status: dict = {
        "available": p.exists(),
        "path": str(p),
        "symbols": len(events),
        "events": n_events,
        # Events on disk ≠ research-grade prices. Verification must PASS.
        "research_grade": False,
        "share_adjust_types": sorted(_VALID_TYPES),
        "rejected_types": rejected,
        "todo_path": str(todo_path()),
        "never_invents": True,
        "honesty": (
            "QuantTerm never invents corporate actions from price gaps. "
            "Fill factor/type from official NSE filings into the todo CSV, then ca-ingest."
        ),
    }
    if verify:
        try:
            v = refresh_adjustment_verify(sample=sample)
        except Exception as exc:
            v = {"passed": False, "note": f"verify failed: {exc}", "ca_events_loaded": len(events)}
    else:
        v = _read_verify_cache() or {}
    if v:
        status["adjustment_verified"] = bool(v.get("passed"))
        status["gap_rate"] = v.get("gap_rate")
        status["verify_checked"] = v.get("checked")
        status["verify_still_flagged"] = v.get("still_flagged")
        status["verify_note"] = v.get("note")
        status["verify_flagged"] = list(v.get("flagged") or [])[:10]
    else:
        status["adjustment_verified"] = False
        status["verify_note"] = "verify not run yet — python main.py ca-ingest --verify"
    # Earn ledger research_grade only after share-count events exist AND verify PASS.
    status["research_grade"] = bool(n_events) and bool(status.get("adjustment_verified"))
    todo = todo_path()
    status["todo_available"] = todo.exists()
    if todo.exists():
        try:
            # header + rows
            lines = todo.read_text(encoding="utf-8").strip().splitlines()
            status["todo_gaps"] = max(0, len(lines) - 1)
        except Exception:
            status["todo_gaps"] = None
    else:
        status["todo_gaps"] = 0
    if not status["available"] or n_events < 1:
        status["next_action"] = (
            "python main.py ca-ingest --from-gaps  "
            "→ fill factor/type from NSE filings in logs/ca_events.todo.csv  "
            "→ python main.py ca-ingest --source logs/ca_events.todo.csv"
        )
    elif not status.get("adjustment_verified"):
        status["next_action"] = (
            "Ledger has events but adjustment is not verified — "
            "python main.py ca-ingest --verify  (or add more rows from --from-gaps)"
        )
    else:
        status["next_action"] = "NONE"
    return status


def validate_event_rows(rows) -> list[dict]:
    """Return cleaned serialisable event rows; drop invalid / cash-dividend entries."""
    import pandas as pd

    cleaned: list[dict] = []
    seen: set[tuple] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            sym = str(row.get("symbol", "")).strip().upper()
            factor_raw = row.get("factor")
            if factor_raw is None or str(factor_raw).strip() == "":
                continue
            factor = float(factor_raw)
            ex = pd.Timestamp(row.get("ex_date") or row.get("ex_date_guess"))
            typ = str(row.get("type", "split")).lower().strip()
            if not sym or factor <= 0 or factor == 1.0 or pd.isna(ex):
                continue
            if typ in _REJECTED_TYPES or typ not in _VALID_TYPES:
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
        "note": "Share-count events only (split/bonus/consolidation). Cash dividends rejected.",
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
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows = _coerce_rows(json.loads(text))
    return merge_events(rows, path=dest, source=f"ingest:{src.name}")


def export_gap_todo(
    *,
    sample: int = 400,
    path=None,
    threshold_pct: float | None = None,
) -> dict:
    """Scan RAW bhav for phantom gaps → operator TODO CSV.

    Never invents ``factor`` or ``type``. Only records where a gap was seen and
    a guessed ex-date (the later bar). Operator fills factor/type from NSE
    filings, then ``ca-ingest --source`` the filled CSV.
    """
    from core.data_integrity import _GAP_PCT, phantom_gaps

    thr = float(threshold_pct) if threshold_pct is not None else float(_GAP_PCT)
    out_path = Path(path) if path else todo_path()
    existing = load_events()
    rows: list[dict] = []
    scanned = 0
    try:
        from data.bhavcopy_store import iter_raw_frames, store_symbols

        symbols = store_symbols()[: max(1, int(sample))]
        wanted = set(symbols)
        for sym, df in iter_raw_frames():
            if sym not in wanted:
                continue
            scanned += 1
            if df is None or getattr(df, "empty", True) or "close" not in df.columns:
                continue
            # Skip symbols that already have at least one ledger event —
            # operator can still re-export with a larger sample later.
            if existing.get(sym):
                continue
            gaps = phantom_gaps(df["close"].to_numpy(dtype=float), thr)
            if not gaps:
                continue
            for g in gaps[:3]:
                idx = int(g["index"])
                try:
                    ex_guess = str(df.index[idx].date())
                except Exception:
                    ex_guess = ""
                rows.append({
                    "symbol": sym,
                    "ex_date": ex_guess,
                    "factor": "",  # OPERATOR MUST FILL — never invented
                    "type": "",   # split | bonus | consolidation
                    "pct_move": g.get("pct"),
                    "note": "Fill factor+type from NSE CA filing; do not guess",
                })
            if len(rows) >= 500:
                break
    except Exception as exc:
        return {
            "written": False,
            "path": str(out_path),
            "gaps": 0,
            "scanned": scanned,
            "error": str(exc),
            "never_invents": True,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "ex_date", "factor", "type", "pct_move", "note"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "written": True,
        "path": str(out_path),
        "gaps": len(rows),
        "scanned": scanned,
        "sample": int(sample),
        "threshold_pct": thr,
        "never_invents": True,
        "next_action": (
            f"Fill factor+type in {out_path.name} from official NSE filings, then: "
            f"python main.py ca-ingest --source {out_path}"
        ),
    }


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
        typ = str(e.get("type", "split")).lower()
        if typ in _REJECTED_TYPES or typ not in _VALID_TYPES:
            continue
        divisor[idx < e["ex_date"]] *= float(e["factor"])
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
