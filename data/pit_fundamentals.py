"""Point-in-time fundamentals ledger (statement metrics with AVAILABLE_AT).

Canonical file: ``logs/pit_fundamentals.json``

Each row MUST carry ``available_at`` — the public broadcast/filing date of the
underlying result. Never map screener ``fetched_at`` into ``available_at``.

Metrics are stored only when sourced from official NSE result XBRL (or an
operator file that already carries honest available_at). Missing stays missing.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "pit_fundamentals.json"

_NUMERIC = (
    "revenue_from_operations",
    "other_income",
    "operating_profit",
    "profit_before_tax",
    "profit_after_tax",
    "comprehensive_income",
    "basic_eps",
    "diluted_eps",
    "face_value",
    "paid_up_equity_capital",
    "debt_equity_ratio",
)


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("QT_PIT_FUNDAMENTALS_FILE")
    return Path(override) if override else _DEFAULT_PATH


def _coerce_rows(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        nested = raw.get("rows") or raw.get("fundamentals") or raw.get("data") or []
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
    return []


def _parse_available_at(raw: Any) -> str | None:
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


def validate_rows(rows) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        # Hard refuse mapping fetched_at → available_at
        if "fetched_at" in row and not (
            row.get("available_at") or row.get("available_ts") or row.get("broadCastDate")
        ):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        avail = _parse_available_at(
            row.get("available_at")
            or row.get("available_ts")
            or row.get("broadCastDate")
            or row.get("exchdisstime")
            or row.get("filingDate")
        )
        if not sym or not avail:
            continue
        item: dict[str, Any] = {"symbol": sym, "available_at": avail}
        for key in (
            "isin", "period", "period_start", "period_end", "relating_to",
            "financial_year", "consolidated", "audited", "source", "source_url",
            "xbrl_url", "seq_id", "security_id", "unit", "currency",
            "revision_status", "first_known_at", "source_id", "source_hash",
            "filing_id", "superseded_by_row_id",
            "period_kind", "quarterly_usable", "consol_basis", "reporting_frequency",
            "parser_version", "raw_hash", "ingested_at", "field_quality",
        ):
            if row.get(key) not in (None, ""):
                item[key] = row[key]
        if "source_hash" not in item:
            blob_src = json.dumps(
                {k: row.get(k) for k in (
                    "symbol", "available_at", "period_end", "xbrl_url", "seq_id",
                    "revenue_from_operations", "profit_after_tax", "basic_eps",
                )},
                sort_keys=True, default=str,
            )
            item["source_hash"] = hashlib.sha256(blob_src.encode()).hexdigest()[:16]
        item.setdefault("first_known_at", avail)
        item.setdefault("revision_status", row.get("revision_status") or "original")
        n_metrics = 0
        for key in _NUMERIC:
            if row.get(key) in (None, ""):
                continue
            try:
                item[key] = float(row[key])
                n_metrics += 1
            except (TypeError, ValueError):
                continue
        if n_metrics == 0:
            # Row without metrics is not a fundamentals observation
            continue
        item["n_metrics"] = n_metrics
        blob = json.dumps(
            {k: item.get(k) for k in (
                "symbol", "available_at", "period_end", "consolidated", "seq_id", "xbrl_url",
            )},
            sort_keys=True, default=str,
        )
        item["row_id"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
        cleaned.append(item)
    seen: set[str] = set()
    out: list[dict] = []
    for row in cleaned:
        if row["row_id"] in seen:
            continue
        seen.add(row["row_id"])
        out.append(row)
    out.sort(key=lambda r: (r["symbol"], r["available_at"], r.get("period_end") or "", r["row_id"]))
    return out


def write_fundamentals(rows, path: str | Path | None = None, *, source: str = "operator") -> dict:
    p = ledger_path(path)
    cleaned = validate_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "key_requirement": "AVAILABLE_AT",
        "never_uses_fetched_at_as_available_at": True,
        "numeric_fields": list(_NUMERIC),
        "rows": cleaned,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return ledger_status(p)


def merge_fundamentals(rows, path: str | Path | None = None, *, source: str = "operator_merge") -> dict:
    p = ledger_path(path)
    existing: list[dict] = []
    if p.exists():
        try:
            existing = _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            existing = []
    return write_fundamentals(list(existing) + list(rows or []), path=p, source=source)


def ingest_from_path(source_path, *, dest=None) -> dict:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"PIT fundamentals source not found: {src}")
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".csv":
        import csv
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows = _coerce_rows(json.loads(text))
    cleaned = validate_rows(rows)
    if not cleaned:
        raise ValueError(f"no valid PIT fundamentals rows in {src}")
    return merge_fundamentals(cleaned, path=dest, source=f"ingest:{src.name}")


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
            "note": "no PIT fundamentals ledger on file",
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
            "note": f"PIT fundamentals ledger unreadable ({exc})",
        }
    rows = validate_rows(_coerce_rows(raw))
    source = str(raw.get("source") or "") if isinstance(raw, dict) else ""
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
        "date_range": [min(dates), max(dates)] if dates else [None, None],
        "key_requirement": "AVAILABLE_AT",
        "metric_coverage": {
            k: sum(1 for r in rows if r.get(k) is not None) for k in _NUMERIC
        },
    }


def _asof_str(as_of) -> str:
    import pandas as pd
    try:
        return str(pd.Timestamp(as_of).date())
    except Exception:
        return str(as_of).strip()[:10]


_ROWS_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _load_rows(path: str | Path | None = None) -> list[dict]:
    p = ledger_path(path)
    if not p.exists():
        return []
    key = str(p.resolve())
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    hit = _ROWS_CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        rows = validate_rows(_coerce_rows(json.loads(p.read_text(encoding="utf-8"))))
    except Exception:
        rows = []
    _ROWS_CACHE[key] = (mtime, rows)
    return rows


def known_as_of(symbol: str, as_of, path: str | Path | None = None) -> list[dict]:
    """Every fundamentals row for symbol with available_at <= as_of (original + later restatements)."""
    sym = str(symbol or "").strip().upper()
    asof = _asof_str(as_of)
    out = []
    for row in _load_rows(path):
        if row["symbol"] != sym:
            continue
        if row["available_at"] <= asof:
            out.append(dict(row))
    return out


def get_period_as_of(
    symbol: str,
    period_end,
    as_of,
    path: str | Path | None = None,
) -> dict | None:
    """Version of one fiscal period that was public as of ``as_of``.

    A later restatement (newer available_at) is invisible before it was filed.
    """
    pe = _asof_str(period_end)
    best = None
    for row in known_as_of(symbol, as_of, path=path):
        if _asof_str(row.get("period_end") or "") != pe:
            continue
        if best is None or row["available_at"] > best["available_at"]:
            best = dict(row)
        elif row["available_at"] == best["available_at"]:
            # Canonical preference: CONSOLIDATED over STANDALONE on the same day.
            if str(row.get("consol_basis") or "") == "CONSOLIDATED" and str(
                best.get("consol_basis") or ""
            ) != "CONSOLIDATED":
                best = dict(row)
    return best


def get_fundamentals(symbol: str, as_of, path: str | Path | None = None) -> dict | None:
    """Latest fundamentals row for symbol with available_at <= as_of."""
    asof = _asof_str(as_of)
    best = None
    for row in known_as_of(symbol, as_of, path=path):
        avail = row["available_at"]
        if avail > asof:
            continue
        if best is None or avail > best["available_at"]:
            best = dict(row)
        elif avail == best["available_at"]:
            row_pe = str(row.get("period_end") or "")
            best_pe = str(best.get("period_end") or "")
            row_c = str(row.get("consol_basis") or "") == "CONSOLIDATED"
            best_c = str(best.get("consol_basis") or "") == "CONSOLIDATED"
            if row_pe > best_pe or (row_pe == best_pe and row_c and not best_c):
                best = dict(row)
    return best


def get_prior_period(
    current: dict | None,
    as_of,
    path: str | Path | None = None,
) -> dict | None:
    """Most recent earlier period for the same symbol known by ``as_of``."""
    if not current:
        return None
    pe = str(current.get("period_end") or "")
    best = None
    for row in known_as_of(current["symbol"], as_of, path=path):
        other = str(row.get("period_end") or "")
        if not other or other >= pe:
            continue
        if best is None or other > str(best.get("period_end") or ""):
            best = dict(row)
        elif other == str(best.get("period_end") or "") and row["available_at"] >= best["available_at"]:
            best = dict(row)
    return best


def fundamentals_with_ratios(symbol: str, as_of, path: str | Path | None = None) -> dict | None:
    """Current known row + read-time ratios + lineage. No surprise field."""
    from data.pit_ratios import derive_ratios, lineage_for

    cur = get_fundamentals(symbol, as_of, path=path)
    if not cur:
        return None
    prior = get_prior_period(cur, as_of, path=path)
    derived = derive_ratios(cur, prior)
    return {
        "as_of": _asof_str(as_of),
        "current": cur,
        "prior": prior,
        "ratios": derived,
        "lineage": {
            name: lineage_for(name, derived)
            for name in ("revenue_growth", "eps_growth", "pat_margin")
        },
    }


def content_hash(path: str | Path | None = None) -> str | None:
    p = ledger_path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()
