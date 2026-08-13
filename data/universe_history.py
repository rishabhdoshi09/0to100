"""Point-in-time NSE universe membership ledger.

Canonical file: ``logs/universe_history.json``
Each membership row: ``{symbol, listed, delisted?}``.

Without a ledger, callers must use today's survivors and treat results as
survivorship-biased. This module can bootstrap an *inferred* ledger from local
bhav coverage (first/last session dates). That clears the missing-source gap and
is better than survivors-only, but it is NOT a substitute for an official NSE
listing/delisting archive — consumers see ``source=bhav_inferred`` and
``research_grade=False`` in metadata.

Official / operator archives are ingested via ``ingest_from_path`` (never invented).
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "universe_history.json"
_INFERRED_SOURCES = frozenset({"", "bhav_inferred"})


def history_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("QT_UNIVERSE_HISTORY_FILE")
    return Path(override) if override else _DEFAULT_PATH


def is_research_grade_source(source: str | None) -> bool:
    src = str(source or "").strip()
    if not src or src in _INFERRED_SOURCES:
        return False
    if src.startswith("bhav_"):
        return False
    return True


def _coerce_payload(raw: Any) -> tuple[list[dict], dict]:
    meta: dict = {}
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)], meta
    if isinstance(raw, dict):
        meta = {
            "schema_version": raw.get("schema_version"),
            "source": raw.get("source"),
            "note": raw.get("note"),
            "generated_at": raw.get("generated_at"),
        }
        rows = raw.get("rows") or raw.get("symbols") or raw.get("membership") or []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], meta
    return [], meta


def validate_membership_rows(rows) -> list[dict]:
    import pandas as pd

    cleaned: list[dict] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            sym = str(row.get("symbol", "")).strip().upper()
            listed = pd.Timestamp(row.get("listed") or row.get("listing_date") or row.get("list_date"))
            if not sym or pd.isna(listed):
                continue
            item = {"symbol": sym, "listed": str(listed.date())}
            delisted_raw = row.get("delisted") or row.get("delisting_date") or row.get("delist_date")
            if delisted_raw:
                delisted = pd.Timestamp(delisted_raw)
                if pd.isna(delisted) or delisted.date() <= listed.date():
                    continue
                item["delisted"] = str(delisted.date())
            # Last write wins per symbol.
            if sym in seen:
                cleaned = [r for r in cleaned if r["symbol"] != sym]
            seen.add(sym)
            cleaned.append(item)
        except Exception:
            continue
    cleaned.sort(key=lambda r: r["symbol"])
    return cleaned


def write_universe_history(
    rows,
    path: str | Path | None = None,
    *,
    source: str = "operator",
    note: str = "",
) -> dict:
    p = history_path(path)
    cleaned = validate_membership_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "note": note,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": cleaned,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return ledger_status(p)


def merge_universe_history(
    rows,
    path: str | Path | None = None,
    *,
    source: str = "operator_merge",
    note: str = "",
) -> dict:
    p = history_path(path)
    existing: list[dict] = []
    if p.exists():
        try:
            existing, _meta = _coerce_payload(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            existing = []
    return write_universe_history(
        list(existing) + list(rows or []),
        path=p,
        source=source,
        note=note,
    )


def ingest_from_path(source_path, *, dest=None, source: str | None = None, note: str = "") -> dict:
    """Ingest an operator/NSE listing archive (JSON or CSV) into the canonical ledger.

    Never invents membership. Replaces a non-research-grade (bhav-inferred) ledger
    outright so inferred dates are not mixed with official rows. Merges into an
    existing research-grade ledger.
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"universe history source not found: {src}")
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".csv":
        import csv
        from io import StringIO

        rows = list(csv.DictReader(StringIO(text)))
    else:
        rows, _meta = _coerce_payload(json.loads(text))
    cleaned = validate_membership_rows(rows)
    if not cleaned:
        raise ValueError(f"no valid membership rows in {src}")
    dest_path = history_path(dest)
    src_label = source or f"ingest:{src.name}"
    note_text = note or f"Ingested from {src.name}"
    if dest_path.exists():
        existing = ledger_status(dest_path)
        if existing.get("research_grade"):
            return merge_universe_history(
                cleaned,
                path=dest_path,
                source=src_label,
                note=note_text,
            )
    return write_universe_history(
        cleaned,
        path=dest_path,
        source=src_label,
        note=note_text,
    )


def ledger_status(path: str | Path | None = None) -> dict:
    p = history_path(path)
    if not p.exists():
        return {
            "available": False,
            "path": str(p),
            "rows": 0,
            "survivorship_complete": False,
            "source": "",
            "note": "no membership history on file",
            "research_grade": False,
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": True,
            "path": str(p),
            "rows": 0,
            "survivorship_complete": False,
            "source": "",
            "note": f"membership history unreadable ({exc})",
            "research_grade": False,
        }
    rows, meta = _coerce_payload(raw)
    cleaned = validate_membership_rows(rows)
    source = str(meta.get("source") or ("operator" if isinstance(raw, list) else ""))
    note = str(meta.get("note") or "")
    if source == "bhav_inferred" and not note:
        note = (
            "Membership inferred from local bhav first/last sessions — better than "
            "today's survivors only, but not an official NSE listing/delisting archive."
        )
    completeness = raw.get("completeness") if isinstance(raw, dict) else None
    if not isinstance(completeness, dict):
        completeness = {}

    has_any_delist = any(bool(r.get("delisted")) for r in cleaned)
    survivor_only = bool(completeness.get("reconstructed_from_survivors_only", False))

    if "survivorship_complete" in completeness:
        survivorship_complete = bool(completeness.get("survivorship_complete"))
    elif source == "bhav_inferred":
        # Operational PIT intervals from local bhav — NOT research-grade.
        survivorship_complete = bool(cleaned)
    elif is_research_grade_source(source) and has_any_delist and not survivor_only:
        # Official/operator archive that includes delisting evidence.
        survivorship_complete = True
    else:
        # Fail closed for official listing-only masters (e.g. EQUITY_L survivors).
        survivorship_complete = False

    research_grade = (
        bool(cleaned)
        and is_research_grade_source(source)
        and survivorship_complete
        and not survivor_only
        and has_any_delist
    )
    return {
        "available": True,
        "path": str(p),
        "rows": len(cleaned),
        "survivorship_complete": survivorship_complete,
        "source": source or "operator",
        "note": note,
        "research_grade": research_grade,
        "generated_at": meta.get("generated_at") or "",
        "completeness": completeness,
    }


def build_from_bhav(
    *,
    path: str | Path | None = None,
    inactive_after_days: int = 30,
    min_sessions: int = 5,
    force: bool = False,
) -> dict:
    """Bootstrap universe history from local official bhav coverage.

    listed = first session in store; delisted set when last session is older than
    the store's latest day by ``inactive_after_days``. Never invents symbols that
    are not present in the local bhav cache. Never overwrites a research-grade ledger.
    """
    dest = history_path(path)
    if dest.exists():
        status = ledger_status(dest)
        # Never overwrite an official / research-grade ledger with bhav inference.
        if status.get("research_grade") or is_research_grade_source(status.get("source")):
            status["built"] = False
            status["reason"] = "refusing_to_overwrite_research_grade"
            return status
        if not force:
            status["built"] = False
            status["reason"] = "ledger_already_present"
            return status

    from data.bhavcopy_runtime import ensure_loaded
    from data import bhavcopy_store as BS

    ensure_loaded(rebuild_from_local=False)
    spans = BS.symbol_date_spans()
    if not spans:
        return {
            "available": False,
            "built": False,
            "reason": "bhav_store_empty",
            "path": str(dest),
            "rows": 0,
            "survivorship_complete": False,
            "source": "",
            "note": "bhav store empty — cannot infer membership",
            "research_grade": False,
        }

    latest = max(item["last"] for item in spans.values())
    cutoff = latest - timedelta(days=max(1, int(inactive_after_days)))
    rows: list[dict] = []
    for sym, span in spans.items():
        if int(span.get("sessions") or 0) < min_sessions:
            continue
        first: date = span["first"]
        last: date = span["last"]
        row: dict[str, str] = {"symbol": sym, "listed": first.isoformat()}
        if last < cutoff:
            row["delisted"] = (last + timedelta(days=1)).isoformat()
        rows.append(row)

    status = write_universe_history(
        rows,
        path=dest,
        source="bhav_inferred",
        note=(
            "Inferred from local NSE bhav first/last sessions. Replace with an "
            "official listing/delisting archive for research-grade survivorship."
        ),
    )
    status["built"] = True
    status["reason"] = "built_from_bhav"
    status["store_latest"] = latest.isoformat()
    return status
