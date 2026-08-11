"""Historical security identity ledger (research).

A trading symbol is NOT permanent identity. This module stores only what
legitimate NSE source evidence can establish (EQUITY_L listing master +
symbol-change file). Missing transitions stay unknown — never guessed.

Canonical file: ``logs/security_identity.json``
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _ROOT / "logs" / "security_identity.json"
_EQUITY_L_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_SYMBOLCHANGE_URL = "https://archives.nseindia.com/content/equities/symbolchange.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}

SCHEMA_VERSION = 1


def identity_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("QT_SECURITY_IDENTITY_FILE")
    return Path(override) if override else _DEFAULT_PATH


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_nse_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_equity_l(*, session: requests.Session | None = None) -> tuple[list[dict], dict]:
    """Fetch current NSE equity listing master (listing date + ISIN)."""
    sess = session or requests.Session()
    resp = sess.get(_EQUITY_L_URL, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    raw = resp.content
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for r in reader:
        sym = str(r.get("SYMBOL") or "").strip().upper()
        series = str(r.get(" SERIES") or r.get("SERIES") or "").strip().upper()
        isin = str(r.get("ISIN NUMBER") or r.get("ISIN") or "").strip().upper()
        listed = _parse_nse_date(str(r.get(" DATE OF LISTING") or r.get("DATE OF LISTING") or ""))
        if not sym or not listed:
            continue
        if series and series != "EQ":
            continue
        security_id = f"isin:{isin}" if isin.startswith("INE") or isin.startswith("IN") else f"sym:{sym}"
        rows.append({
            "security_id": security_id,
            "symbol": sym,
            "series": series or "EQ",
            "isin": isin or None,
            "valid_from": listed,
            "valid_to": None,  # unknown unless a later symbol-change closes it
            "listing_date": listed,
            "delisting_date": None,  # EQUITY_L is current members only — unknown for delisted
            "tradable": True,
            "provenance": "nse_equity_l",
        })
    meta = {
        "source_url": _EQUITY_L_URL,
        "source_sha256": _sha256_bytes(raw),
        "n_rows": len(rows),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return rows, meta


def fetch_symbol_changes(*, session: requests.Session | None = None) -> tuple[list[dict], dict]:
    """Fetch NSE symbol-change file. Unparseable rows are skipped (unknown stays unknown)."""
    sess = session or requests.Session()
    resp = sess.get(_SYMBOLCHANGE_URL, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    raw = resp.content
    text = raw.decode("utf-8", errors="replace")
    changes: list[dict] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        # Format observed: COMPANY, OLD_SYMBOL, NEW_SYMBOL, DD-MMM-YYYY
        company, old_sym, new_sym, dt_raw = parts[0], parts[1], parts[2], parts[3]
        old_sym = old_sym.strip().upper()
        new_sym = new_sym.strip().upper()
        when = _parse_nse_date(dt_raw)
        if not old_sym or not new_sym or not when or old_sym == new_sym:
            continue
        if not re.fullmatch(r"[A-Z0-9&-]{1,20}", old_sym):
            continue
        if not re.fullmatch(r"[A-Z0-9&-]{1,20}", new_sym):
            continue
        changes.append({
            "event": "symbol_change",
            "old_symbol": old_sym,
            "new_symbol": new_sym,
            "effective_date": when,
            "company": company,
            "provenance": "nse_symbolchange",
        })
    meta = {
        "source_url": _SYMBOLCHANGE_URL,
        "source_sha256": _sha256_bytes(raw),
        "n_rows": len(changes),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return changes, meta


def build_identity_ledger(
    equity_rows: list[dict],
    symbol_changes: list[dict],
    *,
    source_meta: dict | None = None,
) -> dict:
    """Compose ledger. Does not invent delisting or unstated transitions."""
    by_symbol: dict[str, dict] = {}
    for row in equity_rows:
        by_symbol[row["symbol"]] = dict(row)

    # Apply symbol changes only as explicit evidence: close old symbol interval,
    # open new if already in EQUITY_L (otherwise new may be unknown / not current).
    for ch in sorted(symbol_changes, key=lambda x: x["effective_date"]):
        old_s, new_s, when = ch["old_symbol"], ch["new_symbol"], ch["effective_date"]
        if old_s in by_symbol:
            prev = by_symbol[old_s]
            # Only set valid_to if currently open and change is after valid_from
            if prev.get("valid_to") is None and when >= str(prev.get("valid_from") or ""):
                prev["valid_to"] = when
                prev["tradable"] = False
                prev["symbol_change_to"] = new_s
        # Link new symbol's security_id to old ISIN when new is present and old had ISIN
        if new_s in by_symbol and old_s in by_symbol:
            old = by_symbol[old_s]
            new = by_symbol[new_s]
            if old.get("isin") and not new.get("isin"):
                new["isin"] = old["isin"]
                new["security_id"] = old["security_id"]
            new["symbol_change_from"] = old_s
            new.setdefault("valid_from", when)

    securities = sorted(by_symbol.values(), key=lambda r: r["symbol"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "nse_equity_l+nse_symbolchange",
        "source_meta": source_meta or {},
        "note": (
            "Identity rows are evidence-backed only. Delisting dates are unknown "
            "unless supplied by a separate official archive. Unknown transitions "
            "are omitted, never fabricated."
        ),
        "symbol_changes": symbol_changes,
        "securities": securities,
        "completeness": {
            "has_isin_for_current_eq": True,
            "has_official_delistings": False,
            "symbol_lineage_complete": False,  # honest: delistings + full lineage not proven
        },
    }


def write_identity_ledger(ledger: dict, path: str | Path | None = None) -> Path:
    p = identity_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(p)
    return p


def load_identity_ledger(path: str | Path | None = None) -> dict:
    p = identity_path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def materialize_from_nse(*, path: str | Path | None = None) -> dict:
    """Fetch NSE masters and write the identity ledger. Network only at ingest time."""
    sess = requests.Session()
    equity_rows, eq_meta = fetch_equity_l(session=sess)
    changes, ch_meta = fetch_symbol_changes(session=sess)
    ledger = build_identity_ledger(
        equity_rows,
        changes,
        source_meta={"equity_l": eq_meta, "symbolchange": ch_meta},
    )
    out = write_identity_ledger(ledger, path=path)
    return {
        "path": str(out),
        "n_securities": len(ledger["securities"]),
        "n_symbol_changes": len(ledger["symbol_changes"]),
        "completeness": ledger["completeness"],
        "source_meta": ledger["source_meta"],
    }


def resolve_as_of(symbol: str, as_of: str, ledger: dict | None = None) -> dict[str, Any]:
    """Resolve symbol → security_id as of date. Unknown → blocked, never guessed."""
    ledger = ledger if ledger is not None else load_identity_ledger()
    sym = str(symbol or "").strip().upper()
    as_of_s = str(as_of)[:10]
    if not ledger or not sym:
        return {"status": "UNKNOWN", "security_id": None, "symbol": sym, "as_of": as_of_s}
    for row in ledger.get("securities") or []:
        if row.get("symbol") != sym:
            continue
        vf = str(row.get("valid_from") or "")[:10]
        vt = row.get("valid_to")
        vt_s = str(vt)[:10] if vt else None
        if vf and as_of_s < vf:
            return {"status": "NOT_YET_LISTED", "security_id": row.get("security_id"),
                    "symbol": sym, "as_of": as_of_s}
        if vt_s and as_of_s >= vt_s:
            return {"status": "SYMBOL_ENDED", "security_id": row.get("security_id"),
                    "symbol": sym, "as_of": as_of_s, "changed_to": row.get("symbol_change_to")}
        return {
            "status": "OK",
            "security_id": row.get("security_id"),
            "symbol": sym,
            "series": row.get("series"),
            "as_of": as_of_s,
            "listing_date": row.get("listing_date"),
            "delisting_date": row.get("delisting_date"),
        }
    return {"status": "UNKNOWN", "security_id": None, "symbol": sym, "as_of": as_of_s}


def ledger_status(path: str | Path | None = None) -> dict:
    p = identity_path(path)
    ledger = load_identity_ledger(p)
    comp = (ledger or {}).get("completeness") or {}
    return {
        "available": bool(ledger),
        "path": str(p),
        "n_securities": len((ledger or {}).get("securities") or []),
        "n_symbol_changes": len((ledger or {}).get("symbol_changes") or []),
        "symbol_lineage_complete": bool(comp.get("symbol_lineage_complete")),
        "has_official_delistings": bool(comp.get("has_official_delistings")),
        "source": (ledger or {}).get("source"),
    }
