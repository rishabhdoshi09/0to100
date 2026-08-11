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
_DELISTED_URL = "https://nsearchives.nseindia.com/content/equities/delisted.csv"
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
        # NSE headers often have leading spaces (" ISIN NUMBER").
        r = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        sym = str(r.get("SYMBOL") or "").strip().upper()
        series = str(r.get("SERIES") or "").strip().upper()
        isin = str(r.get("ISIN NUMBER") or r.get("ISIN") or "").strip().upper()
        listed = _parse_nse_date(str(r.get("DATE OF LISTING") or ""))
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


def _parse_delist_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    got = _parse_nse_date(raw)
    if got:
        return got
    for fmt in ("%d-%b-%y", "%d-%B-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_delisted(*, session: requests.Session | None = None) -> tuple[list[dict], dict]:
    """Official NSE delisting archive."""
    import csv
    import io
    sess = session or requests.Session()
    resp = sess.get(_DELISTED_URL, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    raw = resp.content
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace")))
    rows: list[dict] = []
    for r in reader:
        sym = str(r.get("Symbol") or r.get("SYMBOL") or "").strip().upper()
        when = _parse_delist_date(str(r.get("Delisted Date") or ""))
        if not sym or not when:
            continue
        rows.append({
            "symbol": sym,
            "delisted": when,
            "delist_type": str(r.get("Type of Delisting") or "").strip(),
            "company": str(r.get("Company") or "").strip(),
            "provenance": "nse_delisted",
        })
    meta = {
        "source_url": _DELISTED_URL,
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
    try:
        delisted_rows, de_meta = fetch_delisted(session=sess)
    except Exception as exc:
        delisted_rows, de_meta = [], {"error": str(exc)}

    # Apply official delisting dates onto matching securities (evidence only).
    by_delist = {d["symbol"]: d for d in delisted_rows}
    for row in equity_rows:
        d = by_delist.get(row["symbol"])
        if d:
            row["delisting_date"] = d["delisted"]
            if row.get("valid_to") is None:
                row["valid_to"] = d["delisted"]
                row["tradable"] = False

    # Delisted-only names (not in EQUITY_L): record with unknown listing unless skipped
    known = {r["symbol"] for r in equity_rows}
    for d in delisted_rows:
        if d["symbol"] in known:
            continue
        equity_rows.append({
            "security_id": f"sym:{d['symbol']}",
            "symbol": d["symbol"],
            "series": "EQ",
            "isin": None,
            "valid_from": None,  # unknown listing — not invented
            "valid_to": d["delisted"],
            "listing_date": None,
            "delisting_date": d["delisted"],
            "tradable": False,
            "provenance": "nse_delisted",
        })

    ledger = build_identity_ledger(
        equity_rows,
        changes,
        source_meta={"equity_l": eq_meta, "symbolchange": ch_meta, "delisted": de_meta},
    )
    ledger["source"] = "nse_equity_l+nse_symbolchange+nse_delisted"
    lin = lineage_coverage_report(ledger)
    ledger["completeness"] = {
        "has_isin_for_current_eq": any(bool(r.get("isin")) for r in ledger["securities"]),
        "has_official_delistings": bool(delisted_rows),
        "symbol_lineage_complete": bool(lin.get("symbol_lineage_complete")),
        "isin_confirmed_rate": lin.get("isin_confirmed_rate"),
        "lineage_by_status": lin.get("by_status"),
    }
    out = write_identity_ledger(ledger, path=path)
    return {
        "path": str(out),
        "n_securities": len(ledger["securities"]),
        "n_symbol_changes": len(ledger["symbol_changes"]),
        "n_delisted": len(delisted_rows),
        "completeness": ledger["completeness"],
        "source_meta": ledger["source_meta"],
        "lineage": {
            "symbol_lineage_complete": lin.get("symbol_lineage_complete"),
            "isin_confirmed_rate": lin.get("isin_confirmed_rate"),
            "by_status": lin.get("by_status"),
        },
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


def lineage_coverage_report(ledger: dict | None = None, *, focus_symbols: set[str] | None = None) -> dict:
    """Classify each official symbol-change row; never invent missing links.

    Statuses: CONFIRMED | PARTIAL | CONFLICT | UNRESOLVED
    """
    from product.plain_language import PlainCard, render_layers

    ledger = ledger if ledger is not None else load_identity_ledger()
    if not ledger:
        return {
            "available": False,
            "symbol_lineage_complete": False,
            "n_changes": 0,
            "by_status": {},
            "unresolved": [],
            "focus": {},
            "user_facing": render_layers(PlainCard(
                label="Stock history link",
                state="NOT_READY",
                explanation=(
                    "The company's ticker/history changed, and QuantTerm cannot yet prove "
                    "that the old and new records represent the same security."
                ),
                implication="Unresolved lineage blocks RESEARCH_GRADE until evidenced.",
                technical="UNRESOLVED_LINEAGE; identity ledger missing",
                internal_key="UNRESOLVED_LINEAGE",
                internal_value="MISSING_LEDGER",
            )),
        }

    by_sym = {r.get("symbol"): r for r in (ledger.get("securities") or []) if r.get("symbol")}
    classified = []
    by_status = {"CONFIRMED": 0, "PARTIAL": 0, "CONFLICT": 0, "UNRESOLVED": 0}
    for ch in ledger.get("symbol_changes") or []:
        old_s = str(ch.get("old_symbol") or "").upper()
        new_s = str(ch.get("new_symbol") or "").upper()
        when = ch.get("effective_date")
        old = by_sym.get(old_s) or {}
        new = by_sym.get(new_s) or {}
        old_isin = old.get("isin")
        new_isin = new.get("isin")
        status = "PARTIAL"
        note = "Official symbol-change row present."
        if old_isin and new_isin and old_isin == new_isin:
            status = "CONFIRMED"
            note = "Old and new symbols share the same ISIN evidence."
        elif old_isin and new_isin and old_isin != new_isin:
            status = "CONFLICT"
            note = f"ISIN mismatch old={old_isin} new={new_isin}."
        elif not old_isin and not new_isin:
            status = "PARTIAL"
            note = "Rename evidenced by NSE symbolchange.csv; ISIN link not available."
        elif old_isin or new_isin:
            status = "PARTIAL"
            note = "Only one side has ISIN evidence; link not fully closed."
        if not when or not old_s or not new_s:
            status = "UNRESOLVED"
            note = "Incomplete transition fields."
        row = {
            "old_symbol": old_s,
            "new_symbol": new_s,
            "security_id": new.get("security_id") or old.get("security_id"),
            "isin": new_isin or old_isin,
            "valid_from": when,
            "valid_to": None,
            "event": "symbol_change",
            "source": ch.get("provenance") or "nse_symbolchange",
            "evidence_reference": (ledger.get("source_meta") or {}).get("symbolchange", {}).get("source_sha256"),
            "status": status,
            "note": note,
        }
        classified.append(row)
        by_status[status] = by_status.get(status, 0) + 1

    focus_symbols = {s.upper() for s in (focus_symbols or set())}
    focus_hits = [
        r for r in classified
        if r["old_symbol"] in focus_symbols or r["new_symbol"] in focus_symbols
    ]
    # Complete when every transition is officially evidenced and none CONFLICT/UNRESOLVED.
    # PARTIAL (official rename without dual-ISIN closure) is allowed for completeness of
    # *available evidence*; isin_confirmed_rate is reported separately.
    complete = (
        bool(classified)
        and by_status.get("CONFLICT", 0) == 0
        and by_status.get("UNRESOLVED", 0) == 0
    )
    isin_confirmed_rate = (
        by_status.get("CONFIRMED", 0) / len(classified) if classified else 0.0
    )
    focus_blocking = [r for r in focus_hits if r["status"] in {"CONFLICT", "UNRESOLVED"}]

    return {
        "available": True,
        "symbol_lineage_complete": complete,
        "isin_confirmed_rate": round(isin_confirmed_rate, 4),
        "n_changes": len(classified),
        "by_status": by_status,
        "unresolved": [r for r in classified if r["status"] in {"UNRESOLVED", "CONFLICT"}],
        "partial": [r for r in classified if r["status"] == "PARTIAL"][:50],
        "confirmed": by_status.get("CONFIRMED", 0),
        "focus_symbols": sorted(focus_symbols),
        "focus_transitions": focus_hits,
        "focus_blocking": focus_blocking,
        "user_facing": render_layers(PlainCard(
            label="Stock history link",
            state="GOOD" if complete and not focus_blocking else ("CAUTION" if not focus_blocking else "NOT_READY"),
            explanation=(
                "Official rename notices are on file. Matching ISINs confirm some links; "
                "others remain evidence-backed renames without dual-ISIN closure."
                if complete else
                "The company's ticker/history changed, and QuantTerm cannot yet prove "
                "that the old and new records represent the same security."
            ),
            implication=(
                "No conflicting lineage blocks research; ISIN confirmation rate is reported separately."
                if complete and not focus_blocking else
                "Unresolved lineage on names that matter for a test can block RESEARCH_GRADE."
            ),
            technical=(
                f"UNRESOLVED_LINEAGE complete={complete} isin_confirmed_rate={isin_confirmed_rate:.3f} "
                f"by_status={by_status} focus_blocking={len(focus_blocking)}"
            ),
            internal_key="UNRESOLVED_LINEAGE",
            internal_value="COMPLETE" if complete else "PARTIAL",
        )),
    }


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
