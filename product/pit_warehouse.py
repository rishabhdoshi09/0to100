"""Point-in-time evidence warehouse.

Stores dated company evidence. The eligibility rule is available_from <= T,
never period_end <= T. acquired_at is QuantTerm's download time and does not
grant historical availability.

Revisions are separate rows. History is never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.due_diligence.provenance import SOURCE_TRUST, classify_source_type
from product.pit_availability import PIT_UNVERIFIED, available_to_engine_at_t

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("QT_PIT_WAREHOUSE") or ROOT / "logs" / "product" / "pit_warehouse.db")

DOC_ANNUAL_REPORT = "ANNUAL_REPORT"
DOC_QUARTERLY_RESULT = "QUARTERLY_RESULT"
DOC_INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"
DOC_SHAREHOLDING_PATTERN = "SHAREHOLDING_PATTERN"
DOC_CORPORATE_ANNOUNCEMENT = "CORPORATE_ANNOUNCEMENT"
DOC_CREDIT_RATING = "CREDIT_RATING"
DOC_EXCHANGE_FILING = "EXCHANGE_FILING"
DOC_OTHER = "OTHER"

# Live research uses latest eligible. Replay uses latest as of T.
QUERY_LIVE = "LIVE"
QUERY_AS_OF = "AS_OF"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evidence_id(
    *,
    symbol: str,
    evidence_type: str,
    source_identity: str,
    publication_date: str = "",
    period_end: str = "",
    revision: int = 1,
) -> str:
    raw = "|".join([
        str(symbol or "").upper(),
        str(evidence_type or ""),
        str(source_identity or ""),
        str(publication_date or "")[:10],
        str(period_end or "")[:10],
        str(int(revision)),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def classify_document(text: str) -> str:
    blob = str(text or "").lower()
    if "annual report" in blob:
        return DOC_ANNUAL_REPORT
    if "shareholding" in blob or "share holding" in blob:
        return DOC_SHAREHOLDING_PATTERN
    if "investor presentation" in blob or "earnings presentation" in blob:
        return DOC_INVESTOR_PRESENTATION
    if "credit rating" in blob or "rating rationale" in blob:
        return DOC_CREDIT_RATING
    if any(tok in blob for tok in (
        "financial result", "quarterly result", "audited result",
        "un-audited result", "unaudited result",
    )):
        return DOC_QUARTERLY_RESULT
    if "filing" in blob or "intimation" in blob:
        return DOC_EXCHANGE_FILING
    return DOC_CORPORATE_ANNOUNCEMENT


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target))
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            period_start TEXT,
            period_end TEXT,
            publication_date TEXT,
            filing_date TEXT,
            exchange_timestamp TEXT,
            available_from TEXT,
            acquired_at TEXT,
            source TEXT,
            source_url TEXT,
            source_identity TEXT,
            source_trust INTEGER,
            raw_artifact_id TEXT,
            parser_version TEXT,
            extracted_json TEXT,
            supersedes TEXT,
            revision INTEGER DEFAULT 1,
            pit_status TEXT,
            document_type TEXT,
            reason_code TEXT,
            created_at TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_pit_sym_avail ON evidence(symbol, available_from)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pit_type ON evidence(evidence_type)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            symbol TEXT,
            source_url TEXT,
            content_sha TEXT,
            local_path TEXT,
            bytes INTEGER,
            document_type TEXT,
            acquired_at TEXT,
            parser_version TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS conflicts (
            conflict_id TEXT PRIMARY KEY,
            symbol TEXT,
            fact_key TEXT,
            left_evidence_id TEXT,
            right_evidence_id TEXT,
            left_value TEXT,
            right_value TEXT,
            winner_evidence_id TEXT,
            resolution TEXT,
            created_at TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            cache_key TEXT PRIMARY KEY,
            symbol TEXT,
            as_of TEXT,
            kind TEXT,
            fingerprint TEXT,
            generation INTEGER,
            payload_json TEXT,
            created_at TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    return con


def generation(*, path: Path | None = None) -> int:
    con = _connect(path)
    row = con.execute("SELECT value FROM meta WHERE key='generation'").fetchone()
    con.close()
    return int((row["value"] if row else 0) or 0)


def _bump_generation(con: sqlite3.Connection, symbol: str = "") -> int:
    row = con.execute("SELECT value FROM meta WHERE key='generation'").fetchone()
    nxt = int((row["value"] if row else 0) or 0) + 1
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('generation', ?)", (str(nxt),))
    if symbol:
        con.execute("DELETE FROM snapshots WHERE symbol=?", (str(symbol).upper(),))
    else:
        con.execute("DELETE FROM snapshots")
    return nxt


def persist(row: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Insert or keep an evidence row. Never mutates extracted facts in place."""
    item = dict(row)
    symbol = str(item.get("symbol") or "").upper()
    if not symbol:
        return item
    pub = str(item.get("publication_date") or "")[:10]
    filing = str(item.get("filing_date") or "")[:10]
    available = str(item.get("available_from") or pub or filing)[:10]
    if not available:
        item["pit_status"] = PIT_UNVERIFIED
        item["available_from"] = ""
        item["reason_code"] = item.get("reason_code") or "PUBLICATION_DATE_UNKNOWN"
    else:
        item["available_from"] = available
        item.setdefault("pit_status", "INDEXED")
    eid = str(item.get("evidence_id") or "") or evidence_id(
        symbol=symbol,
        evidence_type=str(item.get("evidence_type") or DOC_OTHER),
        source_identity=str(item.get("source_identity") or item.get("source_url") or ""),
        publication_date=available,
        period_end=str(item.get("period_end") or ""),
        revision=int(item.get("revision") or 1),
    )
    item["evidence_id"] = eid
    item["symbol"] = symbol
    item.setdefault("acquired_at", _now())
    item.setdefault("created_at", _now())
    item.setdefault("revision", 1)
    item.setdefault("parser_version", "pit_warehouse.v1")
    source = str(item.get("source") or "")
    url = str(item.get("source_url") or "")
    kind = classify_source_type(source, url)
    item.setdefault("source_trust", SOURCE_TRUST.get(kind, 20))
    con = _connect(path)
    existing = con.execute("SELECT evidence_id FROM evidence WHERE evidence_id=?", (eid,)).fetchone()
    if existing:
        con.close()
        item["deduped"] = True
        return item
    con.execute(
        """INSERT INTO evidence (
            evidence_id, symbol, evidence_type, period_start, period_end,
            publication_date, filing_date, exchange_timestamp, available_from,
            acquired_at, source, source_url, source_identity, source_trust,
            raw_artifact_id, parser_version, extracted_json, supersedes,
            revision, pit_status, document_type, reason_code, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            eid, symbol, item.get("evidence_type"), item.get("period_start"),
            item.get("period_end"), item.get("publication_date"), item.get("filing_date"),
            item.get("exchange_timestamp"), item.get("available_from"),
            item.get("acquired_at"), item.get("source"), item.get("source_url"),
            item.get("source_identity"), item.get("source_trust"),
            item.get("raw_artifact_id"), item.get("parser_version"),
            json.dumps(item.get("extracted") or {}, default=str),
            item.get("supersedes"), int(item.get("revision") or 1),
            item.get("pit_status"), item.get("document_type") or item.get("evidence_type"),
            item.get("reason_code"), item.get("created_at"),
        ),
    )
    _bump_generation(con, symbol)
    con.commit()
    con.close()
    item["deduped"] = False
    return item


def _row(raw: sqlite3.Row) -> dict[str, Any]:
    item = dict(raw)
    try:
        item["extracted"] = json.loads(item.pop("extracted_json") or "{}")
    except Exception:
        item["extracted"] = {}
    return item


def get_evidence(
    symbol: str,
    *,
    as_of: str,
    evidence_types: Sequence[str] | None = None,
    path: Path | None = None,
    unsafe: bool = False,
) -> list[dict[str, Any]]:
    """Temporal query. Future artifacts are invisible unless unsafe=True."""
    con = _connect(path)
    q = "SELECT * FROM evidence WHERE symbol=?"
    args: list[Any] = [str(symbol).upper()]
    if evidence_types:
        q += " AND evidence_type IN (%s)" % ",".join("?" * len(evidence_types))
        args.extend(evidence_types)
    if not unsafe:
        q += " AND available_from IS NOT NULL AND available_from != '' AND available_from <= ? AND pit_status != ?"
        args.extend([str(as_of)[:10], PIT_UNVERIFIED])
    q += " ORDER BY available_from DESC, source_trust DESC"
    rows = [_row(r) for r in con.execute(q, args)]
    con.close()
    if unsafe:
        return rows
    # Defense in depth: re-check the contract.
    kept = []
    for row in rows:
        check = available_to_engine_at_t(
            as_of=as_of,
            period_end=row.get("period_end"),
            publication_date=row.get("publication_date") or row.get("available_from"),
            filing_date=row.get("filing_date"),
            acquired_at=None,
        )
        if check.get("available_to_engine_at_T"):
            kept.append(row)
    return kept


def get_evidence_raw(symbol: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """Diagnostic only. Includes unverified and future rows."""
    return get_evidence(symbol, as_of="9999-12-31", path=path, unsafe=True)


def counts(*, path: Path | None = None) -> dict[str, int]:
    con = _connect(path)
    n = int(con.execute("SELECT count(*) FROM evidence").fetchone()[0])
    dated = int(con.execute(
        "SELECT count(*) FROM evidence WHERE available_from IS NOT NULL AND available_from != '' AND pit_status != ?",
        (PIT_UNVERIFIED,),
    ).fetchone()[0])
    symbols = int(con.execute("SELECT count(DISTINCT symbol) FROM evidence").fetchone()[0])
    unverified = int(con.execute(
        "SELECT count(*) FROM evidence WHERE pit_status=?",
        (PIT_UNVERIFIED,),
    ).fetchone()[0])
    con.close()
    return {"rows": n, "dated": dated, "symbols": symbols, "unverified": unverified}


def persist_artifact(row: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    item = dict(row)
    aid = str(item.get("artifact_id") or "")
    if not aid:
        raw = "|".join([
            str(item.get("symbol") or "").upper(),
            str(item.get("source_url") or item.get("local_path") or ""),
            str(item.get("content_sha") or ""),
        ])
        aid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    item["artifact_id"] = aid
    item.setdefault("acquired_at", _now())
    con = _connect(path)
    existing = con.execute("SELECT artifact_id FROM artifacts WHERE artifact_id=?", (aid,)).fetchone()
    if existing:
        con.close()
        item["deduped"] = True
        return item
    con.execute(
        """INSERT INTO artifacts (
            artifact_id, symbol, source_url, content_sha, local_path,
            bytes, document_type, acquired_at, parser_version
        ) VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            aid, str(item.get("symbol") or "").upper(), item.get("source_url"),
            item.get("content_sha"), item.get("local_path"), item.get("bytes"),
            item.get("document_type"), item.get("acquired_at"), item.get("parser_version"),
        ),
    )
    con.commit()
    con.close()
    item["deduped"] = False
    return item


def record_conflict(row: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Keep both source records. Record which fact won and why."""
    item = dict(row)
    cid = str(item.get("conflict_id") or "") or evidence_id(
        symbol=str(item.get("symbol") or ""),
        evidence_type="CONFLICT",
        source_identity=str(item.get("fact_key") or ""),
        publication_date=str(item.get("left_evidence_id") or ""),
        period_end=str(item.get("right_evidence_id") or ""),
    )
    item["conflict_id"] = cid
    item.setdefault("created_at", _now())
    con = _connect(path)
    existing = con.execute("SELECT conflict_id FROM conflicts WHERE conflict_id=?", (cid,)).fetchone()
    if existing:
        con.close()
        item["deduped"] = True
        return item
    con.execute(
        """INSERT INTO conflicts (
            conflict_id, symbol, fact_key, left_evidence_id, right_evidence_id,
            left_value, right_value, winner_evidence_id, resolution, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            cid, str(item.get("symbol") or "").upper(), item.get("fact_key"),
            item.get("left_evidence_id"), item.get("right_evidence_id"),
            str(item.get("left_value") or ""), str(item.get("right_value") or ""),
            item.get("winner_evidence_id"), item.get("resolution"), item.get("created_at"),
        ),
    )
    con.commit()
    con.close()
    return item


def resolve_by_authority(left: Mapping[str, Any], right: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Prefer higher source_trust. Do not silently overwrite either row."""
    lt = int(left.get("source_trust") or 0)
    rt = int(right.get("source_trust") or 0)
    if lt > rt:
        winner, loser, why = left, right, "higher_source_trust"
    elif rt > lt:
        winner, loser, why = right, left, "higher_source_trust"
    else:
        left_date = str(left.get("available_from") or "")
        right_date = str(right.get("available_from") or "")
        if left_date > right_date:
            winner, loser, why = left, right, "later_available_from_same_trust"
        else:
            winner, loser, why = right, left, "later_available_from_same_trust"
    record_conflict({
        "symbol": left.get("symbol") or right.get("symbol"),
        "fact_key": left.get("fact_key") or right.get("fact_key") or "value",
        "left_evidence_id": left.get("evidence_id"),
        "right_evidence_id": right.get("evidence_id"),
        "left_value": left.get("value"),
        "right_value": right.get("value"),
        "winner_evidence_id": winner.get("evidence_id"),
        "resolution": why,
    }, path=path)
    return {"winner": dict(winner), "loser": dict(loser), "resolution": why}


def warehouse_fingerprint(*, path: Path | None = None) -> str:
    tally = counts(path=path)
    raw = json.dumps({"counts": tally, "generation": generation(path=path)}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_key(symbol: str, as_of: str, kind: str, fingerprint: str) -> str:
    raw = "|".join([str(symbol).upper(), str(as_of)[:10], kind, fingerprint])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_cached_snapshot(
    symbol: str,
    as_of: str,
    kind: str,
    fingerprint: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    key = cache_key(symbol, as_of, kind, fingerprint)
    con = _connect(path)
    row = con.execute("SELECT * FROM snapshots WHERE cache_key=?", (key,)).fetchone()
    gen = generation(path=path)
    if row and int(row["generation"] or 0) == gen:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = None
        con.close()
        return payload if isinstance(payload, dict) else None
    con.close()
    return None


def put_cached_snapshot(
    symbol: str,
    as_of: str,
    kind: str,
    fingerprint: str,
    payload: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> str:
    key = cache_key(symbol, as_of, kind, fingerprint)
    gen = generation(path=path)
    con = _connect(path)
    con.execute(
        """INSERT OR REPLACE INTO snapshots (
            cache_key, symbol, as_of, kind, fingerprint, generation, payload_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            key, str(symbol).upper(), str(as_of)[:10], kind, fingerprint,
            gen, json.dumps(dict(payload), default=str), _now(),
        ),
    )
    con.commit()
    con.close()
    return key
