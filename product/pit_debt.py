"""First-class PIT_DATA_DEBT store.

Tracks missing evidence, affected decisions, information value, source
candidates and retry state. Quiet-period consumption prefers structured
XBRL and never runs inside the live entry window.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "product" / "pit_data_debt.db"

OPEN = "OPEN"
RETRY = "RETRY"
ACQUIRED = "ACQUIRED"
BLOCKED = "BLOCKED"
UNAVAILABLE = "UNAVAILABLE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    con = connect(path or DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS debt (
            debt_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            as_of TEXT,
            category TEXT NOT NULL,
            missing TEXT,
            affected_decision TEXT,
            information_value TEXT,
            acquisition_cost TEXT,
            source_candidates TEXT,
            retry_state TEXT,
            attempts INTEGER,
            last_attempt TEXT,
            last_error TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    return con


def upsert(
    *,
    symbol: str,
    category: str,
    as_of: str = "",
    missing: Sequence[str] | None = None,
    affected_decision: str = "",
    information_value: str = "MEDIUM",
    acquisition_cost: str = "BOUNDED_NSE",
    source_candidates: Sequence[str] | None = None,
    retry_state: str = OPEN,
    path: Path | None = None,
) -> dict[str, Any]:
    name = str(symbol).upper()
    day = str(as_of)[:10]
    debt_id = f"{name}:{category}:{day or 'any'}"
    now = _now()
    con = _connect(path)
    existing = con.execute("SELECT * FROM debt WHERE debt_id=?", (debt_id,)).fetchone()
    payload = {
        "debt_id": debt_id,
        "symbol": name,
        "as_of": day,
        "category": category,
        "missing": json.dumps(list(missing or [])),
        "affected_decision": affected_decision,
        "information_value": information_value or "MEDIUM",
        "acquisition_cost": acquisition_cost,
        "source_candidates": json.dumps(list(source_candidates or ["NSE XBRL"])),
        "retry_state": retry_state,
        "attempts": int(existing["attempts"]) if existing else 0,
        "last_attempt": existing["last_attempt"] if existing else "",
        "last_error": existing["last_error"] if existing else "",
        "created_at": existing["created_at"] if existing else now,
        "updated_at": now,
    }
    con.execute(
        """INSERT OR REPLACE INTO debt (
            debt_id, symbol, as_of, category, missing, affected_decision,
            information_value, acquisition_cost, source_candidates, retry_state,
            attempts, last_attempt, last_error, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        tuple(payload[k] for k in (
            "debt_id", "symbol", "as_of", "category", "missing", "affected_decision",
            "information_value", "acquisition_cost", "source_candidates", "retry_state",
            "attempts", "last_attempt", "last_error", "created_at", "updated_at",
        )),
    )
    con.commit()
    con.close()
    return payload


def mark_attempt(debt_id: str, *, ok: bool, error: str = "", path: Path | None = None) -> None:
    con = _connect(path)
    row = con.execute("SELECT * FROM debt WHERE debt_id=?", (debt_id,)).fetchone()
    if not row:
        con.close()
        return
    con.execute(
        """UPDATE debt SET attempts=?, last_attempt=?, last_error=?, retry_state=?, updated_at=?
           WHERE debt_id=?""",
        (
            int(row["attempts"] or 0) + 1,
            _now(),
            error[:240],
            ACQUIRED if ok else (BLOCKED if "403" in error or "429" in error else RETRY),
            _now(),
            debt_id,
        ),
    )
    con.commit()
    con.close()


def open_items(*, limit: int = 20, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    rows = con.execute(
        """SELECT * FROM debt WHERE retry_state IN (?,?)
           ORDER BY CASE information_value WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
                    attempts ASC LIMIT ?""",
        (OPEN, RETRY, int(limit)),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def ingest_coverage_debt(
    decisions: Sequence[Mapping[str, Any]],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Turn a walk-forward coverage map into retryable debt rows."""
    from product.pit_coverage import data_debt

    summary = data_debt(decisions)
    n = 0
    for row in summary.get("rows") or []:
        missing = list(row.get("missing") or [])
        if not missing:
            continue
        info = "HIGH" if "FINANCIALS" in missing else "MEDIUM"
        upsert(
            symbol=str(row.get("symbol") or ""),
            category="FINANCIALS" if "FINANCIALS" in missing else (missing[0] if missing else "OTHER"),
            as_of=str(row.get("as_of") or ""),
            missing=missing,
            affected_decision=str(row.get("decision") or ""),
            information_value=info,
            source_candidates=["NSE integrated-filing XBRL", "NSE quarterly XBRL"],
            path=path,
        )
        n += 1
    return {"written": n, "summary": {k: v for k, v in summary.items() if k != "rows"}}


def consume(*, limit: int = 2, entry_window: bool = False, warehouse_path=None) -> dict[str, Any]:
    """Bounded quiet-period work. Never starves the live entry window."""
    from product.pit_backfill import consume_data_debt

    return consume_data_debt(
        limit=limit,
        entry_window=entry_window,
        warehouse_path=warehouse_path,
    )
