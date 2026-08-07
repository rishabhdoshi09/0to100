"""Durable EOD option-chain / OI / IV history.

Live chain fetches are ephemeral. This store persists one snapshot per
(symbol, as_of_date, expiry) so PCR/IV/OI can be studied over time.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "logs" / "options" / "eod_chains.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    expiry TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    source TEXT NOT NULL,
    pcr REAL,
    max_pain REAL,
    atm_iv REAL,
    spot REAL,
    strike_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of, expiry)
);
CREATE INDEX IF NOT EXISTS idx_chain_symbol_asof ON chain_snapshots(symbol, as_of);
"""


def db_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else _DEFAULT_DB


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = db_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save_chain_snapshot(
    symbol: str,
    *,
    as_of: str | date | None = None,
    expiry: str,
    rows: list[dict[str, Any]],
    source: str = "nse",
    pcr: float | None = None,
    max_pain: float | None = None,
    atm_iv: float | None = None,
    spot: float | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol required")
    if not expiry:
        raise ValueError("expiry required")
    as_of_s = str(as_of or date.today())
    captured = datetime.now(timezone.utc).isoformat()
    payload = {
        "symbol": sym,
        "as_of": as_of_s,
        "expiry": str(expiry),
        "rows": list(rows or []),
        "pcr": pcr,
        "max_pain": max_pain,
        "atm_iv": atm_iv,
        "spot": spot,
        "source": source,
    }
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO chain_snapshots(
                symbol, as_of, expiry, captured_at, source,
                pcr, max_pain, atm_iv, spot, strike_count, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, as_of, expiry) DO UPDATE SET
                captured_at=excluded.captured_at,
                source=excluded.source,
                pcr=excluded.pcr,
                max_pain=excluded.max_pain,
                atm_iv=excluded.atm_iv,
                spot=excluded.spot,
                strike_count=excluded.strike_count,
                payload_json=excluded.payload_json
            """,
            (
                sym,
                as_of_s,
                str(expiry),
                captured,
                source,
                pcr,
                max_pain,
                atm_iv,
                spot,
                len(rows or []),
                json.dumps(payload),
            ),
        )
        conn.commit()
    return {
        "symbol": sym,
        "as_of": as_of_s,
        "expiry": str(expiry),
        "strike_count": len(rows or []),
        "source": source,
        "path": str(db_path(path)),
    }


def load_chain(symbol: str, *, as_of: str | None = None, path: str | Path | None = None) -> dict[str, Any] | None:
    sym = str(symbol or "").strip().upper()
    with _connect(path) as conn:
        if as_of:
            row = conn.execute(
                "SELECT payload_json FROM chain_snapshots WHERE symbol=? AND as_of=? "
                "ORDER BY expiry DESC LIMIT 1",
                (sym, str(as_of)),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT payload_json FROM chain_snapshots WHERE symbol=? "
                "ORDER BY as_of DESC, expiry DESC LIMIT 1",
                (sym,),
            ).fetchone()
    if not row:
        return None
    return json.loads(row["payload_json"])


def history(symbol: str, *, days: int = 30, path: str | Path | None = None) -> list[dict[str, Any]]:
    sym = str(symbol or "").strip().upper()
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT symbol, as_of, expiry, pcr, max_pain, atm_iv, spot, strike_count, source, captured_at
            FROM chain_snapshots
            WHERE symbol=?
            ORDER BY as_of DESC, expiry DESC
            LIMIT ?
            """,
            (sym, max(1, int(days) * 3)),
        ).fetchall()
    out = [dict(r) for r in rows]
    # Keep ~days unique as_of values.
    seen: set[str] = set()
    trimmed: list[dict] = []
    for item in out:
        if item["as_of"] in seen:
            continue
        seen.add(item["as_of"])
        trimmed.append(item)
        if len(seen) >= max(1, int(days)):
            break
    return trimmed


def store_status(path: str | Path | None = None) -> dict[str, Any]:
    p = db_path(path)
    if not p.exists():
        return {
            "available": False,
            "path": str(p),
            "symbols": 0,
            "snapshots": 0,
            "latest_as_of": "",
        }
    with _connect(path) as conn:
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) AS n FROM chain_snapshots").fetchone()["n"]
        snaps = conn.execute("SELECT COUNT(*) AS n FROM chain_snapshots").fetchone()["n"]
        latest = conn.execute("SELECT MAX(as_of) AS d FROM chain_snapshots").fetchone()["d"]
    return {
        "available": snaps > 0,
        "path": str(p),
        "symbols": int(symbols or 0),
        "snapshots": int(snaps or 0),
        "latest_as_of": str(latest or ""),
    }
