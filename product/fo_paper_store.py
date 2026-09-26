"""Durable SQLite state for the NSE F&O options paper lane."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime_paths import ensure_logs_path


SCHEMA_VERSION = 2


class FoPaperStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else ensure_logs_path("product", "fo_paper.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fo_open_positions (
                    trade_id TEXT PRIMARY KEY,
                    option_symbol TEXT NOT NULL UNIQUE,
                    underlying TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fo_closed_trades (
                    trade_id TEXT PRIMARY KEY,
                    option_symbol TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    context_key TEXT NOT NULL DEFAULT '',
                    evidence_lane TEXT NOT NULL DEFAULT 'FORWARD_PAPER',
                    settled_at TEXT NOT NULL DEFAULT '',
                    production_evidence_eligible INTEGER NOT NULL DEFAULT 0,
                    net_pnl REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fo_paper_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in self.conn.execute("PRAGMA table_info(fo_closed_trades)").fetchall()
            }
            if "net_pnl" not in columns:
                self.conn.execute(
                    "ALTER TABLE fo_closed_trades ADD COLUMN net_pnl REAL NOT NULL DEFAULT 0"
                )
                # Backfill from the durable JSON written by schema v1.
                rows = self.conn.execute(
                    "SELECT trade_id, payload_json FROM fo_closed_trades"
                ).fetchall()
                for row in rows:
                    payload = self._decode(row["payload_json"])
                    net_pnl = float((payload or {}).get("net_pnl") or 0.0)
                    self.conn.execute(
                        "UPDATE fo_closed_trades SET net_pnl = ? WHERE trade_id = ?",
                        (net_pnl, str(row["trade_id"])),
                    )
            self.conn.execute(
                "INSERT OR REPLACE INTO fo_paper_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _json(payload: Mapping[str, Any]) -> str:
        return json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _decode(raw: Any) -> dict[str, Any] | None:
        try:
            payload = json.loads(str(raw or ""))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def load_positions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM fo_open_positions ORDER BY updated_at, trade_id"
        ).fetchall()
        return [
            payload for row in rows
            if (payload := self._decode(row["payload_json"])) is not None
        ]

    def replace_positions(self, positions: Iterable[Mapping[str, Any]]) -> None:
        payloads = [dict(row) for row in positions]
        with self.conn:
            self.conn.execute("DELETE FROM fo_open_positions")
            for row in payloads:
                self.conn.execute(
                    """
                    INSERT INTO fo_open_positions(
                        trade_id, option_symbol, underlying, payload_json, updated_at
                    ) VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        str(row.get("trade_id") or ""),
                        str(row.get("option_symbol") or ""),
                        str(row.get("underlying") or ""),
                        self._json(row),
                    ),
                )

    def append_trades(self, trades: Iterable[Mapping[str, Any]]) -> int:
        inserted = 0
        with self.conn:
            for raw in trades:
                row = dict(raw)
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO fo_closed_trades(
                        trade_id, option_symbol, underlying, context_key,
                        evidence_lane, settled_at, production_evidence_eligible,
                        net_pnl, payload_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("trade_id") or ""),
                        str(row.get("option_symbol") or ""),
                        str(row.get("underlying") or ""),
                        str(row.get("context_key") or ""),
                        str(row.get("evidence_lane") or "FORWARD_PAPER"),
                        str(row.get("settled_at") or ""),
                        1 if bool(row.get("production_evidence_eligible")) else 0,
                        float(row.get("net_pnl") or 0.0),
                        self._json(row),
                    ),
                )
                inserted += int(cur.rowcount or 0)
        return inserted

    def load_trades(
        self,
        *,
        limit: int = 1000,
        production_only: bool = False,
    ) -> list[dict[str, Any]]:
        sql = "SELECT payload_json FROM fo_closed_trades"
        params: list[Any] = []
        if production_only:
            sql += " WHERE production_evidence_eligible = 1"
        sql += " ORDER BY settled_at DESC, created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        return [
            payload for row in rows
            if (payload := self._decode(row["payload_json"])) is not None
        ]

    def realized_pnl(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(net_pnl), 0) FROM fo_closed_trades"
        ).fetchone()
        return float(row[0] or 0.0)

    def status(self) -> dict[str, Any]:
        open_count = int(self.conn.execute("SELECT COUNT(*) FROM fo_open_positions").fetchone()[0])
        closed_count = int(self.conn.execute("SELECT COUNT(*) FROM fo_closed_trades").fetchone()[0])
        eligible_count = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM fo_closed_trades WHERE production_evidence_eligible = 1"
            ).fetchone()[0]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self.path),
            "open_positions": open_count,
            "closed_trades": closed_count,
            "production_evidence_trades": eligible_count,
            "realized_pnl": round(self.realized_pnl(), 2),
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FoPaperStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
