"""Durable SQLite state for the NSE F&O options paper lane."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime_paths import ensure_logs_path, logs_path


SCHEMA_VERSION = 2


class FoPaperStore:
    def __init__(self, path: Path | str | None = None, *, read_only: bool = False) -> None:
        self.path = Path(path) if path is not None else (
            logs_path("product", "fo_paper.sqlite3") if read_only
            else ensure_logs_path("product", "fo_paper.sqlite3")
        )
        if read_only:
            # Dashboard projections must not migrate tables or replace metadata:
            # those writes wait behind the paper worker's active transaction.
            # mode=ro also fails without creating an empty ledger when absent.
            self.conn = sqlite3.connect(
                self.path.absolute().as_uri() + "?mode=ro", uri=True, timeout=0.5,
            )
            self.conn.row_factory = sqlite3.Row
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        from core.atomic_io import make_sqlite_durable
        make_sqlite_durable(self.conn)
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
        # No tolerant filtering here: a skipped row becomes an invisible open
        # position, and the next commit_cycle() would DELETE it permanently.
        rows = self.conn.execute(
            "SELECT trade_id, option_symbol, underlying, payload_json "
            "FROM fo_open_positions ORDER BY updated_at, trade_id"
        ).fetchall()
        positions: list[dict[str, Any]] = []
        for row in rows:
            payload = self._decode(row["payload_json"])
            if payload is None or any(
                str(payload.get(key) or "") != str(row[key])
                for key in ("trade_id", "option_symbol", "underlying")
            ):
                raise ValueError("F&O paper open-position record is corrupt or mismatched")
            positions.append(payload)
        return positions

    def _replace_positions_in_transaction(
        self,
        payloads: list[dict[str, Any]],
    ) -> None:
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

    def _append_trades_in_transaction(
        self,
        payloads: list[dict[str, Any]],
    ) -> int:
        inserted = 0
        for row in payloads:
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

    def replace_positions(self, positions: Iterable[Mapping[str, Any]]) -> None:
        payloads = [dict(row) for row in positions]
        with self.conn:
            self._replace_positions_in_transaction(payloads)

    def append_trades(self, trades: Iterable[Mapping[str, Any]]) -> int:
        payloads = [dict(row) for row in trades]
        with self.conn:
            return self._append_trades_in_transaction(payloads)

    def commit_cycle(
        self,
        trades: Iterable[Mapping[str, Any]],
        positions: Iterable[Mapping[str, Any]],
    ) -> int:
        """Atomically persist settlements and the resulting open-position book.

        A process death must never leave a trade durable in fo_closed_trades
        while the same trade remains durable in fo_open_positions. That split
        state can replay a settlement after restart and distort paper risk/equity.
        SQLite rolls both sides back together on any failure.
        """
        trade_payloads = [dict(row) for row in trades]
        position_payloads = [dict(row) for row in positions]
        with self.conn:
            inserted = self._append_trades_in_transaction(trade_payloads)
            self._replace_positions_in_transaction(position_payloads)
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

    def premium_deployed_on_session(self, session: str) -> float:
        """Gross daily entry premium, refusing unknowable persisted exposure.

        This value is a hard new-entry spending gate. A corrupt closed-trade
        payload must never be silently skipped: a restart would then give the
        F&O lane a fresh premium budget despite earlier trades that day.
        """
        wanted = str(session or "")[:10]
        if not wanted:
            return 0.0
        total = 0.0
        seen_trade_ids: set[str] = set()
        for table in ("fo_open_positions", "fo_closed_trades"):
            rows = self.conn.execute(
                f"SELECT trade_id, payload_json FROM {table}"
            ).fetchall()
            for row in rows:
                payload = self._decode(row["payload_json"])
                if payload is None:
                    raise ValueError(
                        f"F&O {table} contains corrupt premium-accounting data"
                    )
                trade_id = str(payload.get("trade_id") or "").strip()
                if not trade_id or trade_id != str(row["trade_id"]):
                    raise ValueError(
                        f"F&O {table} has mismatched premium-accounting identity"
                    )
                opened_at = str(payload.get("opened_at") or "")
                if len(opened_at) < 10:
                    # Without the original entry date a prior deployment cannot
                    # be excluded from today's limit. Require reconciliation.
                    raise ValueError(
                        f"F&O {table} has missing entry date for premium accounting"
                    )
                if opened_at[:10] != wanted or trade_id in seen_trade_ids:
                    continue
                try:
                    entry = float(payload["entry_price"])
                    quantity_raw = payload["quantity"]
                    quantity = int(quantity_raw)
                    valid = (
                        math.isfinite(entry) and entry > 0
                        and not isinstance(quantity_raw, bool)
                        and quantity > 0 and float(quantity_raw) == quantity
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError(
                        f"F&O {table} has invalid current-session premium accounting"
                    )
                total += entry * quantity
                if not math.isfinite(total):
                    raise ValueError("F&O daily premium exposure overflow")
                seen_trade_ids.add(trade_id)
        return total

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
