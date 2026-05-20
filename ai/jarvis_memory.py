"""
Persistent cross-session SQLite memory for JARVIS.
Database stored at logs/jarvis_memory.db.
"""

import os
import sqlite3
import threading
import time
import json
import requests
from datetime import datetime, timezone
from pathlib import Path


_DB_PATH = Path("logs/jarvis_memory.db")


class JarvisMemory:
    def __init__(self, db_path: str | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS facts (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        category   TEXT NOT NULL DEFAULT 'fact',
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS sessions (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        summary       TEXT NOT NULL,
                        timestamp     TEXT NOT NULL,
                        message_count INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS price_alerts (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol      TEXT NOT NULL,
                        price       REAL NOT NULL,
                        direction   TEXT NOT NULL CHECK(direction IN ('above','below')),
                        created_at  TEXT NOT NULL,
                        triggered_at TEXT
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str, category: str = "fact") -> None:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO facts (key, value, category, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value      = excluded.value,
                            category   = excluded.category,
                            updated_at = excluded.updated_at
                        """,
                        (key, str(value), category, self._now_iso()),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] remember error: {exc}")

    def recall(self, category: str | None = None, limit: int = 20) -> list[dict]:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    if category:
                        rows = conn.execute(
                            "SELECT * FROM facts WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                            (category, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT * FROM facts ORDER BY updated_at DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] recall error: {exc}")
            return []

    def forget(self, key: str) -> None:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute("DELETE FROM facts WHERE key = ?", (key,))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] forget error: {exc}")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def summarize_and_save_session(self, messages: list[dict]) -> None:
        try:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            summary = "No summary available."
            if api_key and messages:
                conversation_text = "\n".join(
                    f"{m.get('role','?')}: {m.get('content','')}" for m in messages
                )
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Summarize the following trading assistant conversation "
                                "in exactly 3 concise sentences. Focus on key decisions, "
                                "symbols discussed, and any user preferences revealed."
                            ),
                        },
                        {"role": "user", "content": conversation_text},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.3,
                }
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=20,
                )
                resp.raise_for_status()
                summary = resp.json()["choices"][0]["message"]["content"].strip()

            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT INTO sessions (summary, timestamp, message_count) VALUES (?, ?, ?)",
                        (summary, self._now_iso(), len(messages)),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] summarize_and_save_session error: {exc}")

    def get_recent_sessions(self, n: int = 3) -> list[dict]:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (n,)
                    ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] get_recent_sessions error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Price alerts
    # ------------------------------------------------------------------

    def set_price_alert(self, symbol: str, price: float, direction: str) -> None:
        if direction not in ("above", "below"):
            raise ValueError("direction must be 'above' or 'below'")
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO price_alerts (symbol, price, direction, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (symbol.upper(), float(price), direction, self._now_iso()),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] set_price_alert error: {exc}")

    def check_price_alerts(self, prices: dict[str, float]) -> list[dict]:
        triggered: list[dict] = []
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM price_alerts WHERE triggered_at IS NULL"
                    ).fetchall()
                    now = self._now_iso()
                    for row in rows:
                        symbol = row["symbol"]
                        ltp = prices.get(symbol)
                        if ltp is None:
                            continue
                        fire = (row["direction"] == "above" and ltp >= row["price"]) or (
                            row["direction"] == "below" and ltp <= row["price"]
                        )
                        if fire:
                            conn.execute(
                                "UPDATE price_alerts SET triggered_at = ? WHERE id = ?",
                                (now, row["id"]),
                            )
                            alert = dict(row)
                            alert["triggered_at"] = now
                            alert["ltp"] = ltp
                            triggered.append(alert)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] check_price_alerts error: {exc}")
        return triggered

    def get_active_alerts(self) -> list[dict]:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM price_alerts WHERE triggered_at IS NULL ORDER BY created_at DESC"
                    ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            print(f"[JarvisMemory] get_active_alerts error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Morning context builder
    # ------------------------------------------------------------------

    def build_morning_context(self) -> str:
        try:
            sessions = self.get_recent_sessions(n=3)
            facts = self.recall(limit=20)
            alerts = self.get_active_alerts()

            lines: list[str] = ["=== JARVIS PERSISTENT MEMORY ==="]

            # Session summaries
            if sessions:
                latest = sessions[0]
                lines.append(f"Last session: {latest['summary']}")
                if len(sessions) > 1:
                    older_bullets = []
                    for s in sessions[1:3]:
                        older_bullets.append(f"  • {s['summary']}")
                    lines.append("Earlier sessions:")
                    lines.extend(older_bullets)
            else:
                lines.append("Last session: No previous sessions recorded.")

            # Facts
            if facts:
                lines.append("User facts:")
                for f in facts:
                    lines.append(f"  - {f['key']}: {f['value']}")
            else:
                lines.append("User facts: None stored yet.")

            # Price alerts
            if alerts:
                lines.append("Active price alerts:")
                for a in alerts:
                    lines.append(
                        f"  • {a['symbol']} {a['direction']} ₹{a['price']:,.2f}"
                    )
            else:
                lines.append("Active price alerts: None.")

            lines.append("=== END MEMORY ===")
            return "\n".join(lines)
        except Exception as exc:
            print(f"[JarvisMemory] build_morning_context error: {exc}")
            return "=== JARVIS PERSISTENT MEMORY ===\n[Error loading memory]\n=== END MEMORY ==="
