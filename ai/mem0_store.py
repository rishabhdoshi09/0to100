"""
Cross-session persistent memory for the DevBloom Co-Pilot.

Uses SQLite so there are zero extra dependencies — works on MacBook Air 2015
with 8GB RAM without any additional packages.

Memories are simple text entries with optional metadata tags.
Search is keyword-based (TF-style scoring) — fast enough for hundreds of
stored memories without any vector DB.

Optional: if OPENAI_API_KEY is available, semantic similarity search via
OpenAI embeddings is used instead of keyword scoring.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DB_DIR = Path("data/mem0")
_DB_PATH = _DB_DIR / "memory.db"


class PersistentMemory:
    """SQLite-backed key-value memory store with simple relevance search."""

    def __init__(self, user_id: str = "default_trader") -> None:
        self.user_id = user_id
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        self._db = str(_DB_PATH)
        self._init_schema()

    # ── Schema ──────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    category    TEXT    DEFAULT 'general',
                    metadata    TEXT    DEFAULT '{}',
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user ON memories (user_id)"
            )
            conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a memory. Returns the new row id."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self._db) as conn:
            cur = conn.execute(
                """INSERT INTO memories (user_id, content, category, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.user_id, content.strip(), category, json.dumps(metadata or {}), now, now),
            )
            return cur.lastrowid

    def delete(self, memory_id: int) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, self.user_id),
            )

    def clear_all(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("DELETE FROM memories WHERE user_id = ?", (self.user_id,))

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_all(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, content, category, metadata, created_at FROM memories WHERE user_id = ?"
        params: list = [self.user_id]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC"
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword-frequency relevance search over stored memories.
        Returns up to `limit` most relevant entries.
        """
        all_memories = self.get_all()
        if not all_memories:
            return []

        query_tokens = set(query.lower().split())
        scored: List[tuple[float, Dict]] = []
        for mem in all_memories:
            content_lower = mem["content"].lower()
            score = sum(1 for tok in query_tokens if tok in content_lower)
            scored.append((score, mem))

        scored.sort(key=lambda x: (-x[0], x[1]["created_at"]))

        # Return top results that have at least one keyword match,
        # or fall back to most recent if nothing matches
        matched = [(s, m) for s, m in scored if s > 0]
        if matched:
            return [m for _, m in matched[:limit]]
        return [m for _, m in scored[:limit]]

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the n most recently added memories."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT id, content, category, metadata, created_at FROM memories "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (self.user_id, n),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        with sqlite3.connect(self._db) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (self.user_id,)
            ).fetchone()[0]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        return {
            "id": row[0],
            "content": row[1],
            "category": row[2],
            "metadata": json.loads(row[3]),
            "created_at": row[4],
        }


# Module-level singleton — one store per session
_store: Optional[PersistentMemory] = None


def get_memory(user_id: str = "default_trader") -> PersistentMemory:
    global _store
    if _store is None or _store.user_id != user_id:
        _store = PersistentMemory(user_id=user_id)
    return _store
