"""Durable owner-control queue shared by the retail UI and autonomy process."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import time

ENABLE_PAPER_AUTO = "ENABLE_PAPER_AUTO"
PAUSE_NEW_PAPER_ENTRIES = "PAUSE_NEW_PAPER_ENTRIES"
RESUME_NEW_PAPER_ENTRIES = "RESUME_NEW_PAPER_ENTRIES"
REFRESH_DATA_NOW = "REFRESH_DATA_NOW"
RUN_SCAN_NOW = "RUN_SCAN_NOW"
RUN_CYCLE_NOW = "RUN_CYCLE_NOW"
REFRESH_NEWS_NOW = "REFRESH_NEWS_NOW"
RUN_RESEARCH_NOW = "RUN_RESEARCH_NOW"
RUN_LONG_TERM_SCAN_NOW = "RUN_LONG_TERM_SCAN_NOW"
REFRESH_LONG_TERM_NOW = "REFRESH_LONG_TERM_NOW"
TRACK_LONG_TERM_IDEA = "TRACK_LONG_TERM_IDEA"
HALT_AUTONOMY = "HALT_AUTONOMY"
RESUME_AUTONOMY = "RESUME_AUTONOMY"
OBSERVE_ONLY_TODAY = "OBSERVE_ONLY_TODAY"
CLEAR_OBSERVE_ONLY = "CLEAR_OBSERVE_ONLY"
VALID_CONTROLS = {
    ENABLE_PAPER_AUTO, PAUSE_NEW_PAPER_ENTRIES, RESUME_NEW_PAPER_ENTRIES,
    REFRESH_DATA_NOW, RUN_SCAN_NOW, RUN_CYCLE_NOW, REFRESH_NEWS_NOW, RUN_RESEARCH_NOW,
    RUN_LONG_TERM_SCAN_NOW, REFRESH_LONG_TERM_NOW, TRACK_LONG_TERM_IDEA,
    HALT_AUTONOMY, RESUME_AUTONOMY, OBSERVE_ONLY_TODAY, CLEAR_OBSERVE_ONLY,
}
PENDING = "PENDING"
PROCESSED = "PROCESSED"
FAILED = "FAILED"


@dataclass(frozen=True)
class Control:
    control_id: str
    control_type: str
    requested_at: float
    requested_by: str
    value: str
    reason: str
    status: str
    processed_at: float | None = None


class ControlStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS controls (
          control_id TEXT PRIMARY KEY, control_type TEXT NOT NULL, requested_at REAL NOT NULL,
          requested_by TEXT NOT NULL, value TEXT, reason TEXT, status TEXT NOT NULL,
          processed_at REAL, result TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_controls_status ON controls(status, requested_at);
        """)
        self._db.commit()

    @staticmethod
    def _row(r) -> Control:
        return Control(r["control_id"], r["control_type"], r["requested_at"], r["requested_by"],
                       r["value"] or "", r["reason"] or "", r["status"], r["processed_at"])

    def request(self, control_type: str, *, value="", reason="", requested_by="owner") -> Control:
        if control_type not in VALID_CONTROLS:
            raise ValueError(f"unsupported control {control_type}")
        cid = uuid.uuid4().hex[:16]
        now = time()
        with self._lock:
            self._db.execute("INSERT INTO controls VALUES(?,?,?,?,?,?,?,?,?)",
                             (cid, control_type, now, requested_by, json.dumps(value, default=str),
                              reason, PENDING, None, ""))
            self._db.commit()
            row = self._db.execute("SELECT * FROM controls WHERE control_id=?", (cid,)).fetchone()
        return self._row(row)

    def pending(self, limit=50) -> list[Control]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM controls WHERE status=? ORDER BY requested_at LIMIT ?",
                                    (PENDING, limit)).fetchall()
        return [self._row(r) for r in rows]

    def finish(self, control_id: str, *, ok=True, result="") -> None:
        with self._lock:
            self._db.execute("UPDATE controls SET status=?, processed_at=?, result=? WHERE control_id=?",
                             (PROCESSED if ok else FAILED, time(), result, control_id))
            self._db.commit()

    def recent(self, limit=50) -> list[Control]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM controls ORDER BY requested_at DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def close(self):
        with self._lock:
            self._db.close()


def request_control(control_type: str, *, value="", reason="", root=None) -> Control:
    from research.autonomy import default_root
    store = ControlStore(Path(root or default_root()) / "controls.db")
    try:
        return store.request(control_type, value=value, reason=reason)
    finally:
        store.close()
