"""Durable observation ledger for the learning-policy ladder.

A policy's sample size must count *independent* observations. Two failure
modes made it count copies instead:

* Every paper cycle re-freezes the same rejected names. Owner's Mac
  2026-11-11: 36 cycles froze 175 symbols as 3,744 forward-ledger rows, each
  settled and ingested on its own, so ``REJECT::LOW_QUALITY_SETUP`` reached
  n=2,989 from a few hundred symbol-days.
* The taken-trade ingest kept ``sorted(keys)[-2000:]``, the alphabetically
  last keys, so beyond 2,000 trades early symbols would be ingested again.

``claim()`` is an ``INSERT OR IGNORE`` on ``(scope, observation_key)``; it
returns True once per key, forever. Callers claim BEFORE updating a policy, so
a crash between the two can only under-count, never double-count.

``distinct_sessions()`` feeds the session-diversity floor in
``learning_policy_store``: outcomes from one session share one market path,
so a policy needs evidence spread across sessions, not just a large n.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS observations ("
    " scope TEXT NOT NULL,"
    " observation_key TEXT NOT NULL,"
    " session TEXT NOT NULL,"
    " recorded_at TEXT NOT NULL,"
    " PRIMARY KEY (scope, observation_key))"
)


def ledger_path_for(policy_file: str | Path) -> Path:
    """The ledger lives beside the policy store it guards (tests isolate both)."""
    p = Path(policy_file)
    return p.with_name(f"{p.stem}.observations.sqlite3")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(_SCHEMA)
    return conn


def claim(path: str | Path, scope: str, observation_key: str, session: str) -> bool:
    """Record ``observation_key`` under ``scope``. True only the first time."""
    if not scope or not observation_key:
        raise ValueError("scope and observation_key are required")
    with _LOCK:
        conn = _connect(Path(path))
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO observations(scope, observation_key, session, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (str(scope), str(observation_key), str(session or ""),
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()


def seen(path: str | Path, scope: str, observation_key: str) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute(
                "SELECT 1 FROM observations WHERE scope=? AND observation_key=?",
                (str(scope), str(observation_key)),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def distinct_sessions(path: str | Path, scope: str) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session) FROM observations WHERE scope=? AND session<>''",
                (str(scope),),
            ).fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()


def count(path: str | Path, scope: str) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute("SELECT COUNT(*) FROM observations WHERE scope=?", (str(scope),)).fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()
