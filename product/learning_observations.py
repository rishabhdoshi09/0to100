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

Exactly once across crashes (cross-review A-3, 2026-11-21). "Claim, then
apply" under-counted on a crash in between: the claim made every retry a
duplicate, so the learning event was lost for good. Claims are now
TWO-PHASE:

* ``claim(..., payload=...)`` records the observation as CLAIMED together
  with everything needed to apply it (one SQLite commit);
* the caller applies it and then ``mark_applied()``;
* a CLAIMED row left by a crash is replayed by its owner
  (``learning_policy_store.recover_claims``), which first checks whether the
  apply already landed, so it is applied exactly once.

Rows from before this change are LEGACY: still deduplicated, never replayed
(whether they were applied is unknowable). ``writer_lock()`` serialises
claim → apply → mark across processes.

The OUTBOX (``enqueue_event`` / ``pending_events`` / ``finish_event``) is
the durable hand-off from settlement to learning: an event is written BEFORE
the forward ledger marks its row settled, so a crash after the ledger write
cannot strand a settled outcome that learning never saw.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

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


LEGACY = "LEGACY"        # claimed before two-phase claims existed: dedupe only, never replayed
CLAIMED = "CLAIMED"      # claimed, apply not yet confirmed
APPLIED = "APPLIED"      # applied (or single-phase: the claim itself is the effect)

PENDING = "PENDING"      # outbox: settled, learning not yet confirmed
DONE = "DONE"            # outbox: learning applied (or deliberately skipped, with its reason)
QUARANTINED = "QUARANTINED"  # outbox: corrupt or repeatedly failing: kept visible, never applied
MAX_EVENT_ATTEMPTS = 5

_OUTBOX = (
    "CREATE TABLE IF NOT EXISTS outbox ("
    " event_id TEXT PRIMARY KEY,"
    " kind TEXT NOT NULL,"
    " payload TEXT NOT NULL,"
    " state TEXT NOT NULL,"
    " attempts INTEGER NOT NULL DEFAULT 0,"
    " last_error TEXT NOT NULL DEFAULT '',"
    " result TEXT NOT NULL DEFAULT '',"
    " created_at TEXT NOT NULL,"
    " finished_at TEXT NOT NULL DEFAULT '')"
)


def _schema_current(conn: sqlite3.Connection) -> bool:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
    if not {"state", "payload"} <= cols:
        return False
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='outbox'").fetchone() is not None


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    # Claims and applications must reach stable storage BEFORE the policy and ledger writes
    # that follow them: with every write in the chain a real barrier, a power cut can only
    # lose a SUFFIX of it, never apply a policy whose claim was lost (lifecycle audit F2,
    # 24 Nov: one observation credited twice). FULL + F_FULLFSYNC per commit, barriered
    # checkpoints (make_sqlite_checkpoint_safe alone adds F_FULLFSYNC only below FULL).
    from core.atomic_io import make_sqlite_durable
    make_sqlite_durable(conn)
    if _schema_current(conn):
        return conn
    # Create / migrate under ONE write transaction: two processes opening a
    # legacy ledger at once must not both see the old columns and both ALTER
    # (overnight review L-09). BEGIN IMMEDIATE serialises them; the second
    # re-reads the schema inside its own transaction and finds nothing to do.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(_SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
        if "state" not in cols:
            # Existing claims cannot say whether their apply landed: LEGACY.
            conn.execute(f"ALTER TABLE observations ADD COLUMN state TEXT NOT NULL DEFAULT '{LEGACY}'")
        if "payload" not in cols:
            conn.execute("ALTER TABLE observations ADD COLUMN payload TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS observations_state ON observations(state)")
        conn.execute(_OUTBOX)
        conn.execute("CREATE INDEX IF NOT EXISTS outbox_state ON outbox(state)")
        conn.commit()
    except BaseException:
        conn.rollback()
        conn.close()
        raise
    return conn


@contextlib.contextmanager
def writer_lock(path: str | Path) -> Iterator[None]:
    """Cross-process exclusive lock for claim → apply → mark (one writer per policy store).

    Not re-entrant: never take it twice in one call chain.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.with_name(target.name + ".lock"), "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def claim(path: str | Path, scope: str, observation_key: str, session: str,
          payload: Mapping[str, Any] | None = None) -> bool:
    """Record ``observation_key`` under ``scope``. True only the first time.

    Without ``payload`` the claim is single-phase (APPLIED at once: the claim
    is the whole effect). With ``payload`` it is CLAIMED until
    ``mark_applied``; the payload is what a replay needs to apply it.
    """
    if not scope or not observation_key:
        raise ValueError("scope and observation_key are required")
    state = APPLIED if payload is None else CLAIMED
    body = "" if payload is None else json.dumps(dict(payload), sort_keys=True, default=str)
    with _LOCK:
        conn = _connect(Path(path))
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO observations(scope, observation_key, session, recorded_at, state, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(scope), str(observation_key), str(session or ""),
                 datetime.now(timezone.utc).isoformat(), state, body),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()


def claim_state(path: str | Path, scope: str, observation_key: str) -> str:
    """'' when unclaimed, else LEGACY / CLAIMED / APPLIED."""
    target = Path(path)
    if not target.exists():
        return ""
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute("SELECT state FROM observations WHERE scope=? AND observation_key=?",
                               (str(scope), str(observation_key))).fetchone()
            return "" if row is None else str(row[0] or LEGACY)
        finally:
            conn.close()


def claim_payload(path: str | Path, scope: str, observation_key: str) -> dict[str, Any] | None:
    """The payload stored with a claim; None when unclaimed, empty, or not a JSON object."""
    target = Path(path)
    if not target.exists():
        return None
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute("SELECT payload FROM observations WHERE scope=? AND observation_key=?",
                               (str(scope), str(observation_key))).fetchone()
        finally:
            conn.close()
    if row is None or not row[0]:
        return None
    try:
        body = json.loads(row[0])
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def mark_applied(path: str | Path, scope: str, observation_key: str) -> None:
    with _LOCK:
        conn = _connect(Path(path))
        try:
            conn.execute("UPDATE observations SET state=? WHERE scope=? AND observation_key=? AND state=?",
                         (APPLIED, str(scope), str(observation_key), CLAIMED))
            conn.commit()
        finally:
            conn.close()


def claimed_unapplied(path: str | Path) -> list[dict[str, Any]]:
    """Every CLAIMED row (a crash between claim and apply): scope, key, session, payload."""
    target = Path(path)
    if not target.exists():
        return []
    with _LOCK:
        conn = _connect(target)
        try:
            rows = conn.execute("SELECT scope, observation_key, session, payload FROM observations "
                                "WHERE state=? ORDER BY recorded_at", (CLAIMED,)).fetchall()
        finally:
            conn.close()
    out = []
    for scope, key, session, payload in rows:
        try:
            body = json.loads(payload) if payload else {}
        except ValueError:
            body = None
        out.append({"scope": scope, "observation_key": key, "session": session, "payload": body})
    return out


def scope_payloads(path: str | Path, scope: str, *, states: tuple[str, ...] = (CLAIMED, APPLIED),
                   contains: str = "") -> list[dict[str, Any]]:
    """Units of ``scope`` in ``states`` that carry a payload (optionally one containing ``contains``).

    One read: observation_key, state and the parsed payload (None when it is not JSON).
    """
    target = Path(path)
    if not target.exists() or not states:
        return []
    marks = ",".join("?" * len(states))
    sql = (f"SELECT observation_key, state, payload FROM observations "
           f"WHERE scope=? AND state IN ({marks}) AND payload != ''")
    args: list[Any] = [str(scope), *states]
    if contains:
        sql += " AND instr(payload, ?) > 0"
        args.append(str(contains))
    with _LOCK:
        conn = _connect(target)
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
    out = []
    for key, state, payload in rows:
        try:
            body = json.loads(payload)
        except ValueError:
            body = None
        out.append({"observation_key": key, "state": state, "payload": body})
    return out


# ── outbox: the durable settlement → learning hand-off ────────────────────────

def enqueue_event(path: str | Path, event_id: str, kind: str, payload: Mapping[str, Any]) -> bool:
    """Durably record a learning event (idempotent on event_id). True when new."""
    if not event_id or not kind:
        raise ValueError("event_id and kind are required")
    with _LOCK:
        conn = _connect(Path(path))
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO outbox(event_id, kind, payload, state, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(event_id), str(kind), json.dumps(dict(payload), sort_keys=True, default=str), PENDING,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()


def upsert_pending_event(path: str | Path, event_id: str, kind: str, payload: Mapping[str, Any]) -> str:
    """Queue an event, REPLACING the payload of a still-PENDING one (overnight review L-02).

    Call it only for a settlement that is not yet committed: a PENDING event
    for such a row was never committed either (a crash before the ledger
    write), so the new measurement supersedes it. Returns NEW, REPLACED, or
    the state of an event that was already finished (DONE / QUARANTINED),
    which is left untouched.
    """
    if not event_id or not kind:
        raise ValueError("event_id and kind are required")
    body = json.dumps(dict(payload), sort_keys=True, default=str)
    with _LOCK:
        conn = _connect(Path(path))
        try:
            conn.execute("BEGIN IMMEDIATE")          # read-decide-write as one cross-process step
            row = conn.execute("SELECT state FROM outbox WHERE event_id=?", (str(event_id),)).fetchone()
            if row is None:
                conn.execute("INSERT INTO outbox(event_id, kind, payload, state, created_at) VALUES (?, ?, ?, ?, ?)",
                             (str(event_id), str(kind), body, PENDING, datetime.now(timezone.utc).isoformat()))
                result = "NEW"
            elif str(row[0]) == PENDING:
                conn.execute("UPDATE outbox SET payload=?, kind=? WHERE event_id=? AND state=?",
                             (body, str(kind), str(event_id), PENDING))
                result = "REPLACED"
            else:
                result = str(row[0])
            conn.commit()
            return result
        finally:
            conn.close()


def event_state(path: str | Path, event_id: str) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    with _LOCK:
        conn = _connect(target)
        try:
            row = conn.execute("SELECT state FROM outbox WHERE event_id=?", (str(event_id),)).fetchone()
            return "" if row is None else str(row[0])
        finally:
            conn.close()


def pending_events(path: str | Path, kind: str | None = None) -> list[dict[str, Any]]:
    """PENDING events, oldest first. A payload that is not valid JSON is returned as None (corrupt)."""
    target = Path(path)
    if not target.exists():
        return []
    with _LOCK:
        conn = _connect(target)
        try:
            q = "SELECT event_id, kind, payload, attempts FROM outbox WHERE state=?"
            args: tuple = (PENDING,)
            if kind:
                q += " AND kind=?"
                args += (str(kind),)
            rows = conn.execute(q + " ORDER BY created_at, event_id", args).fetchall()
        finally:
            conn.close()
    out = []
    for event_id, k, payload, attempts in rows:
        try:
            body = json.loads(payload)
            if not isinstance(body, dict):
                body = None
        except ValueError:
            body = None
        out.append({"event_id": event_id, "kind": k, "payload": body, "attempts": int(attempts or 0)})
    return out


def finish_event(path: str | Path, event_id: str, *, result: str = "", error: str = "") -> str:
    """Mark an event DONE (``error`` empty) or count a failed attempt (QUARANTINED after the limit)."""
    with _LOCK:
        conn = _connect(Path(path))
        try:
            now = datetime.now(timezone.utc).isoformat()
            if not error:
                conn.execute("UPDATE outbox SET state=?, result=?, finished_at=?, last_error='' "
                             "WHERE event_id=? AND state=?", (DONE, str(result)[:200], now, str(event_id), PENDING))
                state = DONE
            else:
                row = conn.execute("SELECT attempts FROM outbox WHERE event_id=?", (str(event_id),)).fetchone()
                attempts = int((row or [0])[0] or 0) + 1
                state = QUARANTINED if attempts >= MAX_EVENT_ATTEMPTS else PENDING
                conn.execute("UPDATE outbox SET attempts=?, last_error=?, state=?, "
                             "finished_at=CASE WHEN ?=? THEN ? ELSE finished_at END WHERE event_id=? AND state=?",
                             (attempts, str(error)[:300], state, state, QUARANTINED, now, str(event_id), PENDING))
            conn.commit()
            return state
        finally:
            conn.close()


def quarantine_event(path: str | Path, event_id: str, reason: str) -> None:
    """A corrupt or ineligible event: kept for audit, never applied."""
    with _LOCK:
        conn = _connect(Path(path))
        try:
            conn.execute("UPDATE outbox SET state=?, last_error=?, finished_at=? WHERE event_id=? AND state=?",
                         (QUARANTINED, str(reason)[:300], datetime.now(timezone.utc).isoformat(),
                          str(event_id), PENDING))
            conn.commit()
        finally:
            conn.close()


def outbox_counts(path: str | Path) -> dict[str, int]:
    target = Path(path)
    if not target.exists():
        return {}
    with _LOCK:
        conn = _connect(target)
        try:
            return {str(s): int(n) for s, n in conn.execute("SELECT state, COUNT(*) FROM outbox GROUP BY state")}
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
