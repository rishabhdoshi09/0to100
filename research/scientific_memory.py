"""
📚 Scientific Memory — the system stops storing models and starts storing SCIENCE.

Not a Knowledge Base (a wiki of facts) but a research lab's memory: it holds
hypotheses AND failed hypotheses, evidence, drift, retirements, recoveries,
confidence, and historical context — the difference between a fact store and an
institution that remembers what it tried, what worked, and what it has since
stopped believing.

An edge that survives the harness is not a number to tuck into a config; it is a
BELIEF the institution now holds — with the evidence behind it, how confident it
is, what it depends on, and when it was last checked. This is the store the whole
Research OS was building toward: hypotheses are generated, tested (harness +
registry), monitored (drift + edge-timeline), and — here — the survivors are
preserved as durable, self-revalidating knowledge, while the failures are kept as
NEGATIVE knowledge so they are never blindly re-tried.

    Belief:  "Breakouts work in healthy breadth"
    Status:  ACTIVE       Evidence: 184 trades       Confidence: HIGH
    Created: 2026-07      Last validated: 2026-10     Drift: STABLE
    Depends on: feature schema fs_20d1805d…, signal 'breakout'

A belief is not static: `revalidate()` re-checks it against fresh evidence and
drift, and the lifecycle transitions are MECHANICAL (a decaying ACTIVE belief
demotes itself to WATCH; a dead one RETIREs) so the memory can never quietly keep
asserting something the tape has stopped supporting.

Pure lifecycle logic (`_next_status`) is unit-tested; the SQLite layer uses a
monkeypatchable path and fails safe (a broken store degrades to "no directives",
never to a false claim). It owns NO statistical logic — it records what the
harness/drift decided. The Brain only renders these evidence objects.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "scientific_memory.db"

# Lifecycle states. REJECTED is first-class NEGATIVE knowledge ("what doesn't
# work") — preserved so a hypothesis generator never wastes a cycle re-testing it.
ACTIVE = "ACTIVE"
WATCH = "WATCH"
RETIRED = "RETIRED"
REJECTED = "REJECTED"
_STATES = (ACTIVE, WATCH, RETIRED, REJECTED)

_DDL = """
CREATE TABLE IF NOT EXISTS beliefs (
    belief_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    signal TEXT,
    status TEXT NOT NULL,
    evidence_n INTEGER,
    confidence TEXT,               -- HIGH | MEDIUM | LOW
    ev_r REAL,                     -- expected value per trade, R
    drift_status TEXT,             -- STABLE | DECAYING | RECOVERING | ...
    schema_version TEXT,           -- feature schema it was validated under
    hypothesis_id TEXT,            -- link to research/registry experiment
    dependencies TEXT,             -- json list
    notes TEXT,
    created_at TEXT NOT NULL,
    last_validated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status);
CREATE INDEX IF NOT EXISTS idx_beliefs_signal ON beliefs(signal);
CREATE TABLE IF NOT EXISTS belief_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id TEXT NOT NULL,
    at TEXT NOT NULL,
    status TEXT NOT NULL,
    evidence_n INTEGER,
    ev_r REAL,
    event TEXT                     -- CREATED | REVALIDATED
);
CREATE INDEX IF NOT EXISTS idx_belief_events_bid ON belief_events(belief_id, at);
CREATE INDEX IF NOT EXISTS idx_belief_events_at ON belief_events(at);
"""


def _record_event(c: sqlite3.Connection, bid: str, at: str, status: str,
                  evidence_n, ev_r, event: str) -> None:
    """Append a status snapshot to the belief's history — the substrate for the
    Time Machine (reconstruct THAT day's beliefs, not today's) and for 'promoted
    / retired this week' activity. Best-effort."""
    try:
        c.execute("INSERT INTO belief_events (belief_id, at, status, evidence_n, "
                  "ev_r, event) VALUES (?,?,?,?,?,?)",
                  (bid, at, status, evidence_n, ev_r, event))
    except Exception:
        pass


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


# ══════════════════════════════════════════════════════════════════════════════
# Pure lifecycle logic
# ══════════════════════════════════════════════════════════════════════════════

def belief_id(statement: str, signal: str | None = None) -> str:
    """Deterministic id — the same belief (statement + signal) always hashes to
    the same id, so revalidation updates in place and a claim can't be recorded
    twice under two ids."""
    payload = json.dumps({"s": statement.strip().lower(),
                          "sig": (signal or "").strip().lower()}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _next_status(current: str, drift_status: str | None, ev_r: float | None,
                 confidence: str | None) -> str:
    """The mechanical belief lifecycle. Terminal NEGATIVE knowledge (REJECTED)
    and RETIRED never silently reactivate. An ACTIVE belief whose signal is
    DECAYING demotes to WATCH; a DEAD signal (edge-timeline) RETIREs it. A WATCH
    belief earns ACTIVE back only on a healthy, non-decaying, positive-EV read.
    Pure — no I/O."""
    if current == REJECTED:
        return REJECTED                        # negative knowledge is permanent
    ds = (drift_status or "").upper()
    ev = ev_r if ev_r is not None else 0.0

    if ds == "DEAD" or (ev <= 0 and ds == "DECAYING"):
        return RETIRED
    if current == RETIRED:
        # only a clearly healthy re-read revives a retired belief
        return ACTIVE if (ds in ("STRENGTHENING", "RECOVERING") and ev > 0
                          and confidence in ("HIGH", "MEDIUM")) else RETIRED
    if ds == "DECAYING":
        return WATCH
    if current == WATCH:
        healthy = ds in ("STABLE", "STRENGTHENING", "RECOVERING", "")
        return ACTIVE if (healthy and ev > 0 and confidence in ("HIGH", "MEDIUM")) else WATCH
    return ACTIVE


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def record_belief(statement: str, signal: str | None = None, *,
                  status: str = ACTIVE, evidence_n: int = 0,
                  confidence: str = "LOW", ev_r: float | None = None,
                  drift_status: str = "STABLE", schema_version: str | None = None,
                  hypothesis_id: str | None = None, dependencies=None,
                  notes: str = "") -> str:
    """Record (or refresh the metadata of) a belief. Idempotent on belief_id:
    re-recording the same statement+signal updates its evidence but preserves its
    identity, created_at, and — importantly — never resurrects a REJECTED/RETIRED
    belief into ACTIVE by the back door (status only relaxes through revalidate).
    Returns the belief_id."""
    if status not in _STATES:
        status = ACTIVE
    bid = belief_id(statement, signal)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        c = _conn()
        try:
            row = c.execute("SELECT status, created_at FROM beliefs WHERE belief_id=?",
                            (bid,)).fetchone()
            deps = json.dumps(list(dependencies or []))
            if row is None:
                c.execute(
                    "INSERT INTO beliefs (belief_id, statement, signal, status, "
                    "evidence_n, confidence, ev_r, drift_status, schema_version, "
                    "hypothesis_id, dependencies, notes, created_at, "
                    "last_validated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (bid, statement, signal, status, int(evidence_n), confidence,
                     ev_r, drift_status, schema_version, hypothesis_id, deps,
                     notes, now, now))
                _record_event(c, bid, now, status, int(evidence_n), ev_r, "CREATED")
            else:
                # keep a terminal status terminal on a plain re-record
                keep = row["status"] if row["status"] in (REJECTED, RETIRED) else status
                c.execute(
                    "UPDATE beliefs SET statement=?, signal=?, status=?, "
                    "evidence_n=?, confidence=?, ev_r=?, drift_status=?, "
                    "schema_version=?, hypothesis_id=?, dependencies=?, notes=?, "
                    "last_validated_at=? WHERE belief_id=?",
                    (statement, signal, keep, int(evidence_n), confidence, ev_r,
                     drift_status, schema_version, hypothesis_id, deps, notes,
                     now, bid))
                if keep != row["status"]:
                    _record_event(c, bid, now, keep, int(evidence_n), ev_r,
                                  "REVALIDATED")
            c.commit()
        finally:
            c.close()
    except Exception:
        pass
    return bid


def revalidate(belief_id_: str, *, evidence_n: int | None = None,
               confidence: str | None = None, ev_r: float | None = None,
               drift_status: str | None = None) -> dict:
    """Re-check a belief against fresh evidence + drift and transition its status
    mechanically (see `_next_status`). Stamps last_validated_at. Fail-open →
    {'status':'error'|'not_found'}."""
    try:
        c = _conn()
        try:
            row = c.execute("SELECT * FROM beliefs WHERE belief_id=?",
                            (belief_id_,)).fetchone()
            if row is None:
                return {"status": "not_found"}
            new_status = _next_status(
                row["status"], drift_status if drift_status is not None else row["drift_status"],
                ev_r if ev_r is not None else row["ev_r"],
                confidence if confidence is not None else row["confidence"])
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            new_n = int(evidence_n) if evidence_n is not None else row["evidence_n"]
            new_ev = ev_r if ev_r is not None else row["ev_r"]
            c.execute(
                "UPDATE beliefs SET status=?, evidence_n=?, confidence=?, ev_r=?, "
                "drift_status=?, last_validated_at=? WHERE belief_id=?",
                (new_status, new_n,
                 confidence if confidence is not None else row["confidence"],
                 new_ev,
                 drift_status if drift_status is not None else row["drift_status"],
                 now, belief_id_))
            # always log a revalidation event (status may or may not have moved —
            # the history needs the check-in either way for the Time Machine).
            _record_event(c, belief_id_, now, new_status, new_n, new_ev,
                          "REVALIDATED")
            c.commit()
        finally:
            c.close()
        return {"status": "revalidated", "belief_status": new_status,
                "from": row["status"]}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def promote_from_experiment(hypothesis_id: str, statement: str,
                            signal: str | None, *, evidence_n: int,
                            confidence: str, ev_r: float,
                            schema_version: str | None = None,
                            dependencies=None) -> str:
    """Bridge from the experiment registry: a PROMOTED hypothesis becomes an
    ACTIVE belief, carrying its evidence and the schema version it was validated
    under. This is the seam that turns a passed experiment into retained
    knowledge (Idea → Experiment → Validated → Knowledge Base → Brain)."""
    bid = record_belief(statement, signal, status=ACTIVE, evidence_n=evidence_n,
                        confidence=confidence, ev_r=ev_r, drift_status="STABLE",
                        schema_version=schema_version, hypothesis_id=hypothesis_id,
                        dependencies=dependencies,
                        notes="promoted from experiment " + hypothesis_id)
    # record the provenance spine in the Evidence Graph (fail-open — the belief
    # stands on its own even if the graph write fails).
    try:
        from research import evidence_graph as _eg
        _eg.record_promotion(statement, hypothesis_id, bid,
                             hypothesis_label=statement, belief_label=statement,
                             schema_version=schema_version, evidence_n=evidence_n)
    except Exception:
        pass
    return bid


def sync_from_registry(min_n: int = 30) -> dict:
    """Auto-sync: every PROMOTED experiment in the registry becomes an ACTIVE
    belief here (with its Evidence-Graph provenance), closing the loop Idea →
    Experiment → Validated → Scientific Memory → Brain WITHOUT a manual bridge.
    Idempotent and lifecycle-safe: a belief that already exists is left ALONE
    (never clobbered back to ACTIVE — its drift lifecycle is authoritative).
    Returns {'synced': n}. Fail-open."""
    try:
        from research import registry as _reg
    except Exception:
        return {"synced": 0}
    synced = 0
    try:
        for exp in _reg.list_experiments(status="PROMOTED"):
            hid = exp.get("hypothesis_id") or ""
            name = exp.get("name") or hid
            try:
                result = json.loads(exp.get("result") or "{}")
            except Exception:
                result = {}
            signal = result.get("signal")
            # idempotency + lifecycle safety: skip anything already remembered
            if get_belief(belief_id(name, signal)) is not None:
                continue
            evidence_n = int(result.get("n") or result.get("evidence_n") or 0)
            if evidence_n < min_n:
                continue                       # below the harness floor — no belief
            ev_r = float(result.get("ev_r") or result.get("mean_r") or 0.0)
            conf = result.get("confidence") or (
                "HIGH" if evidence_n >= 100 else "MEDIUM")
            promote_from_experiment(
                hid, name, signal, evidence_n=evidence_n, confidence=conf,
                ev_r=ev_r, schema_version=result.get("schema_version"))
            synced += 1
    except Exception:
        return {"synced": synced}
    return {"synced": synced}


def record_negative(statement: str, signal: str | None = None, *,
                    evidence_n: int = 0, notes: str = "") -> str:
    """Preserve NEGATIVE knowledge — a claim the evidence rejected. Kept so the
    system (and a future hypothesis generator) never wastes a cycle re-testing
    what's already been disproven."""
    return record_belief(statement, signal, status=REJECTED, evidence_n=evidence_n,
                         confidence="LOW", drift_status="", notes=notes)


def is_known_dead(statement: str, signal: str | None = None) -> bool:
    """Has this exact claim already been REJECTED or RETIRED? A cheap guard for a
    hypothesis generator to skip settled ground. Fail-open → False."""
    b = get_belief(belief_id(statement, signal))
    return bool(b and b["status"] in (REJECTED, RETIRED))


def get_belief(belief_id_: str) -> dict | None:
    try:
        c = _conn()
        try:
            row = c.execute("SELECT * FROM beliefs WHERE belief_id=?",
                            (belief_id_,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["dependencies"] = json.loads(d["dependencies"] or "[]")
            return d
        finally:
            c.close()
    except Exception:
        return None


def list_beliefs(status: str | None = None) -> list[dict]:
    try:
        c = _conn()
        try:
            if status:
                rows = c.execute("SELECT * FROM beliefs WHERE status=? ORDER BY "
                                 "ev_r DESC", (status,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM beliefs ORDER BY status, "
                                 "ev_r DESC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["dependencies"] = json.loads(d["dependencies"] or "[]")
                out.append(d)
            return out
        finally:
            c.close()
    except Exception:
        return []


def belief_history(belief_id_: str) -> list[dict]:
    """Ordered status history for one belief (oldest→newest). Fail-open → []."""
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT at, status, evidence_n, ev_r, event FROM "
                             "belief_events WHERE belief_id=? ORDER BY id ASC",
                             (belief_id_,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            c.close()
    except Exception:
        return []


def beliefs_as_of(iso_date: str) -> list[dict]:
    """TIME MACHINE — reconstruct each belief's status AS OF a date (not today's).
    For every belief created on/before the date, its status is the last recorded
    event with at ≤ date; beliefs not yet created are omitted. This is what lets
    the system answer 'what did we believe on 12 Jan?' years later. Fail-open."""
    cutoff = str(iso_date)
    try:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT be.belief_id, be.status, be.evidence_n, be.ev_r, be.at, "
                "b.statement, b.signal FROM belief_events be "
                "JOIN beliefs b ON b.belief_id = be.belief_id "
                "WHERE be.at <= ? ORDER BY be.belief_id, be.id",
                (cutoff,)).fetchall()
        finally:
            c.close()
    except Exception:
        return []
    latest: dict[str, dict] = {}
    for r in rows:                                    # last event ≤ cutoff wins
        latest[r["belief_id"]] = {"belief_id": r["belief_id"],
                                  "statement": r["statement"], "signal": r["signal"],
                                  "status": r["status"], "evidence_n": r["evidence_n"],
                                  "ev_r": r["ev_r"], "as_of_event": r["at"]}
    return sorted(latest.values(), key=lambda d: (d["status"], -(d["ev_r"] or 0)))


def recent_activity(days: int = 7) -> dict:
    """What the research org did lately — beliefs newly promoted (CREATED as
    ACTIVE), moved to WATCH, RETIRED, or REJECTED within the window. Feeds the
    dashboard's 'this week' line. Fail-open → zeros."""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    tally = {"promoted": 0, "to_watch": 0, "retired": 0, "rejected": 0}
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT status, event FROM belief_events WHERE at >= ?",
                             (cutoff,)).fetchall()
        finally:
            c.close()
    except Exception:
        return tally
    for r in rows:
        if r["event"] == "CREATED" and r["status"] == ACTIVE:
            tally["promoted"] += 1
        elif r["status"] == WATCH:
            tally["to_watch"] += 1
        elif r["status"] == RETIRED:
            tally["retired"] += 1
        elif r["status"] == REJECTED:
            tally["rejected"] += 1
    return tally


def overdue_for_review(max_age_days: int = 30) -> list[dict]:
    """ACTIVE/WATCH beliefs whose last validation is older than max_age_days —
    RESEARCH DEBT (a belief still asserted but not re-checked in a while).
    Fail-open → []."""
    from datetime import datetime
    out = []
    for b in list_beliefs():
        if b["status"] not in (ACTIVE, WATCH):
            continue
        lv = b.get("last_validated_at")
        try:
            age = (datetime.now() - datetime.fromisoformat(str(lv)[:19])).days
        except Exception:
            continue
        if age > max_age_days:
            out.append({"belief_id": b["belief_id"], "statement": b["statement"],
                        "status": b["status"], "age_days": age})
    return sorted(out, key=lambda d: -d["age_days"])


# ══════════════════════════════════════════════════════════════════════════════
# Brain-facing evidence objects (thin — the KB records, the Brain renders)
# ══════════════════════════════════════════════════════════════════════════════

def belief_directives(max_items: int = 2) -> list[dict]:
    """Surface beliefs that just changed character: an ACTIVE belief now on WATCH
    (its edge is decaying — the institution's own knowledge is being questioned),
    or a freshly RETIRED one. Demote-only, evidence-gated. Fail-open → []."""
    dirs: list[dict] = []
    for b in list_beliefs(WATCH):
        dirs.append({"severity": "warn",
                     "text": f"📚 Belief under review: “{b['statement']}” — its "
                             f"edge is decaying ({b.get('ev_r', 0) or 0:+.2f}R, "
                             f"n={b.get('evidence_n', 0)}). Trust it less until it "
                             f"stabilises."})
        if len(dirs) >= max_items:
            return dirs
    for b in list_beliefs(RETIRED)[:max_items - len(dirs)]:
        dirs.append({"severity": "info",
                     "text": f"📚 Retired belief: “{b['statement']}” — stopped "
                             f"working; no longer acted on."})
        if len(dirs) >= max_items:
            break
    return dirs
