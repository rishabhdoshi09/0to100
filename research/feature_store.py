"""
🧊 Feature Store — immutable, versioned SNAPSHOTS of every observation.

The piece almost nobody builds, and the one that makes research reproducible
forever: feature vectors are FROZEN at observation time and never recomputed
with tomorrow's code. An experiment run in 2026 reads exactly the numbers that
existed in 2026, stamped with the schema version that produced them — so a
feature improvement 18 months later can't silently rewrite history and break
reproducibility.

Contract:
  • snapshot()   — canonicalise → validate → FREEZE. An observation_id is
                   write-once: re-snapshotting the same id is refused (the vector
                   is immutable). The realised OUTCOME is the one thing added
                   later (it's the label, learned after the fact — not a
                   recomputation of the features).
  • set_outcome()— attach/settle the label on an existing frozen observation.
  • load_matrix()— the research consumption API: aligned X / y / ids for a kind,
                   plus the set of schema versions the rows were frozen under
                   (mixed versions are surfaced, never silently blended).
  • feature_coverage() — data-quality lens: fill-rate + problem counts per
                   feature, so a decaying data feed is caught as a data problem,
                   not misread as a decaying edge.

Thin SQLite layer on a monkeypatchable path; fail-open on reads, and writes
return a status dict instead of raising (a broken store must never crash a scan).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from research import feature_schema as _S
from core.runtime_paths import logs_path

_DB_PATH = logs_path("feature_store.db")

_WRITE_BATCH = threading.local()

# observation kinds — the whole point is that a REJECTION or NEAR_MISS is as much
# an observation as a TRADE (non-event learning needs them on equal footing).
KINDS = ("SCAN", "TRADE", "REJECTION", "NEAR_MISS", "DECISION")

_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,               -- observation time (point-in-time)
    symbol TEXT,
    kind TEXT NOT NULL,
    outcome REAL,                   -- realised label (R or %), filled later
    outcome_meta TEXT,              -- json provenance for the settled label
    schema_version TEXT NOT NULL,   -- which schema froze this vector
    features TEXT NOT NULL,         -- json canonical vector
    validation TEXT,                -- json problems list
    reason TEXT,                    -- REJECTION cause code (structured)
    subtype TEXT,                   -- NEAR_MISS kind: ALMOST | FADED
    meta TEXT,                      -- json: gap detail, extra structured context
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_kind ON observations(kind, ts);
CREATE INDEX IF NOT EXISTS idx_obs_symbol ON observations(symbol);
CREATE INDEX IF NOT EXISTS idx_obs_reason ON observations(kind, reason);
"""

# Columns added after the table's first release — migrated in on connect so an
# existing store keeps working without a manual step.
_MIGRATIONS = (("reason", "TEXT"), ("subtype", "TEXT"), ("meta", "TEXT"), ("outcome_meta", "TEXT"))


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    have = {r["name"] for r in c.execute("PRAGMA table_info(observations)")}
    for col, decl in _MIGRATIONS:
        if col not in have:
            c.execute(f"ALTER TABLE observations ADD COLUMN {col} {decl}")
    c.commit()
    return c


# ══════════════════════════════════════════════════════════════════════════════
# Write path — freeze once, settle the label later
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def feature_write_batch():
    """Commit a bounded group of immutable observations in one transaction.

    Decision-board construction can freeze dozens of observations at once. On
    removable/sparsebundle storage, opening and durably committing SQLite once
    per row creates extreme fsync latency. This context preserves the exact
    write-once snapshot contract while amortizing that durability cost across
    one board. Nested callers reuse the existing transaction.
    """
    existing = getattr(_WRITE_BATCH, "connection", None)
    if existing is not None:
        yield
        return

    connection = _conn()
    _WRITE_BATCH.connection = connection
    try:
        connection.execute("BEGIN")
        yield
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        finally:
            raise
    finally:
        try:
            delattr(_WRITE_BATCH, "connection")
        except AttributeError:
            pass
        connection.close()


def snapshot(observation_id: str, symbol: str, kind: str, raw_features: dict,
             ts: str | None = None, outcome: float | None = None,
             ages: dict | None = None, reason: str | None = None,
             subtype: str | None = None, meta: dict | None = None) -> dict:
    """Freeze one observation. Canonicalises + validates the raw features, then
    writes them WRITE-ONCE under the current schema version. `reason` (a
    structured REJECTION cause), `subtype` (a NEAR_MISS kind), and `meta` (json
    context such as a threshold gap) are frozen alongside. Returns
    {status, schema_version, problems}:
      • 'frozen' — newly stored.
      • 'exists' — id already frozen; nothing changed (immutability enforced).
      • 'error'  — store failure (fail-open, never raises).
    IMPOSSIBLE/STALE features are stored too (with their validation flags) so the
    data-quality problem is auditable — but the caller sees them in `problems`."""
    if kind not in KINDS:
        return {"status": "error", "reason": f"unknown kind {kind!r}",
                "schema_version": _S.SCHEMA_VERSION, "problems": []}
    try:
        canonical = _S.canonicalize(raw_features or {})
        report = _S.validate_vector(canonical, ages)
        batched = getattr(_WRITE_BATCH, "connection", None)
        c = batched or _conn()
        owns_connection = batched is None
        try:
            exists = c.execute("SELECT 1 FROM observations WHERE observation_id=?",
                               (observation_id,)).fetchone()
            if exists:
                return {"status": "exists", "schema_version": _S.SCHEMA_VERSION,
                        "problems": report.problems}
            c.execute(
                "INSERT INTO observations (observation_id, ts, symbol, kind, "
                "outcome, schema_version, features, validation, reason, subtype, "
                "meta, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (observation_id, ts or time.strftime("%Y-%m-%dT%H:%M:%S"),
                 (symbol or "").upper(), kind,
                 float(outcome) if outcome is not None else None,
                 _S.SCHEMA_VERSION, json.dumps(canonical),
                 json.dumps(report.problems), reason, subtype,
                 json.dumps(meta) if meta is not None else None,
                 time.strftime("%Y-%m-%dT%H:%M:%S")))
            if owns_connection:
                c.commit()
        finally:
            if owns_connection:
                c.close()
        return {"status": "frozen", "schema_version": _S.SCHEMA_VERSION,
                "problems": report.problems, "valid": report.ok}
    except Exception as exc:
        return {"status": "error", "reason": str(exc),
                "schema_version": _S.SCHEMA_VERSION, "problems": []}


def set_outcome(
    observation_id: str,
    outcome: float,
    *,
    outcome_meta: dict | None = None,
) -> dict:
    """Settle the realised label without rewriting decision-time features.

    outcome_meta records whether the label came from a taken paper trade,
    counterfactual forward path, replay, or another explicit evidence lane.
    Re-setting the exact same value is idempotent; a conflicting rewrite is
    refused because history must not change after learning has consumed it.
    """
    try:
        c = _conn()
        try:
            row = c.execute(
                "SELECT outcome, outcome_meta FROM observations WHERE observation_id=?",
                (observation_id,),
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            if row["outcome"] is not None:
                if abs(float(row["outcome"]) - float(outcome)) <= 1e-12:
                    return {"status": "exists", "outcome": float(row["outcome"])}
                return {
                    "status": "conflict",
                    "reason": "settled outcome is immutable",
                    "existing": float(row["outcome"]),
                    "attempted": float(outcome),
                }
            c.execute(
                "UPDATE observations SET outcome=?, outcome_meta=? WHERE observation_id=?",
                (
                    float(outcome),
                    json.dumps(dict(outcome_meta or {}), default=str),
                    observation_id,
                ),
            )
            c.commit()
        finally:
            c.close()
        return {"status": "settled", "outcome": float(outcome)}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Read path — reproducible research consumption
# ══════════════════════════════════════════════════════════════════════════════

def get_observation(observation_id: str) -> dict | None:
    try:
        c = _conn()
        try:
            row = c.execute("SELECT * FROM observations WHERE observation_id=?",
                            (observation_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["features"] = json.loads(d["features"] or "{}")
            d["validation"] = json.loads(d["validation"] or "[]")
            d["meta"] = json.loads(d["meta"]) if d.get("meta") else None
            d["outcome_meta"] = (
                json.loads(d["outcome_meta"]) if d.get("outcome_meta") else {}
            )
            return d
        finally:
            c.close()
    except Exception:
        return None


def load_matrix(kind: str | None = None, feature_names=None,
                require_outcome: bool = False) -> dict:
    """Aligned design matrix for research. Returns:
      X               — (n × d) float array, missing values as NaN
      features        — the d feature names, in order
      ids             — observation ids, row-aligned
      y               — (n,) outcomes when require_outcome, else None
      schema_versions — the DISTINCT schema versions across the rows (a set); a
                        caller seeing >1 knows the rows aren't strictly
                        comparable and can filter, rather than blend blindly.
    Fail-open → empty matrix."""
    feats = list(feature_names) if feature_names else list(_S.FEATURE_NAMES)
    try:
        c = _conn()
        try:
            q = "SELECT observation_id, features, outcome, schema_version FROM observations"
            args: tuple = ()
            clauses = []
            if kind:
                clauses.append("kind=?"); args = args + (kind,)
            if require_outcome:
                clauses.append("outcome IS NOT NULL")
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY ts ASC"
            rows = c.execute(q, args).fetchall()
        finally:
            c.close()
    except Exception:
        rows = []
    X, ids, y, versions = [], [], [], set()
    for r in rows:
        fv = json.loads(r["features"] or "{}")
        X.append([_as_float(fv.get(name)) for name in feats])
        ids.append(r["observation_id"])
        y.append(r["outcome"])
        versions.add(r["schema_version"])
    return {"X": np.array(X, dtype=float) if X else np.empty((0, len(feats))),
            "features": feats, "ids": ids,
            "y": np.array(y, dtype=float) if (require_outcome and y) else None,
            "schema_versions": versions}


def load_observations(
    *,
    kind: str | None = None,
    require_outcome: bool = False,
    before_ts: str | None = None,
    schema_version: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return immutable observations with provenance for evidence research.

    Unlike load_matrix this preserves timestamp, symbol, meta, validation and
    schema identity. before_ts is a strict point-in-time cutoff: rows at or
    after the query decision are excluded so a historical analogue can never
    look into its own future.
    """
    try:
        c = _conn()
        try:
            q = (
                "SELECT observation_id, ts, symbol, kind, outcome, outcome_meta, schema_version, "
                "features, validation, reason, subtype, meta, created_at "
                "FROM observations"
            )
            clauses: list[str] = []
            args: list = []
            if kind:
                clauses.append("kind=?")
                args.append(kind)
            if require_outcome:
                clauses.append("outcome IS NOT NULL")
            if before_ts:
                clauses.append("ts < ?")
                args.append(str(before_ts))
            if schema_version:
                clauses.append("schema_version=?")
                args.append(str(schema_version))
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY ts ASC"
            if limit is not None:
                q += " LIMIT ?"
                args.append(max(0, int(limit)))
            rows = c.execute(q, tuple(args)).fetchall()
        finally:
            c.close()
    except Exception:
        return []

    out: list[dict] = []
    for row in rows:
        item = dict(row)
        try:
            item["features"] = json.loads(item.get("features") or "{}")
        except Exception:
            item["features"] = {}
        try:
            item["validation"] = json.loads(item.get("validation") or "[]")
        except Exception:
            item["validation"] = []
        try:
            item["meta"] = json.loads(item.get("meta") or "{}")
        except Exception:
            item["meta"] = {}
        try:
            item["outcome_meta"] = json.loads(item.get("outcome_meta") or "{}")
        except Exception:
            item["outcome_meta"] = {}
        out.append(item)
    return out

def _as_float(v) -> float:
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def observation_counts() -> dict:
    """Counts by kind + settled/unsettled, and the distinct schema versions in
    the store — the data-health headline. Fail-open → {}."""
    try:
        c = _conn()
        try:
            by_kind = {k: {"total": 0, "settled": 0} for k in KINDS}
            for r in c.execute("SELECT kind, COUNT(*) n, "
                               "SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) s "
                               "FROM observations GROUP BY kind").fetchall():
                by_kind[r["kind"]] = {"total": int(r["n"] or 0),
                                      "settled": int(r["s"] or 0)}
            versions = [r["schema_version"] for r in c.execute(
                "SELECT DISTINCT schema_version FROM observations").fetchall()]
            total = int((c.execute("SELECT COUNT(*) n FROM observations")
                         .fetchone() or {"n": 0})["n"] or 0)
        finally:
            c.close()
        return {"total": total, "by_kind": by_kind,
                "schema_versions": versions,
                "current_schema": _S.SCHEMA_VERSION,
                "on_current_schema": _S.SCHEMA_VERSION in versions}
    except Exception:
        return {}


def feature_coverage(kind: str | None = None) -> list[dict]:
    """Per-feature data-quality: fill-rate and count of each problem verdict
    across stored observations. A feature whose fill-rate collapses or whose
    IMPOSSIBLE count spikes is a DATA problem — surfacing it here stops a broken
    feed from being misread downstream as a decaying edge. Fail-open → []."""
    try:
        c = _conn()
        try:
            q = "SELECT features, validation FROM observations"
            args: tuple = ()
            if kind:
                q += " WHERE kind=?"; args = (kind,)
            rows = c.execute(q, args).fetchall()
        finally:
            c.close()
    except Exception:
        return []
    n = len(rows)
    if not n:
        return []
    present = {name: 0 for name in _S.FEATURE_NAMES}
    problems = {name: {} for name in _S.FEATURE_NAMES}
    for r in rows:
        fv = json.loads(r["features"] or "{}")
        for name in _S.FEATURE_NAMES:
            if fv.get(name) is not None:
                present[name] += 1
        for name, verdict, _reason in json.loads(r["validation"] or "[]"):
            problems.setdefault(name, {})
            problems[name][verdict] = problems[name].get(verdict, 0) + 1
    out = []
    for name in _S.FEATURE_NAMES:
        out.append({"feature": name, "n": n,
                    "fill_rate": round(present[name] / n, 3),
                    "problems": problems.get(name, {})})
    return sorted(out, key=lambda d: d["fill_rate"])
