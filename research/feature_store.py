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
import time
from pathlib import Path

import numpy as np

from research import feature_schema as _S

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "feature_store.db"

# observation kinds — the whole point is that a REJECTION or NEAR_MISS is as much
# an observation as a TRADE (non-event learning needs them on equal footing).
KINDS = ("SCAN", "TRADE", "REJECTION", "NEAR_MISS")

_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,               -- observation time (point-in-time)
    symbol TEXT,
    kind TEXT NOT NULL,
    outcome REAL,                   -- realised label (R or %), filled later
    schema_version TEXT NOT NULL,   -- which schema froze this vector
    features TEXT NOT NULL,         -- json canonical vector
    validation TEXT,                -- json problems list
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_kind ON observations(kind, ts);
CREATE INDEX IF NOT EXISTS idx_obs_symbol ON observations(symbol);
"""


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
# Write path — freeze once, settle the label later
# ══════════════════════════════════════════════════════════════════════════════

def snapshot(observation_id: str, symbol: str, kind: str, raw_features: dict,
             ts: str | None = None, outcome: float | None = None,
             ages: dict | None = None) -> dict:
    """Freeze one observation. Canonicalises + validates the raw features, then
    writes them WRITE-ONCE under the current schema version. Returns
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
        c = _conn()
        try:
            exists = c.execute("SELECT 1 FROM observations WHERE observation_id=?",
                               (observation_id,)).fetchone()
            if exists:
                return {"status": "exists", "schema_version": _S.SCHEMA_VERSION,
                        "problems": report.problems}
            c.execute(
                "INSERT INTO observations (observation_id, ts, symbol, kind, "
                "outcome, schema_version, features, validation, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (observation_id, ts or time.strftime("%Y-%m-%dT%H:%M:%S"),
                 (symbol or "").upper(), kind,
                 float(outcome) if outcome is not None else None,
                 _S.SCHEMA_VERSION, json.dumps(canonical),
                 json.dumps(report.problems),
                 time.strftime("%Y-%m-%dT%H:%M:%S")))
            c.commit()
        finally:
            c.close()
        return {"status": "frozen", "schema_version": _S.SCHEMA_VERSION,
                "problems": report.problems, "valid": report.ok}
    except Exception as exc:
        return {"status": "error", "reason": str(exc),
                "schema_version": _S.SCHEMA_VERSION, "problems": []}


def set_outcome(observation_id: str, outcome: float) -> dict:
    """Settle the realised label on a frozen observation. This is NOT a
    recomputation — the features stay exactly as frozen; only the after-the-fact
    result is attached. Fail-open."""
    try:
        c = _conn()
        try:
            row = c.execute("SELECT 1 FROM observations WHERE observation_id=?",
                            (observation_id,)).fetchone()
            if not row:
                return {"status": "not_found"}
            c.execute("UPDATE observations SET outcome=? WHERE observation_id=?",
                      (float(outcome), observation_id))
            c.commit()
        finally:
            c.close()
        return {"status": "settled"}
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


def _as_float(v) -> float:
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


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
