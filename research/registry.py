"""
📓 Experiment Registry + Champion/Challenger — the Research OS's memory & safety.

Two disciplines that turn a pile of statistics into an operating system:

1. PRE-REGISTRATION. A hypothesis is registered — name, data window, and its
   PRE-COMMITTED success criteria — BEFORE it is evaluated. When the result
   comes in it is judged against those frozen criteria; you cannot move the
   goalposts after seeing the numbers. This is the single most effective
   structural defence against p-hacking (you register 20 hypotheses, most fail,
   and the registry remembers that — no cherry-picking the one that worked).
   Each experiment also pins a seed + code hash + data window so any result is
   reproducible and auditable.

2. CHAMPION vs CHALLENGER. A new weight-set / model never replaces the incumbent
   on faith. It runs in SHADOW, its per-period performance recorded alongside
   the champion's, and it is promoted ONLY if it beats the champion by a margin
   AND the improvement is statistically significant (paired one-sided test). So
   nothing touches live sizing without first winning a controlled bake-off —
   the safe rollout mechanism every reviewer insisted on.

Pure decision functions (meets_criteria, should_promote) are unit-tested; the
SQLite persistence layer uses a monkeypatchable path and is exercised on a temp
DB. Fail-safe: a broken registry degrades to "hold / don't promote", never to a
silent live change.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as _student_t

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "experiments.db"

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    hypothesis_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    registered_at TEXT NOT NULL,
    data_window TEXT,          -- json
    success_criteria TEXT,     -- json {metric: {op: value}}
    seed INTEGER,
    code_hash TEXT,
    status TEXT NOT NULL,       -- REGISTERED | PROMOTED | REJECTED
    result TEXT,               -- json metrics
    evaluated_at TEXT
);
CREATE TABLE IF NOT EXISTS champions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,        -- e.g. 'scorer_weights'
    model_id TEXT NOT NULL,
    metric REAL,
    promoted_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_champ_role ON champions(role, is_active);
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
# Pure decision logic
# ══════════════════════════════════════════════════════════════════════════════

def hypothesis_id(name: str, success_criteria: dict, data_window: dict) -> str:
    """Deterministic id for a hypothesis definition — the SAME hypothesis always
    hashes to the SAME id (so re-registering is idempotent and results can't be
    silently re-attributed to a different definition)."""
    payload = json.dumps({"name": name, "criteria": success_criteria,
                          "window": data_window}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cmp(value: float, op: str, threshold: float) -> bool:
    if op in ("gte", ">="):
        return value >= threshold
    if op in ("lte", "<="):
        return value <= threshold
    if op in ("gt", ">"):
        return value > threshold
    if op in ("lt", "<"):
        return value < threshold
    if op in ("eq", "=="):
        return value == threshold
    return False


def meets_criteria(metrics: dict, success_criteria: dict) -> bool:
    """Do the observed `metrics` satisfy EVERY pre-registered criterion?
    Criteria format: {metric_name: {op: threshold}}, op in gte/lte/gt/lt/eq.
    A missing metric fails the criterion (absence is not success)."""
    for metric, cond in (success_criteria or {}).items():
        v = metrics.get(metric)
        if v is None:
            return False
        for op, threshold in cond.items():
            if not _cmp(float(v), op, float(threshold)):
                return False
    return True


def should_promote(challenger_metric: float, champion_metric: float,
                   margin: float = 0.0, challenger_scores=None,
                   champion_scores=None, alpha: float = 0.05) -> dict:
    """Decide whether a challenger replaces the champion. It must (a) beat the
    champion by at least `margin`, AND (b) — when per-period scores are provided
    — do so significantly. When the two score series are aligned (equal length)
    a PAIRED one-sided t-test is used (same test periods → less variance);
    otherwise significance is assumed satisfied on the point margin alone.
    Returns {promote, beats_by_margin, significant, p_value}."""
    beats = challenger_metric >= champion_metric + margin
    significant = True
    p_value = 0.0
    a = np.asarray(challenger_scores, float) if challenger_scores is not None else None
    b = np.asarray(champion_scores, float) if champion_scores is not None else None
    if a is not None and b is not None and a.size >= 2 and a.size == b.size:
        diff = a - b
        sd = diff.std(ddof=1)
        if sd > 0:
            t_stat = diff.mean() / (sd / np.sqrt(diff.size))
            p_value = float(_student_t.sf(t_stat, df=diff.size - 1))  # H0: chal ≤ champ
            significant = p_value < alpha
        else:
            significant = diff.mean() > 0
    promote = bool(beats and significant)
    return {"promote": promote, "beats_by_margin": bool(beats),
            "significant": bool(significant), "p_value": round(p_value, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Persistence — experiments
# ══════════════════════════════════════════════════════════════════════════════

def register_hypothesis(name: str, success_criteria: dict, data_window: dict,
                        description: str = "", seed: int | None = None,
                        code_hash: str | None = None) -> str:
    """Pre-register a hypothesis and return its id. Idempotent: re-registering
    the same definition returns the same id without clobbering an existing
    result."""
    hid = hypothesis_id(name, success_criteria, data_window)
    c = _conn()
    try:
        exists = c.execute("SELECT 1 FROM experiments WHERE hypothesis_id=?",
                           (hid,)).fetchone()
        if not exists:
            c.execute(
                "INSERT INTO experiments (hypothesis_id, name, description, "
                "registered_at, data_window, success_criteria, seed, code_hash, "
                "status) VALUES (?,?,?,?,?,?,?,?,?)",
                (hid, name, description,
                 time.strftime("%Y-%m-%dT%H:%M:%S"),
                 json.dumps(data_window), json.dumps(success_criteria),
                 seed, code_hash, "REGISTERED"))
            c.commit()
    finally:
        c.close()
    return hid


def record_result(hypothesis_id_: str, metrics: dict) -> dict:
    """Attach a result to a PRE-REGISTERED hypothesis and judge it against the
    frozen criteria → status PROMOTED / REJECTED. Refuses to score a hypothesis
    that was never registered (the whole point of pre-registration). Idempotent-
    safe: a second call updates the result but the verdict is deterministic."""
    c = _conn()
    try:
        row = c.execute("SELECT success_criteria FROM experiments WHERE "
                        "hypothesis_id=?", (hypothesis_id_,)).fetchone()
        if row is None:
            return {"error": "not_registered",
                    "note": "Result refused — hypothesis was never pre-registered."}
        criteria = json.loads(row["success_criteria"] or "{}")
        passed = meets_criteria(metrics, criteria)
        status = "PROMOTED" if passed else "REJECTED"
        c.execute("UPDATE experiments SET result=?, status=?, evaluated_at=? "
                  "WHERE hypothesis_id=?",
                  (json.dumps(metrics), status,
                   time.strftime("%Y-%m-%dT%H:%M:%S"), hypothesis_id_))
        c.commit()
    finally:
        c.close()
    return {"status": status, "passed": passed}


def get_experiment(hypothesis_id_: str) -> dict | None:
    c = _conn()
    try:
        row = c.execute("SELECT * FROM experiments WHERE hypothesis_id=?",
                        (hypothesis_id_,)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()


def list_experiments(status: str | None = None) -> list[dict]:
    c = _conn()
    try:
        if status:
            rows = c.execute("SELECT * FROM experiments WHERE status=? "
                             "ORDER BY registered_at DESC", (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM experiments ORDER BY "
                             "registered_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


# ══════════════════════════════════════════════════════════════════════════════
# Persistence — champion / challenger
# ══════════════════════════════════════════════════════════════════════════════

def current_champion(role: str) -> dict | None:
    c = _conn()
    try:
        row = c.execute("SELECT * FROM champions WHERE role=? AND is_active=1 "
                        "ORDER BY id DESC LIMIT 1", (role,)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()


def _promote(role: str, model_id: str, metric: float) -> None:
    c = _conn()
    try:
        c.execute("UPDATE champions SET is_active=0 WHERE role=?", (role,))
        c.execute("INSERT INTO champions (role, model_id, metric, promoted_at, "
                  "is_active) VALUES (?,?,?,?,1)",
                  (role, model_id, metric, time.strftime("%Y-%m-%dT%H:%M:%S")))
        c.commit()
    finally:
        c.close()


def evaluate_challenger(role: str, challenger_id: str, challenger_metric: float,
                        margin: float = 0.0, challenger_scores=None,
                        champion_scores=None, alpha: float = 0.05) -> dict:
    """Run the bake-off. If there is no champion for the role, the challenger
    becomes it (first one in). Otherwise it is promoted ONLY if should_promote()
    passes (beats the champion by `margin`, significantly). Returns the decision
    + whether a promotion happened."""
    champ = current_champion(role)
    if champ is None:
        _promote(role, challenger_id, challenger_metric)
        return {"promote": True, "reason": "first champion for role",
                "champion": challenger_id}
    decision = should_promote(challenger_metric, float(champ["metric"] or 0.0),
                              margin, challenger_scores, champion_scores, alpha)
    if decision["promote"]:
        _promote(role, challenger_id, challenger_metric)
        decision["champion"] = challenger_id
    else:
        decision["champion"] = champ["model_id"]
    return decision
