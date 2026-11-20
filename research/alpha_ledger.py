"""Durable alpha accounting for the live BUY gate's family-wise tests (research-only).

``scan.signal_backtest.REPORT_ALPHA`` (0.05 / 52) is a union-bound budget: at
most 52 family-wise looks per calendar year at alpha / 52 each keeps
P(any false admission in that year) <= 5% under any dependence. A constant
cannot enforce that. Forced reruns, restarts, extra report dates or a changed
family would each spend budget unseen. This ledger makes every look real:

* program: the predeclared hypothesis program. Its id hashes the family
  method, evaluation rulebook, horizon, sampling rule and family-membership
  rule. Membership itself is recorded per look; a program change (new
  method, horizon, rulebook) opens a NEW program with its own budget and is
  attested as such, never silently merged.
* look: one walk-forward report generation. It is reserved BEFORE the family
  correction runs, in one SQLite transaction (BEGIN IMMEDIATE), so a forced
  rerun, a crash after reservation, or the same data re-run under a new
  report SHA all spend budget. Nothing is ever refunded.
* year: the IST calendar year of the look. Budgets do not combine across
  years, and the bound is stated per year.
* attestation: the report carries {program_id, year, look_index, look_id,
  alpha}. The live gate re-reads the ledger and refuses eligibility when the
  attestation is missing, unknown, mismatched, or beyond the annual budget.

The ledger is accounting, not evidence. It lives under logs/research and is
never a production learning ledger.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

LOOKS_PER_YEAR = 52
ANNUAL_ALPHA = 0.05
LEDGER_FILE = "alpha_ledger.db"
IST = timezone(timedelta(hours=5, minutes=30))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS looks(
    program_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    look_index INTEGER NOT NULL,
    look_id TEXT NOT NULL,
    alpha REAL NOT NULL,
    reserved_at TEXT NOT NULL,
    family_size INTEGER,
    members_sha256 TEXT,
    program TEXT NOT NULL,
    PRIMARY KEY (program_id, year, look_index),
    UNIQUE (look_id)
);
"""


def ledger_path() -> Path:
    from core.runtime_paths import logs_path

    return logs_path("research", LEDGER_FILE)


def program_id(program: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(program), sort_keys=True, default=str).encode()).hexdigest()[:20]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(_SCHEMA)
    return conn


def _year(now: datetime) -> int:
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return now.astimezone(IST).year


def reserve(program: Mapping[str, Any], look_id: str, *, now: datetime | None = None,
            family_size: int | None = None, members_sha256: str = "",
            path: Path | None = None) -> dict[str, Any]:
    """Spend one look of this program's annual budget (atomic; never refunded).

    Returns the attestation. ``alpha`` is 0.0 once the year's 52 looks are
    used: the report then admits nothing. Reserving the same ``look_id``
    twice returns the existing look, so a report write retry does not spend
    twice, while a regenerated report (a new look_id) always does.
    """
    now = now or datetime.now(timezone.utc)
    pid, year = program_id(program), _year(now)
    with closing(_connect(path or ledger_path())) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT program_id, year, look_index, alpha FROM looks WHERE look_id=?",
                               (look_id,)).fetchone()
            if row:
                conn.execute("ROLLBACK")
                return _attestation(row[0], row[1], row[2], look_id, row[3])
            used = conn.execute("SELECT COUNT(*) FROM looks WHERE program_id=? AND year=?",
                                (pid, year)).fetchone()[0]
            index = used + 1
            alpha = ANNUAL_ALPHA / LOOKS_PER_YEAR if index <= LOOKS_PER_YEAR else 0.0
            conn.execute("INSERT INTO looks(program_id, year, look_index, look_id, alpha, reserved_at, family_size,"
                         " members_sha256, program) VALUES(?,?,?,?,?,?,?,?,?)",
                         (pid, year, index, look_id, alpha, now.astimezone(timezone.utc).isoformat(),
                          family_size, members_sha256, json.dumps(dict(program), sort_keys=True, default=str)))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return _attestation(pid, year, index, look_id, alpha)


def _attestation(pid: str, year: int, index: int, look_id: str, alpha: float) -> dict[str, Any]:
    spent = min(index, LOOKS_PER_YEAR) * (ANNUAL_ALPHA / LOOKS_PER_YEAR)
    return {"program_id": pid, "year": int(year), "look_index": int(index), "look_id": look_id,
            "alpha": float(alpha), "annual_alpha": ANNUAL_ALPHA, "looks_per_year": LOOKS_PER_YEAR,
            "spent_through_this_look": round(spent, 8), "exhausted": index > LOOKS_PER_YEAR}


_VERIFY_CACHE: dict[tuple, str] = {}


def verify(attestation: Mapping[str, Any] | None, *, program: Mapping[str, Any] | None = None,
           path: Path | None = None) -> str:
    """'' when the attestation matches a recorded, in-budget look; else why the budget is unprovable.

    Cached per (attestation, program, ledger file identity): looks are only ever
    appended, so an unchanged file gives an unchanged answer.
    """
    p = path or ledger_path()
    try:
        st = p.stat()
        key = (json.dumps(dict(attestation or {}), sort_keys=True, default=str),
               json.dumps(dict(program or {}), sort_keys=True, default=str), str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    out = _verify(attestation, program=program, path=p)
    if key is not None:
        if len(_VERIFY_CACHE) > 512:
            _VERIFY_CACHE.clear()
        _VERIFY_CACHE[key] = out
    return out


def _verify(attestation: Mapping[str, Any] | None, *, program: Mapping[str, Any] | None, path: Path) -> str:
    att = dict(attestation or {})
    if not att:
        return "the evidence carries no alpha-budget attestation"
    if program is not None and att.get("program_id") != program_id(program):
        return "the evidence was spent under a different hypothesis program than the current gate"
    p = path
    if not p.exists():
        return "the alpha ledger is unavailable, so the budget cannot be proven"
    try:
        with closing(_connect(p)) as conn:
            row = conn.execute("SELECT program_id, year, look_index, alpha FROM looks WHERE look_id=?",
                               (str(att.get("look_id") or ""),)).fetchone()
    except sqlite3.DatabaseError as exc:
        return f"the alpha ledger is unreadable ({type(exc).__name__})"
    if row is None:
        return "the evidence's look is not recorded in the alpha ledger"
    if (row[0], int(row[1]), int(row[2])) != (att.get("program_id"), int(att.get("year") or 0),
                                              int(att.get("look_index") or 0)):
        return "the evidence's alpha attestation does not match the ledger"
    if float(row[3]) <= 0.0 or int(row[2]) > LOOKS_PER_YEAR:
        return f"the annual alpha budget was exhausted at look {int(row[2])} of {LOOKS_PER_YEAR}"
    if abs(float(att.get("alpha") or 0.0) - float(row[3])) > 1e-15:
        return "the evidence's alpha differs from the ledger's"
    return ""


def summary(*, path: Path | None = None) -> dict[str, Any]:
    p = path or ledger_path()
    if not p.exists():
        return {"programs": {}}
    with closing(_connect(p)) as conn:
        rows = conn.execute("SELECT program_id, year, COUNT(*), SUM(alpha) FROM looks GROUP BY program_id, year")
        return {"programs": {f"{r[0]}:{r[1]}": {"looks": r[2], "alpha_spent": round(r[3] or 0.0, 8),
                                                 "remaining_looks": max(0, LOOKS_PER_YEAR - r[2])}
                             for r in rows}}
