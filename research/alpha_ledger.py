"""Durable alpha accounting for the live BUY gate's family-wise tests (research-only).

``scan.signal_backtest.REPORT_ALPHA`` (0.05 / 52) is a union-bound budget: at
most 52 family-wise looks per calendar year at alpha / 52 each keeps
P(any false admission in that year) <= 5% under any dependence. A constant
cannot enforce that. Forced reruns, restarts, extra report dates or a changed
family would each spend budget unseen. This ledger makes every look real:

* scope: ONE global annual budget for every look that can admit a BUY,
  across ALL program versions. A program is the predeclared hypothesis
  program (hash of the family method, rulebook, horizon, sampling, membership
  rule and report schema). Changing it does NOT open a fresh budget:
  sequential program mutation, parallel programs, or keeping whichever
  program looked best all spend from the same 52 looks. The guarantee is
  P(any false walk-forward BUY admission in a calendar year) <= 5%,
  whichever program versions ran. Research and challenger programs that
  cannot admit a BUY do not reserve looks and are outside this guarantee.
* binding: each look stores a hash of its report's pre-test identity
  (generation time, code SHA, universe, data span, family membership and
  size). A look_id reused with a different program or binding is refused,
  and verify() recomputes the binding from the report itself, so an
  attestation copied onto another report never verifies.
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
    binding_sha256 TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (program_id, year, look_index),
    UNIQUE (look_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_global_index_per_year ON looks(year, look_index);
"""
BINDING_FIELDS = ("generated_at", "code_sha", "universe_sha256", "data_first_session", "data_last_session",
                  "members_sha256", "family_size")


class LedgerConflict(RuntimeError):
    """A look_id was presented again with a different program or report binding."""


def binding_sha256(binding: Mapping[str, Any] | None) -> str:
    """Hash of a report's pre-test identity; '' when no field is present."""
    b = {k: (dict(binding or {})).get(k) for k in BINDING_FIELDS}
    if all(v in (None, "") for v in b.values()):
        return ""
    return hashlib.sha256(json.dumps(b, sort_keys=True, default=str).encode()).hexdigest()


def ledger_path() -> Path:
    """logs/research/alpha_ledger.db; QT_ALPHA_LEDGER overrides it (the hermetic test suite isolates per test)."""
    import os

    override = os.environ.get("QT_ALPHA_LEDGER", "").strip()
    if override:
        return Path(override)
    from core.runtime_paths import logs_path

    return logs_path("research", LEDGER_FILE)


def program_id(program: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(program), sort_keys=True, default=str).encode()).hexdigest()[:20]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA synchronous=FULL")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(looks)")}
    if cols and "binding_sha256" not in cols:            # research-ledger migration (column add only)
        conn.execute("ALTER TABLE looks ADD COLUMN binding_sha256 TEXT NOT NULL DEFAULT ''")
    conn.executescript(_SCHEMA)
    return conn


def _year(now: datetime) -> int:
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return now.astimezone(IST).year


def reserve(program: Mapping[str, Any], look_id: str, *, now: datetime | None = None,
            family_size: int | None = None, members_sha256: str = "", binding: Mapping[str, Any] | None = None,
            path: Path | None = None) -> dict[str, Any]:
    """Spend one look of the GLOBAL annual budget (atomic; never refunded).

    Returns the attestation. ``alpha`` is 0.0 once the year's 52 looks (all
    programs together) are used: the report then admits nothing. Reserving
    the same ``look_id`` again returns the existing look only when program
    and report binding are identical (a write retry); anything else raises
    LedgerConflict, so a look can never be re-pointed at another report.
    """
    now = now or datetime.now(timezone.utc)
    pid, year = program_id(program), _year(now)
    bsha = binding_sha256(binding)
    with closing(_connect(path or ledger_path())) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT program_id, year, look_index, alpha, binding_sha256, family_size,"
                               " members_sha256 FROM looks WHERE look_id=?", (look_id,)).fetchone()
            if row:
                conn.execute("ROLLBACK")
                if row[0] != pid or (row[4] or "") != bsha or (row[5], row[6] or "") != (family_size, members_sha256):
                    raise LedgerConflict(f"look_id {look_id} is already recorded for another program or report")
                return _attestation(row[0], row[1], row[2], look_id, row[3], row[4], row[5], row[6])
            used = conn.execute("SELECT COUNT(*) FROM looks WHERE year=?", (year,)).fetchone()[0]
            index = used + 1
            alpha = ANNUAL_ALPHA / LOOKS_PER_YEAR if index <= LOOKS_PER_YEAR else 0.0
            conn.execute("INSERT INTO looks(program_id, year, look_index, look_id, alpha, reserved_at, family_size,"
                         " members_sha256, program, binding_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (pid, year, index, look_id, alpha, now.astimezone(timezone.utc).isoformat(),
                          family_size, members_sha256, json.dumps(dict(program), sort_keys=True, default=str), bsha))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return _attestation(pid, year, index, look_id, alpha, bsha, family_size, members_sha256)


def _attestation(pid: str, year: int, index: int, look_id: str, alpha: float, bsha: str = "",
                 family_size: int | None = None, members_sha256: str = "") -> dict[str, Any]:
    spent = min(index, LOOKS_PER_YEAR) * (ANNUAL_ALPHA / LOOKS_PER_YEAR)
    return {"program_id": pid, "year": int(year), "look_index": int(index), "look_id": look_id,
            "alpha": float(alpha), "annual_alpha": ANNUAL_ALPHA, "looks_per_year": LOOKS_PER_YEAR,
            "scope": "global: all program versions share one annual budget",
            "binding_sha256": bsha or "", "family_size": family_size, "members_sha256": members_sha256 or "",
            "spent_through_this_look": round(spent, 8), "exhausted": index > LOOKS_PER_YEAR}


_VERIFY_CACHE: dict[tuple, tuple[str, float]] = {}
POSITIVE_TTL_S = 60.0


def _file_identity(p: Path) -> tuple:
    """Identity of the ledger AND its sidecars: inode, size, mtime and ctime (ctime cannot be preserved).

    A same-size replacement, a rollback copied in place, a rename over the
    file, or writes landing in a -journal / -wal sidecar all change it.
    """
    out = []
    for f in (p, p.with_name(p.name + "-journal"), p.with_name(p.name + "-wal")):
        try:
            st = f.stat()
            out.append((f.name, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        except OSError:
            out.append((f.name, None))
    return tuple(out)


def verify(attestation: Mapping[str, Any] | None, *, program: Mapping[str, Any] | None = None,
           binding: Mapping[str, Any] | None = None, path: Path | None = None) -> str:
    """'' when the attestation matches a recorded, in-budget look bound to this report; else why not.

    ``binding``: the report's own pre-test identity (BINDING_FIELDS). When
    given, its hash must equal the ledger's for this look, so an attestation
    copied onto a different report fails. Cache: keyed on the ledger's and
    its sidecars' full file identity; a positive answer is reused for at
    most POSITIVE_TTL_S seconds. Every failure fails closed.
    """
    import time as _time

    p = path or ledger_path()
    key = (json.dumps(dict(attestation or {}), sort_keys=True, default=str),
           json.dumps(dict(program or {}), sort_keys=True, default=str),
           json.dumps(dict(binding or {}), sort_keys=True, default=str), str(p), _file_identity(p))
    hit = _VERIFY_CACHE.get(key)
    if hit is not None and (hit[0] != "" or _time.monotonic() - hit[1] < POSITIVE_TTL_S):
        return hit[0]
    out = _verify(attestation, program=program, binding=binding, path=p)
    if len(_VERIFY_CACHE) > 512:
        _VERIFY_CACHE.clear()
    _VERIFY_CACHE[key] = (out, _time.monotonic())
    return out


def _verify(attestation: Mapping[str, Any] | None, *, program: Mapping[str, Any] | None,
            binding: Mapping[str, Any] | None = None, path: Path) -> str:
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
            row = conn.execute("SELECT program_id, year, look_index, alpha, binding_sha256, family_size,"
                               " members_sha256 FROM looks WHERE look_id=?",
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
    if not (row[4] or ""):
        return "the ledger's look carries no report binding"
    if str(att.get("binding_sha256") or "") != row[4]:
        return "the evidence's report binding differs from the ledger's"
    if (att.get("family_size"), str(att.get("members_sha256") or "")) != (row[5], row[6] or ""):
        return "the evidence's family membership differs from the ledger's"
    if binding is not None and binding_sha256(binding) != row[4]:
        return "the alpha attestation belongs to a different report (binding mismatch)"
    return ""


def summary(*, path: Path | None = None) -> dict[str, Any]:
    p = path or ledger_path()
    if not p.exists():
        return {"programs": {}}
    with closing(_connect(p)) as conn:
        years = conn.execute("SELECT year, COUNT(*), SUM(alpha), COUNT(DISTINCT program_id) FROM looks GROUP BY year")
        programs = conn.execute("SELECT program_id, year, COUNT(*) FROM looks GROUP BY program_id, year")
        return {"years": {str(r[0]): {"looks": r[1], "alpha_spent": round(r[2] or 0.0, 8),
                                       "programs": r[3], "remaining_looks": max(0, LOOKS_PER_YEAR - r[1])}
                          for r in years},
                "programs": {f"{r[0]}:{r[1]}": {"looks": r[2]} for r in programs}}
