"""Shared SQLite connection policy for QuantTerm product stores.

WAL + busy_timeout are the default. Callers still own schema CREATE.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path, *, timeout: float = 30.0, wal: bool = True) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target), timeout=timeout)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def ensure_columns(con: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Add missing columns on an existing table. Safe after a killed CREATE.

    ``table`` and column names must be literals from our schema code.
    """
    if not table.isidentifier() or not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    existing = {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if not existing:
        return
    for name, decl in columns.items():
        if not name.isidentifier():
            raise ValueError("invalid column")
        if name in existing:
            continue
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def integrity_ok(path: str | Path) -> dict[str, object]:
    con = connect(path)
    try:
        row = con.execute("PRAGMA integrity_check").fetchone()
        wal = con.execute("PRAGMA journal_mode").fetchone()
        result = str(row[0] if row else "")
        return {
            "path": str(path),
            "ok": result.lower() == "ok",
            "integrity": result,
            "journal_mode": str(wal[0] if wal else ""),
        }
    finally:
        con.close()


def bootstrap_product_stores() -> dict[str, object]:
    """Create/upgrade required product schemas. Safe on existing DBs."""
    opened: list[str] = []
    errors: list[str] = []
    loaders = (
        ("pit_warehouse", "product.pit_warehouse", "_connect"),
        ("decision_journal", "product.decision_journal", "_connect"),
        ("candidate_lifecycle", "product.candidate_lifecycle", "_connect"),
        ("opportunity_memory", "product.opportunity_memory", "_connect"),
        ("learning_ledger", "product.learning_ledger", "_connect"),
        ("decision_freeze", "product.decision_freeze", "_connect"),
        ("pit_debt", "product.pit_debt", "_connect"),
        ("job_store", "research.autonomy.job_store", "JobStore"),
    )
    for name, module, attr in loaders:
        try:
            mod = __import__(module, fromlist=[attr])
            obj = getattr(mod, attr)
            if attr == "JobStore":
                obj()
            else:
                con = obj()
                con.close()
            opened.append(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}"[:200])
    return {"ok": not errors, "opened": opened, "errors": errors}
