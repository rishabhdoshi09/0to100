"""SQLite connection policy and product-store bootstrap."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from product.sqlite_runtime import bootstrap_product_stores, connect, integrity_ok


def test_connect_uses_wal_and_busy_timeout(tmp_path):
    db = tmp_path / "runtime.db"
    con = connect(db)
    try:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
        con.execute("CREATE TABLE t(x INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
        con.commit()
    finally:
        con.close()
    assert str(mode).upper() == "WAL"
    assert int(timeout) >= 30000
    info = integrity_ok(db)
    assert info["ok"] is True
    assert str(info["integrity"]).lower() == "ok"


def test_existing_db_keeps_rows_across_reconnect(tmp_path):
    db = tmp_path / "keep.db"
    con = connect(db)
    con.execute("CREATE TABLE t(x INTEGER)")
    con.execute("INSERT INTO t VALUES (7)")
    con.commit()
    con.close()
    con = connect(db)
    try:
        assert con.execute("SELECT x FROM t").fetchone()[0] == 7
    finally:
        con.close()


def test_interrupted_create_then_bootstrap(tmp_path):
    """Killing mid-DDL must not require a hand-built schema on restart."""
    db = tmp_path / "partial.db"
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    con = connect(db)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY)")
        con.commit()
    finally:
        con.close()
    from product import pit_warehouse

    pit_warehouse._connect(db).close()
    con = connect(db)
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(evidence)").fetchall()}
    finally:
        con.close()
    assert "evidence_id" in cols
    assert "available_from" in cols
    assert integrity_ok(db)["ok"] is True


def test_bootstrap_product_stores_reports_opened():
    result = bootstrap_product_stores()
    assert result["ok"] is True
    assert "pit_warehouse" in result["opened"]
    assert "decision_journal" in result["opened"]
    assert "job_store" in result["opened"]
    assert result["errors"] == []
