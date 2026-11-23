"""Deterministic SQLite damage for tests (Stage 1 finding F1, 22-23 Nov 2026).

``corrupt_index`` reproduces F1 exactly: the table holds the current rows, and
the PRIMARY KEY index is given the b-tree of a WITHOUT ROWID table holding the
same (key, rowid) records plus stale keys that point at rowids the table later
re-used, as after a power cut that persisted index pages but lost table pages.
``integrity_check`` then reports ``wrong # of entries in index
sqlite_autoindex_freezes_1``, while ``quick_check`` (which skips index/table
comparison by design) still reports ``ok``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

FREEZE_SCHEMA = """CREATE TABLE freezes (
            freeze_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            symbol TEXT,
            as_of TEXT,
            decision TEXT,
            frozen_at TEXT,
            payload_json TEXT NOT NULL
        )"""
STALE_KEYS = [f"dec_stale{i:019x}" for i in range(4)]


def freeze_rows(n: int) -> list[tuple]:
    return [(f"dec_{i:024x}", f"fp{i:020d}", f"SYM{i}", "2026-11-19", "WATCH",
             f"2026-11-20T04:05:{i % 60:02d}+00:00", json.dumps({"i": i, "blob": "x" * (i % 50)}))
            for i in range(1, n + 1)]


def _drop_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def healthy_freeze_db(path: Path, n: int = 200) -> Path:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(FREEZE_SCHEMA)
    con.executemany("INSERT INTO freezes VALUES (?,?,?,?,?,?,?)", freeze_rows(n))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    _drop_sidecars(path)
    return path


def corrupt_index(path: Path, n: int = 200, stale_at: int = 150) -> Path:
    healthy_freeze_db(path, n)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ix (freeze_id TEXT, rid INTEGER, PRIMARY KEY (freeze_id, rid)) WITHOUT ROWID")
    con.executemany("INSERT INTO ix VALUES (?,?)", [(r[0], i + 1) for i, r in enumerate(freeze_rows(n))]
                    + [(k, stale_at + i) for i, k in enumerate(STALE_KEYS)])
    con.commit()
    roots = dict(con.execute("SELECT name, rootpage FROM sqlite_master "
                             "WHERE name IN ('ix', 'sqlite_autoindex_freezes_1')").fetchall())
    con.execute("PRAGMA writable_schema=ON")
    con.execute("UPDATE sqlite_master SET rootpage=? WHERE name='sqlite_autoindex_freezes_1'", (roots["ix"],))
    con.execute("UPDATE sqlite_master SET rootpage=? WHERE name='ix'", (roots["sqlite_autoindex_freezes_1"],))
    con.commit()
    con.close()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE ix")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    _drop_sidecars(path)
    return path
