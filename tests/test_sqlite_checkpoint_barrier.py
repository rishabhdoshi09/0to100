"""Stage 1 finding F1 (22-23 Nov 2026): WAL checkpoints must be real write barriers on macOS.

A power cut on 19 Nov (the host drained to 0% and went to AutoPowerOff) left the
decision-freeze ledger with index pages from one transaction and table pages
from an older one. macOS fsync() does not flush the drive cache; only
F_FULLFSYNC does, and SQLite uses it only when asked. These tests pin that every
WAL store asks, at the cost the store can afford.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

import core.atomic_io as aio

REPO = Path(__file__).resolve().parents[1]


def _flags(con) -> tuple[int, int, int]:
    return (con.execute("PRAGMA checkpoint_fullfsync").fetchone()[0],
            con.execute("PRAGMA fullfsync").fetchone()[0],
            con.execute("PRAGMA synchronous").fetchone()[0])


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(aio.sys, "platform", "darwin")


@pytest.mark.parametrize("sync, expect", [("OFF", (1, 1, 0)), ("NORMAL", (1, 1, 1)), ("FULL", (1, 0, 2))])
def test_the_barrier_costs_nothing_per_commit(tmp_path, darwin, sync, expect):
    con = sqlite3.connect(tmp_path / "s.db")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(f"PRAGMA synchronous={sync}")
    aio.make_sqlite_checkpoint_safe(con)
    # checkpoint syncs always become F_FULLFSYNC; every sync only where commits never sync
    assert _flags(con) == expect
    con.close()


def test_no_effect_off_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(aio.sys, "platform", "linux")
    con = sqlite3.connect(tmp_path / "s.db")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    aio.make_sqlite_checkpoint_safe(con)
    assert _flags(con)[:2] == (0, 0)
    con.close()


def test_the_shared_connection_policy_sets_the_barrier(tmp_path, darwin):
    from product.sqlite_runtime import connect

    con = connect(tmp_path / "store.db")
    assert _flags(con) == (1, 1, 1)
    con.close()
    plain = connect(tmp_path / "rollback.db", wal=False)    # rollback journal: not a WAL checkpoint store
    assert _flags(plain)[:2] == (0, 0)
    plain.close()


def test_the_f1_writer_gets_the_barrier(tmp_path, darwin):
    from product import decision_freeze

    con = decision_freeze._connect(tmp_path / "counterfactuals.jsonl.freeze.db")
    assert _flags(con) == (1, 1, 1)
    con.close()


def test_the_busiest_store_keeps_its_commit_cost(tmp_path, darwin):
    from research.autonomy.job_store import JobStore

    store = JobStore(tmp_path / "jobs.db")
    # FULL (SQLite's default): only the checkpoint syncs change, commits do not slow down
    assert _flags(store._db) == (1, 0, 2)


_WAL = re.compile(r"""\.execute\(\s*["']PRAGMA journal_mode\s*=\s*WAL["']""")
_BARRIER = ("make_sqlite_checkpoint_safe(", "make_sqlite_durable(")
# Development tooling, installed and run outside the desk (see docs/runbooks).
_EXEMPT = {"scripts/quantterm_relay.py"}


def test_every_wal_store_in_the_repo_opts_into_the_barrier():
    missing = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", "venv/", ".venv/", "node_modules/", "frontend/")) or rel in _EXEMPT:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if _WAL.search(line) and not any(b in "\n".join(lines[i:i + 8]) for b in _BARRIER):
                missing.append(f"{rel}:{i + 1}")
    assert not missing, f"WAL stores without a checkpoint barrier (finding F1): {missing}"
