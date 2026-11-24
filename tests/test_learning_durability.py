"""Lifecycle audit of 24 Nov (F2, F6): every write in the learning chain is a real barrier.

A WAIT decision later filled is credited GOOD_WAIT through: the observation claim (SQLite),
the policy save (JSON), the ingest dedupe keys (JSON), the counterfactual ledger rewrite
(JSONL). With plain fsync on macOS a power cut could persist a later write and lose an
earlier one: the audit's model kept the policy but lost its claim and the ledger rewrite,
and the retry credited one observation twice. With each write durable before the next, a
power cut can only lose a SUFFIX of the chain. Candidate transitions likewise commit durably
before the settlement clears the flag that would replay them.
"""
from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(tmp_path / "counterfactuals.jsonl"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "learning_policies.json"))
    import pandas as pd
    from core import session_window

    days = [d.date().isoformat() for d in pd.bdate_range("2026-07-01", "2026-09-30")]
    monkeypatch.setattr(session_window, "official_dates", lambda: days)
    yield tmp_path
    from product import sqlite_runtime
    sqlite_runtime.close_keepers()


def _flags(con) -> tuple[int, int, int]:
    return (con.execute("PRAGMA synchronous").fetchone()[0], con.execute("PRAGMA fullfsync").fetchone()[0],
            con.execute("PRAGMA checkpoint_fullfsync").fetchone()[0])


def test_observation_claims_commit_with_a_full_flush(isolated):
    from product import learning_observations as LO

    with closing(LO._connect(isolated / "obs.sqlite3")) as con:
        sync, full, ckpt = _flags(con)
    assert sync == 2                                            # FULL: every commit syncs
    if sys.platform == "darwin":
        assert (full, ckpt) == (1, 1)                           # ... with F_FULLFSYNC


def test_candidate_transitions_commit_with_a_full_flush(isolated):
    from product import candidate_lifecycle as CL

    with closing(CL._connect(isolated / "candidates.db")) as con:
        sync, full, ckpt = _flags(con)
    assert sync == 2
    if sys.platform == "darwin":
        assert (full, ckpt) == (1, 1)


def test_every_write_of_a_good_wait_credit_is_durable(isolated, monkeypatch):
    from core import atomic_io
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF

    CF.freeze_decision(symbol="TCS", reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                       entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": "w"})
    writes = []
    real = atomic_io.atomic_write_text

    def spy(path, *args, **kwargs):
        writes.append((Path(path).name, bool(kwargs.get("durable"))))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(atomic_io, "atomic_write_text", spy)
    out = core.note_later_entry("TCS", as_of="2026-09-15")
    assert out and writes, (out, writes)                        # the chain really ran
    rows = [json.loads(line) for line in (isolated / "counterfactuals.jsonl").read_text().splitlines() if line]
    assert [r.get("classification") for r in rows] == ["GOOD_WAIT"]
    names = {name for name, _durable in writes}
    assert {"counterfactuals.jsonl", "learning_policies.json"} <= names, names
    assert [w for w in writes if not w[1]] == []                 # not one plain-fsync write in the chain


def test_a_fill_queued_before_the_archive_and_credited_later_is_durable(isolated, monkeypatch):
    """The other two rewrites: a fill the archive cannot verify YET is queued on its WAIT
    (_queue_unverified_fills), then credited once it verifies (retry_pending_good_waits)."""
    import pandas as pd
    from core import atomic_io, session_window
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF

    # no archive yet: nothing is verifiable (past the archive the verified holiday
    # calendar counts sessions, so a covered year would verify at once)
    monkeypatch.setattr(session_window, "official_dates", lambda: [])
    CF.freeze_decision(symbol="INFY", reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                       entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": "w"})
    writes = []
    real = atomic_io.atomic_write_text

    def spy(path, *args, **kwargs):
        writes.append((Path(path).name, bool(kwargs.get("durable"))))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(atomic_io, "atomic_write_text", spy)
    core.note_later_entry("INFY", as_of="2026-09-15")              # past the archive: queued
    rows = [json.loads(line) for line in (isolated / "counterfactuals.jsonl").read_text().splitlines() if line]
    assert rows[0].get("later_entry_pending", {}).get("fill_session") == "2026-09-15"
    full = [d.date().isoformat() for d in pd.bdate_range("2026-07-01", "2026-09-30")]
    monkeypatch.setattr(session_window, "official_dates", lambda: full)
    out = core.retry_pending_good_waits(today="2026-09-16")
    assert out["credited"] == 1, out
    rows = [json.loads(line) for line in (isolated / "counterfactuals.jsonl").read_text().splitlines() if line]
    assert rows[0].get("classification") == "GOOD_WAIT" and "later_entry_pending" not in rows[0]
    assert writes and [w for w in writes if not w[1]] == []


def test_the_closed_trade_dedupe_keys_are_written_durably(isolated, monkeypatch):
    """Defence in depth: every sink is idempotent per outcome, but the keys that say a close
    was ingested are written with the same barrier as everything else in the chain."""
    from types import SimpleNamespace

    from core import atomic_io
    from product import _paper_learning_loop_core as core

    monkeypatch.setattr(core, "ingested_path", lambda: isolated / "ingested_trades.json")
    monkeypatch.setattr(core, "ingest_closed_trade", lambda row, **kw: {"policy": "stub"})
    writes = []
    real = atomic_io.atomic_write_text

    def spy(path, *args, **kwargs):
        writes.append((Path(path).name, bool(kwargs.get("durable"))))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(atomic_io, "atomic_write_text", spy)
    book = SimpleNamespace(closed=[{"symbol": "TCS", "entry_date": "2026-09-01", "exit_date": "2026-09-05",
                                    "entry": 100.0, "exit": 104.0, "qty": 10, "pnl": 40.0}])
    assert core.ingest_closed_book(book)["applied"] == 1
    assert ("ingested_trades.json", True) in writes and [w for w in writes if not w[1]] == []
