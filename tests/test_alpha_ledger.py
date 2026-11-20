"""The annual alpha budget is global, report-bound and enforced by a durable ledger."""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone

import pytest

from research import alpha_ledger as AL

PROGRAM = {"family_method": "Holm", "eval_spec": "sim:h10:be2:gap", "horizon_sessions": 10,
           "sampling": "even_stride", "report_schema": 5, "membership": ">= 30 trades"}
T = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)


def _binding(tag="r", universe="u" * 64):
    return {"generated_at": f"2026-11-20T01:00:{tag}", "code_sha": "sha", "universe_sha256": universe,
            "data_first_session": "2025-11-03", "data_last_session": "2026-11-18",
            "members_sha256": "m" * 64, "family_size": 44}


def _reserve(path, look_id, program=PROGRAM, now=T, binding=None):
    return AL.reserve(program, look_id, now=now, path=path, family_size=44, members_sha256="m" * 64,
                      binding=binding or _binding(look_id))


def _verify(att, path, program=PROGRAM, binding="from_att"):
    b = _binding(att["look_id"]) if binding == "from_att" else binding
    return AL.verify(att, program=program, binding=b, path=path)


def test_the_fifty_third_look_of_a_year_gets_no_alpha_and_never_verifies(tmp_path):
    path = tmp_path / "ledger.db"
    atts = [_reserve(path, f"look-{i}") for i in range(53)]
    assert [a["look_index"] for a in atts] == list(range(1, 54))
    assert all(a["alpha"] == pytest.approx(0.05 / 52) for a in atts[:52]) and atts[52]["alpha"] == 0.0
    assert _verify(atts[51], path) == ""
    assert "exhausted at look 53" in _verify(atts[52], path)


@pytest.mark.parametrize("pattern", ["sequential", "parallel"])
def test_program_mutation_never_opens_a_fresh_budget(tmp_path, pattern):
    """One global budget: changing the method/rulebook (or running variants side by side) shares the 52 looks."""
    path = tmp_path / "ledger.db"
    programs = [{**PROGRAM, "family_method": f"variant-{k}"} for k in range(5)]
    atts = []
    for i in range(53):
        prog = programs[i // 13] if pattern == "sequential" else programs[i % 5]
        atts.append(_reserve(path, f"look-{i}", program=prog))
    assert atts[52]["alpha"] == 0.0 and atts[52]["look_index"] == 53
    assert len({a["program_id"] for a in atts}) == 5
    assert sum(a["alpha"] for a in atts) <= 0.05 + 1e-12          # union bound across all program versions


def test_keeping_the_best_looking_program_is_covered_by_the_global_bound(tmp_path):
    """Selecting among programs after seeing results: every look of every program spent from one 5%."""
    path = tmp_path / "ledger.db"
    spent = 0.0
    for k in range(8):                                            # 8 candidate programs, 6 looks each
        prog = {**PROGRAM, "family_method": f"candidate-{k}"}
        for i in range(6):
            spent += _reserve(path, f"look-{k}-{i}", program=prog)["alpha"]
    assert spent <= 0.05 + 1e-12
    assert AL.summary(path=path)["years"]["2026"]["looks"] == 48


def test_a_program_change_does_not_carry_old_attestations_into_the_new_gate(tmp_path):
    path = tmp_path / "ledger.db"
    old = _reserve(path, "look-old")
    changed = {**PROGRAM, "family_method": "Holm v2"}
    assert "different hypothesis program" in _verify(old, path, program=changed)


def test_a_write_retry_is_free_but_a_rerun_spends(tmp_path):
    path = tmp_path / "ledger.db"
    first = _reserve(path, "look-A")
    retry = _reserve(path, "look-A")
    rerun = _reserve(path, "look-B", binding=_binding("look-A"))                 # same data, new report id
    assert retry == first and rerun["look_index"] == 2


@pytest.mark.parametrize("change", ["program", "membership", "binding"])
def test_a_reused_look_id_with_a_different_identity_is_refused(tmp_path, change):
    path = tmp_path / "ledger.db"
    _reserve(path, "look-X")
    with pytest.raises(AL.LedgerConflict):
        if change == "program":
            _reserve(path, "look-X", program={**PROGRAM, "family_method": "other"})
        elif change == "membership":
            AL.reserve(PROGRAM, "look-X", now=T, path=path, family_size=45, members_sha256="m" * 64,
                       binding=_binding("look-X"))
        else:
            _reserve(path, "look-X", binding=_binding("look-X", universe="v" * 64))


def test_an_attestation_replayed_onto_another_report_never_verifies(tmp_path):
    path = tmp_path / "ledger.db"
    att = _reserve(path, "look-1")
    assert _verify(att, path) == ""
    other_report = _binding("look-1", universe="w" * 64)                         # different universe
    assert "different report" in _verify(att, path, binding=other_report)
    forged = {**att, "binding_sha256": AL.binding_sha256(other_report)}
    assert "binding differs" in _verify(forged, path, binding=other_report)


@pytest.mark.parametrize("tamper", [{"look_index": 1}, {"year": 2025}, {"alpha": 0.05}, {"look_id": "made-up"},
                                    {"family_size": 10}, {"members_sha256": "x"}])
def test_a_tampered_attestation_never_verifies(tmp_path, tamper):
    path = tmp_path / "ledger.db"
    _reserve(path, "look-0")
    att = _reserve(path, "look-2")
    assert _verify({**att, **tamper}, path, binding=_binding("look-2")) != ""


def test_no_ledger_no_attestation_or_an_unbound_look_is_unprovable(tmp_path):
    att = {"look_id": "x", "program_id": AL.program_id(PROGRAM), "year": 2026, "look_index": 1}
    assert "unavailable" in AL.verify(att, program=PROGRAM, path=tmp_path / "missing.db")
    assert "no alpha-budget attestation" in AL.verify(None, program=PROGRAM, path=tmp_path / "missing.db")
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"not sqlite" * 50)
    assert "unreadable" in AL.verify(att, program=PROGRAM, path=bad)
    unbound = AL.reserve(PROGRAM, "legacy", now=T, path=tmp_path / "l.db")       # reserved without a binding
    assert "no report binding" in AL.verify(unbound, program=PROGRAM, path=tmp_path / "l.db")


def test_a_same_size_replacement_with_restored_mtime_cannot_serve_a_stale_positive(tmp_path):
    a, b = tmp_path / "ledger.db", tmp_path / "other.db"
    att = _reserve(a, "look-1")
    _reserve(b, "look-9")
    assert _verify(att, a) == ""                                                 # cached positive
    st = a.stat()
    shutil.copyfile(b, a)                                                        # same inode, new content
    os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns))                             # mtime restored
    assert a.stat().st_size == st.st_size
    assert "not recorded" in _verify(att, a)


def test_a_rollback_copy_or_a_sidecar_write_invalidates_the_cache(tmp_path):
    path = tmp_path / "ledger.db"
    backup = tmp_path / "backup.db"
    _reserve(path, "look-0")
    shutil.copyfile(path, backup)                                                # before look-1 existed
    att = _reserve(path, "look-1")
    assert _verify(att, path) == ""
    os.replace(backup, path)                                                     # rollback: rename over
    assert "not recorded" in _verify(att, path)
    att2 = _reserve(path, "look-2")
    assert _verify(att2, path) == ""
    before = AL._file_identity(path)
    path.with_name(path.name + "-wal").write_bytes(b"x")                         # a WAL sidecar appears
    assert AL._file_identity(path) != before                                     # cache key turns over


def test_concurrent_reserve_and_verify_never_see_a_stale_positive(tmp_path):
    path = tmp_path / "ledger.db"
    atts = [_reserve(path, "seed")]
    barrier, errors = threading.Barrier(8), []

    def reserver(i):
        barrier.wait()
        atts.append(_reserve(path, f"look-{i}"))

    def verifier():
        barrier.wait()
        for _ in range(20):
            if _verify(atts[0], path) != "":
                errors.append("seed stopped verifying")

    threads = [threading.Thread(target=reserver, args=(i,)) for i in range(6)] + \
              [threading.Thread(target=verifier) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and sorted(a["look_index"] for a in atts) == list(range(1, 8))


def test_budgets_are_per_calendar_year_in_ist(tmp_path):
    path = tmp_path / "ledger.db"
    a = _reserve(path, "look-dec", now=datetime(2026, 12, 31, 10, 0, tzinfo=timezone.utc))
    b = _reserve(path, "look-jan", now=datetime(2026, 12, 31, 18, 40, tzinfo=timezone.utc))   # 1 Jan 00:10 IST
    assert (a["year"], b["year"], b["look_index"]) == (2026, 2027, 1)


def test_the_walk_forward_report_spends_and_attests_a_bound_look():
    import scan.signal_backtest as SB
    from research import alpha_ledger
    from scan.unified_scanner import evidence_provenance_note
    from tests.evidence_fixtures import proven_provenance

    prov = proven_provenance()
    assert evidence_provenance_note(prov) == ""
    assert prov["alpha_accounting"]["program_id"] == alpha_ledger.program_id(SB.gate_program())
    assert "different report" in evidence_provenance_note({**prov, "universe_sha256": "z" * 64})
    assert "no alpha-budget attestation" in evidence_provenance_note({**prov, "alpha_accounting": None})


def test_a_legacy_per_program_ledger_is_globalised_without_refunding_any_look(tmp_path):
    """A ledger written by the per-program code: every old look stays spent, none verifies, reserve works."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE looks(program_id TEXT NOT NULL, year INTEGER NOT NULL, look_index INTEGER NOT NULL,"
                 " look_id TEXT NOT NULL, alpha REAL NOT NULL, reserved_at TEXT NOT NULL, family_size INTEGER,"
                 " members_sha256 TEXT, program TEXT NOT NULL, PRIMARY KEY (program_id, year, look_index),"
                 " UNIQUE (look_id))")
    legacy = []
    for k in range(30):                        # 3 programs x 30 looks: 90 per-program looks in 2026
        for p in ("pA", "pB", "pC"):
            legacy.append({"program_id": p, "year": 2026, "look_index": k + 1, "look_id": f"{p}-{k}",
                           "alpha": 0.05 / 52})
            conn.execute("INSERT INTO looks VALUES(?,?,?,?,?,?,?,?,?)",
                         (p, 2026, k + 1, f"{p}-{k}", 0.05 / 52, f"2026-03-01T00:{k:02d}:00+00:00", 44, "m", "{}"))
    conn.commit()
    conn.close()

    att = _reserve(path, "look-new")
    assert att["look_index"] == 91 and att["alpha"] == 0.0                     # the 90 legacy looks are spent
    assert all(_verify(a, path, binding=None) != "" for a in legacy[:3])       # no legacy look ever verifies
    with closing(sqlite3.connect(path)) as c:
        idx = [r[0] for r in c.execute("SELECT look_index FROM looks WHERE year=2026 ORDER BY look_index")]
        spent = c.execute("SELECT SUM(alpha) FROM looks WHERE year=2026").fetchone()[0]
    assert idx == list(range(1, 92)) and spent <= 0.05 + 1e-12


def test_an_exhausted_budget_admits_nothing_in_the_family_correction():
    import scan.signal_backtest as SB
    rows = {"REAL": {"expectancy_r_precise": 0.5, "se_r": 0.01, "df": 30, "inference": "OVERLAP_ROBUST"}}
    SB.apply_family_correction(rows, {}, alpha=0.0)
    assert rows["REAL"]["family_significant"] is False


# ── crash / restart integrity of look reservation ─────────────────────────────

def _stub_walk_forward(monkeypatch, tmp_path, n_sym=3):
    """A real run_backtest over small frames with a stub scanner (fast, deterministic)."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    import data.bhavcopy_store as BS
    import scan.signal_backtest as SB
    import scan.unified_scanner as US

    idx = pd.bdate_range("2025-01-01", periods=200)
    rng = np.random.default_rng(1)
    frames = {}
    for k in range(n_sym):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        frames[f"S{k}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                                        "volume": np.full(200, 1e6)}, index=idx)

    class Stub:
        def __init__(self, *a, **k):
            pass

        def _analyze(self, sym, hist):
            px = float(hist["close"].iloc[-1])
            return SimpleNamespace(signals=["S"], entry=px * 1.001, stop=px * 0.95, target=px * 1.08, confidence=0.0)

    monkeypatch.setattr(US, "UnifiedScanner", Stub)
    monkeypatch.setattr(BS, "store_symbols", lambda: sorted(frames))
    monkeypatch.setattr(BS, "get_ohlcv", lambda s: frames.get(s))
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: [d.date() for d in idx])
    monkeypatch.setattr(SB, "_nifty_regime_series", lambda: None)
    monkeypatch.setattr(SB, "_OUT", tmp_path / "signal_backtest.json")
    return SB


def _looks():
    p = AL.ledger_path()
    if not p.exists():
        return []
    import sqlite3
    with closing(sqlite3.connect(p)) as c:
        return [r[0] for r in c.execute("SELECT look_index FROM looks ORDER BY look_index")]


def test_a_discarded_thin_run_spends_no_look(monkeypatch, tmp_path):
    SB = _stub_walk_forward(monkeypatch, tmp_path)                 # 3 symbols < MIN_SYMBOLS_TO_SAVE
    rep = SB.run_backtest(lookback_sessions=100)
    assert rep["signals"] and not SB._OUT.exists()                 # measured, then discarded
    acc = rep["evidence_provenance"]["alpha_accounting"]
    assert acc["alpha"] == 0.0 and acc["unreserved"] == "REPORT_NOT_SAVED" and _looks() == []
    assert not any(v.get("family_significant") for v in rep["signals"].values())
    from scan.unified_scanner import evidence_provenance_note
    assert evidence_provenance_note(rep["evidence_provenance"]) != ""      # can never certify a BUY


def test_a_crash_after_reservation_spends_the_look_and_a_rerun_spends_another(monkeypatch, tmp_path):
    SB = _stub_walk_forward(monkeypatch, tmp_path)
    monkeypatch.setattr(SB, "MIN_SYMBOLS_TO_SAVE", 1)
    real = SB.apply_family_correction

    def crash(*a, **k):
        raise RuntimeError("killed between reservation and test")

    monkeypatch.setattr(SB, "apply_family_correction", crash)
    with pytest.raises(RuntimeError):
        SB.run_backtest(lookback_sessions=100)
    assert _looks() == [1] and not SB._OUT.exists()                # spent, nothing written, never refunded
    monkeypatch.setattr(SB, "apply_family_correction", real)
    rep = SB.run_backtest(lookback_sessions=100)
    assert _looks() == [1, 2] and SB._OUT.exists()
    assert rep["evidence_provenance"]["alpha_accounting"]["look_index"] == 2


def test_a_hard_kill_mid_migration_leaves_the_legacy_ledger_intact(tmp_path):
    """A kill inside the one-time migration rolls back; the next open migrates cleanly."""
    import sqlite3
    import subprocess
    import sys
    import textwrap

    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE looks(program_id TEXT NOT NULL, year INTEGER NOT NULL, look_index INTEGER NOT NULL,"
                     " look_id TEXT NOT NULL, alpha REAL NOT NULL, reserved_at TEXT NOT NULL, family_size INTEGER,"
                     " members_sha256 TEXT, program TEXT NOT NULL, PRIMARY KEY (program_id, year, look_index),"
                     " UNIQUE (look_id))")
        for k in range(10):
            for p in ("pA", "pB"):
                conn.execute("INSERT INTO looks VALUES(?,?,?,?,?,?,?,?,?)",
                             (p, 2026, k + 1, f"{p}-{k}", 0.05 / 52, f"2026-03-01T00:{k:02d}:00+00:00", 4, "m", "{}"))
        conn.commit()
    before = sorted(sqlite3.connect(path).execute("SELECT look_id, look_index FROM looks").fetchall())
    child = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(os.getcwd())!r})
        from research import alpha_ledger as AL
        class Bomb:                       # dies on the 7th renumbered row, inside the transaction
            calls = 0
            def __ge__(self, other):
                Bomb.calls += 1
                if Bomb.calls == 7:
                    os._exit(9)
                return True
            def __rtruediv__(self, other):
                return other / 52
        AL.LOOKS_PER_YEAR = Bomb()
        AL._connect(AL.Path({str(path)!r}))
    """)
    rc = subprocess.run([sys.executable, "-c", child], cwd=os.getcwd()).returncode
    assert rc == 9
    with closing(sqlite3.connect(path)) as c:                      # opening rolls the hot journal back
        assert sorted(c.execute("SELECT look_id, look_index FROM looks").fetchall()) == before
        assert not c.execute("SELECT 1 FROM sqlite_master WHERE name='one_global_index_per_year'").fetchone()
    att = AL.reserve(PROGRAM, "after-kill", now=T, path=path, family_size=44, members_sha256="m" * 64,
                     binding=_binding("after-kill"))
    assert att["look_index"] == 21                                 # all 20 legacy looks still spent
