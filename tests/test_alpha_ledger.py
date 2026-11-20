"""The annual alpha budget is enforced by a durable ledger, not by a constant."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from research import alpha_ledger as AL

PROGRAM = {"family_method": "Holm", "eval_spec": "sim:h10:be2:gap", "horizon_sessions": 10,
           "sampling": "even_stride", "report_schema": 5, "membership": ">= 30 trades"}
T = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)


def test_the_fifty_third_look_of_a_year_gets_no_alpha_and_never_verifies(tmp_path):
    path = tmp_path / "ledger.db"
    atts = [AL.reserve(PROGRAM, f"look-{i}", now=T, path=path) for i in range(53)]
    assert [a["look_index"] for a in atts] == list(range(1, 54))
    assert all(a["alpha"] == pytest.approx(0.05 / 52) for a in atts[:52])
    assert atts[52]["alpha"] == 0.0 and atts[52]["exhausted"] is True
    assert AL.verify(atts[51], program=PROGRAM, path=path) == ""
    assert "exhausted at look 53" in AL.verify(atts[52], program=PROGRAM, path=path)
    assert sum(a["alpha"] for a in atts) <= 0.05 + 1e-12                      # the union bound holds


def test_a_forced_rerun_on_the_same_data_spends_budget_but_a_write_retry_does_not(tmp_path):
    path = tmp_path / "ledger.db"
    first = AL.reserve(PROGRAM, "look-A", now=T, path=path)
    retry = AL.reserve(PROGRAM, "look-A", now=T, path=path)                  # same report, re-written
    rerun = AL.reserve(PROGRAM, "look-B", now=T, path=path)                  # same data, new report SHA
    assert retry == first and rerun["look_index"] == 2


def test_changing_family_membership_cannot_reset_the_budget(tmp_path):
    path = tmp_path / "ledger.db"
    for i in range(52):
        AL.reserve(PROGRAM, f"look-{i}", now=T, path=path, family_size=40, members_sha256=f"m{i % 3}")
    more_members = AL.reserve(PROGRAM, "look-new-family", now=T, path=path, family_size=90, members_sha256="other")
    assert more_members["alpha"] == 0.0


def test_a_program_change_is_a_new_budget_and_old_attestations_do_not_carry_over(tmp_path):
    path = tmp_path / "ledger.db"
    old = AL.reserve(PROGRAM, "look-old", now=T, path=path)
    changed = {**PROGRAM, "family_method": "Holm alpha/52 v2"}
    assert "different hypothesis program" in AL.verify(old, program=changed, path=path)
    new = AL.reserve(changed, "look-new", now=T, path=path)
    assert new["program_id"] != old["program_id"] and new["look_index"] == 1


def test_a_rolled_back_report_keeps_its_original_look(tmp_path):
    path = tmp_path / "ledger.db"
    older = AL.reserve(PROGRAM, "look-older", now=T, path=path)
    AL.reserve(PROGRAM, "look-newer", now=T, path=path)
    assert AL.verify(older, program=PROGRAM, path=path) == ""                 # the look happened; still counted
    assert AL.summary(path=path)["programs"][f"{older['program_id']}:2026"]["looks"] == 2


@pytest.mark.parametrize("tamper", [{"look_index": 1}, {"year": 2025}, {"alpha": 0.05}, {"look_id": "made-up"},
                                    {"program_id": "x"}])
def test_a_tampered_or_unknown_attestation_never_verifies(tmp_path, tamper):
    path = tmp_path / "ledger.db"
    AL.reserve(PROGRAM, "look-1", now=T, path=path)
    att = AL.reserve(PROGRAM, "look-2", now=T, path=path)
    assert AL.verify({**att, **tamper}, program=None if "program_id" in tamper else PROGRAM, path=path) != ""


def test_no_ledger_or_no_attestation_means_the_budget_is_unprovable(tmp_path):
    att = {"look_id": "x", "program_id": AL.program_id(PROGRAM), "year": 2026, "look_index": 1}
    assert "unavailable" in AL.verify(att, program=PROGRAM, path=tmp_path / "missing.db")
    assert "no alpha-budget attestation" in AL.verify(None, program=PROGRAM, path=tmp_path / "missing.db")
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"not sqlite" * 50)
    assert "unreadable" in AL.verify(att, program=PROGRAM, path=bad)


def test_budgets_are_per_calendar_year_in_ist(tmp_path):
    path = tmp_path / "ledger.db"
    late_dec = datetime(2026, 12, 31, 18, 40, tzinfo=timezone.utc)          # 1 Jan 2027 00:10 IST
    a = AL.reserve(PROGRAM, "look-dec", now=datetime(2026, 12, 31, 10, 0, tzinfo=timezone.utc), path=path)
    b = AL.reserve(PROGRAM, "look-jan", now=late_dec, path=path)
    assert (a["year"], b["year"], b["look_index"]) == (2026, 2027, 1)


def test_concurrent_reservations_get_distinct_contiguous_looks(tmp_path):
    path = tmp_path / "ledger.db"
    AL.reserve(PROGRAM, "seed", now=T, path=path)
    barrier, out = threading.Barrier(10), []

    def worker(i):
        barrier.wait()
        out.append(AL.reserve(PROGRAM, f"look-{i}", now=T, path=path)["look_index"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(out) == list(range(2, 12))


def test_the_walk_forward_report_spends_and_attests_a_look(monkeypatch):
    import scan.signal_backtest as SB
    from research import alpha_ledger
    from scan.unified_scanner import evidence_provenance_note
    from tests.evidence_fixtures import proven_provenance

    prov = proven_provenance()
    assert evidence_provenance_note(prov) == ""
    assert prov["alpha_accounting"]["program_id"] == alpha_ledger.program_id(SB.gate_program())
    exhausted = {**prov["alpha_accounting"], "look_index": 99}
    assert "does not match the ledger" in evidence_provenance_note({**prov, "alpha_accounting": exhausted})
    assert "no alpha-budget attestation" in evidence_provenance_note({**prov, "alpha_accounting": None})


def test_an_exhausted_budget_admits_nothing_in_the_family_correction():
    import scan.signal_backtest as SB
    rows = {"REAL": {"expectancy_r_precise": 0.5, "se_r": 0.01, "df": 30, "inference": "OVERLAP_ROBUST"}}
    SB.apply_family_correction(rows, {}, alpha=0.0)
    assert rows["REAL"]["family_significant"] is False
