"""Lifecycle audit of 24 Nov (F1): a GOOD_WAIT credit cut short by a crash is finished, never lost.

A later fill of a waited name claims its unit, updates REJECT::<reason> and
ENTRY_QUALITY::GOOD_WAIT, then rewrites the counterfactual ledger. Nothing on disk
recorded the fill, and the fill is never delivered again. A crash anywhere in that pass
lost the credit for good. The next official settlement then classified the filled WAIT on
its price path, MISSED_WINNER, beside a policy that had already counted GOOD_WAIT. The
unit stayed CLAIMED forever.

The fill is now marked durably first, and every credit is made from the mark. A unit the
credit claimed before marks existed is finished with its own fill. A crash BEFORE the mark
(after the paper fill was saved) still loses that one credit. The result is an under-count,
with the ledger and the policies agreeing, never a double count.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FILL = "2026-09-15"
UNIT = "cf:2026-09-10|TCS|WAIT_FOR_PULLBACK"


class _Crash(BaseException):
    """The process dies here: no ``except Exception`` handler runs."""


@pytest.fixture
def wait(monkeypatch, tmp_path):
    import pandas as pd
    from core import outcome_resolver, session_window
    from product import autonomous_loop as loop
    from product import counterfactual_learning as CF
    from product import forward_soak
    from tests.evidence_fixtures import rulebook_outcome

    days = [d.date().isoformat() for d in pd.bdate_range("2026-07-01", "2026-09-30")]
    monkeypatch.setattr(session_window, "official_dates", lambda: days)
    # On its price path alone this WAIT is a missed winner: the mislabel the audit found.
    monkeypatch.setattr(outcome_resolver, "counterfactual_outcome",
                        lambda *a, **k: rulebook_outcome(1.0, "WIN", 10.0))
    monkeypatch.setattr(forward_soak, "settle_pending_from_market", lambda: {})
    monkeypatch.setattr(loop, "LEARNING_PATH", tmp_path / "official-learning.jsonl")
    CF.freeze_decision(symbol="TCS", reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                       entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": "w"})
    yield
    from product import sqlite_runtime
    sqlite_runtime.close_keepers()


def _rows() -> list[dict]:
    from product import counterfactual_learning as CF
    return [json.loads(line) for line in CF.ledger_path().read_text().splitlines() if line.strip()]


def _n(policy_id: str) -> int:
    from tests.test_learning_exactly_once import _n as n
    return n(policy_id)


def _units() -> dict[tuple[str, str], str]:
    import sqlite3
    from product import learning_observations as LO
    from product import learning_policy_store as LPS

    led = LO.ledger_path_for(LPS.policy_path())
    if not led.exists():
        return {}
    con = sqlite3.connect(str(led))
    try:
        return {(s, k): st for s, k, st in con.execute("SELECT scope, observation_key, state FROM observations")}
    finally:
        con.close()


def _restart_and_settle() -> None:
    from product import autonomous_loop as loop
    loop.settle_official_outcomes("2026-09-30")
    loop.settle_official_outcomes("2026-09-30")                 # a second pass changes nothing


def _credited_once() -> None:
    (row,) = _rows()
    assert (row["classification"], row.get("later_entry_verified")) == ("GOOD_WAIT", FILL), row
    assert "later_entry_pending" not in row
    assert (_n("ENTRY_QUALITY::GOOD_WAIT"), _n("REJECT::WAIT_FOR_PULLBACK")) == (1, 1)
    units = _units()
    assert len(units) == 3 and set(units.values()) == {"APPLIED"}, units


def _crash_at(m, point: str) -> None:
    from core import atomic_io
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF
    from product import learning_observations as LO

    ledger = CF.ledger_path()
    writes = [0]
    real_write = atomic_io.atomic_write_text

    def write(path, *a, **k):
        mine = Path(path) == ledger
        if mine:
            writes[0] += 1
            if point == "before-ledger-rewrite" and writes[0] == 2:
                raise _Crash
            if point == "before-marker" and writes[0] == 1:
                raise _Crash
        out = real_write(path, *a, **k)
        if mine and point == "after-marker" and writes[0] == 1:
            raise _Crash
        return out

    real_claim = LO.claim

    def claim(path, scope, key, session, payload=None):
        out = real_claim(path, scope, key, session, payload=payload)
        if point == "after-unit-claim" and scope == "COUNTERFACTUAL":
            raise _Crash
        return out

    real_record = core.record_measured_outcome

    def record(**kw):
        out = real_record(**kw)
        if point == "after-first-policy":
            raise _Crash
        return out

    m.setattr(atomic_io, "atomic_write_text", write)
    m.setattr(LO, "claim", claim)
    m.setattr(core, "record_measured_outcome", record)


@pytest.mark.parametrize("point", ["after-marker", "after-unit-claim", "after-first-policy",
                                   "before-ledger-rewrite"])
def test_a_credit_cut_short_anywhere_after_its_mark_is_finished_once(wait, monkeypatch, point):
    from product import _paper_learning_loop_core as core

    with monkeypatch.context() as m:
        _crash_at(m, point)
        with pytest.raises(_Crash):
            core.note_later_entry("TCS", as_of=FILL)
    (row,) = _rows()
    assert not row.get("classification") and row["later_entry_pending"]["fill_session"] == FILL
    _restart_and_settle()
    _credited_once()


def test_without_a_crash_the_credit_lands_at_once(wait):
    from product import _paper_learning_loop_core as core

    out = core.note_later_entry("TCS", as_of=FILL)
    assert (out["updated"], out["classifications"], out["queued"]) == (1, {"GOOD_WAIT": 1}, 1)
    _credited_once()
    _restart_and_settle()
    _credited_once()


def test_the_fill_is_marked_durably_before_anything_is_credited(wait, monkeypatch):
    from core import atomic_io
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF
    from product import learning_observations as LO

    ledger, events = CF.ledger_path(), []
    real_write, real_claim = atomic_io.atomic_write_text, LO.claim

    def write(path, text, *a, **k):
        if Path(path) == ledger:
            events.append(("ledger", "later_entry_pending" in text, bool(k.get("durable"))))
        return real_write(path, text, *a, **k)

    def claim(path, scope, key, session, payload=None):
        events.append(("claim", scope))
        return real_claim(path, scope, key, session, payload=payload)

    monkeypatch.setattr(atomic_io, "atomic_write_text", write)
    monkeypatch.setattr(LO, "claim", claim)
    core.note_later_entry("TCS", as_of=FILL)
    assert events[0] == ("ledger", True, True), events
    assert events.index(("claim", "COUNTERFACTUAL")) > 0


def test_a_crash_before_the_mark_under_counts_and_never_contradicts(wait, monkeypatch):
    """The residual window: the paper fill is saved, its mark is not. One credit is lost; the
    WAIT settles on its price path and no policy claims otherwise."""
    from product import _paper_learning_loop_core as core

    with monkeypatch.context() as m:
        _crash_at(m, "before-marker")
        with pytest.raises(_Crash):
            core.note_later_entry("TCS", as_of=FILL)
    _restart_and_settle()
    (row,) = _rows()
    assert row["classification"] == "MISSED_WINNER"
    assert (_n("ENTRY_QUALITY::GOOD_WAIT"), _n("REJECT::WAIT_FOR_PULLBACK"), _units()) == (0, 0, {})


def _strand_a_pre_mark_credit(monkeypatch, left: str) -> None:
    """What the pre-fix credit left on disk: the unit claimed with its GOOD_WAIT payload, and
    the ledger row neither marked nor classified."""
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF
    from product import learning_observations as LO

    (row,) = _rows()
    settled = CF.settle(row, forward_return_pct=None, later_entered=True)
    settled["later_entry_verified"] = FILL
    with monkeypatch.context() as m:
        if left == "CLAIMED":
            real_claim = LO.claim

            def claim(path, scope, key, session, payload=None):
                out = real_claim(path, scope, key, session, payload=payload)
                raise _Crash

            m.setattr(LO, "claim", claim)
            with pytest.raises(_Crash):
                core.ingest_counterfactual(settled)
        else:
            core.ingest_counterfactual(settled)
    assert _units()[("COUNTERFACTUAL", UNIT)] == left
    (row,) = _rows()
    assert not row.get("classification") and not row.get("later_entry_pending")


@pytest.mark.parametrize("left", ["CLAIMED", "APPLIED"])
def test_a_credit_cut_short_before_marks_existed_is_finished_with_its_own_fill(wait, monkeypatch, left):
    _strand_a_pre_mark_credit(monkeypatch, left)
    _restart_and_settle()
    _credited_once()


def test_a_recovered_credit_that_fails_again_keeps_the_wait_off_its_price_path(wait, monkeypatch):
    from product import autonomous_loop as loop
    from product import learning_observations as LO

    _strand_a_pre_mark_credit(monkeypatch, "CLAIMED")
    with monkeypatch.context() as m:
        m.setattr(LO, "mark_applied", lambda *a, **k: (_ for _ in ()).throw(OSError("learning store unavailable")))
        out = loop.settle_official_outcomes("2026-09-30")
    (row,) = _rows()
    assert not row.get("classification"), row                      # never priced as a missed winner
    assert row["later_entry_pending"] == {"fill_session": FILL, "recovered_from": "observation_unit"}
    assert out["good_waits"]["errors"] and any(p.get("reason") == "LATER_ENTRY_UNVERIFIED" for p in out["pending"])
    _restart_and_settle()
    _credited_once()                                                # the half-applied policies counted once


def test_a_failed_credit_is_reported_and_stays_queued(wait, monkeypatch, capsys):
    from product import _paper_learning_loop_core as core
    from product import learning_observations as LO

    with monkeypatch.context() as m:
        m.setattr(LO, "mark_applied", lambda *a, **k: (_ for _ in ()).throw(OSError("learning store unavailable")))
        out = core.note_later_entry("TCS", as_of=FILL)
    assert out["updated"] == 0 and out["errors"]
    assert "[COUNTERFACTUAL] later-entry credit" in capsys.readouterr().err
    assert _rows()[0]["later_entry_pending"]["fill_session"] == FILL
    _restart_and_settle()
    _credited_once()


def test_a_price_settlement_never_erases_a_queued_fill(wait, monkeypatch):
    from core import session_window
    from product import _paper_learning_loop_core as core

    monkeypatch.setattr(session_window, "official_dates", lambda: [])    # not verifiable yet: queued
    assert core.note_later_entry("TCS", as_of=FILL)["updated"] == 0
    assert _rows()[0]["later_entry_pending"]["fill_session"] == FILL
    out = core.settle_pending_counterfactuals(forward_return_by_symbol={"TCS": 12.0})
    assert out["updated"] == 0 and not _rows()[0].get("classification")


def test_a_credit_pass_for_one_symbol_leaves_the_others_alone(wait):
    from product import _paper_learning_loop_core as core
    from product import counterfactual_learning as CF

    CF.freeze_decision(symbol="INFY", reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                       entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": "i"})
    rows = _rows()
    rows[1]["later_entry_pending"] = {"fill_session": "2026-09-11"}
    CF.ledger_path().write_text("".join(json.dumps(r) + "\n" for r in rows))
    core.note_later_entry("TCS", as_of=FILL)
    tcs, infy = _rows()
    assert tcs["classification"] == "GOOD_WAIT"
    assert not infy.get("classification") and infy["later_entry_pending"] == {"fill_session": "2026-09-11"}


def test_a_copy_of_a_queued_unit_is_never_priced_and_carries_the_units_label(wait, monkeypatch):
    """Every paper cycle re-freezes a WAIT: a copy of the same unit (day, symbol, reason) frozen
    after the fill was marked has no mark of its own. The price path used to label it
    MISSED_WINNER beside the unit's GOOD_WAIT credit."""
    from core import session_window
    from product import _paper_learning_loop_core as core
    from product import autonomous_loop as loop
    from product import counterfactual_learning as CF

    days = session_window.official_dates()
    monkeypatch.setattr(session_window, "official_dates", lambda: [])    # the fill cannot verify yet
    assert core.note_later_entry("TCS", as_of=FILL)["queued"] == 1
    CF.freeze_decision(symbol="TCS", reason_code="WAIT_FOR_PULLBACK", decision="WAIT",
                       entry=100, stop=94, target=116, as_of="2026-09-10", evidence={"decision_id": "copy-2"})
    first, copy = _rows()
    assert first["later_entry_pending"] and not copy.get("later_entry_pending")
    out = loop.settle_official_outcomes("2026-09-30")
    assert [r.get("classification") for r in _rows()] == [None, None], out
    assert sum(p.get("reason") == "LATER_ENTRY_UNVERIFIED" for p in out["pending"]) == 2
    monkeypatch.setattr(session_window, "official_dates", lambda: days)  # the archive is back
    loop.settle_official_outcomes("2026-09-30")
    rows = _rows()
    assert [(r["classification"], r.get("later_entry_verified")) for r in rows] == [("GOOD_WAIT", FILL)] * 2
    assert not any("later_entry_pending" in r for r in rows)
    assert (_n("ENTRY_QUALITY::GOOD_WAIT"), _n("REJECT::WAIT_FOR_PULLBACK")) == (1, 1)
    _restart_and_settle()
    assert [r["classification"] for r in _rows()] == ["GOOD_WAIT"] * 2
    assert (_n("ENTRY_QUALITY::GOOD_WAIT"), _n("REJECT::WAIT_FOR_PULLBACK")) == (1, 1)


def test_no_wait_is_priced_while_the_good_wait_recovery_is_unavailable(wait, monkeypatch):
    """A credit stranded before marks existed is found only through the learning store. When
    that read fails, pricing the WAIT anyway would settle it MISSED_WINNER beside its claimed
    GOOD_WAIT unit; the WAIT waits a pass instead. A non-WAIT still settles."""
    from product import autonomous_loop as loop
    from product import counterfactual_learning as CF
    from product import learning_observations as LO

    _strand_a_pre_mark_credit(monkeypatch, "CLAIMED")
    CF.freeze_decision(symbol="INFY", reason_code="LOW_QUALITY_SETUP", decision="REJECT",
                       entry=100, stop=94, target=115, as_of="2026-09-10", evidence={"note": "r"})
    with monkeypatch.context() as m:
        m.setattr(LO, "scope_payloads", lambda *a, **k: (_ for _ in ()).throw(OSError("learning store unavailable")))
        out = loop.settle_official_outcomes("2026-09-30")
    assert "error" in out["good_waits"]
    tcs, infy = _rows()
    assert not tcs.get("classification") and infy["classification"] == "MISSED_WINNER"
    assert any(p.get("reason") == "GOOD_WAIT_RECOVERY_UNAVAILABLE" and p.get("symbol") == "TCS"
               for p in out["pending"])
    _restart_and_settle()
    assert _rows()[0]["classification"] == "GOOD_WAIT"
    assert (_n("ENTRY_QUALITY::GOOD_WAIT"), _n("REJECT::WAIT_FOR_PULLBACK")) == (1, 1)
