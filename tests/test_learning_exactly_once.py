"""Settlement → learning is exactly once across crashes (cross-review A-3, 2026-11-21).

Before: the forward ledger was marked settled, THEN learning was ingested, and
inside the ingest an observation was claimed, THEN the policy was written.
A crash at either gap lost the learning event for good (every retry saw
"settled" / "duplicate"). Now:

  outbox event (durable) → ledger settled → per-unit claim (CLAIMED, resumable)
  → per-policy claim (CLAIMED + payload) → policy write (stamped) → APPLIED
  → event DONE

Each test crashes at one boundary, retries, and asserts every independent
observation is applied exactly once: never zero, never twice.
"""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from product import forward_evidence as FE
from product import forward_soak as FS
from product import learning_observations as LO
from product import learning_policy_store as LPS
from product import paper_learning_loop as PLL
from tests.evidence_fixtures import rulebook_outcome
from tests.test_forward_settlement_batch import _freeze_rejections, _open_ids


def _outcome(_symbol, _as_of, _row):
    return rulebook_outcome(-0.4, "LOSS", -2.0)           # every rejection: an avoided loser


def _settle(**kw):
    return FS.settle_pending_from_market(outcome_fn=_outcome, **kw)


def _n(policy_id="REJECT::ENTRY_TOO_EXTENDED") -> int:
    store = LPS.load_policies()
    row = next((p for p in store.get("policies") or [] if p.get("policy_id") == policy_id), {})
    return int(row.get("sample_size") or 0)


def _ledger():
    return LO.ledger_path_for(LPS.policy_path())


def test_a_crash_before_the_ledger_write_settles_and_learns_once_on_retry(monkeypatch):
    _freeze_rejections(4)
    real = FE.attach_settlements
    monkeypatch.setattr(FS, "attach_settlements", lambda items, **k: (_ for _ in ()).throw(OSError("power cut")))
    with pytest.raises(OSError):
        _settle()
    assert len(_open_ids()) == 4 and _n() == 0            # nothing settled, events queued
    assert LO.outbox_counts(_ledger()).get(LO.PENDING) == 4
    monkeypatch.setattr(FS, "attach_settlements", real)
    _settle()
    assert _open_ids() == [] and _n() == 4
    assert LO.outbox_counts(_ledger()) == {LO.DONE: 4}
    _settle()
    assert _n() == 4                                         # a third run changes nothing


def test_a_crash_after_the_ledger_write_is_replayed_not_lost(monkeypatch):
    _freeze_rejections(4)
    real = PLL.ingest_counterfactual
    monkeypatch.setattr(PLL, "ingest_counterfactual", lambda *a, **k: (_ for _ in ()).throw(OSError("power cut")))
    with pytest.raises(RuntimeError):
        _settle()
    assert _open_ids() == [] and _n() == 0                # settled on the ledger, learning queued
    assert LO.outbox_counts(_ledger()).get(LO.PENDING) == 4
    monkeypatch.setattr(PLL, "ingest_counterfactual", real)
    out = _settle()                                          # nothing left to settle: the replay applies them
    assert out["updated"] == 0 and _n() == 4
    assert LO.outbox_counts(_ledger()) == {LO.DONE: 4}


def test_a_crash_between_the_policy_write_and_its_applied_mark_never_double_counts(monkeypatch):
    _freeze_rejections(3)
    real = LO.mark_applied
    armed = {"on": True}

    def flaky(path, scope, key):
        if armed.pop("on", False) and scope.startswith("REJECT::"):
            raise OSError("power cut after the policy write")
        return real(path, scope, key)

    monkeypatch.setattr(LO, "mark_applied", flaky)
    with pytest.raises(RuntimeError):
        _settle()
    # The first policy write landed before the crash; the next apply's recovery
    # confirms it by its stamp, so the later units still count once each.
    assert _n() == 3
    _settle()
    assert _n() == 3                                         # confirmed by its stamp, not applied twice
    assert LO.claimed_unapplied(_ledger()) == []


def test_a_crash_after_the_unit_claim_resumes_the_unit(monkeypatch):
    _freeze_rejections(2)
    real = LPS._apply_measured
    armed = {"on": True}

    def flaky(**kw):
        if armed.pop("on", False):
            raise OSError("power cut before the policy write")
        return real(**kw)

    monkeypatch.setattr(LPS, "_apply_measured", flaky)
    with pytest.raises(RuntimeError):
        _settle()
    _settle()
    assert _n() == 2 and LO.claimed_unapplied(_ledger()) == []


def test_copies_and_repeated_deliveries_are_one_observation_each():
    for cycle in ("c1", "c2", "c3"):                       # three cycles re-freeze the same 3 names
        _freeze_rejections(3, cycle_id=cycle)
    _settle()
    assert _n() == 3
    for ev in LO.pending_events(_ledger()):
        raise AssertionError(f"left pending: {ev}")
    FS.replay_learning_outbox(settled_ids=FS._settled_decision_ids(FE.load_ledger()))
    assert _n() == 3


def test_concurrent_workers_apply_one_unit_once():
    settled = {"symbol": "TCS", "as_of": "2026-09-01", "reason_code": "ENTRY_TOO_EXTENDED",
               "classification": "AVOIDED_LOSER", "eval_spec": rulebook_outcome(0)["eval_spec"],
               "evidence": {"setup_label": "VCP"}}
    errors: list[Exception] = []

    def worker():
        try:
            PLL.ingest_counterfactual(dict(settled))
        except Exception as exc:              # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and _n() == 1


def test_a_corrupt_event_is_quarantined_and_visible():
    _freeze_rejections(1)
    _settle()
    LO.enqueue_event(_ledger(), "counterfactual:forged:spec", FS.LEARNING_EVENT_KIND, {"symbol": "X"})
    out = FS.replay_learning_outbox(settled_ids=FS._settled_decision_ids(FE.load_ledger()))
    assert out["quarantined"] == 1 and LO.outbox_counts(_ledger()).get(LO.QUARANTINED) == 1
    assert _n() == 1


def test_a_legacy_claim_still_dedupes_but_is_never_replayed(tmp_path):
    """Claims written before two-phase claims cannot prove they were applied: LEGACY, never guessed."""
    db = tmp_path / "policies.observations.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE observations (scope TEXT NOT NULL, observation_key TEXT NOT NULL, "
                 "session TEXT NOT NULL, recorded_at TEXT NOT NULL, PRIMARY KEY (scope, observation_key))")
    conn.execute("INSERT INTO observations VALUES ('REJECT::X', 'cf:old', '2026-09-01', 'then')")
    conn.commit()
    conn.close()
    assert LO.claim_state(db, "REJECT::X", "cf:old") == LO.LEGACY
    assert LO.claim(db, "REJECT::X", "cf:old", "2026-09-01", payload={"policy_id": "REJECT::X"}) is False
    assert LO.claimed_unapplied(db) == []


def test_a_legacy_label_is_skipped_not_learned():
    _freeze_rejections(2)
    FS.settle_pending_from_market(outcome_fn=lambda *a: {**rulebook_outcome(-0.4, "LOSS"), "eval_spec": ""})
    assert _n() == 0
    assert json.dumps(LO.outbox_counts(_ledger())) == json.dumps({LO.DONE: 2})


def test_good_wait_credits_only_a_live_wait_of_that_symbol(monkeypatch):
    """A later entry credits a WAIT inside its own horizon, never a hard rejection or a stale wait."""
    import pandas as pd
    from core import session_window
    from product import _paper_learning_loop_core as core
    from product.counterfactual_learning import freeze_decision

    # The session ordinal is verified from the official archive (L-08): give it one.
    monkeypatch.setattr(session_window, "official_dates",
                        lambda: [d.date().isoformat() for d in pd.bdate_range("2026-07-01", "2026-09-15")])
    for decision, day, reason in (("WAIT", "2026-09-10", "WAIT_FOR_PULLBACK"),
                                  ("BLOCK", "2026-09-10", "LOW_QUALITY_SETUP"),
                                  ("WAIT", "2026-08-01", "WAIT_FOR_PULLBACK")):
        freeze_decision(symbol="TCS", reason_code=reason, decision=decision, entry=100, stop=94, target=115,
                        as_of=day, evidence={"note": f"{decision}-{day}"})
    out = core.note_later_entry("TCS", as_of="2026-09-15")
    assert out["updated"] == 1 and out["classifications"] == {"GOOD_WAIT": 1}
    assert _n("ENTRY_QUALITY::GOOD_WAIT") == 1
