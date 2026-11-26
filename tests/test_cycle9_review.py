"""Cycle 9 GPT cross-review (two read-only passes of 161f7cbe..a8cb2eef): each claim as a test.

Part A (learning, settlement, autonomy) A1..A6; part B (maintenance tool, research inputs)
C9B-01..07. A claim is acted on only when its test fails on a8cb2eef; the audit
(docs/audits/2026-11-26-cycle9-checkpoint.md) records the verdict and any disagreement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from product.forward_soak import settle_pending_from_market as _real_forward_settle
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401
from tests.test_good_wait_crash_safety import (FILL, _credited_once, _n, _restart_and_settle, _rows,  # noqa: F401
                                               _strand_a_pre_mark_credit, wait)


# ── A2: the forward settler must not claim a unit whose verified fill is queued ──

def test_A2_forward_settlement_cannot_take_a_queued_good_wait(wait, monkeypatch):
    from product import autonomous_loop as loop
    from product import forward_evidence as FE, forward_soak as FS
    from product import learning_observations as LO, paper_learning_loop as PLL

    FE.freeze_observation({"symbol": "TCS", "decision": "WAIT", "reason_code": "WAIT_FOR_PULLBACK",
                           "entry": 100, "stop": 94, "target": 115},
                          cycle_id="wait-cycle", as_of="2026-09-10", group="WAIT",
                          provenance=FE.REAL_FORWARD_MARKET)
    assert PLL._queue_later_fills("TCS", FILL) == 1
    monkeypatch.setattr(FS, "settle_pending_from_market", _real_forward_settle)
    claim = LO.claim

    def fail_good_wait(path, scope, key, session, payload=None):
        if scope == "COUNTERFACTUAL" and ((payload or {}).get("settled") or {}).get("classification") == "GOOD_WAIT":
            raise OSError("claim interrupted")
        return claim(path, scope, key, session, payload=payload)

    with monkeypatch.context() as m:
        m.setattr(LO, "claim", fail_good_wait)
        loop.settle_official_outcomes("2026-09-30")
    loop.settle_official_outcomes("2026-09-30")
    forward = FE.load_ledger()[0]
    assert (_rows()[0]["classification"], forward.get("counterfactual_classification"),
            _n("ENTRY_QUALITY::GOOD_WAIT")) == ("GOOD_WAIT", "GOOD_WAIT", 1)


# ── A3: every copy of a unit takes the unit's verified fill ──

@pytest.mark.parametrize("interleaving", ["two_markers", "freeze_after_recovery"])
def test_A3_all_wait_copies_use_the_units_verified_fill(wait, monkeypatch, interleaving):
    from core import session_window
    from product import autonomous_loop as loop
    from product import counterfactual_learning as CF, paper_learning_loop as PLL

    def freeze_copy():
        CF.freeze_decision(symbol="TCS", reason_code="WAIT_FOR_PULLBACK", decision="WAIT", entry=100, stop=94,
                           target=116, as_of="2026-09-10", evidence={"decision_id": "copy-2"})

    if interleaving == "two_markers":
        with monkeypatch.context() as m:
            m.setattr(session_window, "official_dates", lambda: [])
            assert PLL.note_later_entry("TCS", as_of=FILL)["queued"] == 1
            freeze_copy()
            assert PLL.note_later_entry("TCS", as_of="2026-09-29")["queued"] == 1
    else:
        assert PLL.note_later_entry("TCS", as_of=FILL)["updated"] == 1
        retry = PLL.retry_pending_good_waits

        def retry_then_freeze(**kwargs):
            out = retry(**kwargs)
            freeze_copy()
            return out

        monkeypatch.setattr(PLL, "retry_pending_good_waits", retry_then_freeze)
    loop.settle_official_outcomes("2026-09-30")
    if interleaving == "two_markers":                 # one pass: the expired copy takes the unit's label
        assert [(r["classification"], r.get("later_entry_verified")) for r in _rows()] == [("GOOD_WAIT", FILL)] * 2
    else:                                             # never priced beside the credited unit
        assert all(r.get("classification") in (None, "GOOD_WAIT") for r in _rows())
    loop.settle_official_outcomes("2026-09-30")
    assert [(r["classification"], r.get("later_entry_verified")) for r in _rows()] == [("GOOD_WAIT", FILL)] * 2


# ── A4: a failed GOOD_WAIT recovery makes the EOD job retry ──

@pytest.mark.parametrize("failure", ["store_unreadable", "credit_fails"])
def test_A4_eod_retries_when_good_wait_recovery_fails(wait, monkeypatch, failure):
    from product import autonomous_loop as loop, learning_observations as LO
    from research.autonomy import job_store as JS, jobs as J, outcome_integrity as OI

    _strand_a_pre_mark_credit(monkeypatch, "CLAIMED")

    def base_handler(_ctx):
        official = loop.settle_official_outcomes("2026-09-30")
        return J.JobResult(JS.SUCCEEDED, "outcomes resolved", unblocks=("OUTCOMES:2026-09-30",),
                           metadata={"official_settlement": official})

    if failure == "store_unreadable":
        monkeypatch.setattr(LO, "scope_payloads", lambda *a, **k: (_ for _ in ()).throw(OSError("store unavailable")))
    else:                                             # one row's credit fails; the holds still read
        monkeypatch.setattr(LO, "mark_applied", lambda *a, **k: (_ for _ in ()).throw(OSError("learning store busy")))
    monkeypatch.setattr(OI, "_BASE_HANDLER", base_handler)
    monkeypatch.setattr(OI, "_unreported_forward_failures", lambda: [])
    result = OI.strict_outcome_resolution(None)
    assert result.status == JS.RETRYABLE_FAILED and not result.unblocks


# ── A6: a stop inside lease acquisition leaves the job pending ──

def test_A6_a_stop_inside_the_lease_leaves_the_paper_job_pending(tmp_path, monkeypatch):
    from research.autonomy import job_store as JS, schedules as SCH
    from research.autonomy.supervisor import Supervisor
    from tests.test_autonomy_console_runtime import _Deps

    sup = Supervisor(tmp_path / "auto", deps=_Deps())
    try:
        assert sup.start()
        job = sup.jobs.enqueue(SCH.PAPER_CYCLE, idempotency_key="snapshot_paper:test")
        lease, dispatched = sup.jobs.lease_due, []

        def stop_then_lease(*args, **kwargs):
            sup.stop()
            return lease(*args, **kwargs)

        monkeypatch.setattr(sup, "enqueue_due", lambda *_: None)
        monkeypatch.setattr(sup, "_manage_live_feed", lambda *_: None)
        monkeypatch.setattr(sup.jobs, "lease_due", stop_then_lease)
        monkeypatch.setattr(sup, "_execute", lambda j: dispatched.append(j.job_id))
        sup.tick()
        stored = sup.jobs.get(job.job_id)
        assert (stored.status, stored.attempt, dispatched) == (JS.PENDING, 0, [])
    finally:
        sup.shutdown()


# ── C9B-01: a removed lock file cannot let a second command in ──

def test_C9B01_a_lock_removed_at_the_rename_cannot_erase_an_apply(f1, monkeypatch):
    from product import sqlite_index_repair as repair

    db, manifest = f1
    manifest = manifest.resolve()
    plan_f1(db, manifest)
    real_replace, armed, applied = os.replace, [True], []

    def replace(src, dst, *args, **kwargs):
        if armed[0] and Path(str(dst)).name == manifest.name:
            armed[0] = False
            for p in manifest.parent.glob(f".{manifest.name.casefold()}*.lock"):
                p.unlink()
            try:
                applied.append(repair.apply(manifest, "test", target_is_copy=True))
            except SystemExit:
                pass
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    try:
        plan_f1(db, manifest)
    except SystemExit:
        pass
    monkeypatch.undo()
    if applied:
        assert json.loads(manifest.read_text()).get("backup") == applied[0]["backup"]


# ── C9B-07: a parent swap cannot create a lock beside the database ──

def test_C9B07_a_parent_swap_cannot_create_anything_beside_the_database(f1, tmp_path, monkeypatch):
    from product import sqlite_index_repair as repair

    db, _ = f1
    ops = tmp_path.resolve() / "ops"
    ops.mkdir()
    manifest = ops / "repair.json"
    plan_f1(db, manifest)
    before = sorted(p.name for p in db.parent.iterdir())
    real_open, swapped = os.open, [False]

    def open_after_parent_swap(path, flags, *args, **kwargs):
        # The swap lands on the first open the lock makes in (or of) the manifest directory,
        # after the place checks passed: the directory becomes a symlink to the database's.
        if not swapped[0] and kwargs.get("dir_fd") is None and str(path).startswith(str(ops)) \
                and str(path) != str(manifest):
            swapped[0] = True
            ops.rename(ops.with_name("ops-original"))
            ops.symlink_to(db.parent.resolve(), target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_after_parent_swap)
    with pytest.raises(SystemExit):
        repair.apply(manifest, "test", target_is_copy=True)
    monkeypatch.undo()
    assert swapped[0]
    assert sorted(p.name for p in db.parent.iterdir()) == before


# ── feature002 (C9B-02..06) ──

from research.feature002 import observe as OBS  # noqa: E402
from research.feature002.provenance import row_input  # noqa: E402
from tests.test_cycle9_feature002_pit import _card, _hist, build  # noqa: E402,F401


def _store_env(monkeypatch, store, events=None):
    from data import bhavcopy_store as bs
    from scan import bulk_fetcher as bf

    monkeypatch.setattr(bs, "_store", store)
    monkeypatch.setattr(bs, "_CA_EVENTS", events or {})
    monkeypatch.setattr(bf, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bf, "_bhav_ok", True)
    monkeypatch.setattr(OBS, "_sector_map", lambda: ({}, "test"))
    monkeypatch.setattr(OBS, "_build_rs_table", lambda *a, **k: None)
    return bs, bf


def test_C9B02_an_overlay_published_after_recorded_at_never_reaches_a_row(monkeypatch):
    bs, _ = _store_env(monkeypatch, {"AAA": _hist(last="2026-08-21", tag="bhavcopy_store")})
    day, rows, clock = "2026-08-24", [], ["2026-08-24T10:30:00+05:30"]
    monkeypatch.setattr(OBS, "is_enabled", lambda: True)
    monkeypatch.setattr(OBS, "_write_hook_event", lambda *_: None)
    monkeypatch.setattr(OBS, "_session_date", lambda _: day)
    monkeypatch.setattr(OBS, "_ist_now_iso", lambda: clock[0])
    monkeypatch.setattr(OBS, "persist_shadow", lambda _c, batch, **kw: rows.extend(batch))

    def publish_during_regime_lookup():
        clock[0] = f"{day}T10:31:00+05:30"
        with bs._lock:
            bs._store["AAA"] = _hist(last=day, overlay=day, tag="bhavcopy_store")
        return ""

    monkeypatch.setattr(OBS, "_regime", publish_during_regime_lookup)
    OBS.observe_production_scan([_card("AAA")], background=False)
    row = rows[0]
    assert row["recorded_at"] >= f"{day}T10:31:00+05:30" or row_input(row)["symbol_hist_as_of"] < day


def test_C9B03_a_split_after_the_cutoff_never_rescales_the_kept_bars(monkeypatch):
    raw = _hist(last="2026-08-24")
    friday = "2026-08-21"
    _store_env(monkeypatch, {"AAA": raw}, {"AAA": [{"ex_date": pd.Timestamp("2026-08-24"), "factor": 2.0,
                                                    "type": "split"}]})
    _, rows = OBS.build_shadow_records([_card("AAA")], session_date=friday, recorded_at=f"{friday}T18:05:00+05:30",
                                       scan_cycle_id="before-split", source="live_scan")
    assert rows[0]["feature_snapshot"]["trend"]["price"] == pytest.approx(raw.loc[friday, "close"])


def test_C9B04_a_store_rebuild_during_the_reads_never_ranks_two_generations(monkeypatch):
    old = {s: _hist(last="2026-08-21", slope=0.2) for s in ("AAA", "BBB")}
    new = {s: _hist(last="2026-08-21", slope=0.8) for s in ("AAA", "BBB")}
    bs, bf = _store_env(monkeypatch, old)
    real_get, switched = bf.get_cached, [False]

    def get_then_publish(symbol):
        frame = real_get(symbol)
        if not switched[0]:
            switched[0] = True
            with bs._lock:
                bs._store = new
                bs._bump_revision_locked()
        return frame

    monkeypatch.setattr(bf, "get_cached", get_then_publish)
    _, rows = OBS.build_shadow_records([_card("AAA"), _card("BBB")], session_date="2026-08-24",
                                       recorded_at="2026-08-24T10:30:00+05:30", scan_cycle_id="switch",
                                       source="live_scan")
    if all(r["trend_rank"] is not None for r in rows):
        prices = tuple(r["feature_snapshot"]["trend"]["price"] for r in rows)
        coherent = [tuple(store[s]["close"].iloc[-1] for s in ("AAA", "BBB")) for store in (old, new)]
        assert prices in coherent


@pytest.mark.parametrize("tags", [(("kite_snapshot", "s1"), ("kite_snapshot", "s2")),
                                  ((None, None), ("yfinance", None))])
def test_C9B05_a_withheld_cross_section_persists_no_trend_rank(build, tags):
    rows = build({s: _hist(tag=tag, snapshot=sid) for s, (tag, sid) in zip(("AAA", "BBB"), tags)})
    assert all(r["trend_rank"] is None and r["trend_pctl_in_set"] is None for r in rows.values())


def test_C9B06_an_excluded_row_does_not_erase_the_ranked_spearman_sample(build, monkeypatch):
    from research.feature002 import evaluate

    frames = {f"K{i}": _hist(tag="kite_snapshot", snapshot="s1", slope=0.2 + i / 10) for i in range(8)}
    frames["FALLBACK"] = _hist(last="2026-08-21", tag="bhavcopy_store")
    rows = [dict(row, ret_5d=i / 100) for i, row in enumerate(build(frames).values())]
    monkeypatch.setattr(evaluate, "list_primary_observations", lambda **_: rows)
    monkeypatch.setattr(evaluate, "counts", lambda **_: {})
    monkeypatch.setattr(evaluate, "maturity", lambda *a, **kw: {"stage": "INTERIM", "verdict": None})
    metric = evaluate.summarize()["rank_metrics"]["spearman_trend"]
    assert metric["n"] == 8 and metric["rho"] is not None


def test_A2_a_unit_decided_elsewhere_first_is_never_stamped_good_wait(wait):
    """The recovery reads the winning claim: a unit another settlement already applied as
    MISSED_WINNER (before holds existed) keeps that verdict; the queued fill is superseded."""
    from product import counterfactual_learning as CF
    from product import paper_learning_loop as PLL
    from product import _paper_learning_loop_core as core
    from tests.evidence_fixtures import rulebook_outcome

    (row,) = _rows()
    priced = CF.settle_rulebook(row, rulebook_outcome(1.0, "WIN", 10.0))
    assert priced["classification"] == "MISSED_WINNER"
    core.ingest_counterfactual(priced)                          # the unit is decided MISSED_WINNER
    assert PLL._queue_later_fills("TCS", FILL) == 1
    out = PLL.retry_pending_good_waits(today="2026-09-30")
    (row,) = _rows()
    assert out.get("superseded") == 1 and row.get("later_entry_superseded_by") == "MISSED_WINNER"
    assert row.get("classification") != "GOOD_WAIT" and "later_entry_pending" not in row
    assert _n("ENTRY_QUALITY::GOOD_WAIT") == 0
