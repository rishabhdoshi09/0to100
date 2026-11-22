"""Quarantine of learning observations proven defective, and the read-only correction plan.

A frozen DECISION vector is never rewritten. An observation whose PIT fills
were computed on a store that had lost official history is quarantined, and
the learning loaders skip it (docs/plans/2026-11-22-learning-observation-correction.md).
"""

from __future__ import annotations

import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
import pytest

from product import learning_observation_correction as LOC
from product import pit_features as PF
from research import feature_store as fs

DECIDED = "2026-09-04T10:00:00+05:30"          # cutoff session 2026-09-03


def _history(n=60, start="2026-06-15"):
    """A young listing whose IPO high (200) sits in its first 20 sessions."""
    idx = pd.bdate_range(start, periods=n)
    close = np.r_[np.linspace(190.0, 160.0, 20), np.linspace(158.0, 150.0, n - 20)]
    high = close + 1.0
    high[3] = 200.0
    return pd.DataFrame({"open": close, "high": high, "low": close - 1.0, "close": close,
                         "volume": 1_000_000.0, "deliv_per": 40.0}, index=idx)


def _flat(n=60, start="2026-06-15"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 500_000.0, "deliv_per": 50.0}, index=idx)


def _pickle(path, store):
    with open(path, "wb") as fh:
        pickle.dump({"store": store, "pending": {}}, fh)
    return path


def _freeze(oid, sym, frame, *, extra_filled=(), tamper=None):
    raw = PF.technical_features(sym, DECIDED, frame=frame)
    filled = sorted(raw)
    if tamper:
        raw.update(tamper)
    assert fs.snapshot(oid, sym, "DECISION", raw, ts=DECIDED,
                       meta={"pit_features": {"filled": filled + list(extra_filled)}})["status"] == "frozen"


@pytest.fixture
def world(tmp_path, monkeypatch):
    db = tmp_path / "feature_store.db"
    monkeypatch.setattr(fs, "_DB_PATH", db)
    full = _history()
    truncated = full.iloc[20:]                      # the store lost the first 20 sessions
    current = {"YNG": truncated, "FLAT": _flat()}
    repaired = {"YNG": full, "FLAT": _flat()}
    cur = _pickle(tmp_path / "current.pkl", current)
    rep = _pickle(tmp_path / "repaired.pkl", repaired)
    ca = tmp_path / "ca_events.json"                # absent: no corporate actions
    return {"db": db, "cur": cur, "rep": rep, "ca": ca, "tmp": tmp_path, "current": current}


# ── quarantine (research.feature_store) ───────────────────────────────────────

def test_quarantine_withholds_an_observation_from_both_learning_loaders(world):
    _freeze("decision::a", "YNG", world["current"]["YNG"])
    _freeze("decision::b", "FLAT", world["current"]["FLAT"])
    before = fs.get_observation("decision::a")
    assert fs.quarantine_observation("decision::a", "TRUNCATED_HISTORY_PIT_FILL", {"why": "test"}) == "quarantined"
    assert {r["observation_id"] for r in fs.load_observations(kind="DECISION")} == {"decision::b"}
    assert fs.load_matrix(kind="DECISION")["ids"] == ["decision::b"]
    assert {r["observation_id"] for r in fs.load_observations(kind="DECISION", include_quarantined=True)} \
        == {"decision::a", "decision::b"}
    assert len(fs.load_matrix(kind="DECISION", include_quarantined=True)["ids"]) == 2
    # the frozen record itself is untouched
    assert fs.get_observation("decision::a") == before
    assert fs.quarantined_ids() == {"decision::a"}


def test_quarantine_is_write_once_needs_a_reason_and_an_existing_row(world):
    _freeze("decision::a", "YNG", world["current"]["YNG"])
    with pytest.raises(ValueError):
        fs.quarantine_observation("decision::a", "  ")
    assert fs.quarantine_observation("decision::nope", "X") == "missing"
    assert fs.quarantine_observation("decision::a", "FIRST", {"n": 1}) == "quarantined"
    assert fs.quarantine_observation("decision::a", "SECOND", {"n": 2}) == "exists"
    con = sqlite3.connect(world["db"])
    assert con.execute("SELECT reason, evidence FROM observation_quarantine").fetchall() == [
        ("FIRST", json.dumps({"n": 1}, sort_keys=True))]


def test_a_store_without_quarantines_loads_everything(world):
    _freeze("decision::a", "YNG", world["current"]["YNG"])
    assert fs.quarantined_ids() == set()
    assert [r["observation_id"] for r in fs.load_observations(kind="DECISION")] == ["decision::a"]


# ── plan (read-only) ──────────────────────────────────────────────────────────

def _plan(world, name="m.json"):
    out = world["tmp"] / name
    return LOC.plan(world["db"], world["cur"], world["rep"], out, world["ca"]), out


def test_plan_finds_the_lost_ipo_high_and_nothing_else(world):
    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    _freeze("decision::flat", "FLAT", world["current"]["FLAT"])
    db_bytes = world["db"].read_bytes()
    m, out = _plan(world)
    assert world["db"].read_bytes() == db_bytes                    # read-only
    assert m["symbols_with_changed_history"] == 1 and m["observations_examined"] == 1
    assert m["complete"] is True and m["unexplained"] == [] and m["not_rederived"] == {}
    [row] = m["affected"]
    assert row["observation_id"] == "decision::yng" and row["settled"] is False
    stored = row["features"]["dist_from_high_pct"]["stored"]
    repaired = row["features"]["dist_from_high_pct"]["repaired"]
    assert stored < 10.0 < 20.0 < repaired                         # the IPO high was in the lost sessions
    assert set(row["features"]) <= set(PF.PRODUCED)
    assert json.loads(out.read_text())["state"] == "planned"


@pytest.mark.parametrize("stored,repaired,material", [
    (-1.0, -1.0035, False),     # the 20-EMA seed effect measured on the production clone
    (50.0, 50.3, False),        # above ABS_TOL but under REL_TOL of the value
    (0.01, 0.04, False),        # under ABS_TOL
    (8.683181, 29.79, True),    # VEDPOWER dist_from_high_pct
    (0.0, 0.2, True),
])
def test_materiality_needs_both_an_absolute_and_a_relative_difference(stored, repaired, material):
    assert LOC._material(stored, repaired) is material


def test_a_stored_value_the_current_store_does_not_reproduce_is_unexplained(world):
    _freeze("decision::odd", "YNG", world["current"]["YNG"], tamper={"atr_pct": 9.99})
    m, _ = _plan(world)
    assert [u["feature"] for u in m["unexplained"]] == ["atr_pct"]
    assert m["complete"] is False


def test_fills_it_cannot_re_derive_make_the_plan_incomplete(world):
    _freeze("decision::ctx", "YNG", world["current"]["YNG"], extra_filled=("rs_percentile",))
    m, _ = _plan(world)
    assert m["not_rederived"] == {"rs_percentile": 1} and m["complete"] is False


def test_the_recompute_reads_frames_as_get_ohlcv_serves_them(world):
    """Stored fills came from CA-adjusted frames: the plan must adjust the same way."""
    world["ca"].write_text(json.dumps([{"symbol": "YNG", "ex_date": "2026-08-03", "factor": 2.0,
                                         "type": "split"}]))
    from data.corporate_actions import adjust_frame, load_events

    served = adjust_frame(world["current"]["YNG"], load_events(world["ca"])["YNG"])
    _freeze("decision::ca", "YNG", served)
    m, _ = _plan(world)
    assert m["unexplained"] == [] and m["ca_events_sha256"] and m["complete"] is True
    # without the table the same rows are not reproduced: the adjustment is what matched
    raw = LOC.plan(world["db"], world["cur"], world["rep"], world["tmp"] / "raw.json",
                   world["tmp"] / "absent.json")
    assert raw["unexplained"] and raw["complete"] is False


def test_a_corrected_value_on_an_existing_date_counts_as_changed_history():
    a = _flat()
    b = a.copy()
    b.iloc[10, b.columns.get_loc("close")] = 101.5          # same dates, one corrected close
    assert LOC._changed_symbols({"X": a, "Y": a}, {"X": b, "Y": a.copy()}) == {"X"}
    # a full build stores volume as int64 where appends left float64: not a change
    as_int = a.copy()
    as_int["volume"] = as_int["volume"].astype("int64")
    as_int.index = as_int.index.as_unit("s")
    assert LOC._changed_symbols({"X": a}, {"X": as_int}) == set()
    # the desk's rolling windows read stored order: a re-sorted frame is examined too
    swapped = pd.concat([a.iloc[:30], a.iloc[[31, 30]], a.iloc[32:]])
    assert LOC._changed_symbols({"X": swapped}, {"X": a}) == {"X"}


def test_plan_refuses_a_missing_feature_store(world):
    with pytest.raises(SystemExit, match="no feature store"):
        LOC.plan(world["tmp"] / "absent.db", world["cur"], world["rep"], world["tmp"] / "m.json", world["ca"])


def test_plan_refuses_live_volumes_paths(world):
    with pytest.raises(SystemExit, match="/Volumes"):
        LOC.plan("/Volumes/QuantTermStorage/QuantTerm/runtime/logs/feature_store.db",
                 world["cur"], world["rep"], world["tmp"] / "m.json", world["ca"])


# ── apply (owner-gated) ───────────────────────────────────────────────────────

def test_apply_needs_words_a_complete_plan_and_unchanged_vectors(world):
    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    m, out = _plan(world)
    with pytest.raises(SystemExit, match="approval"):
        LOC.apply(out, world["db"], "  ", target_is_copy=True)

    incomplete = dict(m, complete=False)
    bad = world["tmp"] / "incomplete.json"
    bad.write_text(json.dumps(incomplete))
    with pytest.raises(SystemExit, match="incomplete"):
        LOC.apply(bad, world["db"], "owner words", target_is_copy=True)

    changed = json.loads(out.read_text())
    changed["affected"][0]["vector_sha256"] = "0" * 64
    moved = world["tmp"] / "moved.json"
    moved.write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="changed since the plan"):
        LOC.apply(moved, world["db"], "owner words", target_is_copy=True)
    assert fs.quarantined_ids() == set()                           # nothing partial


def test_apply_quarantines_exactly_the_plan_and_only_once(world):
    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    _freeze("decision::flat", "FLAT", world["current"]["FLAT"])
    before = {oid: fs.get_observation(oid) for oid in ("decision::yng", "decision::flat")}
    _, out = _plan(world)
    done = LOC.apply(out, world["db"], "approve quarantine", target_is_copy=True)
    assert done["results"] == {"decision::yng": "quarantined"} and done["state"] == "applied"
    assert fs.quarantined_ids() == {"decision::yng"}
    assert {oid: fs.get_observation(oid) for oid in before} == before   # vectors byte-identical
    with pytest.raises(SystemExit, match="state 'applied'"):
        LOC.apply(out, world["db"], "again", target_is_copy=True)


def _store_manifest(world, *, target=None, new=None, ready=True):
    path = world["tmp"] / "store_manifest.json"
    path.write_text(json.dumps({"target_sha256": target or LOC._file_sha(world["cur"]),
                                "new_sha256": new or LOC._file_sha(world["rep"]), "ready": ready}))
    return path


def test_apply_on_a_live_target_needs_a_verified_repair_binding_and_a_stopped_desk(world, monkeypatch):
    import product.bhav_store_rebuild as R

    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    _, unbound = _plan(world)
    with pytest.raises(SystemExit, match="not bound"):
        LOC.apply(unbound, world["db"], "owner words")              # not a copy
    wrong = LOC.plan(world["db"], world["cur"], world["rep"], world["tmp"] / "wrong.json", world["ca"],
                     store_manifest=_store_manifest(world, new="0" * 64))
    assert wrong["store_binding"]["verified"] is False
    bound = world["tmp"] / "bound.json"
    m = LOC.plan(world["db"], world["cur"], world["rep"], bound, world["ca"], store_manifest=_store_manifest(world))
    assert m["store_binding"]["verified"] is True
    monkeypatch.setattr(R, "_desk_running", lambda: ["launchd:com.quantterm.desk:123"])
    with pytest.raises(SystemExit, match="desk is running"):
        LOC.apply(bound, world["db"], "owner words")
    assert fs.quarantined_ids() == set()


# ── exposed: carried features cannot be recomputed, so input windows are compared ──

def test_rows_whose_input_window_differs_are_exposed_whatever_their_kind(world):
    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    assert fs.snapshot("rejection::yng", "YNG", "REJECTION", {"rsi": 40.0}, ts=DECIDED,
                       reason="EDGE_GATE")["status"] == "frozen"
    assert fs.snapshot("rejection::flat", "FLAT", "REJECTION", {"rsi": 40.0}, ts=DECIDED,
                       reason="EDGE_GATE")["status"] == "frozen"
    m, _ = _plan(world)
    assert [r["observation_id"] for r in m["affected"]] == ["decision::yng"]
    # the proven row is not listed twice; the other kinds are listed, unproven
    assert [(r["observation_id"], r["kind"]) for r in m["exposed"]] == [("rejection::yng", "REJECTION")]


def test_exposure_is_any_difference_in_the_input_prefix(tmp_path, monkeypatch):
    """An EMA is seeded at the first bar, so a loss 300 bars back still exposes a
    decision; a change AFTER its cutoff does not (GPT QT-09)."""
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "fs.db")
    idx = pd.bdate_range("2025-06-02", periods=400)
    close = np.linspace(100.0, 140.0, 400)
    full = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": 1e6, "deliv_per": 40.0}, index=idx)
    stale = full.copy()
    stale.iloc[300, stale.columns.get_loc("volume")] = 0.0             # a bad row at bar 300
    cur = _pickle(tmp_path / "c.pkl", {"OLD": full.iloc[20:], "ETF": stale})
    rep = _pickle(tmp_path / "r.pkl", {"OLD": full, "ETF": full})
    late = (idx[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%dT10:00:00+05:30")
    before = (idx[200] + pd.Timedelta(days=1)).strftime("%Y-%m-%dT10:00:00+05:30")
    fs.snapshot("decision::late", "OLD", "DECISION", {"rsi": 50.0}, ts=late)
    fs.snapshot("decision::before_bad_row", "ETF", "DECISION", {"rsi": 50.0}, ts=before)
    fs.snapshot("decision::after_bad_row", "ETF", "DECISION", {"rsi": 50.0}, ts=late)
    m = LOC.plan(tmp_path / "fs.db", cur, rep, tmp_path / "m.json", tmp_path / "absent.json")
    assert m["complete"] is True
    exposed = {r["observation_id"] for r in m["exposed"]}
    assert exposed == {"decision::late", "decision::after_bad_row"}


def test_apply_quarantines_exposed_rows_only_when_asked(world):
    _freeze("decision::yng", "YNG", world["current"]["YNG"])
    fs.snapshot("near::yng", "YNG", "NEAR_MISS", {"rsi": 40.0}, ts=DECIDED, subtype="ALMOST")
    _, out = _plan(world)
    clean = world["tmp"] / "copy.json"
    clean.write_text(out.read_text())
    done = LOC.apply(out, world["db"], "approve quarantine", target_is_copy=True)
    assert done["results"] == {"decision::yng": "quarantined"} and done["included_exposed"] is False
    done = LOC.apply(clean, world["db"], "approve exposed too", target_is_copy=True, include_exposed=True)
    assert done["results"] == {"decision::yng": "exists", "near::yng": "quarantined"}
    con = sqlite3.connect(world["db"])
    assert dict(con.execute("SELECT observation_id, reason FROM observation_quarantine")) == {
        "decision::yng": LOC.REASON, "near::yng": LOC.EXPOSED_REASON}


# ── the feature002 shadow ledger: re-derived and reported, never written ──────

def _shadow_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE observations (event_id TEXT PRIMARY KEY, candidate_set_id TEXT, symbol TEXT, "
                "session_date TEXT, feature_snapshot TEXT, eligible_primary INTEGER)")
    con.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def test_the_shadow_ledger_report_finds_the_lost_52_week_high_and_writes_nothing(world):
    from research.feature001.trend_features import compute_trend_features

    fs._conn().close()                                  # an empty feature store
    as_of = "2026-09-03"
    trunc = world["current"]["YNG"]
    snap = {"trend": compute_trend_features(trunc[trunc.index <= as_of]), "rs": {"available": False},
            "hist_as_of": as_of}
    flat = world["current"]["FLAT"]
    flat_snap = {"trend": compute_trend_features(flat[flat.index <= as_of]), "hist_as_of": as_of}
    shadow = _shadow_db(world["tmp"] / "shadow.db", [
        ("e1", "set1", "YNG", "2026-09-04", json.dumps(snap), 1),
        ("e2", "set1", "FLAT", "2026-09-04", json.dumps(flat_snap), 1),
    ])
    before = shadow.read_bytes()
    m = LOC.plan(world["db"], world["cur"], world["rep"], world["tmp"] / "m.json", world["ca"], shadow)
    sh = m["shadow_feature002"]
    assert shadow.read_bytes() == before and sh["written"] is False
    assert sh["examined"] == 1 and sh["unexplained"] == []
    [row] = sh["affected"]
    assert row["event_id"] == "e1" and "dist_from_52w_high_pct" in row["features"]
    assert row["features"]["high_52w"]["repaired"] == 200.0
    assert sh["candidate_sets_with_affected_rows"] == 1


# ── GPT cycle-6 review (QT-02..QT-05, QT-09): sketches, adapted only where noted ──

def test_stale_candidate_cannot_prove_a_defect(world):
    """QT-02: a repaired store that ends a session earlier is not proof."""
    fresh = _flat(59)
    stale = fresh.iloc[:-1].copy()
    fresh.loc[fresh.index[-1], "high"] = 160.0
    _pickle(world["cur"], {"AAA": fresh})
    _pickle(world["rep"], {"AAA": stale})
    _freeze("decision::fresh", "AAA", fresh)

    m, _ = _plan(world)
    assert not m["complete"] or not m["affected"]
    assert m["refusals"][0]["reason"] == "COVERAGE_MISMATCH"


def test_replay_detects_breadth_change_on_unchanged_peer(world):
    """QT-03: a cross-sectional fill on an UNCHANGED symbol is not re-derived, so the plan is incomplete."""
    peer = _flat(51)
    young = peer.copy()
    young.loc[young.index[-1], "close"] = 110.0
    cur = {f"P{i:03}": peer for i in range(300)}
    fixed = dict(cur, YNG=young)
    cur["YNG"] = young.iloc[1:]
    sessions = tuple(peer.index.date)
    day = sessions[-1]
    assert PF._market_pass(sessions, list(cur), cur.get)[day] == (0, 300)
    assert PF._market_pass(sessions, list(fixed), fixed.get)[day] == (1, 301)
    ts = f"{day}T19:00:00+05:30"
    vec, prov = PF.fill_missing(
        {}, "P000", ts, frame=peer, producer=PF.CONTEXT_PRODUCER,
        context={"_session": day.isoformat(), "breadth_pct_above_50dma": 0.0},
    )
    fs.snapshot("decision::peer", "P000", "DECISION", vec, ts=ts,
                meta={"pit_features": prov})
    _pickle(world["cur"], cur)
    _pickle(world["rep"], fixed)

    m, _ = _plan(world)
    assert not m["complete"] or m["affected"]


def test_replay_flags_a_fill_that_should_disappear(world):
    """QT-03: a stored value that cannot exist on the complete history is a defect."""
    fixed = _flat(22)
    cur = fixed.copy()
    cur.loc[cur.index[-2], "volume"] = 700_000.0
    _pickle(world["cur"], {"YNG": cur})
    _pickle(world["rep"], {"YNG": fixed})
    _freeze("decision::volume", "YNG", cur)
    assert "volume_z" not in PF.technical_features("YNG", DECIDED, frame=fixed)

    m, _ = _plan(world)
    assert not m["complete"] or m["affected"]


def test_a_vanished_value_is_proven_and_a_decision_past_the_store_is_unexplained(world):
    fixed = _flat(22)
    cur = fixed.copy()
    cur.loc[cur.index[-2], "volume"] = 700_000.0
    _pickle(world["cur"], {"YNG": cur})
    _pickle(world["rep"], {"YNG": fixed})
    inside = (fixed.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%dT10:00:00+05:30")
    raw = PF.technical_features("YNG", inside, frame=cur)
    fs.snapshot("decision::inside", "YNG", "DECISION", raw, ts=inside,
                meta={"pit_features": {"filled": sorted(raw)}})
    _freeze("decision::past_the_store", "YNG", cur)               # cutoff 2026-09-03, store ends in July
    m, _ = _plan(world)
    assert [r["observation_id"] for r in m["affected"]] == ["decision::inside"]
    assert m["affected"][0]["features"]["volume_z"]["repaired"] is None
    assert [u["reason"] for u in m["unexplained"]] == ["AFTER_STORE_COVERAGE"]
    assert m["complete"] is False


def test_quarantined_decision_does_not_calibrate(world, monkeypatch):
    """QT-04."""
    from product import decision_calibration as DC, evidence_intelligence as EI

    path = world["tmp"] / "calibration.json"
    monkeypatch.setenv("QT_CALIBRATION", str(path))
    fs.snapshot("decision::x", "YNG", "DECISION", {}, ts=DECIDED,
                meta={"predicted_p": 0.8})
    fs.quarantine_observation("decision::x", LOC.REASON)
    EI.record_resolved_prediction(
        "x", 1.0, evidence_class="PAPER_FORWARD",
        resolved_at="2026-09-21T18:00:00+05:30",
    )
    assert DC.load_store(path)["observations"] == []


def test_quarantined_rejection_does_not_reach_analysis(world):
    """QT-04."""
    from research import non_event

    fs.snapshot("rejection::x", "YNG", "REJECTION", {},
                ts=DECIDED, reason="OTHER", outcome=4.0)
    fs.quarantine_observation("rejection::x", LOC.EXPOSED_REASON)
    assert non_event.rejection_analysis() == []


def test_a_challenger_fitted_before_a_quarantine_is_not_served(world, monkeypatch, tmp_path):
    """QT-04: the saved coefficients are withheld until the next train() refits."""
    from product import challenger_learning as CL

    store = tmp_path / "challenger.json"
    CL.save({"current": {"status": CL.SHADOW_CANDIDATE, "quarantine_fingerprint": "",
                         "coef": [0.0] * len(CL.MODEL_FEATURES)}}, store)
    _freeze("decision::q", "YNG", world["current"]["YNG"])
    assert CL.score_decision({"symbol": "YNG"}, path=store).get("status") != "STALE_TRAINING_INPUTS"
    fs.quarantine_observation("decision::q", LOC.REASON)
    out = CL.score_decision({"symbol": "YNG"}, path=store)
    assert out["available"] is False and out["status"] == "STALE_TRAINING_INPUTS"


def test_quarantine_batch_is_atomic(world):
    """QT-05."""
    for suffix in ("a", "b"):
        _freeze(f"decision::{suffix}", "YNG", world["current"]["YNG"])
    _, path = _plan(world)
    with sqlite3.connect(world["db"]) as con:
        con.execute("""
            CREATE TRIGGER fail_second BEFORE INSERT ON observation_quarantine
            WHEN NEW.observation_id = 'decision::b'
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        LOC.apply(path, world["db"], "test", target_is_copy=True)
    assert fs.quarantined_ids() == set()


def test_quarantine_rollback_preserves_prior_entry(world):
    """QT-05, adapted: the documented rollback is now `rollback --manifest` (this
    batch's rows only), not a DELETE by reason, which is what the sketch ran."""
    _freeze("decision::prior", "YNG", world["current"]["YNG"])
    fs.quarantine_observation("decision::prior", LOC.REASON)
    _freeze("decision::new", "YNG", world["current"]["YNG"])
    _, path = _plan(world)
    done = LOC.apply(path, world["db"], "test", target_is_copy=True)
    assert done["results"]["decision::prior"] == "exists"
    assert done["results"]["decision::new"] == "quarantined"
    undone = LOC.rollback(path, world["db"], target_is_copy=True)
    assert undone["released"] == 1
    assert fs.quarantined_ids() == {"decision::prior"}


def test_an_interrupted_apply_completes_on_rerun_with_the_same_batch(world, monkeypatch):
    """QT-05: the transaction committed but the final manifest write was lost."""
    _freeze("decision::a", "YNG", world["current"]["YNG"])
    _, path = _plan(world)
    real = LOC._write_json
    calls = {"n": 0}

    def crash_on_second(p, obj):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("power cut after the commit")
        real(p, obj)

    monkeypatch.setattr(LOC, "_write_json", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        LOC.apply(path, world["db"], "test", target_is_copy=True)
    monkeypatch.setattr(LOC, "_write_json", real)
    assert json.loads(path.read_text())["state"] == "applying"
    done = LOC.apply(path, world["db"], "test", target_is_copy=True)
    assert done["results"] == {"decision::a": "exists"} and done["state"] == "applied"
    assert LOC.rollback(path, world["db"], target_is_copy=True)["released"] == 1
    assert fs.quarantined_ids() == set()


def test_exposure_preserves_scanner_input_order(world):
    """QT-09."""
    from scan.unified_scanner import _rsi

    fixed = _history().iloc[:59]
    cur = pd.concat([fixed.iloc[:-2], fixed.iloc[[-1, -2]]])
    old_rsi = _rsi(cur["close"].to_numpy())
    assert old_rsi != _rsi(fixed["close"].to_numpy())
    _pickle(world["cur"], {"YNG": cur})
    _pickle(world["rep"], {"YNG": fixed})
    fs.snapshot("decision::order", "YNG", "DECISION", {"rsi": old_rsi}, ts=DECIDED)

    m, _ = _plan(world)
    assert "decision::order" in {r["observation_id"] for r in m["exposed"]}


def test_a_weekend_decision_is_inside_the_store_but_a_missing_session_is_not():
    """A Sunday decision's calendar cutoff is Saturday: no session the store lacks."""
    fri = pd.Timestamp("2026-11-20")
    assert LOC._unseen_session(fri, pd.Timestamp("2026-11-21").date(), None) is False      # Saturday
    assert LOC._unseen_session(fri, pd.Timestamp("2026-11-23").date(), None) is True       # Monday
    archive = [pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-25")]                       # Mon/Tue holidays
    assert LOC._unseen_session(fri, pd.Timestamp("2026-11-24").date(), archive) is False
    assert LOC._unseen_session(fri, pd.Timestamp("2026-11-25").date(), archive) is True
