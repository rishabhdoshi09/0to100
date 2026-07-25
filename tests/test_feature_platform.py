"""
Feature Platform + Knowledge Base tests.

The platform's job is reproducibility and data integrity, so these tests are
adversarial about exactly that: a bad feed must be rejected by the PLATFORM (not
the scanner), a frozen vector must be immutable, the schema must be versioned so
old snapshots stay attributable, and the knowledge base must never silently
resurrect a rejected belief or keep asserting a decayed one.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from research import feature_schema as S
from research import feature_store as FS
from research import knowledge as K


class TestFeatureSchema:
    def test_schema_version_is_deterministic_and_tagged(self):
        assert S.SCHEMA_VERSION == S._schema_hash(S.FEATURE_REGISTRY)
        assert S.SCHEMA_TAG.startswith("fs_")

    def test_version_bump_changes_the_schema_hash(self):
        # improving a feature (version bump) must flip the schema fingerprint so
        # snapshots stay attributable to the exact definition that made them
        reg = dict(S.FEATURE_REGISTRY)
        old = reg["rel_strength"]
        reg["rel_strength"] = S.Feature(old.name, old.version + 1, old.dtype,
                                        old.description, old.lo, old.hi)
        assert S._schema_hash(reg) != S.SCHEMA_VERSION

    def test_impossible_value_is_rejected(self):
        # delivery 250% is physically impossible — the PLATFORM catches it
        v, _ = S.FEATURE_REGISTRY["delivery_pct"].validate(250.0)
        assert v == S.IMPOSSIBLE
        v2, _ = S.FEATURE_REGISTRY["rsi"].validate(50.0)
        assert v2 == S.OK

    def test_outlier_vs_impossible_bands(self):
        # RSI 3 is inside hard [0,100] but past the sane soft band → OUTLIER
        assert S.FEATURE_REGISTRY["rsi"].validate(3.0)[0] == S.OUTLIER
        assert S.FEATURE_REGISTRY["rsi"].validate(-1.0)[0] == S.IMPOSSIBLE

    def test_unknown_category_is_impossible(self):
        assert S.FEATURE_REGISTRY["regime"].validate("MOON")[0] == S.IMPOSSIBLE
        assert S.FEATURE_REGISTRY["regime"].validate("DISTRIBUTION")[0] == S.OK

    def test_staleness_flag(self):
        f = S.FEATURE_REGISTRY["vix"]                      # max_age_days=3
        assert f.validate(15.0, age_days=1.0)[0] == S.OK
        assert f.validate(15.0, age_days=9.0)[0] == S.STALE

    def test_canonicalize_drops_unknown_and_fills_missing(self):
        c = S.canonicalize({"rsi": "65", "regime": "MIXED", "junk": 1})
        assert c["rsi"] == 65.0 and c["regime"] == "MIXED"
        assert "junk" not in c and c["crude_usd"] is None   # missing → None
        assert set(c) == set(S.FEATURE_NAMES)

    def test_validate_vector_hard_fails_only_on_impossible_or_stale(self):
        c = S.canonicalize({"rsi": 60})                     # everything else missing
        rep = S.validate_vector(c)
        assert rep.ok is True                               # MISSING alone doesn't fail
        rep2 = S.validate_vector(S.canonicalize({"delivery_pct": 250}))
        assert rep2.ok is False                             # IMPOSSIBLE fails


class TestFeatureStore:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "fs.db")

    def test_snapshot_freezes_and_is_immutable(self):
        r = FS.snapshot("o1", "TCS", "SCAN",
                        {"rsi": 65, "clv": 0.8, "regime": "TRENDING_BULL"})
        assert r["status"] == "frozen" and r["schema_version"] == S.SCHEMA_VERSION
        # write-once: a second snapshot of the same id changes NOTHING
        r2 = FS.snapshot("o1", "TCS", "SCAN", {"rsi": 99})
        assert r2["status"] == "exists"
        assert FS.get_observation("o1")["features"]["rsi"] == 65.0

    def test_bad_feed_is_flagged_by_the_platform(self):
        r = FS.snapshot("o2", "INFY", "SCAN", {"rsi": 50, "delivery_pct": 250})
        assert any(p[0] == "delivery_pct" and p[1] == S.IMPOSSIBLE
                   for p in r["problems"])

    def test_outcome_is_settled_not_recomputed(self):
        FS.snapshot("o3", "WIPRO", "TRADE", {"rsi": 60})
        assert FS.set_outcome("o3", 1.4)["status"] == "settled"
        assert FS.get_observation("o3")["outcome"] == 1.4
        # settling the label must not touch the frozen features
        assert FS.get_observation("o3")["features"]["rsi"] == 60.0
        assert FS.set_outcome("missing", 1.0)["status"] == "not_found"

    def test_load_matrix_is_aligned_and_version_tagged(self):
        FS.snapshot("a", "A", "TRADE", {"rsi": 55, "atr_pct": 3}, outcome=0.5)
        FS.snapshot("b", "B", "TRADE", {"rsi": 70}, outcome=-1.0)  # atr missing → NaN
        m = FS.load_matrix(kind="TRADE", require_outcome=True)
        assert m["X"].shape == (2, len(S.FEATURE_NAMES))
        assert m["y"].tolist() == [0.5, -1.0]
        assert m["schema_versions"] == {S.SCHEMA_VERSION}
        atr_col = m["features"].index("atr_pct")
        assert np.isnan(m["X"][1, atr_col])                 # missing → NaN, aligned

    def test_unknown_kind_refused(self):
        assert FS.snapshot("x", "Z", "GOSSIP", {"rsi": 50})["status"] == "error"

    def test_coverage_surfaces_a_thin_feature(self):
        FS.snapshot("c1", "A", "SCAN", {"rsi": 55})
        FS.snapshot("c2", "B", "SCAN", {"rsi": 60})
        cov = {d["feature"]: d for d in FS.feature_coverage()}
        assert cov["rsi"]["fill_rate"] == 1.0
        assert cov["crude_usd"]["fill_rate"] == 0.0         # never provided


class TestKnowledgeBase:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(K, "_DB_PATH", tmp_path / "kb.db")

    def test_lifecycle_transitions_are_mechanical(self):
        assert K._next_status("ACTIVE", "DECAYING", 0.3, "HIGH") == K.WATCH
        assert K._next_status("ACTIVE", "DEAD", -0.1, "HIGH") == K.RETIRED
        assert K._next_status("WATCH", "STABLE", 0.4, "HIGH") == K.ACTIVE
        assert K._next_status("WATCH", "STABLE", -0.1, "HIGH") == K.WATCH   # no EV
        assert K._next_status("REJECTED", "STRENGTHENING", 0.9, "HIGH") == K.REJECTED

    def test_promoted_experiment_becomes_active_belief(self):
        bid = K.promote_from_experiment(
            "hyp1", "Breakouts work in healthy breadth", "breakout",
            evidence_n=184, confidence="HIGH", ev_r=0.35,
            schema_version=S.SCHEMA_VERSION)
        b = K.get_belief(bid)
        assert b["status"] == K.ACTIVE and b["evidence_n"] == 184
        assert b["schema_version"] == S.SCHEMA_VERSION
        assert b["hypothesis_id"] == "hyp1"

    def test_decaying_belief_demotes_and_surfaces_a_directive(self):
        bid = K.record_belief("X works", "x", status=K.ACTIVE, evidence_n=120,
                              confidence="HIGH", ev_r=0.3)
        K.revalidate(bid, drift_status="DECAYING", ev_r=0.02, evidence_n=140)
        assert K.get_belief(bid)["status"] == K.WATCH
        assert any("under review" in d["text"] for d in K.belief_directives())

    def test_negative_knowledge_is_permanent(self):
        K.record_negative("Buying knives works", "falling_knife")
        assert K.is_known_dead("Buying knives works", "falling_knife") is True
        # a plain re-record must NOT resurrect it to ACTIVE
        K.record_belief("Buying knives works", "falling_knife", status=K.ACTIVE,
                        evidence_n=5, confidence="HIGH", ev_r=0.9)
        assert K.get_belief(K.belief_id("Buying knives works", "falling_knife"))[
            "status"] == K.REJECTED

    def test_idempotent_identity(self):
        a = K.belief_id("same claim", "sig")
        b = K.belief_id("Same Claim", "SIG")               # case-insensitive
        assert a == b

    def test_fail_open_reads(self):
        assert K.list_beliefs() == [] or isinstance(K.list_beliefs(), list)
        assert K.get_belief("nope") is None
        assert isinstance(K.belief_directives(), list)


from research import non_event as NE


class _FakeSignal:
    """Minimal stand-in for a scanner StockSignal (only the fields the mapper
    reads)."""
    def __init__(self, symbol, verdict="WATCH", score=40.0, rsi=60.0,
                 chase_risk=False, pivot_distance_pct=0.0):
        self.symbol = symbol
        self.verdict = verdict
        self.score = score
        self.rsi = rsi
        self.chase_risk = chase_risk
        self.pivot_distance_pct = pivot_distance_pct


class TestNonEvent:
    """The 1,900 stocks you didn't trade are the control group — structured
    causes + two near-miss types + counterfactual verdicts + boundary replay."""

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "fs.db")

    def test_rejection_carries_structured_cause_and_dedupes(self):
        r = NE.capture_rejection("TCS", {"rsi": 85}, "BLOWOFF_RSI",
                                 ts="2026-06-01T09:00:00")
        assert r["status"] == "frozen"
        # same day + symbol + reason → stored once (write-once)
        r2 = NE.capture_rejection("TCS", {"rsi": 84}, "BLOWOFF_RSI",
                                  ts="2026-06-01T12:00:00")
        assert r2["status"] == "exists"
        obs = FS.get_observation("2026-06-01:TCS:REJ:BLOWOFF_RSI")
        assert obs["reason"] == "BLOWOFF_RSI" and obs["kind"] == "REJECTION"

    def test_unknown_reason_falls_back_to_other(self):
        NE.capture_rejection("X", {"rsi": 50}, "MERCURY_RETROGRADE",
                             ts="2026-06-01T09:00:00")
        assert FS.get_observation("2026-06-01:X:REJ:OTHER")["reason"] == "OTHER"

    def test_two_near_miss_types_stored_separately_with_gap(self):
        NE.capture_near_miss("A", {"rsi": 60}, NE.ALMOST,
                             gap={"feature": "rsi", "needed": 72, "observed": 71.6},
                             ts="2026-06-01T09:00:00")
        NE.capture_near_miss("A", {"rsi": 60}, NE.FADED, ts="2026-06-01T09:00:00")
        a = FS.get_observation("2026-06-01:A:NM:ALMOST")
        f = FS.get_observation("2026-06-01:A:NM:FADED")
        assert a["subtype"] == "ALMOST" and a["meta"]["gap"]["observed"] == 71.6
        assert f["subtype"] == "FADED"                       # distinct rows

    def test_reason_verdicts_earning_vs_too_conservative(self):
        # a reason that rejected names which then FELL → EARNING
        for i in range(34):
            NE.capture_rejection(f"D{i}", {"rsi": 85}, "BLOWOFF_RSI",
                                 ts="2026-06-01T09:00:00")
            FS.set_outcome(f"2026-06-01:D{i}:REJ:BLOWOFF_RSI", -3.0 + (i % 3))
        # a reason that rejected names which then ROSE → TOO_CONSERVATIVE
        for i in range(34):
            NE.capture_rejection(f"U{i}", {"rsi": 55}, "LAGGARD",
                                 ts="2026-06-01T09:00:00")
            FS.set_outcome(f"2026-06-01:U{i}:REJ:LAGGARD", 6.0 + (i % 4))
        verdicts = {a["reason"]: a["verdict"] for a in NE.rejection_analysis()}
        assert verdicts["BLOWOFF_RSI"] == "EARNING"
        assert verdicts["LAGGARD"] == "TOO_CONSERVATIVE"
        assert any("too strict" in d["text"] for d in NE.rejection_directives())

    def test_thin_evidence_makes_no_claim(self):
        for i in range(5):                                   # below the 30 floor
            NE.capture_rejection(f"T{i}", {"rsi": 80}, "MACRO",
                                 ts="2026-06-01T09:00:00")
            FS.set_outcome(f"2026-06-01:T{i}:REJ:MACRO", -5.0)
        assert NE.rejection_analysis()[0]["verdict"] == "INSUFFICIENT"

    def test_decision_boundary_replay_no_rescan(self):
        # RSI-cap near-misses just above 72; relaxing the cap 72→74 should flip
        # exactly those whose observed RSI is in (72, 74]
        NE.capture_near_miss("P", {"rsi": 73}, NE.ALMOST,
                             gap={"feature": "rsi", "needed": 72, "observed": 73.0},
                             ts="2026-06-01T09:00:00")
        NE.capture_near_miss("Q", {"rsi": 78}, NE.ALMOST,
                             gap={"feature": "rsi", "needed": 72, "observed": 78.0},
                             ts="2026-06-01T09:00:00")
        FS.set_outcome("2026-06-01:P:NM:ALMOST", 4.0)         # would've been a winner
        rp = NE.replay_threshold("rsi", 72, 74, "ceiling")
        assert rp["would_qualify"] == 1 and "P" in rp["symbols"]
        assert rp["winners"] == 1                            # and it rose

    def test_scan_batch_maps_causes(self):
        results = [
            _FakeSignal("BUYME", verdict="BUY"),             # skipped (executed)
            _FakeSignal("EXT", chase_risk=True),             # → EXTENSION
            _FakeSignal("NEAR", pivot_distance_pct=0.8),     # → ALMOST
            _FakeSignal("MEH", score=20.0),                  # → LOW_CONVICTION
        ]
        counts = NE.record_scan_batch(results, regime="MIXED")
        assert counts == {"extension": 1, "almost": 1, "low_conviction": 1}
        assert FS.get_observation(f"{NE._today()}:EXT:REJ:EXTENSION") is not None
        assert FS.get_observation(f"{NE._today()}:NEAR:NM:ALMOST") is not None

    def test_settle_and_directives_fail_open(self):
        assert NE.settle_outcomes() == 0                     # no bhavcopy in test env
        assert isinstance(NE.rejection_analysis(), list)
        assert isinstance(NE.rejection_directives(), list)
