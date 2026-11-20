"""Invariant 6 on the canonical scanner: a BUY needs a positive measured edge.

Owner's Mac 2026-11-15: the walk-forward backtest measured 13 of 17 signals
as net losers, yet the live scanner kept issuing BUYs. Calibration only
rescaled scores relative to each other; the combo-edge veto existed only in
the legacy auto_scan loop; and any 52-week breakout / high-tight flag became a
BUY regardless of score or evidence.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scan import unified_scanner as US


def _fresh_stamp(days_ago=1):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _provenance(generated_at="fresh", **over):
    import scan.signal_backtest as SB
    from core.evaluation_spec import HORIZON_SESSIONS
    prov = {"generated_at": _fresh_stamp() if generated_at == "fresh" else generated_at,
            "report_schema": SB.REPORT_SCHEMA, "eval_spec": SB._eval_spec_tag(HORIZON_SESSIONS),
            "horizon_sessions": HORIZON_SESSIONS, "family_method": SB.FAMILY_METHOD, "code_sha": "test-sha"}
    prov.update(over)
    return {k: v for k, v in prov.items() if v is not None}


def _snap(edges, generated_at="fresh", **prov):
    snap = {"snapshot_id": "test", "multipliers": {}, "signal_registry": {}, "backtest_edges": edges}
    if generated_at is not None:
        snap["backtest_provenance"] = _provenance(generated_at, **prov)
    return snap


def _ok(mean, lower=0.05, n=60, blocks=24, inference="OVERLAP_ROBUST", family=True):
    """Evidence that meets the full standard: dependence-aware bound over >= 20 blocks, Holm survivor."""
    return {"mean": mean, "lower": lower, "n": n, "blocks": blocks, "inference": inference, "family": family}


def _breakout_frame():
    close = np.linspace(100, 200, 260)
    close[-3] = close[-4] - 2.0
    close[-8] = close[-9] - 2.0
    close[-12] = close[-13] - 2.0
    close[-1] = 212
    vol = np.full(260, 2e6)
    vol[-1] = 5e6
    end = pd.offsets.BDay().rollback(pd.Timestamp(date.today()))
    idx = pd.date_range(end=end, periods=260, freq="B")
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99, "volume": vol}, index=idx)


# ── verdict rule ──────────────────────────────────────────────────────────────

def test_without_the_gate_research_scanners_keep_the_legacy_score_rule():
    for gate in (None, False):              # None = a direct caller that supplies no evidence
        assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], gate=gate) == ("BUY", "")
        assert US._evidence_verdict(80.0, ["MOMENTUM", "CUP_HANDLE"], gate=gate) == ("BUY", "")
        assert US._evidence_verdict(40.0, ["MOMENTUM", "CUP_HANDLE"], gate=gate) == ("WATCH", "")


def test_the_live_gate_with_no_evidence_loaded_fails_closed():
    verdict, why = US._evidence_verdict(80.0, ["MOMENTUM", "CUP_HANDLE"], gate=True)
    assert verdict == "WATCH" and "no walk-forward evidence is loaded" in why
    verdict, why = US._evidence_verdict(40.0, ["BREAKOUT_52W"], gate=True)   # no unproven 52W shortcut
    assert verdict == "WATCH"


def test_52_week_shortcut_needs_its_own_positive_evidence():
    unproven = {"MOMENTUM": _ok(0.05)}                  # 52W has < 30 trades: absent
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=unproven)[0] == "WATCH"
    proven = {"BREAKOUT_52W": _ok(0.20)}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=proven)[0] == "BUY"
    legacy = {"BREAKOUT_52W": 0.20}                      # a bare float proves nothing
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=legacy)[0] == "WATCH"
    losing = {"HIGH_TIGHT_FLAG": -0.30}
    assert US._evidence_verdict(40.0, ["HIGH_TIGHT_FLAG"], signal_edges=losing)[0] == "WATCH"


def test_buy_needs_a_positive_measured_combo_edge():
    edges = {"FLAT_BASE": -0.83, "NR7_COIL": -0.45, "VOL_SQUEEZE": 0.08, "MOMENTUM": -0.03}
    verdict, why = US._evidence_verdict(90.0, ["FLAT_BASE", "NR7_COIL"], signal_edges=edges)
    assert verdict == "WATCH" and "-0.64R" in why
    # Breakeven (NEUTRAL, within noise of zero) is not an edge either: the
    # two survivors of the 2026-11-12 scan measured +0.003R.
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"], signal_edges=edges)
    assert verdict == "WATCH" and "+0.03R" in why and "+0.05R bar" in why
    positive = {"VOL_SQUEEZE": _ok(0.08), "DELIVERY_SPIKE": _ok(0.04, lower=0.01)}
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=positive) == ("BUY", "")
    assert US._MIN_BUY_EDGE_R == 0.05


def test_a_setup_with_no_evidenced_signal_is_watch_not_a_score_only_buy():
    """<30 trades = no claim: before 2026-11-20 such a setup became BUY on score alone."""
    edges = {"FLAT_BASE": -0.83}
    verdict, why = US._evidence_verdict(90.0, ["MOMENTUM", "CUP_HANDLE"], signal_edges=edges)
    assert verdict == "WATCH" and "No measured edge for these signals" in why
    assert "no walk-forward evidence is loaded" not in why


def test_quality_veto_still_outranks_everything():
    edges = {"BREAKOUT_52W": 0.5}
    assert US._evidence_verdict(99.0, ["BREAKOUT_52W", "MOMENTUM"], breakout_watch_veto=True,
                                signal_edges=edges) == ("WATCH", "")


def test_floor_is_tunable(monkeypatch):
    monkeypatch.setattr(US, "_MIN_BUY_EDGE_R", 0.10)
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                signal_edges={"VOL_SQUEEZE": _ok(0.08), "MOMENTUM": _ok(0.06)})[0] == "WATCH"
    monkeypatch.setattr(US, "_MIN_BUY_EDGE_R", 0.0)
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                signal_edges={"VOL_SQUEEZE": _ok(0.08), "MOMENTUM": _ok(-0.03, lower=0.01)})[0] == "BUY"


def test_backtest_edges_keep_only_evidenced_signals():
    report = {"signals": {"A": {"trades": 30, "expectancy_r": -0.2},
                          "B": {"trades": 29, "expectancy_r": 0.9},
                          "C": {"trades": "x", "expectancy_r": 1}}}
    assert US.backtest_edges(report) == {"A": {"mean": -0.2, "lower": None, "n": 30, "blocks": 0, "inference": "", "family": None}}
    assert US.backtest_edges(None) == {}
    assert US.combo_edge_r(["A", "B"], {"A": -0.2}) == -0.2
    assert US.combo_edge_r(["B"], {"A": -0.2}) is None
    precise = {"signals": {"A": {"trades": 40, "expectancy_r": 0.1, "expectancy_r_precise": 0.0812,
                                 "lower_r": -0.01}}}
    assert US.backtest_edges(precise) == {"A": {"mean": 0.0812, "lower": -0.01, "n": 40, "blocks": 0, "inference": "", "family": None}}
    robust = {"signals": {"A": {"trades": 90, "expectancy_r_precise": 0.11, "lower_r": 0.02,
                                "independent_sessions": 23, "inference": "OVERLAP_ROBUST", "family_significant": True}}}
    assert US.backtest_edges(robust) == {"A": {"mean": 0.11, "lower": 0.02, "n": 90, "blocks": 23,
                                               "inference": "OVERLAP_ROBUST", "family": True}}
    assert US.backtest_edges({"signals": {"A": {"trades": 90, "expectancy_r": float("nan")}}}) == {}


# ── scanner integration ───────────────────────────────────────────────────────

def test_scanner_turns_a_losing_buy_into_an_explained_watch():
    losing = {"BREAKOUT_52W": 0.2, "VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}
    r = US.UnifiedScanner(calibration_snapshot=_snap(losing))._analyze("X", _breakout_frame())
    assert r.signals[:3] == ["BREAKOUT_52W", "VOL_SQUEEZE", "POCKET_PIVOT"]
    assert r.verdict == "WATCH" and r.edge_gated is True and r.edge_r == pytest.approx(-0.267, abs=1e-3)
    assert r.reasons[-1].startswith("⚖ Measured edge")
    assert not r.reasons[0].startswith("⚖")             # reasons[0] stays the safety slot


def test_scanner_keeps_an_evidenced_positive_buy():
    edges = {"VOL_SQUEEZE": _ok(0.12), "POCKET_PIVOT": _ok(0.04, lower=0.01)}
    r = US.UnifiedScanner(calibration_snapshot=_snap(edges))._analyze("X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_gated is False and r.edge_r == pytest.approx(0.08)


def test_scanner_with_legacy_float_evidence_shows_why_it_is_not_a_buy():
    r = US.UnifiedScanner(calibration_snapshot=_snap({"VOL_SQUEEZE": 0.12, "POCKET_PIVOT": 0.04}))._analyze(
        "X", _breakout_frame())
    assert r.verdict == "WATCH" and r.edge_gated is True and r.edge_r == pytest.approx(0.08)
    assert "legacy evidence" in r.reasons[-1] and r.edge_lower_r is None


def test_replay_and_backtest_scanners_measure_raw_setups():
    losing = {"BREAKOUT_52W": -0.5, "VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}
    r = US.UnifiedScanner(calibration_snapshot=_snap(losing), evidence_gate=False)._analyze(
        "X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_r is None and r.edge_gated is False


def test_backtest_and_historical_replay_construct_without_the_gate(monkeypatch):
    import product.historical_replay as HR
    import scan.signal_backtest as SB
    import data.bhavcopy_store as store

    seen = []

    class Spy:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("evidence_gate"))

        def _analyze(self, *_a, **_k):
            return None

    monkeypatch.setattr(US, "UnifiedScanner", Spy)
    monkeypatch.setattr(store, "store_symbols", lambda: [])
    SB.run_backtest()
    HR.scan_session("2026-11-13", [], ohlcv_fn=lambda s: None)
    assert seen == [False, False]


def test_snapshot_freezes_the_gate_evidence(tmp_path):
    from scan.calibration_snapshot import build_snapshot

    snap = build_snapshot(
        backtest_report={"signals": {"VOL_SQUEEZE": {"trades": 545, "expectancy_r": 0.08},
                                     "HIGH_TIGHT_FLAG": {"trades": 19, "expectancy_r": 0.33}}},
        live_profile={}, data_identity="t", thesis_hash="t",
    )
    assert snap["backtest_edges"] == {"VOL_SQUEEZE": {"mean": 0.08, "lower": None, "n": 545, "blocks": 0,
                                                      "inference": "", "family": None}}
    assert snap["backtest_generated_at"] == ""                       # undated report: the gate refuses it
    assert snap["backtest_combo_edges"] == {}


def test_scan_record_carries_the_edge_fields():
    from product.scan_store import _record

    r = US.UnifiedScanner(calibration_snapshot=_snap({"VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}))._analyze(
        "X", _breakout_frame())
    row = _record(r, {}, set())
    assert row["verdict"] == "WATCH" and row["status"] != "Ready to trade"
    assert row["edge_gated"] is True and row["edge_r"] == pytest.approx(-0.5)


# ── combinations and confidence (2026-11-15, second pass) ────────────────────

def test_a_positive_average_that_noise_could_explain_is_not_proven():
    edges = {"VOL_SQUEEZE": _ok(0.08, lower=-0.03, n=545),
             "DELIVERY_SPIKE": _ok(0.09, lower=0.01, n=239)}   # lower bounds average -0.01
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=edges)
    assert verdict == "WATCH" and "95% lower bound" in why and "could be noise" in why
    proven = {"VOL_SQUEEZE": _ok(0.12, lower=0.03, n=545),
              "DELIVERY_SPIKE": _ok(0.09, lower=0.02, n=239)}
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=proven) == ("BUY", "")


def test_the_exact_combination_outranks_the_average_of_its_signals():
    edges = {"VOL_SQUEEZE": _ok(0.12, lower=0.03, n=545),
             "POCKET_PIVOT": _ok(0.10, lower=0.02, n=1385)}
    combos = {"POCKET_PIVOT|VOL_SQUEEZE": _ok(-0.20, lower=-0.35, n=64)}
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "POCKET_PIVOT"],
                                        signal_edges=edges, combo_edges=combos)
    assert verdict == "WATCH" and "this exact combination" in why
    ev = US.setup_evidence(["POCKET_PIVOT", "VOL_SQUEEZE"], edges, combos)
    assert ev["source"] == "combination" and ev["mean"] == -0.20
    # Without the combination's own record the signals' average decides.
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "POCKET_PIVOT"], signal_edges=edges) == ("BUY", "")


def test_the_52_week_shortcut_needs_proof_not_just_a_positive_average():
    noisy = {"BREAKOUT_52W": _ok(0.20, lower=-0.05, n=40)}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=noisy)[0] == "WATCH"
    proven = {"BREAKOUT_52W": _ok(0.20, lower=0.05, n=40)}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=proven)[0] == "BUY"


def test_scanner_reports_the_evidence_it_judged_on():
    combos = {"BREAKOUT_52W|POCKET_PIVOT|VOL_SQUEEZE": _ok(0.30, lower=0.10, n=45)}
    edges = {"VOL_SQUEEZE": _ok(-0.5, lower=-0.6, n=545)}
    snap = {**_snap(edges), "backtest_combo_edges": combos}
    r = US.UnifiedScanner(calibration_snapshot=snap)._analyze("X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_source == "combination"
    assert r.edge_r == pytest.approx(0.30) and r.edge_lower_r == pytest.approx(0.10)


# ── fail-closed admission (2026-11-20 review) ────────────────────────────────

INF, NAN = float("inf"), float("nan")


@pytest.mark.parametrize("row,reason", [
    (0.40, "legacy evidence"),                                                     # float only
    ({"mean": 0.40, "lower": 0.10}, "legacy evidence"),                            # missing n
    ({"mean": 0.40, "lower": 0.10, "n": 12, "blocks": 24, "inference": "OVERLAP_ROBUST"}, "only 12 trades"),
    ({"mean": 0.40, "lower": None, "n": 45, "blocks": 1, "inference": "OVERLAP_ROBUST"},
     "uncertainty cannot be measured"),                                            # all on one day
    ({"mean": 0.40, "lower": NAN, "n": 45, "blocks": 24, "inference": "OVERLAP_ROBUST"}, "not a finite number"),
    ({"mean": 0.40, "lower": INF, "n": 45, "blocks": 24, "inference": "OVERLAP_ROBUST"}, "not a finite number"),
    ({"mean": 0.40, "lower": "x", "n": 45, "blocks": 24, "inference": "OVERLAP_ROBUST"}, "not a finite number"),
    ({"mean": 0.40, "lower": 0.10, "n": 45}, "ignores overlapping 10-session windows"),      # pre-rerun report
    ({"mean": 0.40, "lower": 0.10, "n": 45, "blocks": 24, "inference": "WORST_CASE_NO_SESSION_INDEX"},
     "ignores overlapping 10-session windows"),
    ({"mean": 0.40, "lower": 0.10, "n": 45, "blocks": 7, "inference": "OVERLAP_ROBUST"},
     "only 7 independent 10-session windows"),
])
def test_unverifiable_evidence_never_buys_and_says_why(row, reason):
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                        signal_edges={"VOL_SQUEEZE": row, "MOMENTUM": row})
    assert verdict == "WATCH" and reason in why, why
    assert not US._proven(US._evidence_row(row))


@pytest.mark.parametrize("mean", [INF, -INF, NAN, "x", None])
def test_a_non_finite_mean_is_no_evidence(mean):
    assert US._evidence_row(mean) is None
    assert US._evidence_row({"mean": mean, "lower": 0.1, "n": 90}) is None
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                        signal_edges={"VOL_SQUEEZE": {"mean": mean, "n": 90}, "OTHER": _ok(0.2)})
    assert verdict == "WATCH" and "No measured edge" in why


def test_mixed_legacy_and_verified_rows_fail_closed():
    edges = {"VOL_SQUEEZE": _ok(0.30, lower=0.20), "MOMENTUM": 0.30}
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"], signal_edges=edges)
    assert verdict == "WATCH" and "legacy evidence" in why
    ev = US.setup_evidence(["VOL_SQUEEZE", "MOMENTUM"], edges)
    assert ev["inference"] == "MIXED" and ev["n"] == 0 and ev["lower"] is None


def test_only_fully_verified_evidence_proves_a_buy():
    assert US._proven(US._evidence_row(_ok(0.30))) is True
    assert US.unproven_reason(US._evidence_row(_ok(0.30))) == ""


def test_the_research_path_is_unchanged_by_the_live_gate():
    """Backtest and replay (gate off) still measure raw setups, even with legacy evidence around."""
    losing = {"BREAKOUT_52W": 0.9, "VOL_SQUEEZE": 0.9}
    r = US.UnifiedScanner(calibration_snapshot=_snap(losing), evidence_gate=False)._analyze("X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_gated is False


@pytest.mark.parametrize("stamp,reason", [
    (None, "has no provenance record"),
    ("not a date", "unreadable"),
    ("2026-01-01 10:00", "days old"),
    ("2999-01-01T00:00:00+00:00", "dated in the future"),
])
def test_stale_undated_or_future_evidence_cannot_prove_a_buy(stamp, reason):
    edges = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}
    r = US.UnifiedScanner(calibration_snapshot=_snap(edges, generated_at=stamp))._analyze("X", _breakout_frame())
    assert r.verdict == "WATCH" and r.edge_gated is True and reason in r.reasons[-1]
    fresh = US.UnifiedScanner(calibration_snapshot=_snap(edges))._analyze("X", _breakout_frame())
    assert fresh.verdict == "BUY"


def test_the_report_timestamp_format_is_understood():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    assert US.evidence_age_note("2026-11-16 00:51", now=now) == ""            # report format (IST wall time)
    assert "days old" in US.evidence_age_note("2026-10-01 00:51", now=now)


def test_a_bound_that_does_not_survive_the_family_correction_is_not_proof():
    for fam, reason in ((False, "multiple-testing correction"), (None, "multiple-testing correction")):
        edges = {"VOL_SQUEEZE": _ok(0.30, family=fam), "MOMENTUM": _ok(0.30, family=fam)}
        verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"], signal_edges=edges)
        assert verdict == "WATCH" and reason in why
    mixed = {"VOL_SQUEEZE": _ok(0.30), "MOMENTUM": _ok(0.30, family=False)}
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"], signal_edges=mixed)[0] == "WATCH"


# ── decision-time provenance (2026-11-20 review, P0-2) ───────────────────────

@pytest.mark.parametrize("override,reason", [
    ({"report_schema": 3}, "schema 3 predates"),
    ({"report_schema": None}, "schema unknown predates"),
    ({"eval_spec": "sim:h15:be2:gap"}, "different evaluation rulebook"),
    ({"horizon_sessions": 15}, "15-session horizon"),
    ({"family_method": "Benjamini-Hochberg, one-sided edge > 0"}, "predeclared multiple-testing procedure"),
    ({"code_sha": ""}, "does not name the code"),
])
def test_a_fresh_timestamp_on_a_mismatched_report_never_certifies_a_buy(override, reason):
    edges = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}
    r = US.UnifiedScanner(calibration_snapshot=_snap(edges, **override))._analyze("X", _breakout_frame())
    assert r.verdict == "WATCH" and reason in r.reasons[-1], r.reasons[-1]


def test_the_scanner_binds_one_immutable_snapshot_for_the_whole_scan():
    edges = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}
    snap = _snap(edges)
    sc = US.UnifiedScanner(calibration_snapshot=snap)
    first = sc._analyze("X", _breakout_frame())
    snap["backtest_edges"]["VOL_SQUEEZE"]["mean"] = -9.0            # a refresh lands mid-scan
    snap["backtest_provenance"]["report_schema"] = 1
    again = sc._analyze("X", _breakout_frame())
    assert first.verdict == again.verdict == "BUY" and first.edge_r == again.edge_r
    assert first.edge_evidence_id == "test" == again.edge_evidence_id
    fresh = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}
    restarted = US.UnifiedScanner(calibration_snapshot=_snap(fresh))._analyze("X", _breakout_frame())
    assert (restarted.verdict, restarted.edge_r) == (first.verdict, first.edge_r)           # restart: same answer


def test_a_duplicated_signal_cannot_double_its_weight():
    edges = {"VOL_SQUEEZE": _ok(0.30), "MOMENTUM": _ok(-0.50, lower=-0.6)}
    once = US.setup_evidence(["VOL_SQUEEZE", "MOMENTUM"], edges)
    twice = US.setup_evidence(["VOL_SQUEEZE", "VOL_SQUEEZE", "VOL_SQUEEZE", "MOMENTUM"], edges)
    assert once["mean"] == twice["mean"] == pytest.approx(-0.10)


def test_a_partial_or_corrupt_report_file_is_no_evidence(tmp_path, monkeypatch):
    import scan.signal_backtest as SB
    bad = tmp_path / "signal_backtest.json"
    bad.write_text('{"signals": {"VOL_SQUEEZE": {"trades": 90, "expectancy_r_pre')    # truncated write
    monkeypatch.setattr(SB, "_OUT", bad)
    assert SB.load_report() is None
    r = US.UnifiedScanner(calibration_snapshot={"snapshot_id": "none", "multipliers": {}, "signal_registry": {}}
                          )._analyze("X", _breakout_frame())
    assert r.verdict == "WATCH"


def _payload(rows, ident="test", pit=False):
    return {"schema_version": 2, "records": rows, "evidence_identity": {"id": ident} if ident is not None else {},
            **({"pit": True} if pit else {})}


def _row(sym, verdict="BUY", evidence="test"):
    return {"symbol": sym, "verdict": verdict, "status": "Ready to trade" if verdict == "BUY" else "Watch",
            "edge_evidence_id": evidence, "reasons": ["setup"]}


def test_a_legacy_rolled_back_or_mixed_artifact_cannot_certify_a_buy():
    from product.scan_store import bind_evidence, EVIDENCE_WITHHELD
    legacy = bind_evidence(_payload([_row("A")], ident=None))
    assert legacy["records"][0]["verdict"] == "WATCH" and EVIDENCE_WITHHELD in legacy["records"][0]["reasons"]
    mixed = bind_evidence(_payload([_row("A"), _row("B", evidence="older-snapshot"), _row("C", evidence="")]))
    assert [r["verdict"] for r in mixed["records"]] == ["BUY", "WATCH", "WATCH"]
    assert mixed["records"][1]["status"] == "Watch" and mixed["evidence_binding"]["buy_rows_withheld"] == 2
    forged = bind_evidence(_payload([_row("A", evidence="")], ident=None, pit=True))
    assert forged["records"][0]["verdict"] == "WATCH"                         # a pit flag exempts nothing


def _authority(edges, snapshot_id="test", **prov):
    return {"snapshot_id": snapshot_id, "backtest_edges": edges, "backtest_combo_edges": {},
            "backtest_provenance": _provenance(**prov)}


def _srow(sym, signals, verdict="BUY", evidence="test"):
    return {**_row(sym, verdict, evidence), "signals": list(signals)}


PROVEN = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}


def test_load_scan_re_verifies_every_buy_against_the_authoritative_snapshot(tmp_path, monkeypatch):
    import product.scan_store as SS
    from product.recommendations_workspace import build_recommendations_workspace
    payload = _payload([_srow("AAA", ["VOL_SQUEEZE", "POCKET_PIVOT"]), _srow("BBB", ["MOMENTUM", "NR7_COIL"])])
    path = SS.save_scan(payload, tmp_path / "scan.json")
    monkeypatch.setattr(SS, "_evidence_authority", lambda: _authority(PROVEN))
    loaded = SS.load_scan(path)
    assert [r["verdict"] for r in loaded["records"]] == ["BUY", "WATCH"]       # BBB: no proof for its signals
    assert loaded["evidence_binding"]["mode"] == "authoritative"
    original = [dict(r) for r in payload["records"]]
    build_recommendations_workspace(scan_payload=payload)
    assert payload["records"] == original                                     # caller's payload not mutated


@pytest.mark.parametrize("case", ["no_authority", "superseded_evidence", "stale_provenance",
                                  "promoted_verdict", "self_consistent_stale_artifact"])
def test_no_self_consistent_artifact_can_certify_a_buy_without_the_authority(tmp_path, monkeypatch, case):
    import product.scan_store as SS
    rows = [_srow("AAA", ["VOL_SQUEEZE", "POCKET_PIVOT"])]
    authority = _authority(PROVEN)
    if case == "no_authority":
        authority = None
    elif case == "superseded_evidence":                       # the report was replaced after the scan
        authority = _authority({"VOL_SQUEEZE": _ok(0.30, lower=-0.2), "POCKET_PIVOT": _ok(0.30, lower=-0.2)},
                               snapshot_id="newer")
    elif case == "stale_provenance":
        authority = _authority(PROVEN, generated_at="2026-01-02 10:00")
    elif case == "promoted_verdict":                          # a WATCH row edited to BUY after publication
        rows = [_srow("AAA", ["MOMENTUM", "NR7_COIL"])]
    elif case == "self_consistent_stale_artifact":            # old scan, its own id everywhere, old evidence
        rows = [_srow("AAA", ["VOL_SQUEEZE", "POCKET_PIVOT"], evidence="old")]
        authority = _authority({"VOL_SQUEEZE": _ok(0.30, family=False), "POCKET_PIVOT": _ok(0.30, family=False)},
                               snapshot_id="current")
    path = SS.save_scan(_payload(rows, ident="old" if case == "self_consistent_stale_artifact" else "test"),
                        tmp_path / "scan.json")
    monkeypatch.setattr(SS, "_evidence_authority", lambda: authority)
    loaded = SS.load_scan(path)
    assert loaded["records"][0]["verdict"] == "WATCH", case
    assert loaded["records"][0]["reasons"][-1].startswith("⚖ BUY withheld at read time")


def test_the_snapshot_pointer_follows_a_rollback(tmp_path):
    from scan.calibration_snapshot import get_or_create_snapshot, load_current
    old = {"signals": {"A": {"trades": 40, "expectancy_r": 0.1}}, "generated_at": "2026-11-01 10:00"}
    new = {"signals": {"A": {"trades": 45, "expectancy_r": 0.2}}, "generated_at": "2026-11-08 10:00"}
    kw = dict(live_profile={}, data_identity="d", thesis_hash="t", directory=tmp_path)
    first = get_or_create_snapshot(backtest_report=old, **kw)
    second = get_or_create_snapshot(backtest_report=new, **kw)
    assert load_current(directory=tmp_path)["snapshot_id"] == second["snapshot_id"]
    back = get_or_create_snapshot(backtest_report=old, **kw)                  # rollback: a cache hit
    assert back["cache_hit"] and back["snapshot_id"] == first["snapshot_id"]
    assert load_current(directory=tmp_path)["snapshot_id"] == first["snapshot_id"]


def test_the_scan_payload_names_its_evidence_identity():
    from product.scan_store import build_scan_payload
    edges = {"VOL_SQUEEZE": _ok(0.30), "POCKET_PIVOT": _ok(0.30)}
    r = US.UnifiedScanner(calibration_snapshot=_snap(edges))._analyze("X", _breakout_frame())
    payload = build_scan_payload({"X": "X"}, [r], freshness={})
    assert payload["evidence_identity"]["id"] == "test" and payload["records"][0]["edge_evidence_id"] == "test"

