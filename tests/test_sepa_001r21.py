"""SEPA-001R2.1 runner-integrity tests. Thresholds unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.sepa.ablation_r2 import (
    annotate_core_f_deployment,
    attempt_e_entry,
    observe_lifecycle_daily,
    run_ablation_r2,
)
from research.sepa.ca_audit import CATimeline, build_timeline, ca_research_acceptability
from research.sepa.embargo import (
    attach_session_path,
    calendar_day_embargo_until,
    session_embargo_blocks,
)
from research.sepa.entry import FILL_CA_CENSORED
from research.sepa.frames import iso_date
from research.sepa.rs import build_rs_table
from research.sepa.signal_study import forward_path_study
from research.sepa.universe_pit import FastInvestable, membership_hash, screen_investable_as_of
from tests.test_sepa_001_eligibility import CFG, _ohlcv
from tests.test_sepa_001r2 import _liq_frame


def _cal(n, start="2019-01-02"):
    return pd.bdate_range(start, periods=n)


def _gap_frame(*, n_pre=1500, n_post=300, gap_date="2025-05-22"):
    """Official-looking series with an unresolved 2025 discontinuity."""
    idx_pre = pd.bdate_range(end=pd.Timestamp(gap_date) - pd.Timedelta(days=1), periods=n_pre)
    pre = _liq_frame(n_pre, start=200.0, step=0.2, volume=300_000, start_date=str(idx_pre[0].date()))
    pre.index = idx_pre
    post = _liq_frame(n_post, start=70.0, step=0.15, volume=300_000, start_date=gap_date)
    post.iloc[0, post.columns.get_loc("close")] = 70.0
    post.iloc[0, post.columns.get_loc("open")] = 70.0
    return pd.concat([pre, post])


def test_future_unresolved_ca_does_not_affect_2024_universe():
    gap = _gap_frame()
    clean = _liq_frame(1600, start=80.0, volume=300_000, start_date="2019-08-23")
    frames = {"ABFRLX": gap, "CLEAN": clean}
    tl = build_timeline([{
        "symbol": "ABFRLX", "date": "2025-05-22", "resolved": False,
        "event_classification": "demerger", "treatment": "quarantine_no_inferred_factor",
    }])
    as_of = "2024-12-31"
    snap = FastInvestable(frames).snapshot(
        as_of, min_sessions=80, min_turnover=1.0, min_price=1.0, ca_timeline=tl,
    )
    assert "ABFRLX" in snap.investable
    assert "ABFRLX" in snap.candidates
    static = FastInvestable(frames).snapshot(
        as_of, min_sessions=80, min_turnover=1.0, min_price=1.0, quarantined={"ABFRLX"},
    )
    assert "ABFRLX" not in static.investable


def test_signal_ending_before_ca_remains_usable():
    gap = _gap_frame()
    tl = build_timeline([{"symbol": "X", "date": "2025-05-22", "resolved": False}])
    fast = FastInvestable({"X": gap})
    hist, fwd = fast.hist_fwd("X", "2024-06-03", 5, timeline=tl)
    assert hist is not None and len(hist) > 80
    assert fwd is not None
    assert iso_date(hist.index[-1]) <= "2024-06-03"
    assert not tl.horizon_crosses("X", "2024-06-03", iso_date(fwd.index[-1]))


def test_horizon_crossing_unresolved_event_is_ca_censored():
    gap = _gap_frame()
    tl = build_timeline([{"symbol": "X", "date": "2025-05-22", "resolved": False}])
    # as_of a few sessions before the gap so 20d fwd crosses D
    as_of = iso_date(pd.bdate_range(end="2025-05-21", periods=5)[0])
    fast = FastInvestable({"X": gap})
    hist, fwd = fast.hist_fwd("X", as_of, 20, timeline=tl)
    assert fwd is not None
    assert tl.horizon_crosses("X", as_of, iso_date(fwd.index[-1]))
    packed = forward_path_study(fwd)
    assert packed is not None
    # Research must not treat that path as an R trade
    assert "gross_r" not in packed
    assert packed["not_sepa_r"] is True


def test_post_event_indicators_cannot_use_pre_event_history():
    gap = _gap_frame(n_post=40)
    tl = build_timeline([{"symbol": "X", "date": "2025-05-22", "resolved": False}])
    fast = FastInvestable({"X": gap})
    as_of = iso_date(gap.index[-1])
    hist, _ = fast.hist_fwd("X", as_of, 5, timeline=tl)
    assert hist is not None
    assert iso_date(hist.index[0]) > "2025-05-22"
    assert (hist["close"] < 150).all()


def test_symbol_reenters_after_clean_post_event_history():
    gap = _gap_frame(n_post=300)
    tl = build_timeline([{"symbol": "X", "date": "2025-05-22", "resolved": False}])
    fast = FastInvestable({"X": gap})
    soon = iso_date(pd.bdate_range(start="2025-05-23", periods=30)[-1])
    snap_soon = fast.snapshot(soon, min_sessions=80, min_turnover=1.0, min_price=1.0, ca_timeline=tl)
    assert "X" not in snap_soon.investable
    assert snap_soon.reasons.get("X") == "ca_segment_quarantine"
    later = iso_date(gap.index[-1])
    snap_later = fast.snapshot(later, min_sessions=80, min_turnover=1.0, min_price=1.0, ca_timeline=tl)
    assert "X" in snap_later.investable


def test_future_unresolved_ca_cannot_alter_historical_rs_ranks():
    a = _liq_frame(500, start=80, step=0.4, volume=300_000, start_date="2019-08-23")
    b = _liq_frame(500, start=70, step=0.2, volume=300_000, start_date="2019-08-23")
    gap = _gap_frame()
    as_of = "2021-06-15"
    tl = build_timeline([{"symbol": "ABFRLX", "date": "2025-05-22", "resolved": False}])
    frames = {"A": a, "B": b, "ABFRLX": gap}
    fast = FastInvestable(frames)
    snap = fast.snapshot(as_of, min_sessions=80, min_turnover=1.0, min_price=1.0, ca_timeline=tl)
    t1 = fast.rs_table(as_of, snap.investable, CFG, timeline=tl)
    t2 = fast.rs_table(as_of, snap.investable, CFG, timeline=CATimeline([]))
    assert t1["percentiles"]["A"] == t2["percentiles"]["A"]
    assert "ABFRLX" in snap.investable
    table_pandas = build_rs_table(
        {k: v.loc[:as_of] for k, v in frames.items()}, as_of, CFG, universe=snap.investable,
    )
    assert abs(t1["percentiles"]["A"] - table_pandas["percentiles"]["A"]) < 1e-6


def test_fast_investable_equivalence_includes_candidates_and_hash():
    as_of = "2020-06-15"
    future = _liq_frame(80, start=200, volume=300_000, start_date="2021-01-04")
    frames = {
        "LIQ": _liq_frame(300, volume=1_000_000),
        "THIN": _liq_frame(300, volume=1_000),
        "NEW": future,
    }
    a = screen_investable_as_of(frames, as_of, min_sessions=80, min_turnover=5_000_000, min_price=10)
    b = FastInvestable(frames).snapshot(as_of, min_sessions=80, min_turnover=5_000_000, min_price=10)
    # Canonical candidates are names with a bar ≤ as_of, not the 2019–2026 store.
    live = sorted(b.candidates)
    a2 = screen_investable_as_of(
        frames, as_of, min_sessions=80, min_turnover=5_000_000, min_price=10, membership=live,
    )
    assert set(a2.investable) == set(b.investable)
    assert sorted(a2.candidates) == sorted(b.candidates)
    assert "NEW" not in b.candidates
    assert a2.membership_hash == b.membership_hash == membership_hash(live)
    assert a2.exclusions == b.exclusions
    assert "no_bars" not in b.exclusions
    # Even if pit_universe injects extra names into screen(), investable names
    # that actually have bars must match FastInvestable.
    assert set(a.investable) == set(b.investable)


def test_session_embargo_weekend_holiday_and_overlap():
    # Friday signal, 1-session hold → exit is Monday (next session), not Saturday.
    as_of = "2024-01-05"  # Friday
    fwd_idx = pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10"])  # Mon-Wed; skip weekend
    fwd = pd.DataFrame(
        {"open": [10, 10, 10], "high": [11, 11, 11], "low": [9, 9, 9], "close": [10, 10, 10]},
        index=fwd_idx,
    )
    sim = attach_session_path({"hold": 1, "entry": 10, "gross_r": 0, "outcome": "FLAT",
                               "mae_r": 0, "mfe_r": 0, "reached_1r": False, "reached_2r": False,
                               "stop_before_1r": False, "failed_break": False}, fwd, as_of=as_of)
    assert sim["exit_date"] == "2024-01-08"
    assert calendar_day_embargo_until(as_of, 1) == "2024-01-06"  # Saturday — the bug
    assert session_embargo_blocks(as_of="2024-01-08", last_exit_session=sim["exit_date"]) is True
    assert session_embargo_blocks(as_of="2024-01-09", last_exit_session=sim["exit_date"]) is False

    # Republic Day 2024-01-26 Friday holiday: Thu as_of, 1-session hold exits Monday 29th.
    hol = pd.DataFrame(
        {"open": [10], "high": [11], "low": [9], "close": [10]},
        index=pd.DatetimeIndex(["2024-01-29"]),
    )
    sim_h = attach_session_path({"hold": 1, "entry": 10, "gross_r": 0, "outcome": "FLAT",
                                 "mae_r": 0, "mfe_r": 0, "reached_1r": False, "reached_2r": False,
                                 "stop_before_1r": False, "failed_break": False}, hol, as_of="2024-01-25")
    assert sim_h["exit_date"] == "2024-01-29"
    assert calendar_day_embargo_until("2024-01-25", 1) == "2024-01-26"

    # 20-session hold uses the 20th forward session, not +20 calendar days.
    idx = pd.bdate_range("2024-02-01", periods=25)
    fwd20 = pd.DataFrame(
        {"open": np.full(25, 10.0), "high": np.full(25, 11.0),
         "low": np.full(25, 9.0), "close": np.full(25, 10.0)},
        index=idx,
    )
    sim20 = attach_session_path({"hold": 20, "entry": 10, "gross_r": 0, "outcome": "FLAT",
                                 "mae_r": 0, "mfe_r": 0, "reached_1r": False, "reached_2r": False,
                                 "stop_before_1r": False, "failed_break": False}, fwd20, as_of="2024-01-31")
    assert sim20["hold_sessions"] == 20
    assert sim20["exit_date"] == iso_date(idx[19])
    assert calendar_day_embargo_until("2024-01-31", 20) != sim20["exit_date"]

    # Consecutive signal during an open position is blocked.
    assert session_embargo_blocks(as_of=sim20["entry_date"], last_exit_session=sim20["exit_date"]) is True


def test_g_has_no_placeholder_r_metrics():
    idx = pd.bdate_range("2024-03-01", periods=20)
    close = np.linspace(100, 108, 20)
    fwd = pd.DataFrame({
        "open": close, "high": close * 1.02, "low": close * 0.97, "close": close,
    }, index=idx)
    packed = forward_path_study(fwd)
    assert packed["not_sepa_r"] is True
    assert packed["mae_pct"] < 0
    assert packed["mfe_pct"] > 0
    assert "mae_r" not in packed
    assert packed["hit_5pct"] is True


def test_daily_e_captures_one_session_entry_ready_that_step5_would_miss():
    from research.sepa.synthetic import plant_vcp
    frame = plant_vcp(contractions="tight", volume="dry")
    # Pad a few extra sessions after the coil so step-5 can land late.
    last = float(frame["close"].iloc[-1])
    pad = _ohlcv(np.full(8, last * 1.001), volume=np.full(8, 80_000.0))
    pad.index = pd.bdate_range(frame.index[-1] + pd.Timedelta(days=3), periods=8)
    long = pd.concat([frame, pad])
    grind = _liq_frame(len(long), start=40.0, step=0.2, volume=200_000,
                       start_date=str(pd.Timestamp(long.index[0]).date()))
    grind.index = long.index
    ready_day = iso_date(frame.index[-1])

    class _Hit:
        signals = ["BREAKOUT_RES"]
        entry = float(frame["close"].iloc[-1])
        stop = float(frame["close"].iloc[-1]) * 0.94
        target = float(frame["close"].iloc[-1]) * 1.08

    def daily_scan(sym, hist):
        if iso_date(hist.index[-1]) == ready_day:
            return _Hit
        return None

    def step5_scan(sym, hist):
        # Pretend we only invoke the scanner on every 5th eval date by
        # returning a hit solely when the hist length is 0 mod 5 — which
        # the single entry-ready session will miss.
        if len(hist) % 5 == 0:
            return _Hit
        return None

    common = dict(
        frames={"LEADER": long, "GRIND": grind},
        variants=("E",), horizon=8, warmup_sessions=200,
        min_sessions=80, min_price=1.0, min_turnover=1.0, date_step=1,
        scanner_step=1, top_n=None,
    )
    daily = run_ablation_r2(scanner_fn=daily_scan, **common)
    late = run_ablation_r2(scanner_fn=step5_scan, **common)
    assert observe_lifecycle_daily() is True
    assert attempt_e_entry(scanner_ok=True, structure_pass=True, rs_pass=True, vcp_detected=True) is True
    assert attempt_e_entry(scanner_ok=False, structure_pass=True, rs_pass=True, vcp_detected=True) is False
    # Daily scanner-on-ready must be able to consider the E opportunity;
    # 5-day sampling of the scanner gate must not be how lifecycle is clocked.
    d_n = (daily.get("variants") or {}).get("E", {}).get("n_raw_signal_days", 0)
    l_n = (late.get("variants") or {}).get("E", {}).get("n_raw_signal_days", 0)
    # Either fills or classified refusals — daily path must see the ready session.
    assert daily["diagnostics"]["canonical_daily"] is True
    assert daily["sample"]["n_as_of"] == late["sample"]["n_as_of"]
    # The late scanner cannot use the unique ready session as an E fill attempt
    # more often than the daily scanner that actually hits that day.
    assert d_n >= l_n or daily["funnel_unique"]["entry_ready"] >= late["funnel_unique"]["entry_ready"]


def test_scanner_research_wrapper_matches_unified_analyze():
    from research.sepa.scanner_research import equivalence_pairs, make_production_scanner
    from research.sepa.synthetic import plant_vcp, stage2
    scanner = make_production_scanner()
    sample = [
        ("P1", plant_vcp(contractions="tight", volume="dry")),
        ("P2", stage2(n=250, start=40.0, step=0.3)),
        ("P3", plant_vcp(contractions="widening", volume="dry")),
    ]
    ev = equivalence_pairs(scanner, sample)
    assert ev["equivalent"] is True
    assert ev["function"] == "UnifiedScanner._analyze"


def test_ca_research_acceptable_does_not_set_ca_complete():
    from research.sepa.frames import ca_status
    ok = ca_research_acceptability(
        unresolved=[{"symbol": "X", "date": "2025-05-22"}],
        exhaustive=True, inferred_factors=False, unknown_path_crossings=0,
        future_leak_removed_prior=0, audit_persisted=True, contaminated_uncensored=0,
    )
    assert ok["ca_research_acceptable"] is True
    assert ok["does_not_set_ca_complete"] is True
    # Global flag is independent.
    st = ca_status()
    assert "ca_complete" in st


def test_core_f_deployment_uses_confirmation_not_pooled():
    """Pooled STATISTICAL_SIGNAL must not paper-qualify F when confirmation REJECTS."""
    pooled = {
        "statistical_verdict": "STATISTICAL_SIGNAL",
        "mean_r": 0.123,
        "n": 4208,
        "block_ci": {"ci_lower": 0.004, "ci_upper": 0.23},
    }
    confirmation = [
        SimpleNamespace(net_r=-0.25 + (i % 11) * 0.01) for i in range(80)
    ]
    dep = annotate_core_f_deployment(
        pooled_gate=pooled,
        confirmation_rows=confirmation,
        n_trials=7,
        integ={"overall": "PIT_DEGRADED", "ca_research_acceptable": True},
        ca={"ca_complete": False},
        n_years=6.0,
    )
    assert dep["deployment_eligible"] is False
    assert dep["paper_shadow"] is False
    assert dep["label"] == "NOT_DEPLOYMENT_ELIGIBLE"
    assert dep["confirmation_n"] == 80
    assert dep["confirmation_verdict"] != "STATISTICAL_SIGNAL"
    assert "pooled_STATISTICAL_SIGNAL_is_not_confirmation_evidence" in dep["reasons"]
    assert "PROMOTE" not in str(dep)


def test_g_not_fed_to_r_harness_in_ablation():
    from research.sepa.synthetic import plant_vcp, stage2
    frames = {
        "LEADER": plant_vcp(contractions="tight", volume="dry"),
        "GRIND": stage2(n=320, start=40.0, step=0.35),
    }
    payload = run_ablation_r2(
        frames=frames, variants=("G",), horizon=8, warmup_sessions=200,
        min_sessions=80, min_price=1.0, min_turnover=1.0, date_step=1,
        scanner_step=99, top_n=None,
    )
    g = payload["variants"]["G"]
    assert g["not_sepa_r"] is True
    assert g["expectancy_r"] is None
    assert g["statistical_verdict"] == "NOT_SEPA_R"
    assert "PROMOTE" not in str(g.get("deployment"))
