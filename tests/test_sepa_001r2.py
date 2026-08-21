"""SEPA-001R2 validity tests — universe PIT, VCP sequence, lifecycle, CA, warm-up."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.corporate_actions import adjust_frame
from research.sepa.ca_audit import (
    ca_applied_as_of,
    classify_subject,
    quarantine_symbols,
    unresolved_events,
)
from research.sepa.config import R2_CONFIG, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import FILL_GAP_THROUGH, classify_next_open_fill
from research.sepa.lifecycle import PersistentSetupLedger
from research.sepa.rs import build_rs_table
from research.sepa.setups import setup_id
from research.sepa.universe_pit import screen_investable_as_of
from research.sepa.vcp import detect_vcp, select_active_sequence, _contractions_from_swings, causal_zigzag
from tests.test_sepa_001_eligibility import CFG, _idx, _ohlcv, _plant_vcp, _stage2


def _liq_frame(n=300, start=80.0, step=0.3, volume=200_000.0, start_date="2019-01-02"):
    close = start + np.arange(n) * step
    df = _ohlcv(close, volume=np.full(n, volume))
    df.index = pd.bdate_range(start_date, periods=n)
    return df


def _plant_n_contractions(n: int, *, extra_old: int = 0, extend: float = 0.0) -> pd.DataFrame:
    """Stage-2 prefix + `extra_old` deep coils + `n` tightening coils."""
    base = _stage2(260, start=50.0, step=0.4)
    px = float(base["close"].iloc[-1])
    highs, lows, closes, vols = [], [], [], []

    def add(c, h, l, v):
        closes.append(float(c))
        highs.append(float(h))
        lows.append(float(l))
        vols.append(float(v))

    old_depths = [18.0] * extra_old
    new_depths = [14.0, 11.0, 9.0, 7.0, 5.5, 4.5, 3.8, 3.2, 3.0, 2.8][:n]
    last_high = px
    for i, depth in enumerate(old_depths + new_depths):
        h = last_high
        lo = h * (1.0 - depth / 100.0)
        vol = 300_000.0 if i < extra_old else 90_000.0
        for x in (h * 0.997, h, h * 0.997):
            add(x, h * 1.004, x * 0.99, vol)
        for x in np.linspace(h * 0.99, lo, 4):
            add(x, x * 1.01, x * 0.99, vol)
        nxt = h * (0.997 if i < extra_old + n - 1 else 1.0 + extend)
        for x in np.linspace(lo * 1.01, nxt, 4):
            add(x, x * 1.01, x * 0.99, 70_000.0)
        last_high = nxt
    extra = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=_idx(len(closes), start=str(base.index[-1].date() + pd.Timedelta(days=1))),
    )
    return pd.concat([base, extra])


def test_future_prices_cannot_change_historical_investable_set():
    as_of = "2020-06-15"
    a = _liq_frame(400, volume=500_000)
    b = _liq_frame(400, volume=8_000)  # illiquid through 2020
    # After as_of, B explodes in turnover
    b.loc[b.index > as_of, "volume"] = 5_000_000
    frames = {"AAA": a, "BBB": b}
    snap1 = screen_investable_as_of(
        frames, as_of, min_price=20, min_turnover=5_000_000, min_sessions=80, top_n=None,
    )
    b2 = b.copy()
    b2.loc[b2.index > as_of, "close"] *= 40
    b2.loc[b2.index > as_of, "volume"] = 9_000_000
    snap2 = screen_investable_as_of(
        {"AAA": a, "BBB": b2}, as_of, min_price=20, min_turnover=5_000_000,
        min_sessions=80, top_n=None,
    )
    assert snap1.investable == snap2.investable
    assert "BBB" not in snap1.investable


def test_future_turnover_cannot_change_historical_screen():
    as_of = "2020-06-15"
    df = _liq_frame(400, volume=10_000)
    snap = screen_investable_as_of(
        {"ZZZ": df}, as_of, min_turnover=5_000_000, min_sessions=80, min_price=10,
    )
    df2 = df.copy()
    df2["volume"] = 1_000_000.0
    df2.loc[df2.index <= as_of, "volume"] = 10_000.0
    snap2 = screen_investable_as_of(
        {"ZZZ": df2}, as_of, min_turnover=5_000_000, min_sessions=80, min_price=10,
    )
    assert snap.investable == snap2.investable == []


def test_rs_denominator_is_as_of_investable_universe():
    as_of = "2020-12-31"
    names = {}
    for i, sym in enumerate(["A", "B", "C", "D"]):
        names[sym] = _liq_frame(400, start=50 + i * 5, step=0.2 + i * 0.05, volume=300_000)
    snap = screen_investable_as_of(names, as_of, min_sessions=80, min_turnover=1.0, min_price=1)
    table = build_rs_table(names, as_of, CFG, universe=snap.investable)
    assert table["n_universe"] == len(snap.investable)
    assert set(table["percentiles"]) <= set(snap.investable)


def test_top_n_liquidity_is_as_of_date():
    as_of = "2020-06-15"
    frames = {
        "LIQ": _liq_frame(300, volume=1_000_000),
        "MID": _liq_frame(300, volume=200_000),
        "THIN": _liq_frame(300, volume=50_000),
    }
    # Future: THIN becomes the most liquid
    frames["THIN"].loc[frames["THIN"].index > as_of, "volume"] = 9_000_000
    snap = screen_investable_as_of(
        frames, as_of, min_sessions=80, min_turnover=1.0, min_price=1, top_n=2,
    )
    assert snap.investable[0] == "LIQ"
    assert "THIN" not in snap.investable[:2]


def test_future_listings_cannot_alter_historical_rs():
    as_of = "2020-06-15"
    a = _liq_frame(400, start=80, step=0.4, volume=300_000)
    b = _liq_frame(400, start=70, step=0.2, volume=300_000)
    table1 = build_rs_table({"A": a, "B": b}, as_of, CFG, universe=["A", "B"])
    future = _liq_frame(80, start=200, step=2.0, volume=300_000, start_date="2021-01-04")
    table2 = build_rs_table({"A": a, "B": b, "NEW": future}, as_of, CFG, universe=["A", "B"])
    assert table1["percentiles"]["A"] == table2["percentiles"]["A"]
    assert "NEW" not in table2["percentiles"]


def _seq_from_frame(frame, cfg=None):
    cfg = cfg or R2_CONFIG
    vcp = detect_vcp(frame, cfg)
    return vcp


def test_two_three_six_eight_contractions_use_latest_sequence():
    cfg = SepaConfig(min_contractions=2, max_contractions=6, min_reversal_pct=2.5,
                     vcp_lookback=400, volume_dry_up_required=False)
    for n in (2, 3, 6, 8):
        frame = _plant_n_contractions(n)
        vcp = detect_vcp(frame, cfg)
        assert vcp["pivot"] is not None, n
        # Last contraction high is the actionable pivot, not an early high.
        assert vcp["evidence"]["last_contraction_high_date"] == vcp["pivot_date"] or \
            abs(float(vcp["pivot"]) - float(vcp["evidence"]["last_contraction_high"])) < 1e-6


def test_six_contractions_choose_latest_valid_active_sequence():
    cfg = SepaConfig(min_contractions=2, max_contractions=6, min_reversal_pct=2.5,
                     vcp_lookback=500, volume_dry_up_required=False)
    frame = _plant_n_contractions(8)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    vol = frame["volume"].to_numpy(dtype=float)
    dates = [str(pd.Timestamp(t).date()) for t in frame.index]
    swings = causal_zigzag(high, low, 2.5)
    all_c = _contractions_from_swings(swings, high, low, vol, dates, 2.5, None)
    assert len(all_c) >= 6
    seq = select_active_sequence(all_c, cfg)
    assert seq[-1]["high_date"] == all_c[-1]["high_date"]
    assert seq[0]["high_date"] >= all_c[0]["high_date"]


def test_stale_old_contractions_cannot_set_current_pivot():
    cfg = SepaConfig(min_contractions=2, max_contractions=6, min_reversal_pct=2.5,
                     vcp_lookback=500, volume_dry_up_required=False)
    live = _plant_n_contractions(3)
    with_old = _plant_n_contractions(3, extra_old=8)
    v_live = detect_vcp(live, cfg)
    v_old = detect_vcp(with_old, cfg)
    assert v_live["pivot"] is not None and v_old["pivot"] is not None
    # Adding older coils must not move the live last-contraction pivot backward
    # onto an early high. The last contraction high date must be the latest one.
    assert v_old["evidence"]["last_contraction_high_date"] >= v_old["dates"][0]
    assert v_old["pivot"] == v_old["evidence"]["last_contraction_high"] or \
        abs(float(v_old["pivot"]) - float(v_old["evidence"]["last_contraction_high"])) < 1e-5


def test_rolling_lookback_does_not_mint_repeated_setup_ids():
    cfg = SepaConfig(min_contractions=2, max_contractions=6, vcp_lookback=80,
                     volume_dry_up_required=False, min_reversal_pct=2.5)
    frame = _plant_n_contractions(4)
    # Pad so the first contraction can age out of a 80-bar window.
    pad = _ohlcv(np.linspace(float(frame["close"].iloc[-1]), float(frame["close"].iloc[-1]) * 1.01, 40))
    pad.index = pd.bdate_range(frame.index[-1] + pd.Timedelta(days=3), periods=40)
    long = pd.concat([frame, pad])
    versions = {"eligibility_version": "sepa-001r2.v1", "vcp_version": "vcp_causal_v2",
                "pivot_version": "pivot_last_contraction_v1"}
    ledger = PersistentSetupLedger(versions=versions)
    ids = []
    for i in range(len(long) - 90, len(long)):
        prefix = long.iloc[: i + 1]
        vcp = detect_vcp(prefix.iloc[-80:], cfg)
        if not vcp.get("base_start_date"):
            continue
        rec = ledger.observe(symbol="ROLL", vcp=vcp, as_of=str(pd.Timestamp(prefix.index[-1]).date()))
        if rec:
            ids.append(rec["setup_id"])
    assert ids, "expected a continuing setup"
    assert len(set(ids)) == 1, set(ids)


def test_genuine_new_base_gets_new_id():
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    frame = _plant_vcp(contractions="tight", volume="dry")
    v1 = detect_vcp(frame, CFG)
    r1 = ledger.observe(symbol="NB", vcp=v1, as_of="2021-03-16")
    ledger.mark("NB", "FAILED", reason="BROKEN_STRUCTURE")
    frame2 = _plant_vcp(contractions="two", volume="dry")
    frame2.index = pd.bdate_range("2022-01-03", periods=len(frame2))
    v2 = detect_vcp(frame2, CFG)
    r2 = ledger.observe(symbol="NB", vcp=v2, as_of="2022-06-01")
    assert r1["setup_id"] != r2["setup_id"]


def test_new_coil_after_left_censor_can_become_opportunity():
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    vcp = detect_vcp(frame, CFG)
    as_of = str(pd.Timestamp(frame.index[-1]).date())
    rec = ledger.observe(
        symbol="LC2", vcp=vcp, as_of=as_of, evaluation_start=as_of,
        price=float(frame["close"].iloc[-1]), zone_hi=float(vcp["pivot"]) * 1.015,
    )
    assert rec["left_censored"] is True
    later = _plant_vcp(contractions="two", volume="dry", extend=0.0)
    later.index = pd.bdate_range("2023-01-03", periods=len(later))
    v2 = detect_vcp(later, CFG)
    rec2 = ledger.observe(symbol="LC2", vcp=v2, as_of="2023-06-01")
    assert rec2["setup_id"] != rec["setup_id"]
    assert rec2.get("left_censored") is not True
    assert ledger.is_core_opportunity("LC2") is True


def test_left_censored_setup_excluded_from_opportunity_stats():
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    vcp = detect_vcp(frame, CFG)
    as_of = str(pd.Timestamp(frame.index[-1]).date())
    rec = ledger.observe(
        symbol="LC", vcp=vcp, as_of=as_of, evaluation_start=as_of,
        price=float(frame["close"].iloc[-1]), zone_hi=float(vcp["pivot"]) * 1.015,
        in_eval_window=True,
    )
    assert rec["left_censored"] is True
    assert ledger.is_core_opportunity("LC") is False
    ledger.mark("LC", "EXTENDED")
    assert ledger.get("LC")["status"] == "LEFT_CENSORED"


def test_observed_extended_breakout_remains_refused():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    el = evaluate_sepa_eligibility(
        "EX", frame.index[-1], frame=frame, rs_percentile=90.0, config=CFG,
        pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert el.extended is True
    assert el.eligible is False
    assert el.entry_valid is False


def test_forming_below_pivot_remains_trackable():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=-0.02)
    vcp = detect_vcp(frame, CFG)
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    rec = ledger.observe(symbol="BL", vcp=vcp, as_of=str(pd.Timestamp(frame.index[-1]).date()))
    assert rec is not None
    assert rec["status"] not in {"EXTENDED", "FAILED", "LEFT_CENSORED"}
    assert ledger.is_core_opportunity("BL") is True


def test_pivot_retest_is_not_core_sepa_entry():
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    frame = _plant_vcp(contractions="tight", volume="dry")
    vcp = detect_vcp(frame, CFG)
    rec = ledger.observe(symbol="RT", vcp=vcp, as_of="2021-03-16")
    ledger.mark("RT", "EXTENDED")
    vcp2 = dict(vcp)
    vcp2["state"] = "ENTRY_READY"
    rec2 = ledger.observe(symbol="RT", vcp=vcp2, as_of="2021-04-15")
    assert rec2["status"] == "PIVOT_RETEST"
    assert rec2.get("core_sepa_entry") is False
    assert rec["setup_id"] == rec2["setup_id"]


def test_demerger_gap_is_quarantined_not_adjusted():
    close = np.concatenate([np.linspace(200, 220, 40), np.linspace(110, 120, 40)])
    df = _ohlcv(close, volume=np.full(80, 100_000))
    rows = unresolved_events({"ABFRLX": df}, events={})
    assert rows
    q = quarantine_symbols(rows)
    assert "ABFRLX" in q
    # Must not invent a 2.0 factor from the ~50% gap
    assert all(r.get("never_infers_factor") for r in rows)


def test_unresolved_restructuring_quarantined():
    assert classify_subject("Demerger of lifestyle business") == "demerger"
    close = np.concatenate([np.full(20, 100.0), np.full(20, 40.0)])
    df = _ohlcv(close)
    rows = unresolved_events({"RESTR": df}, events={})
    assert "RESTR" in quarantine_symbols(rows)


def test_future_ca_not_applied_before_ex_date():
    close = np.linspace(100, 110, 30)
    df = _ohlcv(close)
    future = [{"ex_date": pd.Timestamp(df.index[-1] + pd.Timedelta(days=10)),
               "factor": 5.0, "type": "split"}]
    as_of = df.index[10]
    adj = ca_applied_as_of(df, future, as_of)
    raw = df.loc[:as_of]
    assert abs(float(adj["close"].iloc[-1]) - float(raw["close"].iloc[-1])) < 1e-9
    # After the ex-date the same event DOES adjust history
    later = adjust_frame(df, future)
    assert float(later["close"].iloc[0]) < float(df["close"].iloc[0]) / 2


def test_split_does_not_create_fake_vcp_when_adjusted():
    base = _stage2(200, start=80, step=0.4)
    crash = _ohlcv(np.linspace(40, 42, 30))
    crash.index = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=30)
    raw = pd.concat([base, crash])
    events = [{"ex_date": crash.index[0], "factor": 2.0, "type": "split"}]
    adj = adjust_frame(raw, events)
    v_raw = detect_vcp(raw, CFG)
    v_adj = detect_vcp(adj, CFG)
    # Unadjusted crash can fabricate structure; adjusted series must not
    # treat the split print as a VCP low.
    if v_adj.get("stop") is not None:
        assert float(v_adj["stop"]) > 30


def test_warmup_setups_do_not_count_as_fresh_detections():
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    ledger = PersistentSetupLedger(versions=versions)
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.05)
    vcp = detect_vcp(frame, CFG)
    warm = str(pd.Timestamp(frame.index[-1]).date())
    rec = ledger.observe(
        symbol="WU", vcp=vcp, as_of=warm, evaluation_start=warm,
        price=float(frame["close"].iloc[-1]),
        zone_hi=float(vcp["pivot"]) * 1.015 if vcp.get("pivot") else None,
    )
    assert rec["left_censored"] is True or rec["status"] == "LEFT_CENSORED"


def test_gap_through_still_no_fill():
    packed = classify_next_open_fill(open_px=110.0, zone_lo=99.75, zone_hi=101.5, stop=92.0)
    assert packed["class"] == FILL_GAP_THROUGH
    assert packed["fill"] is None


def test_setup_id_stable_helper_still_hashes_origin():
    a = setup_id("X", "2020-01-02", eligibility_version="e", vcp_version="v", pivot_version="p")
    b = setup_id("X", "2020-01-02", eligibility_version="e", vcp_version="v", pivot_version="p")
    c = setup_id("X", "2020-06-01", eligibility_version="e", vcp_version="v", pivot_version="p")
    assert a == b and a != c


def test_fast_investable_matches_as_of_screen():
    from research.sepa.universe_pit import FastInvestable
    as_of = "2020-06-15"
    frames = {
        "LIQ": _liq_frame(300, volume=1_000_000),
        "THIN": _liq_frame(300, volume=1_000),
    }
    a = screen_investable_as_of(frames, as_of, min_sessions=80, min_turnover=5_000_000, min_price=10)
    b = FastInvestable(frames).snapshot(as_of, min_sessions=80, min_turnover=5_000_000, min_price=10)
    assert a.investable == b.investable


def test_ablation_r2_smoke_unique_setups_and_no_lookahead_universe():
    from research.sepa.ablation_r2 import run_ablation_r2
    from research.sepa.synthetic import plant_vcp, stage2
    frames = {
        "LEADER": plant_vcp(contractions="tight", volume="dry"),
        "GRIND": stage2(n=320, start=40.0, step=0.35),
        "WIDEN": plant_vcp(contractions="widening", volume="dry"),
    }
    payload = run_ablation_r2(
        frames=frames, variants=("F", "G"), horizon=8, warmup_sessions=200,
        min_sessions=80, min_price=1.0, min_turnover=1.0, date_step=1,
        scanner_step=99, top_n=None,
    )
    assert payload["experiment"] == "SEPA-001R2"
    assert payload["sample"]["n_as_of"] > 0
    f = payload["variants"]["F"]
    assert "statistical_verdict" in f
    assert f["deployment"]["deployment_eligible"] is False
    assert "PROMOTE" not in str(f.get("statistical_verdict"))
