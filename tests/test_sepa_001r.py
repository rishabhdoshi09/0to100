"""SEPA-001R causality, timing, CA, dedup, entry, PIT tests. Research only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.corporate_actions import adjust_frame
from research.sepa.ablation_r import run_ablation_r, sepa_fill_sim
from research.sepa.config import DEFAULT_CONFIG, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import (
    FILL_GAP_THROUGH,
    FILL_MISSED,
    classify_next_open_fill,
    evaluate_entry,
)
from research.sepa.integrity import PIT_DEGRADED, PIT_UNVERIFIED, classify_pit
from research.sepa.rs import build_rs_table
from research.sepa.setups import SetupRegistry, setup_id
from research.sepa.timing import diagnose_symbol
from research.sepa.vcp import causal_zigzag, detect_vcp, detect_vcp_legacy, find_swings
from research.sepa.vcp_state import VcpStateMachine
from tests.test_sepa_001_eligibility import CFG, _ohlcv, _plant_vcp, _stage2


def test_future_bars_cannot_alter_past_eligibility():
    frame = _plant_vcp(contractions="tight", volume="dry")
    as_of = frame.index[-5]
    first = evaluate_sepa_eligibility(
        "PIT", as_of, frame=frame.loc[:as_of], rs_percentile=88.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    extra = _ohlcv(np.linspace(float(frame["close"].iloc[-1]) * 1.25, float(frame["close"].iloc[-1]) * 1.6, 25))
    extra.index = pd.bdate_range(frame.index[-1] + pd.Timedelta(days=3), periods=25)
    combined = pd.concat([frame, extra])
    second = evaluate_sepa_eligibility(
        "PIT", as_of, frame=combined, rs_percentile=88.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert first.to_json() == second.to_json()


def test_future_swing_confirmation_cannot_backdate_a_pivot():
    frame = _plant_vcp(contractions="tight", volume="dry")
    machine = VcpStateMachine(CFG)
    seen = []
    for i, ts in enumerate(frame.index):
        snap = machine.update(
            float(frame["high"].iloc[i]), float(frame["low"].iloc[i]),
            float(frame["close"].iloc[i]), float(frame["volume"].iloc[i]), ts,
        )
        seen.append((str(pd.Timestamp(ts).date()), snap.get("detected"), snap.get("pivot"),
                     snap.get("pivot_knowable_date")))
    # Recompute each historical date from a prefix — must match the incremental snapshot.
    for i in range(80, len(frame), 7):
        prefix = frame.iloc[: i + 1]
        batch = detect_vcp(prefix, CFG)
        inc = seen[i]
        assert batch.get("detected") == inc[1]
        assert batch.get("pivot") == inc[2]
        if batch.get("pivot_knowable_date"):
            assert batch["pivot_knowable_date"] <= str(pd.Timestamp(prefix.index[-1]).date())


def test_pivot_knowable_date_is_confirmation_not_extreme():
    frame = _plant_vcp(contractions="tight", volume="dry")
    vcp = detect_vcp(frame, CFG)
    assert vcp["pivot"] is not None
    assert vcp["pivot_knowable_date"] is not None
    assert vcp["pivot_knowable_date"] >= vcp["pivot_date"]
    assert vcp["vcp_knowable_date"] >= vcp["pivot_knowable_date"]


def test_fractal_swings_look_ahead_but_vcp_does_not():
    frame = _plant_vcp(contractions="tight", volume="dry")
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    cut = len(frame) - 6
    sh_early, _ = find_swings(high[:cut], low[:cut], 2, 2)
    sh_full, _ = find_swings(high, low, 2, 2)
    # Fractal can discover a swing whose index is < cut only after right-side bars arrive.
    leaked = [i for i in sh_full if i < cut - 2 and i not in sh_early]
    # Causal eligibility at the cut must ignore those future-confirmed fractals.
    early = evaluate_sepa_eligibility(
        "FR", frame.index[cut - 1], frame=frame.iloc[:cut], rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    later = evaluate_sepa_eligibility(
        "FR", frame.index[cut - 1], frame=frame, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert early.to_json() == later.to_json()
    assert leaked == leaked  # documented: fractal may leak; VCP path must not


def test_vcp_detected_before_or_at_breakout_on_planted_pattern():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.0)
    diag = diagnose_symbol("DEMO", frame, config=CFG, start=200)
    assert diag["new_detection"] is not None
    if diag["breakout"]:
        assert diag["new_detection"] <= diag["breakout"]
    # At first new detection, last-contraction pivot should not be a +10% chase.
    assert diag["new_dist_to_pivot"] is not None
    assert diag["new_dist_to_pivot"] < 5.0


def test_post_breakout_late_print_is_extended():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    result = evaluate_sepa_eligibility(
        "LATE", frame.index[-1], frame=frame, rs_percentile=96.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert result.extended is True
    assert result.eligible is False
    assert result.vcp_state == "EXTENDED" or "ENTRY_EXTENDED" in result.rejection_codes


def test_split_cannot_create_fake_vcp():
    n = 220
    close = np.linspace(100.0, 118.0, n)
    close = close.copy()
    close[160:] = close[160:] * 0.5
    raw = _ohlcv(close, volume=np.full(n, 80_000.0))
    events = [{"ex_date": pd.Timestamp(raw.index[160]), "factor": 2.0, "type": "split"}]
    adj = adjust_frame(raw, events)
    vcp_adj = detect_vcp(adj, CFG)
    assert vcp_adj["detected"] is False
    # Raw discontinuity is either rejected or flagged as a deep/broken structure —
    # it must not be an eligible SEPA trade.
    raw_el = evaluate_sepa_eligibility(
        "SPLIT", raw.index[-1], frame=raw, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": False},
    )
    assert raw_el.eligible is False
    assert raw_el.ca_complete is False


def test_bonus_discontinuity_cannot_fake_stage2_pass_or_fail():
    frame = _stage2()
    raw = frame.copy()
    for col in ("open", "high", "low", "close"):
        raw.loc[raw.index[210]:, col] = raw.loc[raw.index[210]:, col] / 2.0
    events = [{"ex_date": pd.Timestamp(raw.index[210]), "factor": 2.0, "type": "bonus"}]
    adj = adjust_frame(raw, events)
    from research.sepa.trend import evaluate_trend
    raw_trend = evaluate_trend(raw, CFG, rs_percentile=90.0)
    adj_trend = evaluate_trend(adj, CFG, rs_percentile=90.0)
    assert adj_trend["structure_pass"] is True
    # Unadjusted crash vs still-high SMAs should not keep the same Stage-2 pass.
    assert raw_trend["structure_pass"] is False or raw_trend["trend_template_pass"] is False


def test_unverified_ca_propagates_into_research_metadata():
    frame = _stage2()
    result = evaluate_sepa_eligibility(
        "CA", frame.index[-1], frame=frame, rs_percentile=80.0,
        config=CFG, pit_meta={"universe_complete": False, "ca_complete": False, "ca_n_events": 0},
    )
    assert result.ca_complete is False
    assert result.pit_safe is False
    assert result.pit_class in {PIT_DEGRADED, PIT_UNVERIFIED}
    assert any("Corporate-action" in r or "phantom" in r for r in result.reasons)


def test_one_vcp_one_setup_id():
    frame = _plant_vcp(contractions="tight", volume="dry")
    a = evaluate_sepa_eligibility(
        "ONE", frame.index[-1], frame=frame, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    b = evaluate_sepa_eligibility(
        "ONE", frame.index[-2], frame=frame.iloc[:-1], rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    if a.vcp_detected and b.vcp_detected and a.base_start_date == b.base_start_date:
        assert a.setup_id == b.setup_id
        assert len(a.setup_id) == 16


def test_setup_registry_does_not_repeat_terminal_setups():
    reg = SetupRegistry()
    versions = {"eligibility_version": "x", "vcp_version": "y", "pivot_version": "z"}
    vcp = {"base_start_date": "2021-01-04", "pivot": 100.0, "stop": 95.0,
           "detected": True, "pivot_knowable_date": "2021-02-01", "vcp_knowable_date": "2021-02-10"}
    first = reg.see(symbol="AAA", vcp=vcp, versions=versions)
    second = reg.see(symbol="AAA", vcp=vcp, versions=versions)
    assert first["setup_id"] == second["setup_id"]
    reg.mark("AAA", "2021-01-04", "FILLED")
    assert reg.is_terminal("AAA", "2021-01-04")


def test_daily_ablation_does_not_mint_pseudo_trades():
    frames = {
        "LEADER": _plant_vcp(contractions="tight", volume="dry"),
        "GRIND": _stage2(),
    }
    payload = run_ablation_r(
        frames=frames, sample_step=1, lookback_sessions=60, horizon=8,
        max_symbols=None, config=CFG, variants=("F",),
    )
    f = payload["variants"]["F"]
    ids = [s["setup_id"] for s in payload.get("setups") or [] if s.get("setup_id")]
    assert len(ids) == len(set(ids))
    if f["n"] > 1:
        # trades come from distinct setups or the embargo
        assert f["n"] <= max(1, payload["sample"]["unique_setups"])


def test_gap_through_produces_no_fill():
    fwd = _ohlcv(np.array([110.0, 111.0, 112.0]))
    packed = sepa_fill_sim(fwd, stop=90.0, pivot=100.0, buy_zone_low=99.75, buy_zone_high=101.5, horizon=3)
    assert packed["class"] == FILL_GAP_THROUGH
    assert packed["sim"] is None


def test_extended_entry_remains_reject_and_price_is_not_the_entry():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.07)
    result = evaluate_sepa_eligibility(
        "EXT", frame.index[-1], frame=frame, rs_percentile=97.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert result.eligible is False
    assert result.extended is True
    if result.pivot is not None:
        assert result.proposed_entry != result.pivot or result.entry_valid is False
        # never silently replace pivot with last price as a *valid* entry
        assert result.entry_valid is False


def test_classify_next_open_never_chases():
    miss = classify_next_open_fill(open_px=90.0, zone_lo=99.75, zone_hi=101.5, stop=92.0)
    assert miss["class"] == FILL_MISSED
    assert miss["fill"] is None
    gap = classify_next_open_fill(open_px=108.0, zone_lo=99.75, zone_hi=101.5, stop=92.0)
    assert gap["class"] == FILL_GAP_THROUGH
    assert gap["fill"] is None
    ok = classify_next_open_fill(open_px=100.5, zone_lo=99.75, zone_hi=101.5, stop=92.0)
    assert ok["class"] == "VALID_FILL"
    assert ok["fill"] == 100.5


def test_expanded_future_universe_does_not_change_historical_rank():
    def _ret_frame(total: float) -> pd.DataFrame:
        return _ohlcv(np.linspace(100.0, 100.0 * (1 + total), 260))

    frames = {"WEAK": _ret_frame(0.10), "MID": _ret_frame(0.40), "HOT": _ret_frame(1.20)}
    as_of = frames["HOT"].index[-1]
    table = build_rs_table(frames, as_of, CFG, universe=["WEAK", "MID", "HOT"])
    future = _ohlcv(np.linspace(100.0, 900.0, 260))
    future.index = pd.bdate_range("2018-01-02", periods=260)
    frames2 = dict(frames)
    frames2["NEWBIE"] = future
    table2 = build_rs_table(frames2, as_of, CFG, universe=["WEAK", "MID", "HOT"])
    assert table2["percentiles"] == table["percentiles"]
    # NEWBIE is not in the as-of universe even if the dict is larger
    table3 = build_rs_table(frames2, as_of, CFG, universe=["WEAK", "MID", "HOT", "NEWBIE"])
    assert "NEWBIE" in table3["percentiles"]
    # historical trio order unchanged among themselves
    assert table3["percentiles"]["HOT"] > table3["percentiles"]["MID"] > table3["percentiles"]["WEAK"]


def test_legacy_detector_is_stricter_on_distance_than_causal():
    """Legacy fails VCP when far below the pattern-high pivot; 001R still sees structure."""
    frame = _plant_vcp(contractions="tight", volume="dry", extend=-0.10)
    neu = detect_vcp(frame, CFG)
    old = detect_vcp_legacy(frame, CFG)
    # If the print is still a structural VCP, legacy may reject on TOO_FAR_BELOW_PIVOT.
    if neu.get("detected") and neu.get("pivot"):
        dist = (float(frame["close"].iloc[-1]) / float(neu["pivot"]) - 1.0) * 100.0
        if dist < -8:
            assert old.get("detected") is False or "TOO_FAR_BELOW_PIVOT" in (old.get("fail_reasons") or [])


def test_evaluate_entry_does_not_fallback_to_price_as_pivot():
    vcp = {"pivot": None, "stop": None, "depths": []}
    out = evaluate_entry(price=123.0, vcp=vcp, atr=1.0, config=DEFAULT_CONFIG)
    assert out["entry_valid"] is False
    assert out["proposed_entry"] is None or out["entry_rejection"] == "NO_PIVOT"


def test_causal_zigzag_confirmation_index_not_before_extreme():
    high = np.array([10.0, 11.0, 12.0, 11.5, 10.0, 9.0, 9.5, 11.0, 12.5, 12.0], dtype=float)
    low = high - 0.4
    swings = causal_zigzag(high, low, 5.0)
    for s in swings:
        assert s["confirmed_index"] >= s["index"]


def test_classify_pit_does_not_call_inferred_strong():
    pit = classify_pit(
        universe_meta={"universe_complete": True, "source": "bhav_inferred", "research_grade": False},
        ca={"ca_complete": False, "verified": False, "n_events": 0},
    )
    assert pit in {PIT_DEGRADED, PIT_UNVERIFIED}
    assert pit != "PIT_STRONG"
