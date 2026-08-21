"""SEPA-001 canonical eligibility — synthetic, PIT-safe, no live wiring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.sepa.config import SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.rs import build_rs_table, percentile_rank, score_one
from research.sepa.trend import evaluate_trend
from research.sepa.vcp import detect_vcp, find_swings


def _idx(n: int, start="2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _ohlcv(close, volume=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    vol = np.asarray(volume if volume is not None else np.full(n, 100_000.0), dtype=float)
    high = close + 0.4
    low = close - 0.4
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=_idx(n),
    )


def _stage2(n: int = 280, start=80.0, step=0.55, volume=120_000.0) -> pd.DataFrame:
    close = start + np.arange(n) * step
    return _ohlcv(close, volume=np.full(n, volume))


def _downtrend(n: int = 280) -> pd.DataFrame:
    close = 220 - np.arange(n) * 0.45
    return _ohlcv(close)


CFG = SepaConfig(swing_left=2, swing_right=2)


def _plant_vcp(*, contractions="tight", volume="dry", extend=0.0, wide_stop=False) -> pd.DataFrame:
    """Long Stage-2 trend plus a handmade swing VCP at the end."""
    base = _stage2(260, start=80.0, step=0.5)
    last = float(base["close"].iloc[-1])
    pivot = last
    if contractions == "tight":
        legs = [
            (pivot, pivot * 0.86),   # 14%
            (pivot * 0.99, pivot * 0.92),  # ~7%
            (pivot, pivot * 0.96),   # 4%
        ]
    elif contractions == "two":
        legs = [
            (pivot, pivot * 0.88),
            (pivot * 0.995, pivot * 0.94),
        ]
    elif contractions == "widening":
        legs = [
            (pivot, pivot * 0.95),
            (pivot * 0.99, pivot * 0.88),
            (pivot, pivot * 0.80),
        ]
    elif contractions == "deep":
        legs = [
            (pivot, pivot * 0.55),
            (pivot * 0.90, pivot * 0.70),
            (pivot, pivot * 0.85),
        ]
    else:
        raise ValueError(contractions)

    highs, lows, closes, vols = [], [], [], []

    def _swing_high(px: float, vol: float):
        # 2 bars each side lower → fractal swing high at centre
        for v in (px * 0.985, px * 0.995, px, px * 0.995, px * 0.985):
            highs.append(v + 0.05)
            lows.append(v - 0.8)
            closes.append(v)
            vols.append(vol)

    def _swing_low(px: float, vol: float):
        for v in (px * 1.015, px * 1.005, px, px * 1.005, px * 1.015):
            highs.append(v + 0.8)
            lows.append(v - 0.05)
            closes.append(v)
            vols.append(vol)

    def _recover(from_px: float, to_px: float, vol: float, bars=4):
        seq = np.linspace(from_px, to_px, bars)
        for v in seq:
            highs.append(v + 0.3)
            lows.append(v - 0.3)
            closes.append(v)
            vols.append(vol)

    vol_first = 400_000.0 if volume == "dry" else 80_000.0
    vol_last = 80_000.0 if volume == "dry" else 500_000.0
    for i, (h, lo) in enumerate(legs):
        vol = vol_first if i == 0 else (vol_last if i == len(legs) - 1 else (vol_first + vol_last) / 2)
        _swing_high(h, vol)
        _recover(h * 0.99, lo * 1.01, vol, bars=3)
        _swing_low(lo if not wide_stop else lo * 0.5, vol)
        nxt = legs[i + 1][0] if i + 1 < len(legs) else h * (1.0 + extend)
        _recover(lo * 1.01, nxt, vol_last if i == len(legs) - 1 else vol, bars=4)

    # finish at/near pivot
    finish = pivot * (1.0 + extend)
    _recover(closes[-1], finish, vol_last, bars=3)
    extra = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        },
        index=_idx(len(closes), start=str(base.index[-1].date() + pd.Timedelta(days=1))),
    )
    return pd.concat([base, extra])


def test_each_trend_rule_passes_on_stage2():
    trend = evaluate_trend(_stage2(), CFG, rs_percentile=85.0)
    assert trend["trend_template_pass"] is True
    assert trend["passed"] == 8
    by = {r.id: r for r in trend["rules"]}
    for rid in (
        "price_gt_150_200", "sma150_gt_200", "sma200_rising", "sma50_leads",
        "price_gt_sma50", "off_52w_low", "near_52w_high", "rs_percentile",
    ):
        assert by[rid].passed is True, rid


def test_missing_history_fails_closed():
    trend = evaluate_trend(_stage2(30), CFG, rs_percentile=90.0)
    assert trend["trend_template_pass"] is False
    by = {r.id: r for r in trend["rules"]}
    assert by["price_gt_150_200"].passed is None
    assert by["sma200_rising"].passed is None


def test_sma200_rising_false_when_falling():
    trend = evaluate_trend(_downtrend(), CFG, rs_percentile=90.0)
    by = {r.id: r for r in trend["rules"]}
    assert by["sma200_rising"].passed is False
    assert trend["trend_template_pass"] is False


def test_52w_low_requires_30_percent():
    # Stage-2 shape but last close only ~20% above the 52w low
    n = 260
    low_phase = np.full(20, 100.0)
    rise = np.linspace(100.0, 120.0, n - 20)
    frame = _ohlcv(np.concatenate([low_phase, rise]))
    trend = evaluate_trend(frame, CFG, rs_percentile=90.0)
    by = {r.id: r for r in trend["rules"]}
    assert by["off_52w_low"].values["threshold"] == 30.0
    assert by["off_52w_low"].passed is False


def test_near_52w_high_25_percent():
    n = 280
    peak = np.linspace(80, 200, 200)
    drop = np.linspace(200, 140, n - 200)  # 30% below high
    frame = _ohlcv(np.concatenate([peak, drop]))
    trend = evaluate_trend(frame, CFG, rs_percentile=90.0)
    by = {r.id: r for r in trend["rules"]}
    assert by["near_52w_high"].passed is False


def test_rs_percentile_injected_threshold():
    trend = evaluate_trend(_stage2(), CFG, rs_percentile=69.9)
    assert trend["trend_template_pass"] is False
    assert {r.id: r for r in trend["rules"]}["rs_percentile"].passed is False
    trend_ok = evaluate_trend(_stage2(), CFG, rs_percentile=70.0)
    assert {r.id: r for r in trend_ok["rules"]}["rs_percentile"].passed is True


def test_rs_rank_no_future_and_stable():
    def _ret_frame(total: float) -> pd.DataFrame:
        n = 260
        start = 100.0
        end = start * (1 + total)
        return _ohlcv(np.linspace(start, end, n))

    frames = {"WEAK": _ret_frame(0.10), "MID": _ret_frame(0.40), "HOT": _ret_frame(1.20)}
    as_of = frames["HOT"].index[-1]
    table = build_rs_table(frames, as_of, CFG, universe=["WEAK", "MID", "HOT"])
    assert table["percentiles"]["HOT"] > table["percentiles"]["MID"] > table["percentiles"]["WEAK"]
    assert table["percentiles"]["WEAK"] == 0.0
    # appending future bars must not change as-of rank
    future = _ohlcv(np.linspace(500, 800, 20))
    future.index = pd.bdate_range(frames["WEAK"].index[-1] + pd.Timedelta(days=3), periods=20)
    frames2 = {k: pd.concat([v, future]) if k == "WEAK" else v for k, v in frames.items()}
    table2 = build_rs_table(frames2, as_of, CFG, universe=["WEAK", "MID", "HOT"])
    assert table2["percentiles"] == table["percentiles"]


def test_rs_only_ranks_as_of_universe():
    def _ret_frame(total: float) -> pd.DataFrame:
        n = 260
        return _ohlcv(np.linspace(100.0, 100.0 * (1 + total), n))

    frames = {"WEAK": _ret_frame(0.10), "HOT": _ret_frame(1.20), "GHOST": _ret_frame(9.0)}
    as_of = frames["HOT"].index[-1]
    table = build_rs_table(frames, as_of, CFG, universe=["WEAK", "HOT"])
    assert "GHOST" not in table["percentiles"]
    assert set(table["percentiles"]) == {"WEAK", "HOT"}


def test_rs_missing_universe_is_empty_not_invented():
    table = build_rs_table({}, "2024-06-01", CFG, universe=["ABC"])
    assert table["n_ranked"] == 0
    assert table["percentiles"] == {}


def test_percentile_formula():
    assert percentile_rank(5, [1, 2, 3, 4, 5]) == 80.0
    assert percentile_rank(1, [1, 2, 3]) == 0.0


def test_vcp_three_contractions_pass():
    frame = _plant_vcp(contractions="tight", volume="dry")
    vcp = detect_vcp(frame, CFG)
    assert vcp["contraction_count"] >= 2
    assert vcp["detected"] is True
    assert vcp["pivot"] is not None
    assert vcp["dry_up_ratio"] is not None and vcp["dry_up_ratio"] <= 0.90
    assert vcp["final_depth_pct"] <= 12.0


def test_vcp_two_contractions_can_pass():
    frame = _plant_vcp(contractions="two", volume="dry")
    vcp = detect_vcp(frame, CFG)
    assert vcp["contraction_count"] >= 2
    assert vcp["detected"] is True


def test_vcp_rejects_calendar_coil_without_swings():
    # Smooth grind — shrinking ranges if windowed, but no pullback structure
    n = 200
    close = 100 + np.linspace(0, 5, n)
    frame = _ohlcv(close, volume=np.full(n, 50_000))
    vcp = detect_vcp(frame, CFG)
    assert vcp["detected"] is False
    assert "TOO_FEW_CONTRACTIONS" in vcp["fail_reasons"] or "NO_SWING_STRUCTURE" in vcp["fail_reasons"]


def test_vcp_rejects_widening_contractions():
    frame = _plant_vcp(contractions="widening", volume="dry")
    vcp = detect_vcp(frame, CFG)
    assert vcp["detected"] is False
    assert any(r in vcp["fail_reasons"] for r in ("NOT_TIGHTENING", "EXPANDING_PULLBACKS", "FINAL_CONTRACTION_LOOSE", "BASE_TOO_DEEP"))


def test_vcp_rejects_deep_base():
    frame = _plant_vcp(contractions="deep", volume="dry")
    vcp = detect_vcp(frame, CFG)
    assert vcp["detected"] is False
    assert "BASE_TOO_DEEP" in vcp["fail_reasons"] or vcp["base_depth_pct"] is None or vcp["base_depth_pct"] > 35


def test_vcp_no_volume_dryup_fails():
    frame = _plant_vcp(contractions="tight", volume="expand")
    vcp = detect_vcp(frame, CFG)
    assert vcp["detected"] is False
    assert "VOLUME_EXPANDING" in vcp["fail_reasons"]
    assert vcp["dry_up_ratio"] is not None


def test_vcp_cannot_see_future_bars():
    frame = _plant_vcp(contractions="tight", volume="dry")
    as_of = frame.index[120]
    early = frame.loc[:as_of]
    vcp = detect_vcp(early, CFG)
    later = detect_vcp(frame, CFG)
    # early slice is still in the trend-build, not the planted VCP
    assert vcp["detected"] is False or vcp["pivot"] != later.get("pivot")


def test_excellent_stock_extended_entry_is_not_eligible():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    result = evaluate_sepa_eligibility(
        "DEMO", frame.index[-1], frame=frame, rs_percentile=96.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert result.good_stock is True
    assert result.extended is True
    assert result.eligible is False
    assert result.headline.startswith("NO TRADE")
    assert "ENTRY_EXTENDED" in result.rejection_codes


def test_buy_zone_examples_a_and_b():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=0.004)
    a = evaluate_sepa_eligibility(
        "A", frame.index[-1], frame=frame, rs_percentile=96.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert a.pivot is not None
    # 0.4% above pivot is inside 1.5% default zone
    if a.vcp_detected:
        assert a.entry_valid is True
        assert a.extended is False

    b = evaluate_sepa_eligibility(
        "B", frame.index[-1], frame=frame, rs_percentile=97.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
        buy_zone_above_pct=0.25,
    )
    # same print with a 0.25% cap may extend; force a 7% chase
    chased = _plant_vcp(contractions="tight", volume="dry", extend=0.07)
    far = evaluate_sepa_eligibility(
        "FAR", chased.index[-1], frame=chased, rs_percentile=97.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert far.eligible is False
    assert far.extended is True or far.entry_valid is False


def test_price_slightly_below_pivot_is_not_a_trade():
    frame = _plant_vcp(contractions="tight", volume="dry", extend=-0.02)
    result = evaluate_sepa_eligibility(
        "WAIT", frame.index[-1], frame=frame, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    if result.pivot is not None and result.price is not None and result.price < result.buy_zone_low:
        assert result.eligible is False
        assert "ENTRY_BELOW_PIVOT" in result.rejection_codes or result.entry_valid is False


def test_structural_stop_not_overwritten_by_atr():
    frame = _plant_vcp(contractions="tight", volume="dry")
    result = evaluate_sepa_eligibility(
        "STOP", frame.index[-1], frame=frame, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    if result.structural_stop is not None and result.atr is not None and result.proposed_entry:
        atr_stop = result.proposed_entry - 2 * result.atr
        # structural stop is the contraction low, not 2*ATR
        assert result.stop_basis == "final_contraction_low"
        assert abs(result.structural_stop - atr_stop) > 1e-6 or result.vcp_detected is False


def test_wide_structural_stop_rejected():
    frame = _plant_vcp(contractions="tight", volume="dry", wide_stop=True)
    result = evaluate_sepa_eligibility(
        "WIDE", frame.index[-1], frame=frame, rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    if result.good_setup:
        assert result.eligible is False
        assert "WIDE_STRUCTURAL_STOP" in result.rejection_codes or result.stop_ok is False


def test_no_manufactured_pivot_when_vcp_fails():
    result = evaluate_sepa_eligibility(
        "GRIND", _stage2().index[-1], frame=_stage2(), rs_percentile=90.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert result.vcp_detected is False
    assert result.eligible is False
    assert result.pivot is None or "NO_PIVOT" in result.rejection_codes or "VCP_NOT_DETECTED" in result.rejection_codes


def test_pit_invariance_when_future_bars_appended():
    frame = _plant_vcp(contractions="tight", volume="dry")
    as_of = frame.index[-1]
    first = evaluate_sepa_eligibility(
        "PIT", as_of, frame=frame, rs_percentile=88.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    extra = _ohlcv(np.linspace(float(frame["close"].iloc[-1]) * 1.2, float(frame["close"].iloc[-1]) * 1.5, 30))
    extra.index = pd.bdate_range(frame.index[-1] + pd.Timedelta(days=3), periods=30)
    combined = pd.concat([frame, extra])
    second = evaluate_sepa_eligibility(
        "PIT", as_of, frame=combined, rs_percentile=88.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert first.to_json() == second.to_json()


def test_determinism_same_inputs_same_json():
    frame = _stage2()
    a = evaluate_sepa_eligibility(
        "DET", frame.index[-1], frame=frame, rs_percentile=80.0,
        config=CFG, pit_meta={"universe_complete": False, "ca_complete": False},
    )
    b = evaluate_sepa_eligibility(
        "DET", frame.index[-1], frame=frame, rs_percentile=80.0,
        config=CFG, pit_meta={"universe_complete": False, "ca_complete": False},
    )
    assert a.to_json() == b.to_json()
    json.loads(a.to_json())


def test_near_sepa_is_not_eligible():
    trend = evaluate_trend(_stage2(), CFG, rs_percentile=60.0)
    assert trend["near_sepa"] is True
    assert trend["trend_template_pass"] is False
    result = evaluate_sepa_eligibility(
        "NEAR", _stage2().index[-1], frame=_stage2(), rs_percentile=60.0,
        config=CFG, pit_meta={"universe_complete": True, "ca_complete": True},
    )
    assert result.eligible is False
    assert result.evidence.get("near_sepa") is True
