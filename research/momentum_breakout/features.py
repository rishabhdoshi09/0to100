"""
🧮 Feature computation for the Institutional Momentum Breakout framework.

Six feature GROUPS + trend-extension + weakening/exit features, each built ONLY
from the point-in-time primitives in `pit.py`. Every function returns RAW values
(the framework never collapses to one opaque score); the *_score() helpers turn
raw values into transparent 0–100 component scores whose weights are versioned.

Point-in-time discipline: every lookback ends at the observation bar `i`; the base
uses only bars ≤ i-1; the pivot is the base's PRE-EXISTING resistance (no future
bar). Valuation is CONTEXT only and fails closed to UNAVAILABLE unless a real
publication timestamp proves the data pre-dates the observation.
"""
from __future__ import annotations

import numpy as np

from research.momentum_breakout import pit as P
from research.momentum_breakout import observation as OBS


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — prior upmove & leadership
# ══════════════════════════════════════════════════════════════════════════════

def prior_upmove_features(close, high, low, bench_close, i, cfg) -> dict:
    f = {
        "ret_3m_pct": P.ret_pct(close, i, 63),
        "ret_6m_pct": P.ret_pct(close, i, 126),
        "ret_12m_pct": P.ret_pct(close, i, 252),
        "ret_12m1m_pct": P.ret_pct_skip(close, i, 252, 21),
        "rel_to_bench_pct": P.rel_strength_vs_benchmark(close, bench_close, i,
                                                        cfg.prior_rs_lookback),
        "dist_from_52w_high_pct": P.dist_from_high_pct(high, i, 252),
        "dma50_slope_pct": P.slope_pct(close, i, 50, cfg.trend_slope_ago),
        "dma200_slope_pct": P.slope_pct(close, i, cfg.trend_ma, cfg.trend_slope_ago),
        "prior_upmove_max_dd_pct": P.max_drawdown_pct(close, i, cfg.prior_rs_lookback),
    }
    c_i = float(np.asarray(close, float)[i])
    f["above_20ema"] = bool(c_i > P.ema(close, i, 20)) if np.isfinite(P.ema(close, i, 20)) else None
    f["above_50dma"] = bool(c_i > P.sma(close, i, 50)) if np.isfinite(P.sma(close, i, 50)) else None
    f["above_200dma"] = (bool(c_i > P.sma(close, i, cfg.trend_ma))
                         if np.isfinite(P.sma(close, i, cfg.trend_ma)) else None)
    # all-time-high distance only when history is long enough to be reliable
    if i + 1 >= 252 * 5:
        f["dist_from_ath_pct"] = P.dist_from_high_pct(high, i, i + 1)
    else:
        f["dist_from_ath_pct"] = None
    return f


def leadership_score(f: dict, cfg) -> float:
    """Transparent 0–100. Rewards genuine prior leadership; a stock emerging from a
    long downtrend (negative rel-strength, below 200-DMA) scores near zero."""
    s = 0.0
    rel = f.get("rel_to_bench_pct")
    if rel is not None and np.isfinite(rel):
        s += max(0.0, min(40.0, rel))              # up to +40 for outperformance
    if f.get("above_200dma"):
        s += 20.0
    if f.get("dma200_slope_pct") is not None and np.isfinite(f["dma200_slope_pct"]) \
            and f["dma200_slope_pct"] > 0:
        s += 20.0
    dist = f.get("dist_from_52w_high_pct")
    if dist is not None and np.isfinite(dist):
        s += max(0.0, 20.0 * (1.0 - min(dist, 30.0) / 30.0))   # near highs = leader
    return round(min(100.0, s), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — long-base structure
# ══════════════════════════════════════════════════════════════════════════════

def base_features(open_, high, low, close, volume, base_start_i, base_end_i,
                  pivot, cfg) -> dict:
    """Deterministic base measurements over [base_start_i, base_end_i] (all bars
    strictly BEFORE the breakout bar). No future bar decides an earlier base."""
    h = np.asarray(high, float)[base_start_i: base_end_i + 1]
    l = np.asarray(low, float)[base_start_i: base_end_i + 1]
    c = np.asarray(close, float)[base_start_i: base_end_i + 1]
    o = np.asarray(open_, float)[base_start_i: base_end_i + 1]
    n = c.size
    base_high = float(np.max(h)) if n else float("nan")
    base_low = float(np.min(l)) if n else float("nan")
    depth = ((base_high - base_low) / base_high * 100.0) if base_high > 0 else float("nan")
    tol = pivot * 0.02
    pivot_tests = int(np.sum(h >= pivot - tol)) if n else 0
    time_near_pivot = float(np.mean(h >= pivot - tol) * 100.0) if n else float("nan")
    # rolling range contraction: last third vs first third of the base
    third = max(1, n // 3)
    early_rng = float(np.mean((h[:third] - l[:third]) / c[:third])) if n else float("nan")
    late_rng = float(np.mean((h[-third:] - l[-third:]) / c[-third:])) if n else float("nan")
    range_contraction = (late_rng / early_rng) if (early_rng and early_rng > 0) else float("nan")
    # ATR contraction across the base (uses absolute indices for prior-close TR)
    atr_early = P.atr(high, low, close, base_start_i + min(14, n - 1), 14) if n > 14 else float("nan")
    atr_late = P.atr(high, low, close, base_end_i, 14)
    atr_contraction = (atr_late / atr_early) if (atr_early and np.isfinite(atr_early)
                                                 and atr_early > 0) else float("nan")
    rvol_contraction = P.realised_vol_pct(close, base_end_i, min(n - 1, 20)) if n > 2 else float("nan")
    dryup = P.volume_dryup(volume, base_end_i, n, cfg.base_dryup_ref)
    tight_close = float(np.mean(np.abs(c - o) / c < 0.015) * 100.0) if n else float("nan")
    upper_half = float(np.mean((c - l) > (h - c)) * 100.0) if n else float("nan")
    # higher-low retention: is the last-third min low above the first-third min low?
    higher_lows = bool(np.min(l[-third:]) >= np.min(l[:third])) if n else None
    # failed breakout attempts inside the base: closes that pierced then fell back
    failed_breaks = int(np.sum((h >= pivot) & (c < pivot))) if n else 0
    return {
        "base_high": base_high, "base_low": base_low, "base_depth_pct": depth,
        "pivot": pivot, "pivot_tests": pivot_tests, "time_near_pivot_pct": time_near_pivot,
        "range_contraction_ratio": range_contraction,
        "atr_contraction_ratio": atr_contraction,
        "realised_vol_pct": rvol_contraction,
        "volume_dryup_ratio": dryup,
        "tight_close_freq_pct": tight_close,
        "upper_half_close_freq_pct": upper_half,
        "higher_lows": higher_lows,
        "failed_breakout_attempts": failed_breaks,
        "base_above_rising_200dma": (bool(np.asarray(close, float)[base_end_i] >
                                          P.sma(close, base_end_i, cfg.trend_ma))
                                     if np.isfinite(P.sma(close, base_end_i, cfg.trend_ma))
                                     else None),
        "base_near_52w_high": (P.dist_from_high_pct(high, base_end_i, 252) < 15.0
                               if np.isfinite(P.dist_from_high_pct(high, base_end_i, 252))
                               else None),
    }


def base_quality_score(f: dict, cfg) -> float:
    s = 0.0
    depth = f.get("base_depth_pct")
    if depth is not None and np.isfinite(depth):
        s += max(0.0, 25.0 * (1.0 - min(depth, cfg.max_base_depth_pct) / cfg.max_base_depth_pct))
    for key, pts in (("range_contraction_ratio", 20.0), ("atr_contraction_ratio", 15.0),
                     ("volume_dryup_ratio", 15.0)):
        v = f.get(key)
        if v is not None and np.isfinite(v) and v < 1.0:
            s += pts * min(1.0, (1.0 - v) * 2.0)
    if f.get("higher_lows"):
        s += 10.0
    tn = f.get("time_near_pivot_pct")
    if tn is not None and np.isfinite(tn):
        s += min(15.0, tn / 100.0 * 15.0)
    return round(min(100.0, s), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — breakout quality
# ══════════════════════════════════════════════════════════════════════════════

def breakout_features(open_, high, low, close, volume, i, pivot, atr_val, cfg) -> dict:
    c_i = float(np.asarray(close, float)[i]); o_i = float(np.asarray(open_, float)[i])
    prev_c = float(np.asarray(close, float)[i - 1]) if i > 0 else float("nan")
    dist_above = ((c_i - pivot) / pivot * 100.0) if pivot > 0 else float("nan")
    gap_pct = ((o_i - prev_c) / prev_c * 100.0) if np.isfinite(prev_c) and prev_c > 0 else float("nan")
    ext_atr = ((c_i - pivot) / atr_val) if (atr_val and np.isfinite(atr_val) and atr_val > 0) else float("nan")
    return {
        "close_above_pivot": bool(c_i > pivot),
        "breakout_dist_above_pivot_pct": dist_above,
        "breakout_clv": P.clv(high, low, close, i),
        "breakout_range_atr": (P.true_range(high, low, close, i) / atr_val
                               if (atr_val and np.isfinite(atr_val) and atr_val > 0)
                               else float("nan")),
        "breakout_rvol": P.rel_volume(volume, i, cfg.vol_ref_window),
        "breakout_volume_z": P.volume_z(volume, i, cfg.vol_ref_window),
        "gap_pct": gap_pct,
        "dist_from_20ema_pct": ((c_i / P.ema(close, i, 20) - 1.0) * 100.0
                                if np.isfinite(P.ema(close, i, 20)) and P.ema(close, i, 20) > 0
                                else float("nan")),
        "dist_from_50dma_pct": ((c_i / P.sma(close, i, 50) - 1.0) * 100.0
                                if np.isfinite(P.sma(close, i, 50)) and P.sma(close, i, 50) > 0
                                else float("nan")),
        "is_52w_high_breakout": P.is_new_high(high, i, 252),
        "confirmed_close": bool(c_i >= pivot * (1.0 + cfg.breakout_buffer_pct / 100.0)),
        "extension_atr": ext_atr,
        "overextended": bool(np.isfinite(ext_atr) and ext_atr > cfg.max_extension_atr),
    }


def breakout_quality_score(f: dict, cfg) -> float:
    s = 0.0
    if f.get("confirmed_close"):
        s += 30.0
    clv = f.get("breakout_clv")
    if clv is not None and np.isfinite(clv):
        s += max(0.0, 20.0 * (clv + 1.0) / 2.0)     # strong close = high CLV
    rvol = f.get("breakout_rvol")
    if rvol is not None and np.isfinite(rvol):
        s += min(30.0, max(0.0, (rvol - 1.0) * 30.0))
    if f.get("is_52w_high_breakout"):
        s += 20.0
    if f.get("overextended"):
        s = max(0.0, s - 20.0)                       # chasing is penalised
    return round(min(100.0, s), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — structural stop candidates & initial risk
# ══════════════════════════════════════════════════════════════════════════════

def stop_candidates(high, low, close, i, pivot, base_low, atr_val, cfg) -> dict:
    """Every point-in-time stop candidate + the rule used. Uses only bars ≤ i."""
    lo = np.asarray(low, float)
    swing_low = float(np.min(P.window(low, i, cfg.swing_low_lookback)))
    breakout_bar_low = float(lo[i])
    tight_low = float(np.min(P.window(low, i, 3)))
    short_low = float(np.min(P.window(low, i, cfg.swing_low_lookback)))
    pivot_minus_atr = (pivot - cfg.atr_stop_mult * atr_val
                       if (atr_val and np.isfinite(atr_val)) else float("nan"))
    return {
        "swing_low": swing_low,
        "breakout_bar_low": breakout_bar_low,
        "tight_range_low": tight_low,
        "short_lookback_low": short_low,
        "pivot_minus_atr": pivot_minus_atr,
        "base_support": base_low,
    }


def select_structural_stop(cands: dict, entry_ref: float, cfg) -> tuple:
    """Pick the primary structural stop: the HIGHEST candidate that is still below
    entry (tightest structurally justified risk). Returns (stop, rule_name).
    Deterministic; only signal-time candidates. None if no valid candidate."""
    valid = [(name, v) for name, v in cands.items()
             if v is not None and np.isfinite(v) and 0 < v < entry_ref]
    if not valid:
        return None, "none"
    name, stop = max(valid, key=lambda kv: kv[1])
    return float(stop), name


def risk_efficiency_score(initial_risk_pct: float, cfg) -> float:
    """Tighter (but not implausibly tight) structural risk scores higher."""
    if initial_risk_pct is None or not np.isfinite(initial_risk_pct):
        return 0.0
    if initial_risk_pct <= cfg.min_initial_risk_pct:
        return 50.0
    if initial_risk_pct >= cfg.max_initial_risk_pct:
        return 0.0
    span = cfg.max_initial_risk_pct - cfg.min_initial_risk_pct
    return round(100.0 * (1.0 - (initial_risk_pct - cfg.min_initial_risk_pct) / span), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — sector strength (context assembled PIT by the detector)
# ══════════════════════════════════════════════════════════════════════════════

def sector_strength_score(sector: dict, cfg) -> float:
    if not sector:
        return 0.0
    s = 0.0
    rs = sector.get("sector_rs_pct")
    if rs is not None and np.isfinite(rs):
        s += max(0.0, min(50.0, rs * 5.0))
    br = sector.get("breadth_pct_above_50dma")
    if br is not None and np.isfinite(br):
        s += min(50.0, br / 100.0 * 50.0)
    return round(min(100.0, s), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — participation
# ══════════════════════════════════════════════════════════════════════════════

def participation_features(volume, close, i, base_end_i, base_len, cfg,
                           delivery_pct=None) -> dict:
    return {
        "breakout_rvol": P.rel_volume(volume, i, cfg.vol_ref_window),
        "breakout_volume_z": P.volume_z(volume, i, cfg.vol_ref_window),
        "base_volume_dryup_ratio": P.volume_dryup(volume, base_end_i, base_len,
                                                  cfg.base_dryup_ref),
        # delivery is MISSING (not zero) unless real point-in-time data is supplied
        "delivery_pct": delivery_pct,
        "delivery_available": delivery_pct is not None,
    }


def participation_score(f: dict, cfg) -> float:
    s = 0.0
    rvol = f.get("breakout_rvol")
    if rvol is not None and np.isfinite(rvol):
        s += min(50.0, max(0.0, (rvol - 1.0) * 50.0))
    dry = f.get("base_volume_dryup_ratio")
    if dry is not None and np.isfinite(dry) and dry < 1.0:
        s += min(30.0, (1.0 - dry) * 60.0)
    if f.get("delivery_available") and f.get("delivery_pct") is not None:
        s += min(20.0, float(f["delivery_pct"]) / 100.0 * 20.0)
    return round(min(100.0, s), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Trend extension (chase risk)
# ══════════════════════════════════════════════════════════════════════════════

def trend_extension_features(close, high, i, pivot, atr_val, cfg) -> dict:
    c_i = float(np.asarray(close, float)[i])
    ext_atr = ((c_i - pivot) / atr_val) if (atr_val and np.isfinite(atr_val) and atr_val > 0) else float("nan")
    return {
        "extension_atr": ext_atr,
        "dist_from_50dma_pct": ((c_i / P.sma(close, i, 50) - 1.0) * 100.0
                                if np.isfinite(P.sma(close, i, 50)) and P.sma(close, i, 50) > 0
                                else float("nan")),
        "overextended": bool(np.isfinite(ext_atr) and ext_atr > cfg.max_extension_atr),
    }


def extension_risk_score(f: dict, cfg) -> float:
    """Higher = MORE chase risk (a risk score, not a quality score)."""
    ext = f.get("extension_atr")
    if ext is None or not np.isfinite(ext):
        return 0.0
    return round(min(100.0, max(0.0, ext / cfg.max_extension_atr * 100.0)), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — valuation CONTEXT (never a primary reject; fails closed)
# ══════════════════════════════════════════════════════════════════════════════

def valuation_features(valuation_record, observation_ts, cfg) -> tuple:
    """Return (features_dict, flags_list, valuation_ts).

    `valuation_record` is either None (no PIT valuation data — the current
    repository reality: `data/fundamentals_cache.db` has no publication dates) or a
    dict that MUST carry `available_ts` (the date the financials became public).
    Point-in-time rule: if `available_ts` is missing or AFTER the observation, the
    data is treated as UNAVAILABLE — never forward-filled into the past."""
    flags: list[str] = []
    if not valuation_record:
        return ({"available": False}, [OBS.FLAG_VALUATION_UNAVAILABLE], None)
    avail = valuation_record.get("available_ts")
    if not avail or str(avail) > str(observation_ts):
        # future/unknown publication date → cannot be used at the observation
        return ({"available": False}, [OBS.FLAG_VALUATION_UNAVAILABLE], None)
    age_days = valuation_record.get("age_days")
    if age_days is not None and age_days > 400:
        flags.append(OBS.FLAG_VALUATION_STALE)
    pe = valuation_record.get("pe")
    ps = valuation_record.get("price_to_sales")
    f = {
        "available": True,
        "pe": pe, "price_to_sales": ps,
        "ev_to_sales": valuation_record.get("ev_to_sales"),
        "market_cap_cr": valuation_record.get("market_cap_cr"),
        "sales_growth_pct": valuation_record.get("sales_growth_pct"),
        "earnings_growth_pct": valuation_record.get("earnings_growth_pct"),
        "pe_percentile_own": valuation_record.get("pe_percentile_own"),
    }
    if pe is not None and np.isfinite(pe) and pe >= cfg.extreme_pe:
        flags.append(OBS.FLAG_EXTREME_PE)
    if ps is not None and np.isfinite(ps) and ps >= cfg.extreme_ps:
        flags.append(OBS.FLAG_EXTREME_PS)
    if OBS.FLAG_EXTREME_PE in flags or OBS.FLAG_EXTREME_PS in flags:
        flags.append(OBS.FLAG_HIGH_EXPECTATION_RISK)
    return (f, flags, avail)


# ══════════════════════════════════════════════════════════════════════════════
# Weakening / exit-state features (pre-registered; used by the simulator only)
# ══════════════════════════════════════════════════════════════════════════════

def weakening_state(open_, high, low, close, volume, i, pivot, entry_ema_ref, cfg) -> dict:
    """Momentum-weakening events evaluated AT bar `i` (during a hold). These are
    computed from bars ≤ i and are pre-registered before the experiment runs; the
    simulator never peeks at the full trade to pick the best one."""
    c_i = float(np.asarray(close, float)[i])
    return {
        "close_below_pivot": bool(c_i < pivot),
        "close_below_20ema": bool(np.isfinite(P.ema(close, i, 20)) and c_i < P.ema(close, i, 20)),
        "close_below_50dma": bool(np.isfinite(P.sma(close, i, 50)) and c_i < P.sma(close, i, 50)),
        "high_volume_reversal": bool(P.clv(high, low, close, i) < -0.5
                                     and P.rel_volume(volume, i, cfg.vol_ref_window) > 1.5),
    }
