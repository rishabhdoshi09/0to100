"""Structural VCP from zigzag swing legs — not shrinking calendar windows."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.sepa.config import SepaConfig


def _col(frame, name: str, fallback: str | None = None) -> np.ndarray:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if fallback and fallback in frame.columns:
        return pd.to_numeric(frame[fallback], errors="coerce").to_numpy(dtype=float)
    raise KeyError(name)


def _dates(frame) -> list[str]:
    out = []
    for ts in frame.index:
        try:
            out.append(str(pd.Timestamp(ts).date()))
        except Exception:
            out.append(str(ts))
    return out


def find_swings(high: np.ndarray, low: np.ndarray, left: int, right: int) -> tuple[list[int], list[int]]:
    """Fractal swings (kept for tests / diagnostics). Prefer zigzag_swings for VCP."""
    n = len(high)
    sh, sl = [], []
    if n < left + right + 1:
        return sh, sl
    for i in range(left, n - right):
        hw = high[i - left: i + right + 1]
        lw = low[i - left: i + right + 1]
        if np.isnan(high[i]) or np.isnan(low[i]):
            continue
        if high[i] >= np.nanmax(hw) and high[i] > np.nanmax(hw[:left]):
            sh.append(i)
        if low[i] <= np.nanmin(lw) and low[i] < np.nanmin(lw[:left]):
            sl.append(i)
    return sh, sl


def zigzag_swings(high: np.ndarray, low: np.ndarray, min_pct: float) -> tuple[list[int], list[int]]:
    """Percentage zigzag: a swing exists only after a min_pct reversal."""
    n = len(high)
    sh, sl = [], []
    if n < 5 or min_pct <= 0:
        return sh, sl
    mode = "high"
    ext_i = 0
    ext_px = float(high[0])
    for i in range(1, n):
        h = float(high[i])
        l = float(low[i])
        if mode == "high":
            if h >= ext_px:
                ext_i, ext_px = i, h
            elif ext_px > 0 and (ext_px - l) / ext_px * 100.0 >= min_pct:
                sh.append(ext_i)
                mode = "low"
                ext_i, ext_px = i, l
        else:
            if l <= ext_px:
                ext_i, ext_px = i, l
            elif ext_px > 0 and (h - ext_px) / ext_px * 100.0 >= min_pct:
                sl.append(ext_i)
                mode = "high"
                ext_i, ext_px = i, h
    return sh, sl


def detect_vcp(frame, config: SepaConfig) -> dict[str, Any]:
    empty = {
        "detected": False,
        "contraction_count": 0,
        "depths": [],
        "dates": [],
        "durations": [],
        "highs": [],
        "lows": [],
        "base_depth_pct": None,
        "final_depth_pct": None,
        "tightness": None,
        "vol_first": None,
        "vol_final": None,
        "vol_recent_vs_base": None,
        "dry_up_ratio": None,
        "pivot": None,
        "pivot_index": None,
        "pivot_date": None,
        "pivot_type": None,
        "stop": None,
        "stop_index": None,
        "stop_date": None,
        "quality": None,
        "fail_reasons": ["INSUFFICIENT_HISTORY"],
        "evidence": {},
    }
    if frame is None or len(frame) < 40:
        return empty

    lookback = min(int(getattr(config, "vcp_lookback", 120)), len(frame))
    window = frame.iloc[-lookback:]
    offset = len(frame) - lookback
    high = _col(window, "high", "close")
    low = _col(window, "low", "close")
    close = _col(window, "close")
    vol = _col(window, "volume", "close") if "volume" in window.columns else np.ones(len(window))
    dates = _dates(window)

    min_rev = float(getattr(config, "min_reversal_pct", 2.5))
    sh, sl = zigzag_swings(high, low, min_rev)
    evidence = {
        "swing_highs": len(sh),
        "swing_lows": len(sl),
        "lookback": lookback,
        "min_reversal_pct": min_rev,
    }
    if len(sh) < 1 or len(sl) < 1:
        empty["fail_reasons"] = ["NO_SWING_STRUCTURE"]
        empty["evidence"] = evidence
        return empty

    contractions: list[dict[str, Any]] = []
    lo_i = 0
    for h_idx in sh:
        while lo_i < len(sl) and sl[lo_i] <= h_idx:
            lo_i += 1
        if lo_i >= len(sl):
            break
        l_idx = sl[lo_i]
        h_px = float(high[h_idx])
        l_px = float(low[l_idx])
        if h_px <= 0 or l_px >= h_px:
            continue
        depth = (h_px - l_px) / h_px * 100.0
        if depth < min_rev:
            continue
        vmean = float(np.nanmean(vol[h_idx: l_idx + 1])) if l_idx >= h_idx else float("nan")
        contractions.append({
            "high_index": h_idx,
            "low_index": l_idx,
            "high": h_px,
            "low": l_px,
            "depth": depth,
            "volume": vmean,
            "high_date": dates[h_idx],
            "low_date": dates[l_idx],
            "duration": int(l_idx - h_idx),
        })
        lo_i += 1
        if len(contractions) >= config.max_contractions:
            break

    evidence["raw_contraction_count"] = len(contractions)
    fail: list[str] = []
    if len(contractions) < config.min_contractions:
        fail.append("TOO_FEW_CONTRACTIONS")
        empty["fail_reasons"] = fail
        empty["evidence"] = evidence
        empty["contraction_count"] = len(contractions)
        empty["depths"] = [round(c["depth"], 3) for c in contractions]
        return empty

    seq = contractions[-config.max_contractions:]
    depths = [c["depth"] for c in seq]
    expanding = sum(1 for a, b in zip(depths, depths[1:]) if b > a * config.depth_expand_tol)
    if depths[-1] > depths[0] * config.final_vs_first:
        fail.append("NOT_TIGHTENING")
    if expanding > max(0, len(depths) - 2):
        fail.append("EXPANDING_PULLBACKS")
    if depths[-1] > config.max_final_depth_pct:
        fail.append("FINAL_CONTRACTION_LOOSE")

    first_i = seq[0]["high_index"]
    last_low_i = seq[-1]["low_index"]
    # Pivot is the setup resistance: highest contraction swing high, not a later chase print.
    pivot_local = max(seq, key=lambda c: c["high"])
    pivot_i = int(pivot_local["high_index"])
    pivot = float(pivot_local["high"])
    stop_i = int(seq[-1]["low_index"])
    stop = float(seq[-1]["low"])

    end_i = max(last_low_i, pivot_i)
    window_high = float(np.nanmax(high[first_i: end_i + 1]))
    window_low = float(np.nanmin(low[first_i: end_i + 1]))
    base_depth = (window_high - window_low) / window_high * 100.0 if window_high > 0 else None
    if base_depth is not None and base_depth > config.max_base_depth_pct:
        fail.append("BASE_TOO_DEEP")

    price = float(close[-1])
    if pivot > 0 and price < pivot * config.near_pivot_frac:
        fail.append("TOO_FAR_BELOW_PIVOT")
    if price < stop:
        fail.append("BROKEN_STRUCTURE")

    vol_first = float(seq[0]["volume"])
    vol_final = float(seq[-1]["volume"])
    dry = (vol_final / vol_first) if vol_first > 0 else None
    if dry is not None and dry > config.volume_dry_up_max:
        fail.append("VOLUME_EXPANDING")
    base_vol = float(np.nanmean(vol[first_i: end_i + 1]))
    recent_vol = float(np.nanmean(vol[-10:])) if len(vol) >= 10 else float(np.nanmean(vol))
    recent_vs_base = (recent_vol / base_vol) if base_vol > 0 else None
    tightness = depths[-1] / depths[0] if depths[0] > 0 else None

    quality = 50.0
    if not fail:
        quality = 70.0 + min(20.0, 5.0 * (len(seq) - 2))
        if tightness is not None:
            quality += max(0.0, (1.0 - tightness) * 10.0)
        if dry is not None:
            quality += max(0.0, (1.0 - min(dry, 1.0)) * 10.0)
        quality = min(100.0, quality)

    detected = len(fail) == 0
    return {
        "detected": detected,
        "contraction_count": len(seq),
        "depths": [round(d, 3) for d in depths],
        "dates": [c["low_date"] for c in seq],
        "durations": [int(c["duration"]) for c in seq],
        "highs": [round(c["high"], 4) for c in seq],
        "lows": [round(c["low"], 4) for c in seq],
        "base_depth_pct": None if base_depth is None else round(base_depth, 3),
        "final_depth_pct": round(depths[-1], 3),
        "tightness": None if tightness is None else round(tightness, 4),
        "vol_first": round(vol_first, 2),
        "vol_final": round(vol_final, 2),
        "vol_recent_vs_base": None if recent_vs_base is None else round(recent_vs_base, 4),
        "dry_up_ratio": None if dry is None else round(dry, 4),
        "pivot": round(pivot, 4),
        "pivot_index": int(pivot_i + offset),
        "pivot_date": dates[pivot_i],
        "pivot_type": "vcp_resistance_swing_high",
        "stop": round(stop, 4),
        "stop_index": int(stop_i + offset),
        "stop_date": dates[stop_i],
        "quality": round(quality, 2),
        "fail_reasons": fail,
        "evidence": evidence,
    }
