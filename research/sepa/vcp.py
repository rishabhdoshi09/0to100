"""Structural VCP — causal zigzag + last-contraction pivot.

SEPA-001's `detect_vcp_legacy` is frozen for timing comparison. The 001R
detector never back-dates a swing to before its confirmation bar and uses
the last confirmed contraction high as the actionable pivot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.sepa.config import (
    PIVOT_LAST_CONTRACTION,
    PIVOT_PATTERN_HIGH,
    SepaConfig,
)


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
    """Fractal swings (kept for tests / diagnostics). LOOK-AHEAD: a swing at i
    is only knowable at i+right. Do not use for eligibility timestamps."""
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
    confirmed = causal_zigzag(high, low, min_pct)
    sh = [s["index"] for s in confirmed if s["kind"] == "high"]
    sl = [s["index"] for s in confirmed if s["kind"] == "low"]
    return sh, sl


def causal_zigzag(high: np.ndarray, low: np.ndarray, min_pct: float) -> list[dict[str, Any]]:
    """Confirmed swings with the bar index on which they became knowable.

    The extreme lives at `index`; it is not known until `confirmed_index`.
    Unconfirmed running extremes are omitted — they are not swings yet.
    """
    n = len(high)
    out: list[dict[str, Any]] = []
    if n < 5 or min_pct <= 0:
        return out
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
                out.append({
                    "kind": "high",
                    "index": int(ext_i),
                    "price": float(ext_px),
                    "confirmed_index": int(i),
                })
                mode = "low"
                ext_i, ext_px = i, l
        else:
            if l <= ext_px:
                ext_i, ext_px = i, l
            elif ext_px > 0 and (h - ext_px) / ext_px * 100.0 >= min_pct:
                out.append({
                    "kind": "low",
                    "index": int(ext_i),
                    "price": float(ext_px),
                    "confirmed_index": int(i),
                })
                mode = "high"
                ext_i, ext_px = i, h
    return out


def _empty(reason: str = "INSUFFICIENT_HISTORY") -> dict[str, Any]:
    return {
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
        "pivot_knowable_date": None,
        "vcp_knowable_date": None,
        "base_start_date": None,
        "stop": None,
        "stop_index": None,
        "stop_date": None,
        "quality": None,
        "fail_reasons": [reason],
        "state": "NO_SETUP",
        "broken_out": False,
        "evidence": {},
    }


def _contractions_from_swings(
    swings: list[dict[str, Any]],
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    dates: list[str],
    min_rev: float,
    max_contractions: int | None = None,
) -> list[dict[str, Any]]:
    """All causally confirmed high→low contraction legs in the window.

    Do **not** stop at ``max_contractions``. Truncating here selected the
    *earliest* coils and made ``pivot_last_contraction_v1`` stale.
    ``max_contractions`` is accepted for call-site compatibility and ignored.
    """
    del max_contractions
    sh = [s for s in swings if s["kind"] == "high"]
    sl = [s for s in swings if s["kind"] == "low"]
    contractions: list[dict[str, Any]] = []
    lo_i = 0
    for hs in sh:
        h_idx = int(hs["index"])
        while lo_i < len(sl) and int(sl[lo_i]["index"]) <= h_idx:
            lo_i += 1
        if lo_i >= len(sl):
            break
        ls = sl[lo_i]
        l_idx = int(ls["index"])
        h_px = float(hs["price"])
        l_px = float(ls["price"])
        if h_px <= 0 or l_px >= h_px:
            lo_i += 1
            continue
        depth = (h_px - l_px) / h_px * 100.0
        if depth < min_rev:
            lo_i += 1
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
            "high_knowable_date": dates[int(hs["confirmed_index"])],
            "low_knowable_date": dates[int(ls["confirmed_index"])],
            "high_confirmed_index": int(hs["confirmed_index"]),
            "low_confirmed_index": int(ls["confirmed_index"]),
            "duration": int(l_idx - h_idx),
        })
        lo_i += 1
    return contractions


def _tightening_fail_reasons(seq: list[dict[str, Any]], config: SepaConfig) -> list[str]:
    if len(seq) < 2:
        return []
    depths = [c["depth"] for c in seq]
    fail: list[str] = []
    expanding = sum(1 for a, b in zip(depths, depths[1:]) if b > a * config.depth_expand_tol)
    if depths[-1] > depths[0] * config.final_vs_first:
        fail.append("NOT_TIGHTENING")
    if expanding > max(0, len(depths) - 2):
        fail.append("EXPANDING_PULLBACKS")
    if depths[-1] > config.max_final_depth_pct:
        fail.append("FINAL_CONTRACTION_LOOSE")
    return fail


def select_active_sequence(
    contractions: list[dict[str, Any]],
    config: SepaConfig,
) -> list[dict[str, Any]]:
    """Most recent consecutive contractions ending at the latest low.

    Windows are always anchored on the last confirmed contraction. An older
    valid coil is never preferred over a newer invalid live structure.
    Among windows that pass tightening, the longest (≤ max_contractions) wins.
    """
    if not contractions:
        return []
    n = len(contractions)
    max_n = max(1, int(config.max_contractions))
    min_n = max(1, int(config.min_contractions))
    if n < min_n:
        return list(contractions)
    for k in range(min(max_n, n), min_n - 1, -1):
        cand = contractions[-k:]
        if not _tightening_fail_reasons(cand, config):
            return cand
    return contractions[-min(max_n, n):]


def _choose_pivot(seq: list[dict[str, Any]], pivot_version: str) -> dict[str, Any]:
    if pivot_version == PIVOT_PATTERN_HIGH:
        return max(seq, key=lambda c: c["high"])
    return seq[-1]


def _state_for(
    *,
    fail: list[str],
    seq: list[dict[str, Any]],
    min_n: int,
    price: float,
    pivot: float | None,
    stop: float | None,
    zone_lo: float | None,
    zone_hi: float | None,
) -> str:
    n = len(seq)
    if fail and any(r in fail for r in (
        "BROKEN_STRUCTURE", "BASE_TOO_DEEP", "NOT_TIGHTENING",
        "EXPANDING_PULLBACKS", "FINAL_CONTRACTION_LOOSE", "VOLUME_EXPANDING",
    )):
        if n < min_n:
            return "BASE_FORMING" if n == 1 else "NO_SETUP"
        return "FAILED"
    if n <= 0:
        return "NO_SETUP"
    if n == 1:
        return "CONTRACTION_1"
    if n < min_n:
        return "BASE_FORMING"
    if pivot is None:
        return "VCP_FORMING"
    if stop is not None and price < float(stop):
        return "FAILED"
    if zone_hi is not None and price > float(zone_hi):
        return "EXTENDED"
    if zone_lo is not None and zone_hi is not None and float(zone_lo) <= price <= float(zone_hi):
        return "ENTRY_READY"
    if price > float(pivot):
        return "BROKEN_OUT"
    return "PIVOT_DEFINED"


def _evaluate_structure(
    contractions: list[dict[str, Any]],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    vol: np.ndarray,
    dates: list[str],
    offset: int,
    config: SepaConfig,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    fail: list[str] = []
    if len(contractions) < config.min_contractions:
        fail.append("TOO_FEW_CONTRACTIONS")
        empty = _empty("TOO_FEW_CONTRACTIONS")
        empty["fail_reasons"] = fail
        empty["evidence"] = evidence
        empty["contraction_count"] = len(contractions)
        empty["depths"] = [round(c["depth"], 3) for c in contractions]
        empty["state"] = "CONTRACTION_1" if len(contractions) == 1 else "NO_SETUP"
        if contractions:
            empty["base_start_date"] = contractions[0]["high_date"]
        return empty

    seq = select_active_sequence(contractions, config)
    depths = [c["depth"] for c in seq]
    fail.extend(_tightening_fail_reasons(seq, config))

    first_i = seq[0]["high_index"]
    last_low_i = seq[-1]["low_index"]
    pattern_high = max(seq, key=lambda c: c["high"])
    last_c = seq[-1]
    pivot_c = _choose_pivot(seq, config.pivot_version)
    pivot_i = int(pivot_c["high_index"])
    pivot = float(pivot_c["high"])
    stop_i = int(seq[-1]["low_index"])
    stop = float(seq[-1]["low"])

    end_i = max(last_low_i, pivot_i)
    window_high = float(np.nanmax(high[first_i: end_i + 1]))
    window_low = float(np.nanmin(low[first_i: end_i + 1]))
    base_depth = (window_high - window_low) / window_high * 100.0 if window_high > 0 else None
    if base_depth is not None and base_depth > config.max_base_depth_pct:
        fail.append("BASE_TOO_DEEP")

    price = float(close[-1])
    far_below = bool(pivot > 0 and price < pivot * config.near_pivot_frac)
    if far_below and config.fail_vcp_if_far_below_pivot:
        fail.append("TOO_FAR_BELOW_PIVOT")
    if price < stop:
        fail.append("BROKEN_STRUCTURE")

    vol_first = float(seq[0]["volume"])
    vol_final = float(seq[-1]["volume"])
    dry = (vol_final / vol_first) if vol_first > 0 else None
    if (
        config.volume_dry_up_required
        and dry is not None
        and dry > config.volume_dry_up_max
    ):
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

    zone_lo = pivot * (1.0 - config.buy_zone_below_pct / 100.0)
    zone_hi = pivot * (1.0 + config.buy_zone_above_pct / 100.0)
    detected = len(fail) == 0
    last_bar = len(close) - 1
    broken_out = bool(pivot > 0 and (price > pivot or float(high[last_bar]) > pivot))
    state = _state_for(
        fail=fail, seq=seq, min_n=config.min_contractions, price=price,
        pivot=pivot, stop=stop, zone_lo=zone_lo, zone_hi=zone_hi,
    )
    if detected and state == "FAILED":
        state = "PIVOT_DEFINED"

    vcp_knowable = last_c.get("low_knowable_date")
    pivot_knowable = pivot_c.get("high_knowable_date")
    pivot_type = (
        "last_contraction_resistance"
        if config.pivot_version == PIVOT_LAST_CONTRACTION
        else "vcp_resistance_swing_high"
    )
    evidence = dict(evidence)
    evidence.update({
        "raw_contraction_count": len(contractions),
        "active_sequence_count": len(seq),
        "far_below_pivot": far_below,
        "pattern_high": round(float(pattern_high["high"]), 4),
        "pattern_high_date": pattern_high["high_date"],
        "last_contraction_high": round(float(last_c["high"]), 4),
        "last_contraction_high_date": last_c["high_date"],
        "active_last_high_date": last_c["high_date"],
        "pivot_version": config.pivot_version,
        "volume_dry_up_required": config.volume_dry_up_required,
        "legacy_too_far_below_would_fail": far_below,
        "min_recovery_bounce_unused": True,
    })
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
        "pivot_type": pivot_type,
        "pivot_knowable_date": pivot_knowable,
        "vcp_knowable_date": vcp_knowable,
        "base_start_date": seq[0]["high_date"],
        "stop": round(stop, 4),
        "stop_index": int(stop_i + offset),
        "stop_date": dates[stop_i],
        "quality": round(quality, 2),
        "fail_reasons": fail,
        "state": state,
        "broken_out": broken_out,
        "evidence": evidence,
    }


def detect_vcp(frame, config: SepaConfig) -> dict[str, Any]:
    """Causal VCP snapshot using only bars in `frame` (caller must slice as-of)."""
    if frame is None or len(frame) < 40:
        return _empty()

    lookback = min(int(getattr(config, "vcp_lookback", 120)), len(frame))
    window = frame.iloc[-lookback:]
    offset = len(frame) - lookback
    high = _col(window, "high", "close")
    low = _col(window, "low", "close")
    close = _col(window, "close")
    vol = _col(window, "volume", "close") if "volume" in window.columns else np.ones(len(window))
    dates = _dates(window)

    min_rev = float(getattr(config, "min_reversal_pct", 2.5))
    swings = causal_zigzag(high, low, min_rev)
    evidence = {
        "swing_highs": sum(1 for s in swings if s["kind"] == "high"),
        "swing_lows": sum(1 for s in swings if s["kind"] == "low"),
        "lookback": lookback,
        "min_reversal_pct": min_rev,
        "causal": True,
        "vcp_version": config.vcp_version,
    }
    if evidence["swing_highs"] < 1 or evidence["swing_lows"] < 1:
        empty = _empty("NO_SWING_STRUCTURE")
        empty["fail_reasons"] = ["NO_SWING_STRUCTURE"]
        empty["evidence"] = evidence
        return empty

    contractions = _contractions_from_swings(
        swings, high, low, vol, dates, min_rev, config.max_contractions,
    )
    evidence["raw_contraction_count"] = len(contractions)
    return _evaluate_structure(
        contractions, high, low, close, vol, dates, offset, config, evidence,
    )


def detect_vcp_legacy(frame, config: SepaConfig) -> dict[str, Any]:
    """Frozen SEPA-001 detector: zigzag + pattern-high pivot + 92% near-pivot fail."""
    from dataclasses import replace

    legacy = replace(
        config,
        vcp_version="vcp_swing_v1",
        pivot_version=PIVOT_PATTERN_HIGH,
        fail_vcp_if_far_below_pivot=True,
    )
    out = detect_vcp(frame, legacy)
    if out.get("pivot") is not None:
        out["pivot_type"] = "vcp_resistance_swing_high"
    out["evidence"] = dict(out.get("evidence") or {})
    out["evidence"]["legacy"] = True
    out["evidence"]["causal"] = True
    return out
