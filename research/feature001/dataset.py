"""FEATURE-001 replay: attach frozen Trend+RS vectors to scanner fires.

Research only. Identity calibration (no live/backtest leak). Official bhav.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from research.feature001.constants import (
    EXPERIMENT,
    FAMILY_CATEGORY,
    HORIZON,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    OUT_DIR,
    RS_DELTA_SESSIONS,
    RS_VERSION,
    SAMPLE_STEP,
    SCAN_LOOKBACK,
    TREND_VERSION,
)
from research.feature001.rs_features import compute_rs_features
from research.feature001.trend_features import compute_trend_features
from research.sepa.config import DEFAULT_CONFIG
from research.sepa.frames import iso_date


def identity_scanner():
    """Production detector, unit weights — do not use today's calib on history."""
    from scan.unified_scanner import UnifiedScanner

    sc = UnifiedScanner()
    sc._calib = {}
    sc._regime_calib = {}
    sc._regime = ""
    return sc


def _norm(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    x = (float(val) - lo) / (hi - lo)
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def mom_score_of(hit) -> float:
    mom5 = float(getattr(hit, "momentum_5d", 0.0) or 0.0)
    rsi = float(getattr(hit, "rsi", 50.0) or 50.0)
    vratio = float(getattr(hit, "volume_ratio", 1.0) or 1.0)
    return (_norm(mom5, -5, 10) * 0.40 + _norm(rsi, 30, 70) * 0.35
            + _norm(vratio, 0.5, 3.0) * 0.25) * 100.0


def _nifty_close_local() -> pd.Series | None:
    """Official Nifty 50 already on disk. Never triggers a network build."""
    try:
        from data import index_store
        index_store.build_from_local()
        with index_store._lock:
            df = index_store._store.get("Nifty 50")
        if df is None or len(df) == 0:
            return None
        col = "Close" if "Close" in df.columns else "close"
        s = pd.to_numeric(df[col], errors="coerce")
        idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        s.index = idx
        return s.dropna()
    except Exception:
        return None


def _bench_rel_63(close: pd.Series, nifty: pd.Series | None, as_of: str) -> float | None:
    if nifty is None or close is None or len(close) < 64:
        return None
    try:
        c0 = float(close.iloc[-1])
        c1 = float(close.iloc[-64])
        if c1 <= 0:
            return None
        cutoff = pd.Timestamp(as_of)
        n = nifty[nifty.index <= cutoff]
        if len(n) < 64:
            return None
        n0 = float(n.iloc[-1])
        n1 = float(n.iloc[-64])
        if n1 <= 0:
            return None
        return (c0 / c1 - 1.0) - (n0 / n1 - 1.0)
    except Exception:
        return None


def _calendar(fast) -> list[str]:
    if not fast._dates:
        return []
    all_ns = np.unique(np.concatenate(fast._dates))
    return [str(pd.Timestamp(int(ns)).date()) for ns in all_ns]


def _simulate(entry: float, stop: float, target: float,
              fwd_high, fwd_low, fwd_close) -> dict[str, Any]:
    from scan.signal_backtest import _BT_BREAKEVEN_PCT, _simulate_timed

    outcome, r_mult, e_off, x_off = _simulate_timed(
        entry, stop, target, fwd_high, fwd_low, fwd_close, be_pct=_BT_BREAKEVEN_PCT
    )
    risk = entry - stop
    if outcome == "NO_FILL" or e_off < 0 or risk <= 0:
        return {"outcome": "NO_FILL", "gross_r": 0.0, "net_r": 0.0, "cost_r": 0.0,
                "mae_r": None, "mfe_r": None, "stop_before_1r": None,
                "hit_1r": None, "hit_2r": None, "holding": 0}
    try:
        from core.costs import cost_in_r
        cost_r = float(cost_in_r(risk / entry, "CNC"))
    except Exception:
        cost_r = 0.0
    wh = np.asarray(fwd_high[e_off: x_off + 1], dtype=float)
    wl = np.asarray(fwd_low[e_off: x_off + 1], dtype=float)
    mfe = float((float(np.nanmax(wh)) - entry) / risk) if wh.size else None
    mae = float((entry - float(np.nanmin(wl))) / risk) if wl.size else None
    return {
        "outcome": outcome,
        "gross_r": float(r_mult),
        "net_r": float(r_mult) - cost_r,
        "cost_r": cost_r,
        "mae_r": mae,
        "mfe_r": mfe,
        "stop_before_1r": bool(outcome == "LOSS" and (mfe is None or mfe < 1.0)),
        "hit_1r": bool(mfe is not None and mfe >= 1.0),
        "hit_2r": bool(mfe is not None and mfe >= 2.0),
        "holding": int(x_off - e_off + 1),
    }


def replay(*, max_dates: int | None = None, progress_every: int = 10) -> dict[str, Any]:
    from research.sepa.universe_pit import FastInvestable, load_store_frames
    from research.sepa003.fastrs import FastRS

    t0 = time.time()
    print("FEATURE-001 load_store_frames", flush=True)
    frames = load_store_frames(min_bars=MIN_SESSIONS)
    fast = FastInvestable(frames)
    frs = FastRS(fast, DEFAULT_CONFIG)
    nifty = _nifty_close_local()
    sc = identity_scanner()
    calendar = _calendar(fast)
    if len(calendar) < MIN_SESSIONS + HORIZON + SAMPLE_STEP:
        raise RuntimeError("official history too short for FEATURE-001")
    sample = calendar[MIN_SESSIONS: len(calendar) - HORIZON: SAMPLE_STEP]
    if max_dates is not None:
        sample = sample[: int(max_dates)]

    events: list[dict[str, Any]] = []
    n_analyzed = 0
    n_fires = 0
    rs_cache: dict[str, dict] = {}
    date_index = {d: i for i, d in enumerate(calendar)}

    for di, as_of in enumerate(sample):
        snap = fast.snapshot(
            as_of,
            min_price=MIN_PRICE,
            min_turnover=MIN_TURNOVER,
            min_sessions=MIN_SESSIONS,
        )
        universe = snap.investable
        if as_of not in rs_cache:
            rs_cache[as_of] = frs.table(as_of, universe)
        prev_as_of = None
        i0 = date_index.get(as_of)
        if i0 is not None and i0 >= RS_DELTA_SESSIONS:
            prev_as_of = calendar[i0 - RS_DELTA_SESSIONS]
            if prev_as_of not in rs_cache:
                prev_snap = fast.snapshot(
                    prev_as_of,
                    min_price=MIN_PRICE,
                    min_turnover=MIN_TURNOVER,
                    min_sessions=MIN_SESSIONS,
                )
                rs_cache[prev_as_of] = frs.table(prev_as_of, prev_snap.investable)
        table = rs_cache[as_of]
        prev_table = rs_cache.get(prev_as_of) if prev_as_of else None

        for sym in universe:
            hist, fwd = fast.hist_fwd(sym, as_of, HORIZON)
            if hist is None or fwd is None or len(hist) < MIN_SESSIONS or len(fwd) < 5:
                continue
            hist_scan = hist.iloc[-SCAN_LOOKBACK:] if len(hist) > SCAN_LOOKBACK else hist
            n_analyzed += 1
            try:
                hit = sc._analyze(sym, hist_scan)
            except Exception:
                continue
            if hit is None or not getattr(hit, "signals", None):
                continue
            risk = float(hit.entry) - float(hit.stop)
            if risk <= 0:
                continue
            trend = compute_trend_features(hist_scan)
            close = hist_scan["close"] if "close" in hist_scan.columns else None
            rs = compute_rs_features(
                sym, table, prev_table=prev_table,
                bench_rel_63=_bench_rel_63(close, nifty, as_of) if close is not None else None,
            )
            sim = _simulate(
                float(hit.entry), float(hit.stop), float(hit.target),
                fwd["high"].to_numpy(dtype=float),
                fwd["low"].to_numpy(dtype=float),
                fwd["close"].to_numpy(dtype=float),
            )
            if sim["outcome"] == "NO_FILL":
                continue
            n_fires += 1
            events.append({
                "experiment": EXPERIMENT,
                "symbol": str(sym).upper(),
                "as_of": iso_date(as_of),
                "year": iso_date(as_of)[:4],
                "signals": list(hit.signals),
                "n_signals": len(hit.signals),
                "score": float(hit.score),
                "verdict": hit.verdict,
                "chase_risk": bool(hit.chase_risk),
                "momentum_5d": float(hit.momentum_5d),
                "rsi": float(hit.rsi),
                "volume_ratio": float(hit.volume_ratio),
                "mom_score": round(mom_score_of(hit), 4),
                "above_sma50": bool(hit.above_sma50),
                "above_sma200": bool(hit.above_sma200),
                "price": float(hit.price),
                "entry": float(hit.entry),
                "stop": float(hit.stop),
                "target": float(hit.target),
                "trend": trend,
                "rs": rs,
                **sim,
            })
        if progress_every and (di + 1) % progress_every == 0:
            print(
                f"FEATURE-001 dates {di + 1}/{len(sample)} fires={n_fires} "
                f"analyzed={n_analyzed} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    meta = {
        "experiment": EXPERIMENT,
        "claim_class": "EXPLANATORY",
        "n_frames": len(frames),
        "n_sample_dates": len(sample),
        "n_analyzed": n_analyzed,
        "n_filled_events": len(events),
        "sample_step": SAMPLE_STEP,
        "horizon": HORIZON,
        "scan_lookback": SCAN_LOOKBACK,
        "min_sessions": MIN_SESSIONS,
        "min_price": MIN_PRICE,
        "min_turnover": MIN_TURNOVER,
        "trend_version": TREND_VERSION,
        "rs_version": RS_VERSION,
        "rs_source": "rs_cs_v1",
        "identity_calibration": True,
        "nifty_bench_available": nifty is not None,
        "first_date": sample[0] if sample else None,
        "last_date": sample[-1] if sample else None,
        "seconds": round(time.time() - t0, 1),
        "note": (
            "Shared 5-session grid over full official history. "
            "Not the production last-250-session nightly window. "
            "Not a production change."
        ),
    }
    return {"events": events, "meta": meta}


def explode_families(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        for sig in ev.get("signals") or []:
            rows.append({
                **{k: v for k, v in ev.items() if k != "trend" and k != "rs"},
                "family": sig,
                "family_category": FAMILY_CATEGORY.get(sig, ""),
                "n_structure_passed": (ev.get("trend") or {}).get("n_structure_passed"),
                "structure_pass": (ev.get("trend") or {}).get("structure_pass"),
                "trend_bucket": (ev.get("trend") or {}).get("trend_bucket"),
                "pct_above_sma200": (ev.get("trend") or {}).get("pct_above_sma200"),
                "sma200_slope_pct": (ev.get("trend") or {}).get("sma200_slope_pct"),
                "ma_spread_50_200_pct": (ev.get("trend") or {}).get("ma_spread_50_200_pct"),
                "dist_from_52w_high_pct": (ev.get("trend") or {}).get("dist_from_52w_high_pct"),
                "rs_percentile": (ev.get("rs") or {}).get("rs_percentile"),
                "rs_score": (ev.get("rs") or {}).get("rs_score"),
                "rs_ge_70": (ev.get("rs") or {}).get("rs_ge_70"),
                "rs_bucket": (ev.get("rs") or {}).get("rs_bucket"),
                "rs_pct_chg_21": (ev.get("rs") or {}).get("rs_pct_chg_21"),
                "r63": (ev.get("rs") or {}).get("r63"),
                "r126": (ev.get("rs") or {}).get("r126"),
                "r189": (ev.get("rs") or {}).get("r189"),
                "r252": (ev.get("rs") or {}).get("r252"),
                "bench_rel_63": (ev.get("rs") or {}).get("bench_rel_63"),
                "trend": ev.get("trend"),
                "rs": ev.get("rs"),
            })
    return rows


def persist(payload: dict[str, Any]) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev_path = OUT_DIR / "feature_001_events.jsonl"
    fam_path = OUT_DIR / "feature_001_family_rows.jsonl"
    meta_path = OUT_DIR / "feature_001_dataset_meta.json"
    with ev_path.open("w") as f:
        for ev in payload["events"]:
            f.write(json.dumps(ev, default=str) + "\n")
    rows = explode_families(payload["events"])
    with fam_path.open("w") as f:
        for row in rows:
            slim = {k: v for k, v in row.items() if k not in {"trend", "rs"}}
            f.write(json.dumps(slim, default=str) + "\n")
    meta = dict(payload["meta"])
    meta["n_family_rows"] = len(rows)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return {
        "events": str(ev_path),
        "family_rows": str(fam_path),
        "meta": str(meta_path),
    }


def load_events(path=None) -> list[dict[str, Any]]:
    p = path or (OUT_DIR / "feature_001_events.jsonl")
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
