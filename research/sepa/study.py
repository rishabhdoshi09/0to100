"""SEPA-001R full study orchestration (research only)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from research.sepa.ablation_r import persist_r, run_ablation_r, walk_forward_split
from research.sepa.config import PIVOT_LAST_CONTRACTION, PIVOT_PATTERN_HIGH, SepaConfig
from research.sepa.integrity import research_integrity_report
from research.sepa.rs import build_rs_table, score_one
from research.sepa.synthetic import RESEARCH_CFG, plant_vcp, stage2
from research.sepa.timing import diagnose_symbol


def synthetic_panel() -> dict[str, pd.DataFrame]:
    """A small causal panel: leaders, grinders, failed VCPs, extended coils."""
    return {
        "LEADER": plant_vcp(contractions="tight", volume="dry"),
        "LEADER2": plant_vcp(contractions="two", volume="dry"),
        "LEADER3": plant_vcp(contractions="tight", volume="dry", extend=0.004),
        "EXTENDED": plant_vcp(contractions="tight", volume="dry", extend=0.08),
        "WIDE": plant_vcp(contractions="tight", volume="dry", wide_stop=True),
        "WIDEN": plant_vcp(contractions="widening", volume="dry"),
        "DEEP": plant_vcp(contractions="deep", volume="dry"),
        "GRIND": stage2(),
        "GRIND2": stage2(n=300, start=40.0, step=0.35),
        "EXPANDVOL": plant_vcp(contractions="tight", volume="expand"),
    }


def rs_bucket_study(frames: dict[str, pd.DataFrame], config: SepaConfig, horizon: int = 20) -> dict[str, Any]:
    """Forward 20-session return by RS percentile bucket — independent of VCP gates."""
    buckets = {
        "50-69": (50.0, 70.0),
        "70-79": (70.0, 80.0),
        "80-89": (80.0, 90.0),
        "90-94": (90.0, 95.0),
        "95-99": (95.0, 100.1),
    }
    acc: dict[str, list[float]] = {k: [] for k in buckets}
    names = list(frames)
    # sample every 10th bar to keep this diagnostic cheap
    for sym, df in frames.items():
        n = len(df)
        start = max(260, n - 400)
        for t in range(start, n - horizon, 10):
            hist = df.iloc[:t]
            fwd = df.iloc[t: t + horizon]
            table = build_rs_table(frames, hist.index[-1], config, universe=names)
            pct = (table.get("percentiles") or {}).get(sym)
            if pct is None:
                continue
            r0 = float(hist["close"].iloc[-1])
            r1 = float(fwd["close"].iloc[-1])
            if r0 <= 0:
                continue
            fwd_ret = r1 / r0 - 1.0
            for label, (lo, hi) in buckets.items():
                if lo <= pct < hi:
                    acc[label].append(fwd_ret)
    out = {}
    for label, xs in acc.items():
        arr = np.array(xs, dtype=float)
        out[label] = {
            "n": int(arr.size),
            "mean_fwd_20d": None if arr.size == 0 else round(float(arr.mean()) * 100.0, 3),
            "median_fwd_20d": None if arr.size == 0 else round(float(np.median(arr)) * 100.0, 3),
        }
    return out


def pivot_compare(frames, config: SepaConfig, **kwargs) -> dict[str, Any]:
    a = run_ablation_r(
        frames=frames, config=replace(config, pivot_version=PIVOT_LAST_CONTRACTION),
        variants=("F",), **kwargs,
    )
    b = run_ablation_r(
        frames=frames, config=replace(config, pivot_version=PIVOT_PATTERN_HIGH),
        variants=("F",), **kwargs,
    )
    def _pack(payload):
        f = payload["variants"]["F"]
        return {"n": f.get("n"), "expectancy_r": f.get("expectancy_r"),
                "unique_setups": payload["sample"].get("unique_setups"),
                "fill_attempts": f.get("fill_attempt_counts")}
    return {
        "last_contraction": _pack(a),
        "pattern_high": _pack(b),
        "rationale": (
            "last_contraction is the resistance the coil is under; "
            "pattern_high is the SEPA-001 earliest/highest high. "
            "Chosen for structure, not max R."
        ),
    }


def buyzone_study(frames, config: SepaConfig, **kwargs) -> dict[str, Any]:
    out = {}
    for width in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        payload = run_ablation_r(
            frames=frames, config=config, variants=("F",),
            buy_zone_above_pct=width, **kwargs,
        )
        out[str(width)] = payload["variants"]["F"]
    return out


def vcp_component_study(frames, config: SepaConfig, **kwargs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in (2, 3):
        payload = run_ablation_r(
            frames=frames, config=replace(config, min_contractions=n),
            variants=("F",), **kwargs,
        )
        out[f"min_contractions_{n}"] = payload["variants"]["F"]
    payload = run_ablation_r(
        frames=frames, config=replace(config, volume_dry_up_required=False),
        variants=("F",), **kwargs,
    )
    out["volume_dryup_removed"] = payload["variants"]["F"]
    payload = run_ablation_r(
        frames=frames, config=replace(config, max_final_depth_pct=8.0),
        variants=("F",), **kwargs,
    )
    out["final_contraction_le_8"] = payload["variants"]["F"]
    payload = run_ablation_r(
        frames=frames, config=replace(config, max_base_depth_pct=25.0),
        variants=("F",), **kwargs,
    )
    out["max_base_depth_25"] = payload["variants"]["F"]
    return out


def run_study(*, use_store: bool = True, max_symbols: int | None = 80) -> dict[str, Any]:
    cfg = RESEARCH_CFG
    source = "synthetic_panel"
    frames = None
    universe_meta: dict[str, Any] = {}
    kwargs = dict(sample_step=1, lookback_sessions=80, horizon=12, max_symbols=None)
    if use_store:
        try:
            from data.bhavcopy_store import reload_corporate_actions
            reload_corporate_actions()
        except Exception:
            pass
        try:
            from research.sepa.universe_screen import load_research_frames
            packed = load_research_frames(max_symbols=max_symbols)
            if packed.get("frames"):
                frames = packed["frames"]
                universe_meta = {k: v for k, v in packed.items() if k != "frames"}
                source = "official_nse_bhavcopy"
                from research.sepa.config import DEFAULT_CONFIG
                cfg = DEFAULT_CONFIG
                kwargs = dict(sample_step=1, lookback_sessions=250, horizon=20, max_symbols=None)
        except Exception as exc:
            universe_meta = {"store_error": str(exc)}
    if not frames:
        frames = synthetic_panel()
        source = "synthetic_panel"
        kwargs = dict(sample_step=1, lookback_sessions=80, horizon=12, max_symbols=None)
        cfg = RESEARCH_CFG
    last = max(df.index[-1] for df in frames.values())
    integ = research_integrity_report(frames=frames, as_of=last, verify=True)
    core = run_ablation_r(frames=frames, config=cfg, variants=("A", "B", "C", "D", "E", "F"), **kwargs)
    core["data_source"] = source
    core["universe"] = universe_meta
    core["integrity"] = integ
    core["buy_zone_study"] = buyzone_study(frames, cfg, **kwargs)
    core["vcp_component_study"] = vcp_component_study(frames, cfg, **kwargs)
    core["rs_buckets"] = rs_bucket_study(frames, cfg, horizon=12)
    core["pivot_compare"] = pivot_compare(frames, cfg, **kwargs)
    years = sorted({
        y for stats in (core.get("variants") or {}).values()
        for y in (stats.get("by_year") or {})
    })
    if len(years) >= 2:
        core["walk_forward"] = walk_forward_split(
            core, train_years=set(years[:-1]), test_years={years[-1]},
        )
    else:
        core["walk_forward"] = {
            "note": "Need ≥2 calendar years of fills for an unseen block.",
            "years_seen": years,
        }
    core["timing_live"] = [
        diagnose_symbol(sym, df, config=cfg, start=max(0, len(df) - 140))
        for sym, df in list(frames.items())[:8]
    ]
    # RS threshold 70/80/90 on C+F
    rs_study = {}
    for thr in (70.0, 80.0, 90.0):
        p = run_ablation_r(frames=frames, config=cfg, variants=("C", "F"), rs_threshold=thr, **kwargs)
        rs_study[str(int(thr))] = {k: p["variants"][k] for k in p["variants"]}
    core["rs_threshold_study"] = rs_study
    persist_r(core, name="ablation_001r.json")
    return core
