"""SEPA-001R ablation — daily, setup-deduped, research only. No orders."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.sepa.ablation import SimRow, _net, _scanner_sim, summarize
from research.sepa.config import DEFAULT_CONFIG, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import (
    FILL_EXTENDED,
    FILL_GAP_THROUGH,
    FILL_INVALIDATED,
    FILL_MISSED,
    FILL_VALID,
    classify_next_open_fill,
)
from research.sepa.frames import ca_status, iso_date, pit_universe, slice_as_of
from research.sepa.integrity import research_integrity_report
from research.sepa.rs import build_rs_table
from research.sepa.setups import SetupRegistry
from research.sepa.types import SepaEligibility

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R"


@dataclass
class SimRowR(SimRow):
    setup_id: str = ""
    fill_class: str = ""
    extension_at_fill: float | None = None
    vcp_state: str = ""
    rs_percentile: float | None = None
    unique_setup: bool = True


def _sector(sym: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(sym) or "UNKNOWN") or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _regime_series(close: pd.Series) -> pd.Series:
    try:
        from scan.signal_backtest import classify_regime
        base = classify_regime(close)
    except Exception:
        base = pd.Series("UNKNOWN", index=close.index)
    ret20 = close.pct_change(20)
    out = []
    for ts, lab in base.items():
        r = float(ret20.loc[ts]) if ts in ret20.index and ret20.loc[ts] == ret20.loc[ts] else 0.0
        if lab == "BULL" and r > 0.06:
            out.append("strong_bull")
        elif lab == "BULL":
            out.append("normal_bull")
        elif lab == "CHOP":
            out.append("sideways_choppy")
        elif lab == "BEAR" and r < -0.08:
            out.append("bear_distribution")
        elif lab == "BEAR":
            out.append("correction")
        else:
            out.append("unknown")
    return pd.Series(out, index=base.index)


def _regime_at(series: pd.Series | None, as_of: str) -> str:
    if series is None or len(series) == 0:
        return "unknown"
    try:
        cutoff = pd.Timestamp(as_of)
        sl = series[series.index.normalize() <= cutoff]
        return str(sl.iloc[-1]) if len(sl) else "unknown"
    except Exception:
        return "unknown"


def sepa_fill_sim(
    fwd: pd.DataFrame,
    *,
    stop: float,
    pivot: float | None,
    buy_zone_low: float | None,
    buy_zone_high: float | None,
    horizon: int,
) -> dict[str, Any]:
    """Next-open SEPA fill. Gap through the zone → no trade (not a worse fill)."""
    if fwd is None or len(fwd) < 1:
        return {"class": "NO_BAR", "sim": None}
    o = float(fwd["open"].iloc[0])
    packed = classify_next_open_fill(
        open_px=o, zone_lo=buy_zone_low, zone_hi=buy_zone_high, stop=stop,
    )
    if packed["class"] != FILL_VALID:
        return {"class": packed["class"], "sim": None, "reason": packed.get("reason")}
    entry = float(packed["fill"])
    risk = entry - float(stop)
    if risk <= 0:
        return {"class": FILL_INVALIDATED, "sim": None}
    mae = mfe = 0.0
    failed = False
    n = min(int(horizon), len(fwd))
    for i in range(n):
        h = float(fwd["high"].iloc[i])
        l = float(fwd["low"].iloc[i])
        c = float(fwd["close"].iloc[i])
        mae = max(mae, (entry - l) / risk)
        mfe = max(mfe, (h - entry) / risk)
        if pivot is not None and i <= 2 and c < float(pivot):
            failed = True
        if l <= float(stop):
            return {"class": FILL_VALID, "sim": {
                "entry": entry, "gross_r": -1.0, "hold": i + 1,
                "outcome": "LOSS", "mae_r": mae, "mfe_r": mfe,
                "reached_1r": mfe >= 1, "reached_2r": mfe >= 2,
                "stop_before_1r": mfe < 1, "failed_break": True,
            }}
    last = float(fwd["close"].iloc[n - 1])
    gross = (last - entry) / risk
    return {"class": FILL_VALID, "sim": {
        "entry": entry, "gross_r": gross, "hold": n,
        "outcome": "FLAT" if abs(gross) < 0.05 else ("WIN" if gross > 0 else "LOSS"),
        "mae_r": mae, "mfe_r": mfe,
        "reached_1r": mfe >= 1, "reached_2r": mfe >= 2,
        "stop_before_1r": False, "failed_break": failed,
    }}


def summarize_r(rows: list[SimRowR], *, n_years: float | None = None, n_trials: int = 6) -> dict[str, Any]:
    base = summarize(list(rows), n_years=n_years)
    if not rows:
        base["fill_classes"] = {}
        base["block_ci"] = None
        base["unique_setups"] = 0
        return base
    r = np.array([x.net_r for x in rows], dtype=float)
    ci = None
    try:
        from research.harness import block_bootstrap_mean_ci, evaluate
        ci = block_bootstrap_mean_ci(r, n_boot=800, seed=7)
        harness = evaluate(r, n_trials=n_trials, min_n=30, require_block_ci=False)
        base["harness"] = {
            "verdict": getattr(harness, "verdict", None),
            "insight": getattr(harness, "insight", None),
            "n": getattr(harness, "n", None),
            "mean_r": getattr(harness, "mean_r", None),
            "p_value": getattr(harness, "p_value", None),
            "psr": getattr(harness, "psr", None),
            "dsr": getattr(harness, "dsr", None),
        }
    except Exception as exc:
        base.setdefault("harness", {"verdict": "INCONCLUSIVE", "one_liner": str(exc)})
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.fill_class or "TRADE"] += 1
    ext = [x.extension_at_fill for x in rows if x.extension_at_fill is not None]
    base["block_ci"] = ci
    base["fill_classes"] = dict(counts)
    base["unique_setups"] = len({x.setup_id for x in rows if x.setup_id}) or len(rows)
    base["avg_extension_at_fill"] = None if not ext else round(float(np.mean(ext)), 4)
    return base


def _nifty_regimes():
    try:
        from data.index_store import get_index_ohlcv
        df = get_index_ohlcv("^NSEI") or get_index_ohlcv("NIFTY 50")
        if df is None or len(df) < 70:
            return None
        col = next((c for c in df.columns if c.lower() == "close"), None)
        if col is None:
            return None
        return _regime_series(pd.to_numeric(df[col], errors="coerce"))
    except Exception:
        return None


def run_ablation_r(
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
    sample_step: int = 1,
    lookback_sessions: int = 400,
    horizon: int = 20,
    max_symbols: int | None = None,
    config: SepaConfig | None = None,
    buy_zone_above_pct: float | None = None,
    rs_threshold: float | None = None,
    min_trend_passed: int = 8,
    variants: tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
    integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    if rs_threshold is not None:
        cfg = replace(cfg, rs_threshold=float(rs_threshold))
    loaded_meta: dict[str, Any] = {}
    if frames is None:
        from research.sepa.universe_screen import load_research_frames
        packed = load_research_frames(max_symbols=max_symbols)
        frames = packed["frames"]
        loaded_meta = {k: v for k, v in packed.items() if k != "frames"}
    pit_meta = {"universe_complete": False, "ca_complete": False, "research_grade": False}
    try:
        last = max(df.index[-1] for df in frames.values())
        u = pit_universe(last)
        pit_meta["universe_complete"] = bool(u.get("universe_complete"))
        pit_meta["universe_note"] = u.get("note") or ""
        pit_meta["universe_source"] = u.get("source") or ""
        pit_meta["research_grade"] = bool(u.get("research_grade"))
    except Exception:
        last = None
    ca = ca_status()
    pit_meta["ca_complete"] = bool(ca.get("ca_complete"))
    pit_meta["ca_note"] = ca.get("note") or ""
    pit_meta["ca_verified"] = bool(ca.get("verified"))
    pit_meta["ca_n_events"] = int(ca.get("n_events") or 0)
    integ = integrity if integrity is not None else research_integrity_report(
        frames=frames, as_of=last,
    )

    need_scanner = any(v in variants for v in ("A", "B", "C", "D", "E"))
    scanner = None
    if need_scanner:
        from scan.unified_scanner import UnifiedScanner
        scanner = UnifiedScanner()

    regimes = _nifty_regimes()
    rows: dict[str, list[SimRowR]] = {v: [] for v in variants}
    last_exit = {v: {} for v in variants}
    registries = {v: SetupRegistry() for v in variants}
    sample_dates: set[str] = set()
    n_signals_a = 0
    rs_cache: dict[str, dict] = {}
    setups: list[dict[str, Any]] = []
    fill_counts = {v: defaultdict(int) for v in variants}
    versions = {
        "eligibility_version": cfg.eligibility_version,
        "vcp_version": cfg.vcp_version,
        "pivot_version": cfg.pivot_version,
    }

    def _take(variant: str, sim: dict[str, Any] | None, *, sepa: SepaEligibility,
              as_of: str, t: int, fill_class: str, scan_hit=None):
        if sim is None:
            fill_counts[variant][fill_class] += 1
            return
        if t < last_exit[variant].get(sepa.symbol, -1):
            return
        last_exit[variant][sepa.symbol] = t + int(sim["hold"])
        scanner_fill = variant in ("A", "B", "C", "D")
        if scanner_fill and scan_hit is not None:
            stop_px = float(scan_hit.stop)
            net = _net(sim["gross_r"], sim["entry"], stop_px)
        else:
            stop_px = float(sepa.structural_stop or sim.get("entry", 1) * 0.95)
            net = _net(sim["gross_r"], sim["entry"], stop_px)
        ext = None
        if sepa.pivot:
            ext = (float(sim["entry"]) / float(sepa.pivot) - 1.0) * 100.0
        fill_counts[variant][fill_class] += 1
        rows[variant].append(SimRowR(
            variant=variant, symbol=sepa.symbol, as_of=as_of,
            entry=float(sim["entry"]), stop=stop_px,
            net_r=float(net), gross_r=float(sim["gross_r"]),
            hold=int(sim["hold"]), outcome=str(sim["outcome"]),
            mae_r=float(sim["mae_r"]), mfe_r=float(sim["mfe_r"]),
            reached_1r=bool(sim["reached_1r"]), reached_2r=bool(sim["reached_2r"]),
            stop_before_1r=bool(sim["stop_before_1r"]),
            failed_break=bool(sim["failed_break"]),
            year=as_of[:4], sector=_sector(sepa.symbol),
            regime=_regime_at(regimes, as_of),
            setup_id=sepa.setup_id, fill_class=fill_class,
            extension_at_fill=None if ext is None else round(ext, 4),
            vcp_state=sepa.vcp_state,
            rs_percentile=sepa.rs_percentile,
        ))

    for sym, df in frames.items():
        n = len(df)
        start = max(80, n - horizon - lookback_sessions)
        for t in range(start, n - horizon, sample_step):
            hist = df.iloc[:t]
            fwd = df.iloc[t: t + horizon]
            as_of = iso_date(hist.index[-1])
            sample_dates.add(as_of)
            scan_hit = None
            scanner_ok = False
            if scanner is not None:
                try:
                    scan_hit = scanner._analyze(sym, hist)
                except Exception:
                    scan_hit = None
                scanner_ok = bool(scan_hit is not None and getattr(scan_hit, "signals", None))
                if scanner_ok:
                    n_signals_a += 1
            rs_table = rs_cache.get(as_of)
            if rs_table is None:
                rs_table = build_rs_table(frames, hist.index[-1], cfg, universe=list(frames))
                rs_cache[as_of] = rs_table
            sepa = evaluate_sepa_eligibility(
                sym, hist.index[-1], frame=hist, rs_table=rs_table, config=cfg,
                pit_meta=pit_meta, buy_zone_above_pct=buy_zone_above_pct,
            )
            versions["as_of"] = as_of
            if min_trend_passed >= 8:
                trend_ok = sepa.structure_pass
            else:
                trend_ok = sepa.trend_passed >= min_trend_passed

            if sepa.vcp_detected and sepa.base_start_date:
                rec = {
                    "setup_id": sepa.setup_id,
                    "symbol": sym,
                    "detection_date": as_of,
                    "base_start_date": sepa.base_start_date,
                    "stage2": sepa.structure_pass,
                    "trend_template_pass": sepa.trend_template_pass,
                    "rs": sepa.rs_percentile,
                    "vcp_detected": sepa.vcp_detected,
                    "vcp_state": sepa.vcp_state,
                    "pivot": sepa.pivot,
                    "buy_zone_low": sepa.buy_zone_low,
                    "buy_zone_high": sepa.buy_zone_high,
                    "structural_stop": sepa.structural_stop,
                    "distance_from_pivot_pct": sepa.distance_from_pivot_pct,
                    "eligible": sepa.eligible,
                    "headline": sepa.headline,
                    "sector": _sector(sym),
                    "regime": _regime_at(regimes, as_of),
                    "experiment_version": cfg.eligibility_version,
                    "pivot_version": cfg.pivot_version,
                }
                setups.append(rec)

            if "A" in variants and scanner_ok and scan_hit is not None:
                _take("A", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                        float(scan_hit.target), horizon),
                      sepa=sepa, as_of=as_of, t=t, fill_class="SCANNER", scan_hit=scan_hit)
            if "B" in variants and scanner_ok and trend_ok and scan_hit is not None:
                _take("B", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                        float(scan_hit.target), horizon),
                      sepa=sepa, as_of=as_of, t=t, fill_class="SCANNER", scan_hit=scan_hit)
            if "C" in variants and scanner_ok and trend_ok and sepa.rs_pass and scan_hit is not None:
                _take("C", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                        float(scan_hit.target), horizon),
                      sepa=sepa, as_of=as_of, t=t, fill_class="SCANNER", scan_hit=scan_hit)
            if "D" in variants and scanner_ok and trend_ok and sepa.rs_pass and sepa.vcp_detected and scan_hit is not None:
                _take("D", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                        float(scan_hit.target), horizon),
                      sepa=sepa, as_of=as_of, t=t, fill_class="SCANNER", scan_hit=scan_hit)

            def _sepa_once(variant: str, ok: bool):
                if not ok or not sepa.structural_stop or not sepa.pivot:
                    return
                reg = registries[variant]
                row = reg.see(symbol=sym, vcp={
                    "base_start_date": sepa.base_start_date,
                    "pivot": sepa.pivot,
                    "stop": sepa.structural_stop,
                    "detected": sepa.vcp_detected,
                    "pivot_knowable_date": sepa.pivot_knowable_date,
                    "vcp_knowable_date": sepa.vcp_knowable_date,
                }, versions=versions)
                if row is None:
                    return
                if reg.is_terminal(sym, sepa.base_start_date):
                    return
                if sepa.extended:
                    fill_counts[variant][FILL_EXTENDED] += 1
                    reg.mark(sym, sepa.base_start_date, "EXTENDED")
                    return
                if not sepa.entry_valid or not sepa.stop_ok:
                    return
                packed = sepa_fill_sim(
                    fwd, stop=float(sepa.structural_stop), pivot=sepa.pivot,
                    buy_zone_low=sepa.buy_zone_low, buy_zone_high=sepa.buy_zone_high,
                    horizon=horizon,
                )
                cls = packed["class"]
                if packed.get("sim") is None:
                    fill_counts[variant][cls] += 1
                    reg.mark(sym, sepa.base_start_date, cls)
                    return
                _take(variant, packed["sim"], sepa=sepa, as_of=as_of, t=t, fill_class=cls)
                reg.mark(sym, sepa.base_start_date, "FILLED")

            if "E" in variants:
                _sepa_once("E", scanner_ok and trend_ok and sepa.rs_pass and sepa.vcp_detected)
            if "F" in variants:
                if min_trend_passed >= 8:
                    f_ok = bool(sepa.eligible or (sepa.good_stock and sepa.good_setup and sepa.stop_ok))
                    # First ENTRY_READY day: eligible. Also allow good_stock+setup+stop
                    # only when entry_valid so we do not fill extended prints.
                    f_ok = sepa.trend_template_pass and sepa.vcp_detected and sepa.stop_ok
                else:
                    f_ok = sepa.trend_passed >= min_trend_passed and sepa.vcp_detected and sepa.stop_ok
                _sepa_once("F", f_ok)

    years = max(1.0, len(sample_dates) / 252.0) if sample_dates else 1.0
    n_trials = max(6, len(variants))
    summary = {v: summarize_r(rows[v], n_years=years, n_trials=n_trials) for v in variants}
    for v in variants:
        summary[v]["fill_attempt_counts"] = dict(fill_counts[v])
        summary[v]["unique_setups_seen"] = len({s["setup_id"] for s in setups if s.get("setup_id")})

    # Dedup setup dataset by setup_id keeping first detection.
    uniq: dict[str, dict[str, Any]] = {}
    for s in setups:
        sid = s.get("setup_id") or ""
        if sid and sid not in uniq:
            uniq[sid] = s
    return {
        "experiment": "SEPA-001R",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_hash": cfg.config_hash(),
        "eligibility_version": cfg.eligibility_version,
        "pivot_version": cfg.pivot_version,
        "vcp_version": cfg.vcp_version,
        "pit": pit_meta,
        "integrity": integ,
        "universe": loaded_meta,
        "sample": {
            "symbols": len(frames),
            "sample_step": sample_step,
            "lookback_sessions": lookback_sessions,
            "horizon": horizon,
            "n_as_of": len(sample_dates),
            "scanner_signals": n_signals_a,
            "unique_setups": len(uniq),
        },
        "variants": summary,
        "setups": list(uniq.values()),
        "buy_zone_above_pct": buy_zone_above_pct if buy_zone_above_pct is not None else cfg.buy_zone_above_pct,
        "rs_threshold": cfg.rs_threshold,
        "min_trend_passed": min_trend_passed,
    }


def walk_forward_split(payload: dict[str, Any], *, train_years: set[str], test_years: set[str]) -> dict[str, Any]:
    """Year-block walk-forward over already-collected variant rows is done at summarize time.

    This helper splits setup-level outcomes already in payload['variants']['F']['by_year'].
    """
    out = {"train_years": sorted(train_years), "test_years": sorted(test_years), "by_variant": {}}
    for vid, stats in (payload.get("variants") or {}).items():
        by_year = stats.get("by_year") or {}
        def _sum(years):
            n = exp = 0.0
            k = 0
            for y, rec in by_year.items():
                if y in years:
                    k += int(rec.get("n") or 0)
                    exp += float(rec.get("expectancy_r") or 0) * int(rec.get("n") or 0)
            return {"n": k, "expectancy_r": None if k == 0 else round(exp / k, 4)}
        out["by_variant"][vid] = {"train": _sum(train_years), "test": _sum(test_years)}
    return out


def persist_r(payload: dict[str, Any], name: str = "ablation.json") -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    slim = dict(payload)
    # setups can be large; always write a sidecar
    setups = slim.pop("setups", [])
    path.write_text(json.dumps(slim, indent=2, default=str))
    side = _OUT / "setups.jsonl"
    with side.open("w", encoding="utf-8") as fh:
        for row in setups:
            fh.write(json.dumps(row, default=str) + "\n")
    return path
