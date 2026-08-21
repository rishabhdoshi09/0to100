"""SEPA-001 ablation — research only, never places orders.

Walk-forward on official bhavcopy (or injected frames). Baseline A is the
production scanner; B–E stack SEPA gates on that baseline; F is core SEPA
without requiring a scanner BUY.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.sepa.config import DEFAULT_CONFIG, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.frames import iso_date, pit_universe, ca_status, slice_as_of
from research.sepa.rs import build_rs_table

_OUT = Path(__file__).resolve().parents[2] / "logs" / "sepa_001"


@dataclass
class SimRow:
    variant: str
    symbol: str
    as_of: str
    entry: float
    stop: float
    net_r: float
    gross_r: float
    hold: int
    outcome: str
    mae_r: float
    mfe_r: float
    reached_1r: bool
    reached_2r: bool
    stop_before_1r: bool
    failed_break: bool
    year: str
    sector: str
    regime: str


def _next_open_sim(
    fwd: pd.DataFrame,
    *,
    stop: float,
    pivot: float | None,
    buy_zone_high: float | None,
    horizon: int,
) -> dict[str, Any] | None:
    if fwd is None or len(fwd) < 2:
        return None
    # Signal known at hist close = fwd.iloc[0] is the signal bar if we pass
    # df.iloc[t-1:]? Convention: hist = df.iloc[:t], next bar is df.iloc[t].
    # Caller passes df.iloc[t:t+horizon].
    o = float(fwd["open"].iloc[0])
    if buy_zone_high is not None and o > float(buy_zone_high):
        return None  # gapped through the buy-zone — not a SEPA fill
    if o <= 0:
        return None
    entry = o
    if entry <= stop:
        # gapped through stop — loss at open
        return {
            "entry": entry, "net_r": -1.0, "gross_r": -1.0, "hold": 1,
            "outcome": "LOSS", "mae_r": 1.0, "mfe_r": 0.0,
            "reached_1r": False, "reached_2r": False, "stop_before_1r": True,
            "failed_break": True,
        }
    risk = entry - stop
    mae = 0.0
    mfe = 0.0
    failed = False
    for i in range(len(fwd)):
        h = float(fwd["high"].iloc[i])
        l = float(fwd["low"].iloc[i])
        c = float(fwd["close"].iloc[i])
        mae = max(mae, (entry - l) / risk)
        mfe = max(mfe, (h - entry) / risk)
        if pivot is not None and i <= 2 and c < float(pivot):
            failed = True
        if l <= stop:
            gross = -1.0
            return {
                "entry": entry, "gross_r": gross, "hold": i + 1,
                "outcome": "LOSS", "mae_r": mae, "mfe_r": mfe,
                "reached_1r": mfe >= 1, "reached_2r": mfe >= 2,
                "stop_before_1r": mfe < 1, "failed_break": True,
            }
    last = float(fwd["close"].iloc[-1])
    gross = (last - entry) / risk
    return {
        "entry": entry, "gross_r": gross, "hold": len(fwd),
        "outcome": "FLAT" if abs(gross) < 0.05 else ("WIN" if gross > 0 else "LOSS"),
        "mae_r": mae, "mfe_r": mfe,
        "reached_1r": mfe >= 1, "reached_2r": mfe >= 2,
        "stop_before_1r": False, "failed_break": failed,
    }


def _scanner_sim(fwd: pd.DataFrame, entry: float, stop: float, target: float, horizon: int) -> dict[str, Any] | None:
    from scan.signal_backtest import _simulate_timed

    high = fwd["high"].to_numpy(dtype=float)
    low = fwd["low"].to_numpy(dtype=float)
    close = fwd["close"].to_numpy(dtype=float)
    outcome, r_mult, e_off, x_off = _simulate_timed(entry, stop, target, high, low, close, 0.0)
    if outcome == "NO_FILL":
        return None
    risk = entry - stop
    if risk <= 0:
        return None
    filled = high[: e_off + 1]
    # MAE/MFE after fill
    mae = mfe = 0.0
    for i in range(e_off, min(x_off, len(high) - 1) + 1):
        mae = max(mae, (entry - float(low[i])) / risk)
        mfe = max(mfe, (float(high[i]) - entry) / risk)
    hold = int(x_off - e_off + 1) if x_off >= e_off >= 0 else 0
    return {
        "entry": entry, "gross_r": float(r_mult), "hold": hold,
        "outcome": outcome, "mae_r": mae, "mfe_r": mfe,
        "reached_1r": mfe >= 1, "reached_2r": mfe >= 2,
        "stop_before_1r": outcome == "LOSS" and mfe < 1,
        "failed_break": outcome == "LOSS",
    }


def _net(gross: float, entry: float, stop: float) -> float:
    try:
        from core.costs import net_r
        return float(net_r(gross, (entry - stop) / entry if entry else 0.0, "CNC"))
    except Exception:
        return float(gross)


def summarize(rows: list[SimRow], *, n_years: float | None = None) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0, "eligible_trades": 0, "trades_per_year": 0.0,
            "expectancy_r": None, "total_r": 0.0, "median_r": None,
            "avg_winner": None, "avg_loser": None, "win_rate": None,
            "profit_factor": None, "max_dd_r": 0.0, "max_dd_pct": None,
            "sharpe": None, "sortino": None, "payoff": None,
            "avg_hold": None, "mae": None, "mfe": None,
            "failed_breakout_rate": None, "pct_1r": None, "pct_2r": None,
            "pct_stop_before_1r": None, "by_year": {}, "by_sector": {},
            "by_regime": {}, "harness": None, "note": "no trades",
        }
    r = np.array([x.net_r for x in rows], dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    pf = None
    if losses.size and wins.size:
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else None
    elif wins.size and not losses.size:
        pf = float("inf")
    avg_w = float(wins.mean()) if wins.size else None
    avg_l = float(losses.mean()) if losses.size else None
    payoff = (avg_w / abs(avg_l)) if avg_w is not None and avg_l not in (None, 0) else None
    years = n_years if n_years and n_years > 0 else 1.0
    harness = None
    try:
        from research.harness import evaluate
        harness = evaluate(r, n_trials=6, min_n=30)
        harness = {
            "verdict": getattr(harness, "verdict", None),
            "insight": getattr(harness, "insight", None),
            "n": getattr(harness, "n", None),
            "mean_r": getattr(harness, "mean_r", None),
            "p_value": getattr(harness, "p_value", None),
            "psr": getattr(harness, "psr", None),
            "dsr": getattr(harness, "dsr", None),
        }
    except Exception as exc:
        harness = {"verdict": "INCONCLUSIVE", "one_liner": str(exc)}

    def _bucket(key):
        g: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            g[getattr(row, key) or "UNKNOWN"].append(row.net_r)
        return {k: {"n": len(v), "expectancy_r": float(np.mean(v))} for k, v in sorted(g.items())}

    return {
        "n": len(rows),
        "eligible_trades": len(rows),
        "trades_per_year": round(len(rows) / years, 2),
        "expectancy_r": round(float(r.mean()), 4),
        "total_r": round(float(r.sum()), 4),
        "median_r": round(float(np.median(r)), 4),
        "avg_winner": None if avg_w is None else round(avg_w, 4),
        "avg_loser": None if avg_l is None else round(avg_l, 4),
        "win_rate": round(float((r > 0).mean()) * 100.0, 2),
        "profit_factor": None if pf is None or math.isinf(pf) else round(pf, 3),
        "max_dd_r": round(max_dd, 4),
        "max_dd_pct": None,
        "sharpe": None if std == 0 else round(float(r.mean()) / std, 3),
        "sortino": None if dstd == 0 else round(float(r.mean()) / dstd, 3),
        "payoff": None if payoff is None else round(payoff, 3),
        "avg_hold": round(float(np.mean([x.hold for x in rows])), 2),
        "mae": round(float(np.mean([x.mae_r for x in rows])), 4),
        "mfe": round(float(np.mean([x.mfe_r for x in rows])), 4),
        "failed_breakout_rate": round(float(np.mean([1.0 if x.failed_break else 0.0 for x in rows])) * 100.0, 2),
        "pct_1r": round(float(np.mean([1.0 if x.reached_1r else 0.0 for x in rows])) * 100.0, 2),
        "pct_2r": round(float(np.mean([1.0 if x.reached_2r else 0.0 for x in rows])) * 100.0, 2),
        "pct_stop_before_1r": round(float(np.mean([1.0 if x.stop_before_1r else 0.0 for x in rows])) * 100.0, 2),
        "by_year": _bucket("year"),
        "by_sector": _bucket("sector"),
        "by_regime": _bucket("regime"),
        "harness": harness,
    }


def _sector(sym: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(sym) or "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def run_ablation(
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
    sample_step: int = 10,
    lookback_sessions: int = 350,
    horizon: int = 20,
    max_symbols: int | None = 100,
    config: SepaConfig | None = None,
    buy_zone_above_pct: float | None = None,
    rs_threshold: float | None = None,
    min_trend_passed: int = 8,
    variants: tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    if rs_threshold is not None:
        cfg = replace(cfg, rs_threshold=float(rs_threshold))
    if frames is None:
        from data.bhavcopy_runtime import ensure_loaded
        from data.bhavcopy_store import get_ohlcv, store_symbols
        ensure_loaded(rebuild_from_local=False)
        ranked = []
        for sym in store_symbols() or []:
            df = get_ohlcv(sym)
            if df is None or len(df) < 80 + horizon:
                continue
            try:
                px = float(df["close"].iloc[-1])
                vol = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 0.0
            except Exception:
                continue
            if px < 20:
                continue
            ranked.append((px * vol, str(sym).upper(), df))
        ranked.sort(reverse=True)
        if max_symbols:
            ranked = ranked[: int(max_symbols)]
        frames = {sym: df for _, sym, df in ranked}
    pit_meta = {"universe_complete": False, "ca_complete": False}
    try:
        # as-of last date for status; per-bar RS still uses the as-of slice
        last = max(df.index[-1] for df in frames.values())
        u = pit_universe(last)
        pit_meta["universe_complete"] = bool(u.get("universe_complete"))
        pit_meta["universe_note"] = u.get("note") or ""
    except Exception:
        pass
    ca = ca_status()
    pit_meta["ca_complete"] = bool(ca.get("ca_complete"))
    pit_meta["ca_note"] = ca.get("note") or ""

    from scan.unified_scanner import UnifiedScanner
    scanner = UnifiedScanner()

    rows: dict[str, list[SimRow]] = {v: [] for v in variants}
    last_exit = {v: {} for v in variants}
    sample_dates: set[str] = set()
    n_signals_a = 0
    rs_cache: dict[str, dict] = {}

    for sym, df in frames.items():
        n = len(df)
        start = max(80, n - horizon - lookback_sessions)
        for t in range(start, n - horizon, sample_step):
            hist = df.iloc[:t]
            fwd = df.iloc[t: t + horizon]
            as_of = iso_date(hist.index[-1])
            sample_dates.add(as_of)
            scan_hit = None
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
            # B = scanner + 7 MA/52w rules (RS is variant C). 7/8 study uses min_trend_passed.
            if min_trend_passed >= 8:
                trend_ok = sepa.structure_pass
            else:
                trend_ok = sepa.trend_passed >= min_trend_passed

            def _take(variant: str, sim: dict[str, Any] | None):
                if sim is None:
                    return
                if t < last_exit[variant].get(sym, -1):
                    return
                last_exit[variant][sym] = t + int(sim["hold"])
                scanner_fill = variant in ("A", "B", "C", "D")
                if scanner_fill and scan_hit is not None:
                    stop_px = float(scan_hit.stop)
                    net = _net(sim["gross_r"], sim["entry"], stop_px)
                else:
                    stop_px = float(sepa.structural_stop or sim.get("entry", 1) * 0.95)
                    net = _net(sim["gross_r"], sim["entry"], stop_px)
                rows[variant].append(SimRow(
                    variant=variant, symbol=sym, as_of=as_of,
                    entry=float(sim["entry"]), stop=stop_px,
                    net_r=float(net), gross_r=float(sim["gross_r"]),
                    hold=int(sim["hold"]), outcome=str(sim["outcome"]),
                    mae_r=float(sim["mae_r"]), mfe_r=float(sim["mfe_r"]),
                    reached_1r=bool(sim["reached_1r"]), reached_2r=bool(sim["reached_2r"]),
                    stop_before_1r=bool(sim["stop_before_1r"]),
                    failed_break=bool(sim["failed_break"]),
                    year=as_of[:4], sector=_sector(sym), regime="UNKNOWN",
                ))

            if "A" in variants and scanner_ok and scan_hit is not None:
                _take("A", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop), float(scan_hit.target), horizon))
            if "B" in variants and scanner_ok and trend_ok and scan_hit is not None:
                _take("B", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop), float(scan_hit.target), horizon))
            if "C" in variants and scanner_ok and trend_ok and sepa.rs_pass and scan_hit is not None:
                _take("C", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop), float(scan_hit.target), horizon))
            if "D" in variants and scanner_ok and trend_ok and sepa.rs_pass and sepa.vcp_detected and scan_hit is not None:
                _take("D", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop), float(scan_hit.target), horizon))
            if "E" in variants and scanner_ok and trend_ok and sepa.rs_pass and sepa.vcp_detected and sepa.entry_valid and sepa.stop_ok and sepa.structural_stop:
                # Same gates as D, but fill/stop are structural — not scanner ATR geometry.
                _take("E", _next_open_sim(
                    fwd, stop=float(sepa.structural_stop), pivot=sepa.pivot,
                    buy_zone_high=sepa.buy_zone_high, horizon=horizon,
                ))
            if "F" in variants:
                if min_trend_passed >= 8:
                    f_ok = sepa.eligible and sepa.structural_stop
                else:
                    f_ok = (
                        sepa.trend_passed >= min_trend_passed
                        and sepa.vcp_detected
                        and sepa.entry_valid
                        and sepa.stop_ok
                        and sepa.structural_stop
                    )
                if f_ok:
                    _take("F", _next_open_sim(
                        fwd, stop=float(sepa.structural_stop), pivot=sepa.pivot,
                        buy_zone_high=sepa.buy_zone_high, horizon=horizon,
                    ))

    years = max(1.0, len(sample_dates) / 252.0) if sample_dates else 1.0
    summary = {v: summarize(rows[v], n_years=years) for v in variants}
    payload = {
        "experiment": "SEPA-001",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_hash": cfg.config_hash(),
        "eligibility_version": cfg.eligibility_version,
        "pit": pit_meta,
        "sample": {
            "symbols": len(frames),
            "sample_step": sample_step,
            "lookback_sessions": lookback_sessions,
            "horizon": horizon,
            "n_as_of": len(sample_dates),
            "scanner_signals": n_signals_a,
        },
        "variants": summary,
        "buy_zone_above_pct": buy_zone_above_pct if buy_zone_above_pct is not None else cfg.buy_zone_above_pct,
        "rs_threshold": cfg.rs_threshold,
        "min_trend_passed": min_trend_passed,
    }
    return payload


def run_parameter_studies(frames=None, **kwargs) -> dict[str, Any]:
    """Buy-zone, RS threshold, and 7/8 vs 8/8 studies. Research only."""
    out: dict[str, Any] = {"buy_zone": {}, "rs_threshold": {}, "template": {}}
    for width in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        payload = run_ablation(
            frames=frames, variants=("E", "F"), buy_zone_above_pct=width, **kwargs
        )
        out["buy_zone"][str(width)] = {
            k: payload["variants"][k] for k in payload["variants"]
        }
    for thr in (70.0, 80.0, 90.0):
        payload = run_ablation(
            frames=frames, variants=("C", "F"), rs_threshold=thr, **kwargs
        )
        out["rs_threshold"][str(int(thr))] = {
            k: payload["variants"][k] for k in payload["variants"]
        }
    for need in (7, 8):
        payload = run_ablation(
            frames=frames, variants=("B", "F"), min_trend_passed=need, **kwargs
        )
        out["template"][f"{need}_of_8"] = {
            k: payload["variants"][k] for k in payload["variants"]
        }
    return out


def persist(payload: dict[str, Any], name: str = "ablation.json") -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
