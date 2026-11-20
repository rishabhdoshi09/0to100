"""Read-only, point-in-time diagnosis of why the scanner's signals show no edge (research only).

For each signal it separates "no edge" from "low statistical power", and both
from the usual suspects: trading costs, fills that never trigger, market drift,
and uncorrected corporate actions. It runs on an OFFLINE COPY of the store.

It measures the rules the desk actually uses, not a re-implementation. It
calls the PRODUCTION walk-forward, ``scan.signal_backtest.run_backtest``,
which applies the production scanner with ``evidence_gate=False`` and the one
rulebook, ``core.evaluation_spec.simulate_trade``. Every candidate is captured
through hooks.

Safety:

* It never writes a backtest report. ``MIN_SYMBOLS_TO_SAVE`` is raised for
  the run and restored afterwards.
* It never touches the production alpha ledger. A discarded report reserves
  no look at all, and ``QT_ALPHA_LEDGER`` also points at a throwaway file
  for the run as defence in depth.
* It refuses a copy on an external volume (the live runtime lives there), so
  it cannot add IO to the desk.
* It makes no network calls. The Nifty regime lookup is replaced by an
  equal-weight regime built from the copy.

Point-in-time discipline:

* The scanner sees ``df.iloc[:t]`` only (the production loop). The decision
  date is the last bar it saw.
* An outcome counts only when its window is exactly 10 official sessions
  (production ``window_is_exact``). A trade exists only when the buy-stop
  traded (production ``simulate_trade``). Costs come from ``core.costs``.
* Corporate actions are verified only from the coverage start in
  ``ca_coverage.json``. Each trade gets a tier:
  - STRICT: the decision date is at least 252 sessions after the coverage
    start, so the scanner's 52-week lookback is adjusted too.
  - ENTRY_ONLY: the decision date is after the start, but the lookback is
    partly uncovered.
  - UNCOVERED: before the start. It is reported only as a sensitivity, never
    as evidence.
* Membership: a trade counts only if its symbol was listed and not yet
  delisted on the decision date (``universe_history.json``, official
  bhavcopy).
* Benchmark: an equal-weight index of every stock in the copy that traded
  that day, delisted names included. It runs from the decision close to the
  exit close.

Intervals are nominal 95% overlap-robust intervals per signal, with no
multiplicity correction. This is a diagnosis, not an admission test, and a
POSITIVE_IN_SAMPLE row is only a hypothesis for a prospective test. Nothing
here is a profitability claim.

    python -m research.edge_diagnosis --root <SSD copy root> --symbols 120 \
        --out docs/research/edge_diagnosis_2026-11-20.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

HORIZON = 10
STRICT_LOOKBACK_SESSIONS = 252          # the scanner's 52-week lookback must also be adjusted
MIN_TRADES = 30
MIN_BLOCKS = 20
EW_MIN_NAMES = 50                       # an equal-weight day needs this many stocks
EW_DAILY_CLIP = 0.20                    # winsorise single-stock daily returns (bad prints, residual gaps)
GAP_PCT = 35.0                          # core.data_integrity's phantom-gap fingerprint
TIERS = ("STRICT", "ENTRY_ONLY", "UNCOVERED")
ANY = "ANY_SIGNAL"

# Classification of a signal's net expectancy (R per trade), from its nominal 95% interval.
INSUFFICIENT = "INSUFFICIENT"                 # < 30 trades or < 20 independent blocks: no statement
NEGATIVE_EDGE = "NEGATIVE_EDGE"               # the whole interval is below zero
NO_MATERIAL_EDGE = "NO_MATERIAL_EDGE"         # straddles zero, but even its top is below the BUY floor
UNDERPOWERED = "UNDERPOWERED"                 # straddles zero and reaches the floor: cannot tell
POSITIVE_IN_SAMPLE = "POSITIVE_IN_SAMPLE"     # nominal interval above zero: a hypothesis, never a BUY


def materiality_r() -> float:
    """The live BUY floor (QT_MIN_BUY_EDGE_R): the smallest edge worth trading."""
    try:
        from scan.unified_scanner import _MIN_BUY_EDGE_R
        return float(_MIN_BUY_EDGE_R)
    except Exception:
        return 0.05


def classify(inference: Mapping[str, Any] | None, *, floor: float | None = None) -> str:
    """Verdict on one overlap-robust inference of net R. Pure."""
    inf = dict(inference or {})
    floor = materiality_r() if floor is None else float(floor)
    ci = inf.get("ci95")
    if (inf.get("status") != "OK" or int(inf.get("n") or 0) < MIN_TRADES
            or int(inf.get("independent_sessions") or 0) < MIN_BLOCKS or not ci):
        return INSUFFICIENT
    lo, hi = float(ci[0]), float(ci[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return INSUFFICIENT
    if hi < 0:
        return NEGATIVE_EDGE
    if lo > 0:
        return POSITIVE_IN_SAMPLE
    if hi < floor:
        return NO_MATERIAL_EDGE
    return UNDERPOWERED


def mde(se: float | None, df: int | None, *, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Minimum detectable effect: the true mean a one-sided test at `alpha` finds with `power`."""
    if se is None or df is None or df < 1 or not math.isfinite(float(se)):
        return None
    from research.overlap_inference import _t_ppf
    return float((_t_ppf(1 - alpha, int(df)) + _t_ppf(power, int(df))) * float(se))


def coverage_start(coverage: Mapping[str, Any] | None) -> str | None:
    """Start of the contiguous run of successful corporate-action windows that ends at the latest one."""
    wins = sorted(((w.get("covered_from"), w.get("covered_to"), bool(w.get("success")))
                   for w in (dict(coverage or {}).get("windows") or {}).values()
                   if w.get("covered_from") and w.get("covered_to")), key=lambda w: w[0])
    start = None
    prev_end = None
    for frm, to, ok in wins:
        if not ok:
            start, prev_end = None, None
            continue
        contiguous = prev_end is not None and (date.fromisoformat(frm) - date.fromisoformat(prev_end)).days <= 1
        if start is None or not contiguous:
            start = frm
        prev_end = to
    return start


def membership(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    for r in rows or []:
        try:
            out[str(r["symbol"]).strip().upper()] = (str(r["listed"])[:10], (str(r["delisted"])[:10]
                                                                             if r.get("delisted") else None))
        except Exception:
            continue
    return out


def is_member(member: Mapping[str, tuple[str, str | None]], sym: str, day: str) -> bool | None:
    """Listed on or before `day` and not delisted by it; None when the table has no row."""
    row = member.get(sym)
    if row is None:
        return None
    listed, delisted = row
    return listed <= day and (delisted is None or delisted > day)


def equal_weight_index(frames: Mapping[str, pd.DataFrame], *, min_names: int = EW_MIN_NAMES,
                       clip: float = EW_DAILY_CLIP) -> pd.Series:
    """Level of an equal-weight index of every stock that traded each day (delisted names included)."""
    rets = []
    for sym, df in frames.items():
        if df is None or len(df) < 2 or "close" not in df:
            continue
        c = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df.index).normalize())
        r = c.pct_change().replace([np.inf, -np.inf], np.nan).dropna().clip(-clip, clip)
        rets.append(r.rename(sym))
    if not rets:
        return pd.Series(dtype=float)
    panel = pd.concat(rets, axis=1)
    counts = panel.notna().sum(axis=1)
    daily = panel.mean(axis=1).where(counts >= min_names).dropna()
    return (1.0 + daily).cumprod()


def regime_series(level: pd.Series) -> pd.Series:
    from scan.signal_backtest import classify_regime
    return classify_regime(level) if len(level) else pd.Series(dtype=object)


# ── collection: the production walk-forward with hooks ──────────────────────

def collect(frames: Mapping[str, pd.DataFrame], *, symbols: int, sample_step: int = 5,
            lookback_sessions: int | None = None, horizon: int = HORIZON) -> dict[str, Any]:
    """Run the production walk-forward on `frames` (already CA-adjusted) and capture every candidate.

    Returns {"events": [...], "trades": [...], "provenance": {...}}. ``events``
    holds one row per scanner detection with its fate (RISK_SKIP, NO_FILL,
    FILLED); ``trades`` holds the production ledger rows of filled trades, each
    stamped with its decision date.
    """
    import data.bhavcopy_store as BS
    import scan.signal_backtest as SB
    from scan.unified_scanner import UnifiedScanner

    all_dates = sorted({d for df in frames.values() for d in pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")})
    lookback = int(lookback_sessions or max(1, len(all_dates) - 60 - horizon))
    saved = {"store_symbols": BS.store_symbols, "get_ohlcv": BS.get_ohlcv, "_dates_on_disk": BS._dates_on_disk,
             "min_save": SB.MIN_SYMBOLS_TO_SAVE, "simulate_trade": SB.simulate_trade,
             "regime": SB._nifty_regime_series, "calib": SB._calibration_version,
             "analyze": UnifiedScanner._analyze, "ledger_env": os.environ.get("QT_ALPHA_LEDGER")}
    events: list[dict] = []
    trades: list[dict] = []
    cur: dict[str, Any] = {}

    def analyze(self, sym, hist, *a, **k):
        res = saved["analyze"](self, sym, hist, *a, **k)
        cur.clear()
        if res is not None and getattr(res, "signals", None):
            risk_ok = (res.entry - res.stop) > 0
            cur.update(sym=sym, decision=str(hist.index[-1])[:10], signals=tuple(res.signals), simulated=False,
                       trigger=float(res.entry), decision_close=float(hist["close"].iloc[-1]))
            events.append({"symbol": sym, "decision": cur["decision"], "signals": cur["signals"],
                           "fate": "PENDING" if risk_ok else "RISK_SKIP"})
            cur["event"] = events[-1]
        return res

    def simulate(*a, **k):
        res = saved["simulate_trade"](*a, **k)
        if cur and not cur.get("simulated"):                    # the main trade, not exit variants
            cur["simulated"] = True
            cur["event"]["fate"] = "NO_FILL" if res.outcome == "NO_FILL" else "FILLED"
        return res

    def sink(rec):
        trades.append({**rec, "decision": cur.get("decision"), "trigger_price": cur.get("trigger"),
                       "decision_close": cur.get("decision_close"), "entry_datetime": str(rec["entry_datetime"])[:10],
                       "exit_datetime": str(rec["exit_datetime"])[:10], "signals": list(rec["signals"])})

    with tempfile.TemporaryDirectory(prefix="qt_edge_diag_") as tmp:
        try:
            os.environ["QT_ALPHA_LEDGER"] = str(Path(tmp) / "throwaway_alpha_ledger.db")
            BS.store_symbols = lambda: sorted(frames)
            BS.get_ohlcv = lambda s: (frames.get(str(s).upper()).copy() if str(s).upper() in frames else None)
            BS._dates_on_disk = lambda: [date.fromisoformat(d) for d in all_dates]
            SB.MIN_SYMBOLS_TO_SAVE = 10 ** 9                    # never write a report anywhere
            SB.simulate_trade = simulate
            SB._nifty_regime_series = lambda: None              # no network; regime comes from the EW index
            SB._calibration_version = lambda: "edge_diagnosis"
            UnifiedScanner._analyze = analyze
            t0 = time.perf_counter()
            rep = SB.run_backtest(sample_step=sample_step, lookback_sessions=lookback, horizon=horizon,
                                  max_symbols=symbols, on_trade=sink)
            elapsed = time.perf_counter() - t0
        finally:
            BS.store_symbols, BS.get_ohlcv, BS._dates_on_disk = (saved["store_symbols"], saved["get_ohlcv"],
                                                                 saved["_dates_on_disk"])
            SB.MIN_SYMBOLS_TO_SAVE, SB.simulate_trade = saved["min_save"], saved["simulate_trade"]
            SB._nifty_regime_series, SB._calibration_version = saved["regime"], saved["calib"]
            UnifiedScanner._analyze = saved["analyze"]
            if saved["ledger_env"] is None:
                os.environ.pop("QT_ALPHA_LEDGER", None)
            else:
                os.environ["QT_ALPHA_LEDGER"] = saved["ledger_env"]
    prov = rep.get("evidence_provenance") or {}
    sampled = SB._spread_sample(sorted(frames), symbols)
    return {"events": events, "trades": trades, "sampled_symbols": sampled,
            # Pre-declared experiments the production walk-forward already runs on the same
            # trades (exit constructions, target geometry, confirmation features): reported,
            # never searched. They cover the whole sampled lookback, not one tier.
            "production_experiments": {k: rep.get(k) for k in ("exit_variants", "target_sweep",
                                                              "feature_confirmation", "recommended_target_pct")},
            "provenance": {"elapsed_s": round(elapsed, 1), "lookback_sessions": lookback,
                           "sample_step": sample_step, "horizon_sessions": horizon,
                           "eval_spec": prov.get("eval_spec") or rep.get("eval_spec"),
                           "code_sha": prov.get("code_sha"),
                           "horizon_gap_windows_skipped": prov.get("horizon_gap_windows_skipped"),
                           "data_first_session": all_dates[0] if all_dates else None,
                           "data_last_session": all_dates[-1] if all_dates else None}}


def symbol_end_truncations(frames: Mapping[str, pd.DataFrame], symbols: Iterable[str], last_session: str, *,
                           horizon: int = HORIZON, sample_step: int = 5) -> dict[str, int]:
    """Sample points never evaluated because the symbol stopped trading before a full window."""
    ended = evaluated_lost = 0
    for sym in symbols:
        df = frames.get(sym)
        if df is None or not len(df):
            continue
        if str(df.index[-1])[:10] < last_session:
            ended += 1
            evaluated_lost += max(0, horizon // max(1, sample_step))
    return {"symbols_ended_before_last_session": ended, "approx_sample_points_lost": evaluated_lost}


# ── analysis ────────────────────────────────────────────────────────────────

def tier_map(ordinal: Mapping[str, int], cover: str | None) -> dict[str, str]:
    """Corporate-action tier of every session (STRICT / ENTRY_ONLY / UNCOVERED)."""
    first = min((d for d in ordinal if cover and d >= cover), default=None)
    out = {}
    for d, o in ordinal.items():
        if first is None or d < first:
            out[d] = "UNCOVERED"
        else:
            out[d] = "STRICT" if o - ordinal[first] >= STRICT_LOOKBACK_SESSIONS else "ENTRY_ONLY"
    return out


def _infer(rows: list[dict], key: str, ordinal: Mapping[str, int]) -> dict[str, Any]:
    from research.overlap_inference import cells_from_pairs, overlap_robust_test
    pairs = [(ordinal[r["decision"]], r[key]) for r in rows
             if r.get(key) is not None and r["decision"] in ordinal and math.isfinite(r[key])]
    res = overlap_robust_test(cells_from_pairs(pairs), HORIZON)
    keep = ("n", "independent_sessions", "df", "mean", "se", "ci95", "lower_95_one_sided", "status", "reason")
    out = {k: res.get(k) for k in keep}
    for k in ("mean", "se", "lower_95_one_sided"):
        if out.get(k) is not None:
            out[k] = round(float(out[k]), 4)
    if out.get("ci95"):
        out["ci95"] = [round(float(x), 4) for x in out["ci95"]]
    return out


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    wins, losses = a[a > 0], a[a < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    return {"win_rate": round(float((a > 0).mean()), 3), "avg_win_r": round(avg_win, 3),
            "avg_loss_r": round(avg_loss, 3),
            "payoff_ratio": round(avg_win / abs(avg_loss), 3) if avg_loss else None,
            "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if losses.size else None,
            "quantiles_r": {q: round(float(np.quantile(a, p)), 3)
                            for q, p in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}}


def _signal_report(rows: list[dict], events: list[dict], ordinal: Mapping[str, int], floor: float,
                   sessions_in_tier: int) -> dict[str, Any]:
    fates = Counter(e["fate"] for e in events)
    net = _infer(rows, "net_r", ordinal)
    gross = _infer(rows, "gross_r", ordinal)
    with_bench = [r for r in rows if r.get("excess_ret") is not None]
    excess = _infer(with_bench, "excess_ret", ordinal)
    bench_mean = float(np.mean([r["bench_ret"] for r in with_bench])) if with_bench else None
    cost_mean = float(np.mean([r["cost_r"] for r in rows])) if rows else None
    gross_mean = gross.get("mean")
    sens = {}
    if rows:
        g = np.asarray([r["gross_r"] for r in rows], dtype=float)
        c = np.asarray([r["cost_r"] for r in rows], dtype=float)
        rf = np.asarray([max(1e-6, r["risk_frac"]) for r in rows], dtype=float)
        for label, k in (("cost_x0", 0.0), ("cost_x0.5", 0.5), ("cost_x1", 1.0), ("cost_x2", 2.0)):
            sens[label] = round(float((g - k * c).mean()), 4)
        sens["cost_x1_plus_0.10pct_slippage_each_side"] = round(float((g - c - 0.002 / rf).mean()), 4)
    beta = None
    if len(with_bench) >= MIN_TRADES:
        x = np.asarray([r["bench_ret"] for r in with_bench], dtype=float)
        y = np.asarray([r["gross_ret"] for r in with_bench], dtype=float)
        if float(x.var()) > 0:
            beta = round(float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)), 3)
    by_regime = {}
    for reg in sorted({r.get("regime") or "UNKNOWN" for r in rows}):
        sub = [r for r in rows if (r.get("regime") or "UNKNOWN") == reg]
        by_regime[reg] = {"trades": len(sub), "mean_net_r": round(float(np.mean([r["net_r"] for r in sub])), 4)}
    exits = Counter(r.get("exit_reason") for r in rows)
    net_mde = mde(net.get("se"), net.get("df"))
    need = None
    if net_mde and net.get("independent_sessions"):
        need = int(math.ceil(net["independent_sessions"] * (net_mde / floor) ** 2))
    drivers = []
    if gross_mean is not None and net.get("mean") is not None and gross_mean > 0 >= net["mean"]:
        drivers.append("COSTS_FLIP_SIGN")
    filled = fates.get("FILLED", 0)
    attempted = filled + fates.get("NO_FILL", 0)
    fill_rate = round(filled / attempted, 3) if attempted else None
    if fill_rate is not None and fill_rate < 0.5:
        drivers.append("MOST_ENTRIES_NEVER_TRIGGER")
    if excess.get("mean") is not None and net.get("mean") is not None and bench_mean is not None:
        if bench_mean < 0 and excess["mean"] >= 0:
            drivers.append("MARKET_DRIFT_NOT_SELECTION")
        if excess.get("ci95") and excess["ci95"][1] < 0:
            drivers.append("UNDERPERFORMS_EQUAL_WEIGHT_MARKET")
    def _bucket(rs):
        return {"trades": len(rs), "mean_net_r": round(float(np.mean([r["net_r"] for r in rs])), 4) if rs else None}

    gaps = [r for r in rows if r.get("trigger_gap") is not None]
    entry_quality = {
        "trigger_within_1pct_of_close": _bucket([r for r in gaps if r["trigger_gap"] < 0.01]),
        "trigger_1_to_3pct_above": _bucket([r for r in gaps if 0.01 <= r["trigger_gap"] < 0.03]),
        "trigger_3pct_or_more_above": _bucket([r for r in gaps if r["trigger_gap"] >= 0.03]),
        "filled_at_trigger": _bucket([r for r in rows if (r.get("fill_vs_trigger") or 0.0) <= 0.001]),
        "gapped_through_trigger": _bucket([r for r in rows if (r.get("fill_vs_trigger") or 0.0) > 0.001]),
    }
    fast = sum(1 for r in rows if r.get("fast_failure"))
    return {
        "verdict": classify(net, floor=floor), "gross_verdict": classify(gross, floor=floor),
        "false_breakout_rate": round(fast / len(rows), 4) if rows else None,
        "false_breakout_note": "share of filled trades stopped out within 3 sessions",
        "entry_quality": entry_quality,
        "detections": sum(fates.values()), "risk_skips": fates.get("RISK_SKIP", 0),
        "no_fills": fates.get("NO_FILL", 0), "trades": len(rows), "fill_rate": fill_rate,
        "net_r": net, "gross_r": gross, "mean_cost_r": round(cost_mean, 4) if cost_mean is not None else None,
        "breakeven_cost_multiple": (round(gross_mean / cost_mean, 2) if gross_mean and cost_mean and gross_mean > 0
                                    else None),
        "cost_sensitivity_mean_r": sens, "mde_net_r_80pct_power": round(net_mde, 4) if net_mde else None,
        "blocks_needed_for_floor_approx": need,
        "excess_vs_equal_weight_return": excess, "mean_benchmark_pct": round(bench_mean * 100, 3) if bench_mean is not None
        else None, "beta_vs_equal_weight": beta,
        "distribution": _dist([r["net_r"] for r in rows]), "exit_reasons": dict(exits),
        "by_regime": by_regime, "mean_holding_sessions": round(float(np.mean([r["holding_period"] for r in rows])), 2)
        if rows else None,
        "trades_per_session_sampled": round(len(rows) / max(1, sessions_in_tier), 4),
        "drivers": drivers,
    }


def enrich(trades: list[dict], frames: Mapping[str, pd.DataFrame], ew: pd.Series, regimes: pd.Series,
           ordinal: Mapping[str, int], tiers: Mapping[str, str], member: Mapping[str, tuple[str, str | None]],
           sectors: Mapping[str, str]) -> list[dict]:
    ew_by_day = {str(k)[:10]: float(v) for k, v in ew.items()}
    reg_by_day = {str(k)[:10]: str(v) for k, v in regimes.items()}
    gaps: dict[str, list[str]] = {}
    out = []
    for t in trades:
        sym, d = t["symbol"], t["decision"]
        if d is None or d not in ordinal:
            continue
        if sym not in gaps:
            from core.data_integrity import phantom_gaps
            df = frames.get(sym)
            gaps[sym] = []
            if df is not None and len(df):
                closes = df["close"].to_numpy(dtype=float)
                days = np.asarray(pd.DatetimeIndex(df.index).strftime("%Y-%m-%d"))[~np.isnan(closes)]
                gaps[sym] = [str(days[g["index"]]) for g in phantom_gaps(closes, GAP_PCT)]
        fill, stop, exit_p = float(t["entry_price"]), float(t["stop_price"]), float(t["exit_price"])
        risk_frac = (fill - stop) / fill if fill > 0 else float("nan")
        gross_ret = exit_p / fill - 1.0 if fill > 0 else float("nan")
        cost_r = float(t.get("costs") or 0.0)
        net_ret = gross_ret - cost_r * risk_frac
        b0, b1 = ew_by_day.get(d), ew_by_day.get(t["exit_datetime"])
        bench = (b1 / b0 - 1.0) if b0 and b1 else None    # fraction, decision close -> exit close
        lo = max(0, ordinal[d] - STRICT_LOOKBACK_SESSIONS)
        gap_hit = any(lo <= ordinal.get(g, -1) <= ordinal.get(t["exit_datetime"], ordinal[d] + HORIZON)
                      for g in gaps[sym])
        trig, dclose = t.get("trigger_price"), t.get("decision_close")
        trigger_gap = (float(trig) / float(dclose) - 1.0) if trig and dclose else None
        fill_vs_trigger = (fill / float(trig) - 1.0) if trig else None
        out.append({**t, "net_r": float(t["net_R"]), "gross_r": float(t["gross_R"]), "cost_r": cost_r,
                    "trigger_gap": trigger_gap, "fill_vs_trigger": fill_vs_trigger,
                    "fast_failure": str(t.get("exit_reason")) == "LOSS" and int(t.get("holding_period") or 99) <= 3,
                    "n_signals": len(t.get("signals") or []),
                    "risk_frac": risk_frac, "gross_ret": gross_ret, "net_ret": net_ret, "bench_ret": bench,
                    "excess_ret": (net_ret - bench) if bench is not None else None,
                    "regime": reg_by_day.get(d, "UNKNOWN"), "tier": tiers[d],
                    "member": is_member(member, sym, d), "sector": sectors.get(sym, "UNMAPPED"),
                    "phantom_gap_in_window": gap_hit})
    return out


def analyse(collected: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], *, coverage: Mapping[str, Any],
            universe_rows: list | None, sectors: Mapping[str, str] | None = None,
            floor: float | None = None, ew_min_names: int = EW_MIN_NAMES) -> dict[str, Any]:
    from scan.signal_backtest import realised_session_index
    floor = materiality_r() if floor is None else float(floor)
    all_dates = sorted({d for df in frames.values() for d in pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")})
    ordinal = realised_session_index(all_dates)
    cover = coverage_start(coverage)
    member = membership(universe_rows)
    tiers_by_day = tier_map(ordinal, cover)
    ew = equal_weight_index(frames, min_names=ew_min_names)
    rows = enrich(list(collected["trades"]), frames, ew, regime_series(ew), ordinal, tiers_by_day, member,
                  sectors or {})
    excluded = Counter()
    kept = []
    for r in rows:
        if r["member"] is False:
            excluded["not_a_member_on_decision_date"] += 1
            continue
        if not math.isfinite(r["risk_frac"]) or r["risk_frac"] <= 0:
            excluded["non_positive_risk"] += 1
            continue
        kept.append(r)
    excluded["no_membership_row"] = sum(1 for r in kept if r["member"] is None)
    excluded["phantom_gap_in_window_flagged"] = sum(1 for r in kept if r["phantom_gap_in_window"])
    excluded["no_benchmark_for_window"] = sum(1 for r in kept if r["bench_ret"] is None)
    events = [e for e in collected["events"] if e["decision"] in ordinal]

    def by_signal(rs: list[dict], es: list[dict], sessions: int) -> dict[str, Any]:
        sigs = sorted({s for r in rs for s in r["signals"]} | {s for e in es for s in e["signals"]})
        out = {s: _signal_report([r for r in rs if s in r["signals"]], [e for e in es if s in e["signals"]],
                                 ordinal, floor, sessions) for s in sigs}
        out[ANY] = _signal_report(rs, es, ordinal, floor, sessions)
        return out

    tiers: dict[str, Any] = {}
    for tier in TIERS + ("COVERED_ALL",):
        names = ("STRICT", "ENTRY_ONLY") if tier == "COVERED_ALL" else (tier,)
        rs = [r for r in kept if r["tier"] in names]
        es = [e for e in events if tiers_by_day[e["decision"]] in names]
        sessions = sum(1 for d in ordinal if tiers_by_day[d] in names)
        tiers[tier] = {"sessions": sessions, "trades": len(rs), "signals": by_signal(rs, es, sessions)}
    strict = [r for r in kept if r["tier"] == "STRICT"]
    clean = [r for r in strict if not r["phantom_gap_in_window"]]
    combos: dict[str, list[dict]] = defaultdict(list)
    for r in strict:
        combos["+".join(sorted(r["signals"]))].append(r)
    interactions = {}
    for key, rs in sorted(combos.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < MIN_TRADES:
            continue
        inf = _infer(rs, "net_r", ordinal)
        interactions[key] = {"trades": len(rs), "net_r": inf, "verdict": classify(inf, floor=floor)}
    by_count = {}
    for label, test in (("1_signal", lambda n: n == 1), ("2_signals", lambda n: n == 2),
                        ("3_or_more", lambda n: n >= 3)):
        rs = [r for r in strict if test(r["n_signals"])]
        inf = _infer(rs, "net_r", ordinal)
        by_count[label] = {"trades": len(rs), "net_r": inf, "verdict": classify(inf, floor=floor)}
    sector_rows = defaultdict(list)
    for r in strict:
        sector_rows[r["sector"]].append(r["net_r"])
    sectors_out = {k: {"trades": len(v), "mean_net_r": round(float(np.mean(v)), 4)}
                   for k, v in sorted(sector_rows.items(), key=lambda kv: -len(kv[1]))[:10]}
    return {
        "coverage_start": cover, "materiality_floor_r": floor,
        "sessions_in_copy": len(all_dates), "first_session": all_dates[0] if all_dates else None,
        "last_session": all_dates[-1] if all_dates else None,
        "trades_captured": len(rows), "trades_kept": len(kept), "exclusions": dict(excluded),
        "detections": len(events), "fates": dict(Counter(e["fate"] for e in events)),
        "tiers": tiers,
        "strict_without_phantom_gap_windows": {ANY: _signal_report(clean, [], ordinal, floor,
                                                                   tiers["STRICT"]["sessions"])},
        "strict_by_sector_top10": sectors_out,
        "strict_signal_combinations": interactions,
        "strict_by_signal_count": by_count,
        "production_experiments": collected.get("production_experiments") or {},
        "equal_weight_index": {"first": str(ew.index[0])[:10] if len(ew) else None,
                               "last": str(ew.index[-1])[:10] if len(ew) else None,
                               "min_names_per_day": EW_MIN_NAMES, "daily_clip": EW_DAILY_CLIP},
    }


# ── offline copy loading and the CLI ────────────────────────────────────────

def load_copy(root: str | Path) -> dict[str, Any]:
    """Load an SSD copy: store, corporate-action events (applied on read), coverage and membership.

    The copy becomes this process's runtime root (QT_RUNTIME_ROOT), so every
    path, here and inside the production modules the run calls, resolves
    through core.runtime_paths into the copy.
    """
    root = Path(root).resolve()
    if str(root).startswith("/Volumes/"):
        raise SystemExit(f"refusing {root}: copy the store to the internal SSD first (never read the live runtime)")
    os.environ["QT_RUNTIME_ROOT"] = str(root)
    from core.runtime_paths import logs_path
    from data.corporate_actions import adjust_frame, load_events
    with open(logs_path("bhav", "store_cache.pkl"), "rb") as fh:
        store = pickle.load(fh)["store"]
    events = load_events(logs_path("ca_events.json"))
    frames = {}
    for sym, df in store.items():
        ev = events.get(str(sym).upper())
        frames[str(sym).upper()] = adjust_frame(df.copy(), ev, copy=False) if ev else df
    cov_p, uni_p = logs_path("ca_coverage.json"), logs_path("universe_history.json")
    return {"frames": frames, "coverage": json.loads(cov_p.read_text()) if cov_p.exists() else {},
            "universe_rows": json.loads(uni_p.read_text()) if uni_p.exists() else None,
            "ca_event_symbols": len(events)}


def run(root: str | Path, *, symbols: int, sample_step: int = 5) -> dict[str, Any]:
    from scan.sector_heat import _load_map
    copy = load_copy(root)
    frames = copy["frames"]
    collected = collect(frames, symbols=symbols, sample_step=sample_step)
    report = analyse(collected, frames, coverage=copy["coverage"], universe_rows=copy["universe_rows"],
                     sectors=_load_map())
    report["provenance"] = {**collected["provenance"], "sampled_symbols": len(collected["sampled_symbols"]),
                            "store_symbols": len(frames), "ca_event_symbols": copy["ca_event_symbols"],
                            "survivorship": symbol_end_truncations(frames, collected["sampled_symbols"],
                                                                   report["last_session"],
                                                                   sample_step=sample_step),
                            "universe_history_rows": len(copy["universe_rows"] or []),
                            "method": ("production run_backtest + core.evaluation_spec; nominal overlap-robust "
                                       "95% intervals per signal (research.overlap_inference); no multiplicity "
                                       "correction; in-sample diagnosis only")}
    return report


def summary_table(report: Mapping[str, Any], tier: str = "STRICT") -> str:
    rows = report["tiers"][tier]["signals"]
    lines = ["| Signal | Trades | Blocks | Net R mean [95% CI] | Gross R | MDE (80%) | Fill rate | Excess vs EW % | "
             "Verdict | Drivers |", "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(rows, key=lambda k: (k != ANY, -(rows[k]["trades"])))
    for k in order:
        r = rows[k]
        n = r["net_r"]
        ci = n.get("ci95") or [None, None]
        ex = r["excess_vs_equal_weight_return"].get("mean")
        fmt = (lambda v, d=3: "–" if v is None else f"{v:+.{d}f}")
        lines.append(f"| {k} | {r['trades']} | {n.get('independent_sessions') or 0} | {fmt(n.get('mean'))} "
                     f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(r['gross_r'].get('mean'))} | "
                     f"{fmt(r['mde_net_r_80pct_power'])} | {r['fill_rate'] if r['fill_rate'] is not None else '–'} | "
                     f"{fmt(ex * 100 if ex is not None else None, 2)} | {r['verdict']} | "
                     f"{', '.join(r['drivers']) or '–'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="SSD copy root holding logs/bhav/store_cache.pkl")
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--sample-step", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    # Every runtime path (scanner calibration snapshots, the backtest report path,
    # resolved at import) must point at the COPY, never the live runtime.
    root = Path(args.root).resolve()
    if str(root).startswith("/Volumes/"):
        raise SystemExit(f"refusing {root}: copy the store to the internal SSD first")
    os.environ["QT_RUNTIME_ROOT"] = str(root)
    rep = run(root, symbols=args.symbols, sample_step=args.sample_step)
    text = json.dumps(rep, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(summary_table(rep))
    print(json.dumps({k: rep[k] for k in ("coverage_start", "trades_captured", "trades_kept", "exclusions",
                                          "fates")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
