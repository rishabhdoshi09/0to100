"""Why the desk's setups lose: DESCRIPTIVE attribution on DISCOVERY data only (research only).

This is not a confirmatory test and spends no alpha. It explains the measured
absence of edge so the next hypothesis is chosen for an economic reason, not
by searching outcomes. Rules fixed before running:

* the PRODUCTION walk-forward (edge_diagnosis.collect → scan.signal_backtest),
  one rulebook (core.evaluation_spec), production costs;
* STRICT tier only (whole 52-week lookback corporate-action adjusted), verified
  point-in-time membership;
* DISCOVERY decisions only: decision date before ``HOLDOUT_START``. Decisions on
  or after it are COUNTED, never measured, so the next pre-registered
  hypothesis keeps one untouched year. (Disclosure: the 20 Nov edge diagnosis
  pooled per-signal statistics over every year, the holdout included; no
  statistic of the kinds reported here was computed on the holdout.)
* a small, pre-declared set of slices and counterfactual geometries; no
  parameter search. Intervals are overlap-robust (research.overlap_inference),
  with no multiplicity correction: a slice that looks good is a hypothesis.

Questions answered: where the loss comes from (signal, regime, breadth,
sector, exit reason, holding period, costs); beta versus selection alpha;
whether the stop/target geometry destroys an otherwise positive selection;
entry timing alternatives; and whether fewer, higher-confidence trades do
better.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

HOLDOUT_START = "2025-11-01"
HORIZON = 10
#: Pre-declared counterfactual exit geometries (applied to the SAME fills).
GEOMETRIES = ("PRODUCTION", "TIME_EXIT_10", "STOP_ONLY_10", "STOP_2X_TIME_10")


def breadth_series(frames: Mapping[str, pd.DataFrame], window: int = 50, min_names: int = 50) -> pd.Series:
    """Fraction of names whose close is above their own ``window``-session mean, per day (PIT: uses closes <= day)."""
    above, seen = defaultdict(int), defaultdict(int)
    for df in frames.values():
        if df is None or len(df) < window + 1:
            continue
        c = df["close"].astype(float)
        ma = c.rolling(window).mean()
        ok = ma.notna()
        idx = pd.DatetimeIndex(df.index[ok]).strftime("%Y-%m-%d")
        flags = (c[ok] > ma[ok]).to_numpy()
        for d, f in zip(idx, flags):
            seen[d] += 1
            above[d] += int(f)
    return pd.Series({d: above[d] / seen[d] for d in seen if seen[d] >= min_names}).sort_index()


def _path(frames, sym: str, decision: str, horizon: int = HORIZON):
    """Bars after the decision session: (open, high, low, close) arrays of the next ``horizon`` sessions."""
    df = frames.get(sym)
    if df is None:
        return None
    days = pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")
    pos = np.searchsorted(days, decision, side="right")
    if pos == 0 or days[pos - 1] != decision or pos + horizon > len(df):
        return None
    w = df.iloc[pos:pos + horizon]
    return (w["open"].to_numpy(float) if "open" in w else w["close"].to_numpy(float),
            w["high"].to_numpy(float), w["low"].to_numpy(float), w["close"].to_numpy(float),
            float(df["close"].iloc[pos - 1]))


def _geometry_r(fill: float, stop: float, highs, lows, closes, kind: str, cost_r: float) -> float | None:
    """Net R of one pre-declared exit geometry on the SAME fill (gap-naive: stop/target at their level)."""
    risk = fill - stop
    if risk <= 0:
        return None
    if kind == "TIME_EXIT_10":
        return (closes[-1] - fill) / risk - cost_r
    if kind in ("STOP_ONLY_10", "STOP_2X_TIME_10"):
        s = stop if kind == "STOP_ONLY_10" else fill - 2.0 * risk
        for lo, cl in zip(lows, closes):
            if lo <= s:
                return (s - fill) / risk - cost_r
        return (closes[-1] - fill) / risk - cost_r
    return None


def attribute(rows: list[dict], frames, ordinal, breadth: Mapping[str, float]) -> dict[str, Any]:
    from research.edge_diagnosis import _infer

    def infer(rs, key="net_r"):
        out = _infer(rs, key, ordinal)
        out["n"] = len([r for r in rs if r.get(key) is not None])
        return out

    def by(key_fn, rs, key="net_r", min_n=30):
        groups = defaultdict(list)
        for r in rs:
            groups[key_fn(r)].append(r)
        return {str(k): infer(v, key) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0])) if len(v) >= min_n}

    for r in rows:
        b = breadth.get(r["decision"])
        r["breadth"] = b
        r["breadth_bucket"] = ("UNKNOWN" if b is None else "NARROW_<40%" if b < 0.40
                               else "MIXED_40-60%" if b <= 0.60 else "BROAD_>60%")
        hp = int(r.get("holding_period") or 0)
        r["holding_bucket"] = "1-3" if hp <= 3 else "4-6" if hp <= 6 else "7-10"
        p = _path(frames, r["symbol"], r["decision"])
        r["raw_ret_10"] = r["raw_next_open_ret_10"] = r["mfe_r"] = r["mae_r"] = None
        for g in GEOMETRIES[1:]:
            r[f"geom_{g}"] = None
        if p is not None:
            opens, highs, lows, closes, dclose = p
            risk = r["entry_price"] - r["stop_price"]
            r["raw_ret_10"] = closes[-1] / dclose - 1.0                                  # decision-close entry
            r["raw_next_open_ret_10"] = closes[-1] / opens[0] - 1.0                     # next-open entry
            if risk > 0:
                r["mfe_r"] = (float(np.max(highs)) - r["entry_price"]) / risk
                r["mae_r"] = (float(np.min(lows)) - r["entry_price"]) / risk
                for g in GEOMETRIES[1:]:
                    r[f"geom_{g}"] = _geometry_r(r["entry_price"], r["stop_price"], highs, lows, closes, g,
                                                 r["cost_r"])
        bench = r.get("bench_ret")
        r["raw_excess_10"] = (r["raw_ret_10"] - bench) if (r["raw_ret_10"] is not None and bench is not None) else None

    # beta vs alpha: per-trade OLS of the trade's net return on the benchmark's
    pairs = [(r["bench_ret"], r["net_ret"]) for r in rows if r.get("bench_ret") is not None]
    beta = alpha = None
    if len(pairs) > 30:
        x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        alpha = float(np.mean(y) - beta * np.mean(x))
    exits = defaultdict(lambda: {"n": 0, "sum_r": 0.0})
    for r in rows:
        e = exits[str(r.get("exit_reason"))]
        e["n"] += 1
        e["sum_r"] += r["net_r"]
    total = sum(e["sum_r"] for e in exits.values()) or 1.0
    exit_table = {k: {"n": v["n"], "share_of_trades": round(v["n"] / len(rows), 3),
                      "mean_net_r": round(v["sum_r"] / v["n"], 4),
                      "sum_net_r": round(v["sum_r"], 2)} for k, v in sorted(exits.items())}
    conf = sorted(r["confidence"] for r in rows if r.get("confidence") is not None)
    q80 = conf[int(0.8 * (len(conf) - 1))] if conf else None
    return {
        "trades": len(rows),
        "pooled": {"net_r": infer(rows), "gross_r": infer(rows, "gross_r"),
                   "excess_ret_after_costs": infer(rows, "excess_ret"),
                   "mean_cost_r": round(float(np.mean([r["cost_r"] for r in rows])), 4)},
        "beta_alpha_per_trade": {"beta_vs_equal_weight": None if beta is None else round(beta, 3),
                                 "alpha_per_trade": None if alpha is None else round(alpha, 5),
                                 "note": "OLS of trade net return on the equal-weight market's return over the "
                                         "same window; overlap ignored (descriptive)"},
        "selection_without_geometry": {
            "decision_close_entry_ret_10": infer(rows, "raw_ret_10"),
            "next_open_entry_ret_10": infer(rows, "raw_next_open_ret_10"),
            "decision_close_excess_vs_market_10": infer(rows, "raw_excess_10"),
            "mfe_r_mean": round(float(np.nanmean([r["mfe_r"] for r in rows if r["mfe_r"] is not None])), 3),
            "mae_r_mean": round(float(np.nanmean([r["mae_r"] for r in rows if r["mae_r"] is not None])), 3),
        },
        "exit_geometry_counterfactuals_net_r": {
            "PRODUCTION": infer(rows),
            **{g: infer(rows, f"geom_{g}") for g in GEOMETRIES[1:]},
            "note": "same fills; gap-naive stop at its level (optimistic for stops); costs as production",
        },
        "exit_reason": exit_table,
        "exit_reason_note": f"sum of net R across all trades = {round(total, 2)}",
        "by_signal": by(lambda r: "+".join(sorted(r["signals"])) if len(r["signals"]) == 1 else "MULTI", rows),
        "by_each_signal": {s: infer([r for r in rows if s in r["signals"]])
                           for s in sorted({s for r in rows for s in r["signals"]})
                           if sum(1 for r in rows if s in r["signals"]) >= 30},
        "by_regime": by(lambda r: r["regime"], rows),
        "by_breadth": by(lambda r: r["breadth_bucket"], rows),
        "by_sector_top": dict(sorted(by(lambda r: r["sector"], rows, min_n=60).items(),
                                     key=lambda kv: -kv[1]["n"])[:10]),
        "by_holding_period": by(lambda r: r["holding_bucket"], rows),
        "by_signal_count": by(lambda r: min(3, r["n_signals"]), rows),
        "top_quintile_confidence": {"threshold": q80, "top": infer([r for r in rows if q80 is not None
                                                                    and r["confidence"] >= q80]),
                                    "rest": infer([r for r in rows if q80 is not None and r["confidence"] < q80])},
    }


def detection_selection(events, frames, ordinal, tiers, member, ew, *, holdout_start: str = HOLDOUT_START) -> dict:
    """Selection WITHOUT trade geometry, on EVERY detection (filled or not), entered at the decision close.

    Measuring "selection" on filled trades only is look-ahead conditioning: a
    buy-stop fills only after the price rises through its trigger, so filled
    trades are selected on their own future (it showed +3.1% / 10 sessions).
    Here every STRICT, member, discovery detection with a valid plan counts,
    and the split by fate shows the conditioning itself.
    """
    from research.edge_diagnosis import _infer, is_member

    ew_by_day = {str(k)[:10]: float(v) for k, v in ew.items()}
    days = sorted(ordinal, key=ordinal.get)
    rows = []
    for e in events:
        if e.get("fate") not in ("FILLED", "NO_FILL"):
            continue
        d = e.get("decision")
        if d is None or d not in ordinal or d >= holdout_start or tiers.get(d) != "STRICT":
            continue
        if is_member(member, e["symbol"], d) is not True:
            continue
        p = _path(frames, e["symbol"], d)
        o = ordinal[d]
        if p is None or o + HORIZON >= len(days):
            continue
        end = days[o + HORIZON]
        b0, b1 = ew_by_day.get(d), ew_by_day.get(end)
        r10 = p[3][-1] / p[4] - 1.0
        rows.append({"decision": d, "fate": e["fate"], "r10": r10,
                     "excess10": (r10 - (b1 / b0 - 1.0)) if b0 and b1 else None})
    out = {"all_detections": {"ret_10": _infer(rows, "r10", ordinal), "excess_vs_market_10": _infer(rows, "excess10", ordinal)}}
    for fate in ("FILLED", "NO_FILL"):
        sub_rows = [r for r in rows if r["fate"] == fate]
        out[fate.lower()] = {"n": len(sub_rows), "ret_10": _infer(sub_rows, "r10", ordinal),
                             "excess_vs_market_10": _infer(sub_rows, "excess10", ordinal)}
    return out


def run(root: str | Path, *, symbols: int, sample_step: int = 5, accept_unbound_coverage: bool = False) -> dict:
    from research import edge_diagnosis as ED
    from scan.sector_heat import _load_map
    from scan.signal_backtest import realised_session_index

    copy = ED.load_copy(root, accept_unbound_coverage=accept_unbound_coverage)
    frames = copy["frames"]
    collected = ED.collect(frames, symbols=symbols, sample_step=sample_step)
    all_dates = sorted({d for df in frames.values() for d in pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")})
    ordinal = realised_session_index(all_dates)
    cover, cover_end = ED.coverage_span(copy["coverage"])
    tiers = ED.tier_map(ordinal, cover, cover_end)
    ew = ED.equal_weight_index(frames)
    rows = ED.enrich(list(collected["trades"]), frames, ew, ED.regime_series(ew), ordinal, tiers,
                     ED.membership(copy["universe_rows"]), _load_map())
    strict = [r for r in rows if r["tier"] == "STRICT" and r["member"] is True
              and math.isfinite(r["risk_frac"]) and r["risk_frac"] > 0]
    discovery = [r for r in strict if r["decision"] < HOLDOUT_START]
    holdout_n = sum(1 for r in strict if r["decision"] >= HOLDOUT_START)
    report = attribute(discovery, frames, ordinal, breadth_series(frames))
    report["selection_on_every_detection"] = detection_selection(
        collected["events"], frames, ordinal, tiers, ED.membership(copy["universe_rows"]), ew)
    report["selection_without_geometry"]["WARNING"] = (
        "measured on FILLED trades only: conditioned on the price rising through the trigger (look-ahead); "
        "see selection_on_every_detection for the unconditioned measure")
    report["provenance"] = {**collected["provenance"], "sampled_symbols": len(collected["sampled_symbols"]),
                            "holdout_start": HOLDOUT_START, "holdout_trades_counted_not_measured": holdout_n,
                            "discovery_first_decision": min((r["decision"] for r in discovery), default=None),
                            "discovery_last_decision": max((r["decision"] for r in discovery), default=None),
                            "ca_coverage_binding": copy.get("coverage_binding"),
                            "method": "descriptive attribution; production walk-forward; STRICT tier; discovery only"}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", type=int, default=400)
    ap.add_argument("--accept-unbound-coverage", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run(a.root, symbols=a.symbols, accept_unbound_coverage=a.accept_unbound_coverage)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("trades", "pooled", "beta_alpha_per_trade", "selection_without_geometry",
                                          "exit_geometry_counterfactuals_net_r")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
