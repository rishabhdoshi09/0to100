"""
🏛️📈 Momentum Gauntlet — EXP-003.

Runs the cross-sectional momentum factor (scan.momentum) through the SAME honest
statistical battery as the breakout gauntlet: alpha-vs-Nifty, correlation-aware
block-bootstrap CI, Deflated Sharpe, and a regime split. ONE pre-registered config
(12-1, monthly, top-20) — no lookback/holding sweep (that would be data-mining).

Verdict mapping (same as the main runner):
  harness PROMOTE → PASS · REJECT → FAIL · else → INCONCLUSIVE.
The strategy return stream is monthly, so with only a few years of data the honest
outcome is often INCONCLUSIVE ("need more history") rather than a false PASS.
"""
from __future__ import annotations

import numpy as np

from research import harness as H
from scan import momentum as MOM


def _regime_split(strat, dates_idx, regime_series, bench_index_dates):
    """Bucket each rebalance's return by the Nifty regime at that date."""
    out: dict[str, list] = {}
    if regime_series is None:
        return {}
    for r, di in zip(strat, dates_idx):
        try:
            lbl = str(regime_series.asof(bench_index_dates[di]))
        except Exception:
            lbl = "UNKNOWN"
        out.setdefault(lbl or "UNKNOWN", []).append(r)
    return {k: {"n": len(v), "mean_pct": round(float(np.mean(v)) * 100, 2)}
            for k, v in out.items()}


def run_momentum_gauntlet(top_n: int = 20, lookback: int = 252, skip: int = 21,
                          rebalance: int = 21, min_turnover_cr: float = 5.0,
                          cost_pct: float = 0.32, seed: int = 1,
                          skip_validation: bool = True) -> dict:
    """Build the momentum return series from the bhav store + Nifty, then judge it.
    Returns a report dict (report.build_report-compatible shape is not required —
    this has its own renderer)."""
    from gauntlet import freeze as FZ, registry as REG

    # load the store into THIS process (fresh CLI process)
    try:
        from data.bhavcopy_store import is_ready, build_store, store_symbols, get_ohlcv
        if not is_ready():
            build_store()
        syms = store_symbols()
    except Exception as e:
        return {"aborted": True, "reason": f"store unavailable: {e}"}

    # Nifty benchmark (daily close) as the master calendar
    try:
        from data.index_store import get_index_ohlcv
        nifty_df = get_index_ohlcv("^NSEI")
        col = next((c for c in ("Close", "close") if nifty_df is not None
                    and c in nifty_df.columns), None)
        nifty_close = nifty_df[col] if col else None
    except Exception:
        nifty_close = None
    if nifty_close is None or len(nifty_close) < lookback + skip + rebalance + 2:
        return {"aborted": True, "reason": "Nifty index history unavailable/too short"}

    import pandas as pd
    cal = pd.DatetimeIndex(nifty_close.index)
    bench = nifty_close.to_numpy(dtype=float)

    # align every symbol's close/volume to the Nifty calendar
    closes, volumes = {}, {}
    for s in syms:
        df = get_ohlcv(s)
        if df is None or "close" not in df.columns or len(df) < lookback:
            continue
        c = df["close"].reindex(cal)
        v = (df["volume"].reindex(cal) if "volume" in df.columns
             else pd.Series(np.nan, index=cal))
        closes[s] = c.to_numpy(dtype=float)
        volumes[s] = v.to_numpy(dtype=float)

    series = MOM.build_momentum_series(
        closes, volumes, bench, top_n=top_n, lookback=lookback, skip=skip,
        rebalance=rebalance, min_turnover_cr=min_turnover_cr, cost_pct=cost_pct)

    strat = series["strat_returns"]
    benchr = series["bench_returns"]
    if strat.size < 2:
        return {"aborted": True,
                "reason": f"too few rebalances ({strat.size}) — need more history"}

    frozen = FZ.freeze()
    # monthly returns → min_detectable_r set to a monthly scale (1% edge)
    verdict = H.evaluate(strat, benchmark_returns=benchr, require_block_ci=True,
                         block_ci_seed=seed, min_detectable_r=0.01)
    st = verdict.stats
    ab = st.get("alpha_beta")
    ci = st.get("block_ci") or {"ci_lower": 0.0, "ci_upper": 0.0, "excludes_zero": False}
    ann = MOM.annualise(strat)
    ann_bench = MOM.annualise(benchr)

    try:
        from scan.signal_backtest import _nifty_regime_series
        regimes = _regime_split(strat, series["dates_idx"], _nifty_regime_series(), cal)
    except Exception:
        regimes = {}

    verdict_word = ("PASS" if verdict.verdict == "PROMOTE"
                    else "FAIL" if verdict.verdict == "REJECT" else "INCONCLUSIVE")

    exp = REG.register(frozen["hash"],
                       {"strategy": "momentum_12_1", "rebalances": int(strat.size),
                        "top_n": top_n, "symbols": series["n_symbols"]},
                       seed, extra={"verdict": verdict_word})

    return {"aborted": False, "experiment": exp, "freeze_hash": frozen["hash"],
            "verdict": verdict_word, "harness_verdict": verdict.verdict,
            "n_rebalances": int(strat.size), "n_symbols": series["n_symbols"],
            "avg_names": round(series["avg_names"], 1),
            "avg_turnover_pct": round(series["avg_turnover"] * 100, 1),
            "mean_monthly_pct": round(float(st.get("mean_r", 0)) * 100, 3),
            "ci_monthly_pct": [round(ci["ci_lower"] * 100, 3), round(ci["ci_upper"] * 100, 3)],
            "ci_excludes_zero": bool(ci["excludes_zero"]),
            "deflated_sharpe": round(verdict.dsr, 4), "p_value": verdict.p_value,
            "alpha_monthly_pct": None if ab is None else round(ab["alpha"] * 100, 3),
            "beta": None if ab is None else round(ab.get("beta", 0.0), 3),
            "beats_benchmark": None if ab is None else ab["beats_benchmark"],
            "strategy": ann, "benchmark": ann_bench, "regimes": regimes,
            "insight": verdict.insight}


def to_markdown(r: dict) -> str:
    if r.get("aborted"):
        return f"# Momentum Gauntlet — ABORTED\n\n**Reason:** {r.get('reason')}"
    s, b = r["strategy"], r["benchmark"]
    return "\n".join([
        "# Momentum Gauntlet (EXP-003) — Cross-Sectional 12-1 Momentum",
        f"_experiment `{r['experiment'].get('experiment_id','?')}` · "
        f"freeze `{r['freeze_hash']}`_", "",
        f"## Verdict: **{r['verdict']}**   _({r['insight']})_", "",
        f"- Rebalances: {r['n_rebalances']} (monthly) · liquid universe "
        f"{r['n_symbols']} names · avg {r['avg_names']} held · "
        f"avg turnover {r['avg_turnover_pct']}%/mo",
        f"- Mean monthly: {r['mean_monthly_pct']:+}% · correlation-aware CI "
        f"[{r['ci_monthly_pct'][0]:+}, {r['ci_monthly_pct'][1]:+}]% "
        f"({'excludes 0 ✅' if r['ci_excludes_zero'] else 'includes 0'})",
        f"- **Alpha vs Nifty: {r['alpha_monthly_pct']}%/mo** · beta {r['beta']} · "
        f"beats benchmark: {r['beats_benchmark']}",
        f"- Deflated Sharpe {r['deflated_sharpe']} · p={r['p_value']:.4g}", "",
        f"- **Strategy:** CAGR {s['cagr_pct']}% · vol {s['vol_pct']}% · "
        f"Sharpe {s['sharpe']} · max DD {s['max_dd_pct']}%",
        f"- **Nifty:**    CAGR {b['cagr_pct']}% · vol {b['vol_pct']}% · "
        f"Sharpe {b['sharpe']} · max DD {b['max_dd_pct']}%",
        f"- Regimes: {r['regimes']}", "",
        "_Pre-registered: 12-1 momentum, monthly rebalance, equal-weight top-20, "
        "liquid names, 0.32% round-trip cost on turnover. One config, no sweep._",
    ])


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m gauntlet.momentum")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = run_momentum_gauntlet(top_n=args.top, seed=args.seed)
    if args.json:
        import json
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(to_markdown(rep))
    return 2 if rep.get("aborted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
