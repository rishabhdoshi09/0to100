"""
E2 — Historical Gauntlet Runner.

One path: clean data → trade ledger → the full statistical battery → exactly one
verdict per strategy (PASS / FAIL / INCONCLUSIVE, no intermediate wording).

The battery is the harness, applied honestly:
  expectancy + block-bootstrap CI (correlation-aware) · effective sample size ·
  Deflated Sharpe (deflated across the number of strategies tried) · alpha-vs-beta
  attribution against the benchmark · Benjamini-Hochberg FDR across strategies.
  White's Reality Check is computed at the PANEL level and REPORTED as a
  data-snooping diagnostic (the best strategy's p-value) — it is not a per-strategy
  gate; the per-strategy multiplicity control is DSR-deflation + FDR.

A strategy PASSes only if the harness PROMOTEs it AND it survives FDR. Purged CV
indices are available from the harness for any in-sample scoring; this runner
scores realised outcomes (already out-of-sample by construction of walk-forward),
so it reports the CV capability without needing to refit here.
"""
from __future__ import annotations

import numpy as np

from research import harness as H
from gauntlet import ledger as L


def _profit_factor(rr) -> float:
    r = np.asarray(rr, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def _equity_and_drawdown(records):
    """Modeled equity curve at a fixed 1%-risk-per-trade (additive R curve for the
    drawdown, and a compounded account for a MODELED CAGR clearly labelled as
    model-dependent). Ordered by exit date."""
    recs = sorted(records, key=lambda x: x.exit_datetime)
    r = np.array([x.net_R for x in recs], dtype=float)
    if r.size == 0:
        return {"total_R": 0.0, "max_drawdown_R": 0.0, "max_drawdown_pct": 0.0,
                "modeled_cagr_pct": None, "curve_points": 0}
    curve = np.cumsum(r)                        # r.size ≥ 1 past the early return
    dd_R = float(np.max(np.maximum.accumulate(curve) - curve))
    # modeled account: 1% risk/trade, compounded
    acct = np.cumprod(1.0 + 0.01 * r)
    apeak = np.maximum.accumulate(acct)
    dd_pct = float(np.max((apeak - acct) / apeak) * 100)
    cagr = None
    try:
        import pandas as pd
        d0 = pd.Timestamp(recs[0].exit_datetime)
        d1 = pd.Timestamp(recs[-1].exit_datetime)
        yrs = max((d1 - d0).days / 365.25, 1e-6)
        if yrs > 0.05:
            cagr = float((acct[-1] ** (1.0 / yrs) - 1.0) * 100)
    except Exception:
        cagr = None
    return {"total_R": round(float(curve[-1]), 2), "max_drawdown_R": round(dd_R, 2),
            "max_drawdown_pct": round(dd_pct, 2),
            "modeled_cagr_pct": None if cagr is None else round(cagr, 2),
            "curve_points": int(r.size)}


def _regime_breakdown(records) -> dict:
    out: dict[str, dict] = {}
    by: dict[str, list] = {}
    for x in records:
        by.setdefault(x.regime or "UNKNOWN", []).append(x.net_R)
    for reg, rr in by.items():
        a = np.asarray(rr, dtype=float)
        out[reg] = {"trades": int(a.size), "expectancy_r": round(float(a.mean()), 3)}
    return out


def _bench_in_R(records):
    """Benchmark return per trade expressed in the trade's OWN R units
    (market_pct ÷ risk_frac), paired with the strategy's net R. Returns
    (strat_R, bench_R) over trades where a benchmark return exists."""
    s, b = [], []
    for x in records:
        br = x.benchmark_return_during_trade
        rf = x.risk_frac
        if br is None or rf <= 0:
            continue
        s.append(x.net_R)
        b.append(br / rf)
    return np.asarray(s, dtype=float), np.asarray(b, dtype=float)


def _map_verdict(h_verdict: str, fdr_significant: bool) -> str:
    if h_verdict == "PROMOTE":
        return "PASS" if fdr_significant else "INCONCLUSIVE"
    if h_verdict == "REJECT":
        return "FAIL"
    return "INCONCLUSIVE"          # UNDERPOWERED / INCONCLUSIVE


def evaluate_strategy(records, n_trials: int = 1, seed: int = 1) -> dict:
    """The full battery for ONE strategy's realised ledger. Delegates the entire
    computation to a SINGLE `H.evaluate(...)` call — the block-bootstrap CI, the
    alpha/beta fit, effective-N and the deflated Sharpe are read back off the
    verdict's `stats`, never recomputed here (so the report can never show a CI
    that differs from the one the gate used). FDR significance is decided by the
    runner across strategies and folded in later."""
    r = np.array([x.net_R for x in records], dtype=float)
    _, bench_R = _bench_in_R(records)
    bench_available = bench_R.size >= 3 and float(np.std(bench_R)) > 0
    verdict = H.evaluate(
        r, n_trials=n_trials, require_block_ci=True, block_ci_seed=seed,
        benchmark_returns=(bench_R if bench_available else None))
    st = verdict.stats
    ci = st.get("block_ci") or {"ci_lower": 0.0, "ci_upper": 0.0,
                                "excludes_zero": False}
    ab = st.get("alpha_beta")
    return {
        "n": verdict.n, "n_effective": round(st.get("n_eff", float(verdict.n)), 1),
        "expectancy_r": round(st.get("mean_r", 0.0), 4),
        "ci_lower": round(ci["ci_lower"], 4), "ci_upper": round(ci["ci_upper"], 4),
        "ci_excludes_zero": bool(ci["excludes_zero"]),
        "profit_factor": round(_profit_factor(r), 3),
        "sharpe": round(st.get("sharpe", 0.0), 4),
        "deflated_sharpe": round(verdict.dsr, 4), "p_value": verdict.p_value,
        "alpha": None if ab is None else round(ab["alpha"], 4),
        "beta": None if ab is None else round(ab.get("beta", 0.0), 4),
        "beats_benchmark": None if ab is None else ab["beats_benchmark"],
        "benchmark_tested": bench_available,
        "regime_breakdown": _regime_breakdown(records),
        "equity": _equity_and_drawdown(records),
        "harness_verdict": verdict.verdict,
        "harness_insight": verdict.insight,
    }


def _reality_check_p(by_strategy: dict, seed: int = 1):
    """White's Reality Check across strategies: monthly mean net-R per strategy as
    the (T periods × L strategies) performance matrix. Best-effort — None when the
    panel is too thin."""
    try:
        import pandas as pd
        periods, cols = set(), {}
        for name, recs in by_strategy.items():
            ser = {}
            for x in recs:
                pm = str(pd.Timestamp(x.exit_datetime).to_period("M"))
                ser.setdefault(pm, []).append(x.net_R)
            cols[name] = {k: float(np.mean(v)) for k, v in ser.items()}
            periods |= set(ser)
        if len(cols) < 2 or len(periods) < 6:
            return None
        idx = sorted(periods)
        mat = np.array([[cols[name].get(p, 0.0) for name in cols] for p in idx])
        return H.whites_reality_check(mat, n_boot=1000, seed=seed)["reality_check_p"]
    except Exception:
        return None


def run_gauntlet(ledger=None, trade_source=None, n_trials: int | None = None,
                 seed: int = 1, index_close=None, factor_closes=None,
                 skip_validation: bool = False, factors_enabled: bool = False) -> dict:
    """Orchestrate E4→E1→E2→E5. Returns a raw results dict (report.build_report
    turns it into the committee document). Provide `ledger` (list[TradeRecord]) or
    `trade_source` (iterable of raw trade dicts) for tests / offline runs; with
    neither, the real backtest is driven and enriched from the index store."""
    from gauntlet import freeze as FZ, validator as V, registry as REG

    # ── E4: abort-on-fail dataset gate ────────────────────────────────────────
    validation = {"ok": True, "checks": [], "failed": [], "skipped": True}
    if not skip_validation:
        validation = V.validate(factors_enabled=factors_enabled)
        if not validation["ok"]:
            return {"aborted": True, "reason": "dataset validation failed",
                    "validation": validation, "strategies": {}}

    # ── E6: freeze the config for the run ─────────────────────────────────────
    frozen = FZ.freeze()

    # ── E1: build / accept the ledger ─────────────────────────────────────────
    if ledger is None:
        builder = L.LedgerBuilder()
        if trade_source is not None:
            for rec in trade_source:
                builder(rec)
        else:                                   # drive the real backtest
            # `python -m gauntlet` is a FRESH process — the bhav store isn't in
            # memory yet (build_store ran in a different process). Load it (from
            # the pickle cache if present) or the backtest sees 0 symbols.
            try:
                from data.bhavcopy_store import is_ready, build_store
                if not is_ready():
                    build_store()
            except Exception:
                pass
            from scan.signal_backtest import run_backtest
            run_backtest(on_trade=builder)
        if index_close is None:
            try:
                from data.index_store import get_index_ohlcv
                nifty = get_index_ohlcv("^NSEI")
                col = "Close" if nifty is not None and "Close" in nifty.columns else None
                index_close = nifty[col] if col else None
            except Exception:
                index_close = None
        ledger = builder.finalize(index_close=index_close, factor_closes=factor_closes)

    by_strategy = L.per_strategy(ledger)
    n_trials = n_trials if n_trials is not None else max(1, len(by_strategy))

    # ── E2: per-strategy battery ──────────────────────────────────────────────
    results = {name: evaluate_strategy(recs, n_trials=n_trials, seed=seed)
               for name, recs in by_strategy.items()}

    # FDR across strategies — the multiple-testing correction
    names = list(results)
    fdr = dict.fromkeys(names, False)
    if names:
        bh = H.benjamini_hochberg([results[k]["p_value"] for k in names])
        fdr = {k: bool(sig) for k, sig in zip(names, bh["rejected"])}
    for k in names:
        results[k]["fdr_significant"] = fdr[k]
        results[k]["verdict"] = _map_verdict(results[k]["harness_verdict"], fdr[k])

    reality_p = _reality_check_p(by_strategy, seed=seed)

    # ── E5: register the experiment ───────────────────────────────────────────
    fingerprint = {"n_trades": len(ledger), "n_strategies": len(by_strategy),
                   "strategies": sorted(by_strategy)}
    exp = REG.register(frozen["hash"], fingerprint, seed,
                       extra={"reality_check_p": reality_p})

    return {"aborted": False, "validation": validation, "freeze": frozen,
            "experiment": exp, "n_trades": len(ledger),
            "reality_check_p": reality_p, "strategies": results,
            "n_trials": n_trials, "seed": seed}
