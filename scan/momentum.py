"""
📈 Cross-Sectional Momentum — the factor EXP-002's evidence points to.

EXP-002 killed the short-term breakout signals: costs turned near-breakeven edges
negative, every signal had negative alpha vs Nifty, and high turnover compounded
the cost drag. This module tests the opposite: a LOW-turnover, evidence-backed
factor — buy the strongest 12-1 momentum names, hold a month, rebalance monthly.

Why this is the right next hypothesis (not a tweak):
  • Monthly rebalance ⇒ ~12 decisions/year, not thousands ⇒ the cost drag that
    sank the scalps barely bites.
  • Momentum is one of the most replicated anomalies globally (incl. India), so a
    positive result is far less likely to be data-mining than a bespoke pattern.
  • It is a different STRATEGY CLASS — factor investing, not active trading.

`build_momentum_series()` is PURE over price/volume history: it produces the
strategy's per-period return stream and the paired Nifty return, which the gauntlet
harness then judges (alpha, block-bootstrap CI, DSR, Reality Check, regimes) exactly
like any other claim. No fishing: ONE pre-registered config (12-1, monthly, top-20).
"""
from __future__ import annotations

import numpy as np


def _liquid_symbols(closes: dict, volumes: dict, min_turnover_cr: float) -> list:
    """Keep only names whose MEDIAN daily turnover (price × volume) clears the bar
    — momentum on illiquid microcaps is where untradeable fake edges hide."""
    out = []
    thresh = min_turnover_cr * 1e7          # ₹ crore → ₹
    for s, c in closes.items():
        v = volumes.get(s)
        if c is None or v is None or len(c) < 60 or len(v) < 60:
            continue
        n = min(len(c), len(v))
        turn = np.asarray(c[-n:], float) * np.asarray(v[-n:], float)
        turn = turn[~np.isnan(turn)]
        if turn.size and float(np.median(turn)) >= thresh:
            out.append(s)
    return out


def trend_gate_from(close, window: int = 200):
    """Boolean 'risk-on' gate: True on days the benchmark closes AT/ABOVE its
    `window`-day SMA (the classic trend filter). NaN warm-up ⇒ risk-off. Used to
    put a momentum book into CASH during downtrends — the EXP-004 hypothesis that
    targets EXP-003's 44% drawdown."""
    import pandas as pd
    c = np.asarray(close, dtype=float)
    sma = pd.Series(c).rolling(window).mean().to_numpy()
    gate = c >= sma
    gate[np.isnan(sma)] = False
    return gate


def build_momentum_series(closes: dict, volumes: dict, benchmark,
                          top_n: int = 20, lookback: int = 252, skip: int = 21,
                          rebalance: int = 21, min_price: float = 20.0,
                          min_turnover_cr: float = 5.0,
                          cost_pct: float = 0.32, trend_gate=None) -> dict:
    """Cross-sectional 12-1 momentum, equal-weight top_n, periodic rebalance. PURE.

    `closes`/`volumes`: {symbol: 1-D array-like of daily close/volume}, all aligned
    to `benchmark`'s daily index (same length, same dates — reindex before calling).
    `lookback`/`skip`/`rebalance` are in TRADING DAYS (252≈12m, 21≈1m). Costs are a
    round-trip %, charged on the FRACTION of the book that turns over each rebalance.

    Returns {strat_returns, bench_returns, dates_idx, n_rebalances, avg_turnover,
    avg_names} — the net-of-cost per-period strategy return and the paired benchmark
    return, ready for harness.evaluate(returns, benchmark_returns=...)."""
    bench = np.asarray(benchmark, dtype=float)
    T = bench.size
    syms = _liquid_symbols(closes, volumes, min_turnover_cr)
    if not syms or T < lookback + skip + rebalance + 1:
        return {"strat_returns": np.zeros(0), "bench_returns": np.zeros(0),
                "dates_idx": [], "n_rebalances": 0, "avg_turnover": 0.0,
                "avg_names": 0.0, "n_symbols": len(syms), "pct_invested": 0.0}
    # price matrix aligned to the benchmark calendar: (T dates, S symbols)
    A = np.column_stack([np.asarray(closes[s], dtype=float)[:T] if len(closes[s]) >= T
                         else np.full(T, np.nan) for s in syms])

    strat, bench_r, dates = [], [], []
    turnovers, n_names = [], []
    invested_flags = []
    prev = set()
    start = lookback + skip
    p = start
    while p + rebalance < T:
        # ── EXP-004 trend filter: benchmark below its 200-DMA ⇒ hold CASH ──────
        if trend_gate is not None and not bool(trend_gate[p]):
            strat.append(0.0)                    # cash earns 0 (conservative)
            bench_r.append(float(bench[p + rebalance] / bench[p] - 1.0))
            dates.append(p); turnovers.append(0.0); n_names.append(0)
            invested_flags.append(False)
            prev = set()                         # liquidated to cash
            p += rebalance
            continue
        past = A[p - skip]                       # price 1m ago
        older = A[p - skip - lookback]           # price 12m+1m ago
        now = A[p]                               # price at rebalance
        fut = A[p + rebalance]                   # price 1m forward
        mom = past / older - 1.0
        fwd = fut / now - 1.0
        ok = (~np.isnan(mom) & ~np.isnan(fwd) & (older > 0) & (now > 0)
              & (now >= min_price))
        idx = np.where(ok)[0]
        if idx.size >= top_n:
            winners = idx[np.argsort(mom[idx])[::-1][:top_n]]   # top_n by momentum
            port = float(np.mean(fwd[winners]))
            names = set(winners.tolist())
            turn = len(names - prev) / float(top_n) if prev else 1.0
            cost = turn * (cost_pct / 100.0)     # round-trip on the turned-over slice
            strat.append(port - cost)
            bench_r.append(float(bench[p + rebalance] / bench[p] - 1.0))
            dates.append(p)
            turnovers.append(turn)
            n_names.append(len(names))
            invested_flags.append(True)
            prev = names
        p += rebalance
    return {"strat_returns": np.asarray(strat), "bench_returns": np.asarray(bench_r),
            "dates_idx": dates, "n_rebalances": len(strat),
            "pct_invested": round(float(np.mean(invested_flags)) * 100, 1)
            if invested_flags else 100.0,
            "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
            "avg_names": float(np.mean(n_names)) if n_names else 0.0,
            "n_symbols": len(syms)}


def annualise(period_returns, periods_per_year: float = 12.0) -> dict:
    """Summary stats for a periodic return stream: CAGR, annualised vol/Sharpe,
    and the worst peak-to-trough drawdown on the compounded curve."""
    r = np.asarray(period_returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {"cagr_pct": 0.0, "vol_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    curve = np.cumprod(1.0 + r)
    yrs = r.size / periods_per_year
    cagr = curve[-1] ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(periods_per_year) if r.size > 1 else 0.0
    sharpe = (float(np.mean(r)) * periods_per_year) / vol if vol > 0 else 0.0
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max((peak - curve) / peak)) if curve.size else 0.0
    return {"cagr_pct": round(cagr * 100, 2), "vol_pct": round(vol * 100, 2),
            "sharpe": round(sharpe, 3), "max_dd_pct": round(max_dd * 100, 2)}
