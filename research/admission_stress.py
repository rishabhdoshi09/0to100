"""Null experiments for the live BUY gate's multiple-testing admission (reproducible).

The gate admits a signal or exact combination when its walk-forward mean is
at least +0.05R, its overlap-robust one-sided 95% lower bound is above zero,
and it survives a family correction across every row the gate judges. This
harness measures how often that admits a row whose TRUE edge is zero, under
the structures the real gate faces:

* correlated rows: signals load on shared market and style factors, their
  10-session outcome windows overlap, and combinations reuse their
  constituents' trades;
* family construction: production-like (every row with >= 30 trades) or
  adversarial (combinations chosen by their in-sample mean from a larger
  pool, then tested on the same data);
* missing tests: rows below the trade floor drop out;
* repeated looks: the report is re-run weekly on a rolling 250-session
  window, so the same data are retested 26 times in half a year.

Reported per procedure (none, BH, BY, Holm, Bonferroni): the per-look
probability of at least one false admission, and the probability of at least
one in ANY of the looks, each with a Wilson 95% interval, plus power when one
signal has a real edge. The procedure in production was chosen on theory
before these runs, not tuned on them.

    python -m research.admission_stress --sims 200 --out docs/research/admission_stress_2026-11-20.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np

from research import overlap_inference as OI
from research.overlap_stress import wilson

H = 10
WINDOW = 250
BAR = 0.05
ALPHA = 0.05
MIN_TRADES = 30
PROCEDURES = ("none", "bh", "by", "holm", "bonferroni", "holm_annual")
LOOKS_PER_YEAR = 52


def adjust(pvals: list[float], method: str, alpha: float = ALPHA) -> list[bool]:
    """Rejections (input order) for a family of one-sided p-values."""
    m = len(pvals)
    if m == 0:
        return []
    if method == "holm_annual":                     # Holm at alpha / 52: the annual union bound
        return adjust(pvals, "holm", alpha / LOOKS_PER_YEAR)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [False] * m
    if method == "none":
        return [p < alpha for p in pvals]
    if method == "bonferroni":
        return [p < alpha / m for p in pvals]
    if method == "holm":
        for rank, i in enumerate(order):
            if pvals[i] < alpha / (m - rank):
                out[i] = True
            else:
                break
        return out
    c = 1.0 if method == "bh" else sum(1.0 / k for k in range(1, m + 1))     # BY: arbitrary dependence
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank * alpha / (m * c):
            kmax = rank
    for rank, i in enumerate(order, start=1):
        out[i] = rank <= kmax
    return out


def simulate_rows(rng, *, looks: int, signals: int = 12, edge: dict | None = None, factor: float = 1.5):
    """Per-row daily [sum R, count] arrays over the whole horizon of looks."""
    T = WINDOW + (looks - 1) * 5 + H + 1
    market = rng.normal(size=T)
    styles = rng.normal(size=(3, T))
    regime = np.cumsum(rng.normal(0, 0.15, size=T))                     # slow trend: firing co-moves

    def windows(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        n = T - H - 1
        return (c[H + 1:n + H + 1] - c[1:n + 1]) / H

    wm = windows(market)
    ws = np.array([windows(s) for s in styles])
    n = len(wm)
    style_of = rng.integers(0, 3, size=signals)
    beta = rng.uniform(0.3, 1.2, size=signals)
    lam = rng.uniform(0.25, 1.4, size=signals)
    per_trade = []                                                     # per signal: (day, R) lists
    fire = np.zeros((signals, n), dtype=int)
    for j in range(signals):
        rate = lam[j] * np.exp(0.6 * np.tanh(regime[:n]) * (1 if j % 2 else -1))
        counts = rng.poisson(rate)
        fire[j] = counts
        days = np.repeat(np.arange(n), counts)
        mu = (edge or {}).get(f"S{j}", 0.0)
        r = mu + beta[j] * wm[days] * factor + 0.8 * ws[style_of[j], days] * factor + rng.normal(0, 1.2, size=len(days))
        per_trade.append((days, r))
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j, (days, r) in enumerate(per_trade):
        s = np.bincount(days, weights=r, minlength=n)
        c = np.bincount(days, minlength=n)
        rows[f"S{j}"] = (s, c)
    for a, b in itertools.combinations(range(signals), 2):            # a combination trades when both fire
        both = np.minimum(fire[a], fire[b])
        if both.sum() == 0:
            continue
        days = np.repeat(np.arange(n), both)
        mu = ((edge or {}).get(f"S{a}", 0.0) + (edge or {}).get(f"S{b}", 0.0)) / 2
        r = (mu + 0.5 * (beta[a] + beta[b]) * wm[days] * factor
             + 0.4 * (ws[style_of[a], days] + ws[style_of[b], days]) * factor + rng.normal(0, 1.2, size=len(days)))
        rows[f"S{a}|S{b}"] = (np.bincount(days, weights=r, minlength=n), np.bincount(days, minlength=n))
    return rows


def test_row(s: np.ndarray, c: np.ndarray, start: int) -> tuple[float, float, int] | None:
    """(mean, one-sided p, trades) for one row in one look window; None below the floor."""
    sl = slice(start, start + WINDOW)
    ss, cc = s[sl], c[sl]
    trades = int(cc.sum())
    if trades < MIN_TRADES:
        return None
    cells = {int(i): [float(ss[i]), int(cc[i])] for i in np.flatnonzero(cc)}
    rep = OI.overlap_robust_test(cells, H)
    if rep["mean"] is None or not rep["se"] or rep["df"] is None:
        return float(rep["mean"] or 0.0), 1.0, trades
    p = OI._t_sf(rep["mean"] / rep["se"], rep["df"])
    return float(rep["mean"]), float(p), trades


def one_sim(rng, *, looks: int, adaptive: bool, edge: dict | None = None, signals: int = 12,
            factor: float = 1.5) -> dict:
    rows = simulate_rows(rng, looks=looks, edge=edge, signals=signals, factor=factor)
    true_pos = {k for k in rows if any(part in (edge or {}) for part in k.split("|"))}
    per_look = {m: [] for m in PROCEDURES}
    admitted_true = {m: 0 for m in PROCEDURES}
    for look in range(looks):
        start = look * 5
        tested = {}
        for key, (s, c) in rows.items():
            res = test_row(s, c, start)
            if res is not None:
                tested[key] = res
        if adaptive:
            combos = sorted((k for k in tested if "|" in k), key=lambda k: -tested[k][0])[:20]
            keys = [k for k in tested if "|" not in k] + combos
        else:
            keys = list(tested)
        pvals = [tested[k][1] for k in keys]
        for m in PROCEDURES:
            rej = adjust(pvals, m)
            admitted = [k for k, r in zip(keys, rej) if r and tested[k][0] >= BAR]
            per_look[m].append(sum(1 for k in admitted if k not in true_pos))
            admitted_true[m] += sum(1 for k in admitted if k in true_pos)
    return {"per_look_false": per_look, "true_admissions": admitted_true, "family": len(rows)}


def run(sims: int, looks: int, seed: int, *, adaptive: bool, edge: dict | None = None, signals: int = 12,
        factor: float = 1.5) -> dict:
    rng = np.random.default_rng(seed)
    any_look = {m: 0 for m in PROCEDURES}
    any_year = {m: 0 for m in PROCEDURES}
    looks_total = sims * looks
    true_hits = {m: 0 for m in PROCEDURES}
    for _ in range(sims):
        out = one_sim(rng, looks=looks, adaptive=adaptive, edge=edge, signals=signals, factor=factor)
        for m in PROCEDURES:
            fl = out["per_look_false"][m]
            any_look[m] += sum(1 for x in fl if x > 0)
            any_year[m] += any(x > 0 for x in fl)
            true_hits[m] += out["true_admissions"][m]
    res = {}
    for m in PROCEDURES:
        res[m] = {
            "per_look_false_admission": round(any_look[m] / looks_total, 4),
            "per_look_wilson95": wilson(any_look[m], looks_total),
            f"any_false_admission_in_{looks}_looks": round(any_year[m] / sims, 4),
            "across_looks_wilson95": wilson(any_year[m], sims),
        }
        if edge:
            res[m]["true_row_admissions_per_look"] = round(true_hits[m] / looks_total, 3)
    return {"seed": seed, "sims": sims, "looks": looks, "adaptive_family": adaptive, "signals": signals,
            "factor_scale": factor, "edge": edge or {}, "procedures": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--looks", type=int, default=26)
    ap.add_argument("--out")
    ap.add_argument("--lifetime", action="store_true",
                    help="52 weekly looks, weak dependence (discriminating null), Holm vs the annual budget")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.lifetime:
        out = {"rule": "as above; 6 signals + pairs, factor scale 0.3 (weak dependence), 52 weekly looks",
               "null_52_looks": run(args.sims, 52, 601, adaptive=False, signals=6, factor=0.3),
               "power_52_looks_edge_0.30R": run(max(20, args.sims // 3), 52, 602, adaptive=False, signals=6,
                                                factor=0.3, edge={"S0": 0.30}),
               "versions": {"numpy": np.__version__}}
        out["elapsed_s"] = round(time.time() - t0, 1)
        text = json.dumps(out, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(text)
        return 0
    out = {
        "rule": f"mean >= {BAR}R, overlap-robust one-sided p < alpha after the family procedure, >= {MIN_TRADES} trades",
        "versions": {"numpy": np.__version__},
        "null_production_family": run(args.sims, args.looks, 501, adaptive=False),
        "null_adaptive_family": run(args.sims, args.looks, 502, adaptive=True),
        "power_edge_0.15R": run(max(40, args.sims // 4), args.looks, 503, adaptive=False, edge={"S0": 0.15}),
        "power_edge_0.30R": run(max(40, args.sims // 4), args.looks, 504, adaptive=False, edge={"S0": 0.30}),
    }
    try:
        import scipy
        out["versions"]["scipy"] = scipy.__version__
    except Exception:
        pass
    out["elapsed_s"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
