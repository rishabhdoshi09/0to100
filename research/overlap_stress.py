"""Adversarial null stress test of research.overlap_inference (reproducible).

Every scenario has NO true effect. A "false confirmation" is a two-sided
p < 0.05 with the status OK (>= 20 independent blocks), i.e. what a verdict
would act on. Each scenario reports the empirical rate with a Wilson 95%
interval, its seed, and the same rate for (a) the superseded block t-test and
(b) the same standard error with a normal instead of a Student-t reference,
to show what the df choice buys.

    python -m research.overlap_stress --sims 2000 --out docs/research/overlap_stress_2026-11-20.json

Deterministic: numpy PCG64 with the seeds below.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Callable

import numpy as np

from research import overlap_inference as OI

H = 10


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _windows(z: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(z)])
    return (c[H + 1:n + H + 1] - c[1:n + 1]) / H


def _ar(rng, n, phi):
    z = rng.normal(size=n + 300)
    for i in range(1, len(z)):
        z[i] = phi * z[i - 1] + math.sqrt(1 - phi * phi) * z[i]
    return z[300:]


def _cells(values, ordinals=None, counts=None):
    ordinals = range(len(values)) if ordinals is None else ordinals
    counts = [1] * len(values) if counts is None else counts
    return {int(o): [float(v) * int(c), int(c)] for o, v, c in zip(ordinals, values, counts)}


# Each generator returns (cells, family) where family is None or a list of
# cell dicts tested together (selection scenario).
def s_overlap(rng):
    return _cells(_windows(rng.normal(size=300 + H + 1), 300) + 0.1 * rng.normal(size=300))


def s_heavy_tails(rng):
    shocks = rng.standard_t(3, size=300 + H + 1) / math.sqrt(3.0)
    idio = rng.standard_t(2.5, size=300) * 0.2
    return _cells(_windows(shocks, 300) + idio)


def s_regime(rng):
    return _cells(_windows(_ar(rng, 300 + H + 1, 0.9), 300) + 0.1 * rng.normal(size=300))


def s_concentration(rng):
    # a few sessions carry most of the trades: the weighted mean is driven by them
    counts = np.maximum(1, np.round(rng.lognormal(0.0, 1.6, size=300))).astype(int)
    common = _windows(rng.normal(size=300 + H + 1), 300)
    means = common + rng.normal(size=300) / np.sqrt(counts)
    return _cells(means, counts=counts)


def s_missing(rng):
    n_all = 520
    common = _windows(rng.normal(size=n_all + H + 1), n_all) + 0.1 * rng.normal(size=n_all)
    keep = rng.random(n_all) > 0.4
    gap = rng.integers(0, n_all - 40)
    keep[gap:gap + 30] = False                                   # a month with no data
    idx = np.flatnonzero(keep)
    return _cells(common[idx], ordinals=idx)


def s_duplicates(rng):
    base = _windows(rng.normal(size=160 + H + 1), 160) + 0.1 * rng.normal(size=160)
    vals, ords, o = [], [], 0
    for x in base:                                             # each outcome copied into the next 1-3 sessions
        for _ in range(1 + int(rng.integers(0, 3))):
            vals.append(x)
            ords.append(o)
            o += 1
    return _cells(vals, ordinals=ords)


def s_events(rng):
    n = 300
    z = rng.normal(size=n + H + 1) * 0.3
    for t in range(0, n + H + 1, 20):                          # a market-wide event every ~20 sessions
        t += int(rng.integers(0, 6))
        if t < len(z):
            z[t] += rng.normal(0, 4.0)
    return _cells(_windows(z, n) + 0.1 * rng.normal(size=n))


def s_threshold(rng):                                          # exactly 20 blocks
    return _cells(_windows(rng.normal(size=200 + H + 1), 200) + 0.1 * rng.normal(size=200))


SCENARIOS: dict[str, tuple[int, Callable]] = {
    "overlapping_windows": (101, s_overlap),
    "heavy_tails_t3": (102, s_heavy_tails),
    "regime_persistence_ar0.9": (103, s_regime),
    "cross_sectional_concentration": (104, s_concentration),
    "missing_sessions_40pct_plus_month_gap": (105, s_missing),
    "duplicated_outcomes_lag1to3": (106, s_duplicates),
    "overlapping_event_windows": (107, s_events),
    "exactly_20_blocks": (108, s_threshold),
}


def run_scenario(gen, seed: int, sims: int) -> dict:
    rng = np.random.default_rng(seed)
    robust = naive = normal_ref = ok = 0
    infl = []
    for _ in range(sims):
        cells = gen(rng)
        rep = OI.overlap_robust_test(cells, H)
        if rep["status"] == OI.OK:
            ok += 1
            robust += rep["p_two_sided"] < 0.05
            normal_ref += math.erfc(abs(rep["t"]) / math.sqrt(2)) < 0.05
            infl.append(rep["variance_inflation"])
        p_old = OI.naive_block_t_test(cells, H)
        naive += p_old is not None and p_old < 0.05
    return {"seed": seed, "sims": sims, "verdict_eligible": ok,
            "false_confirmation_rate": round(robust / sims, 4), "wilson95": wilson(robust, sims),
            "superseded_block_t_rate": round(naive / sims, 4), "superseded_wilson95": wilson(naive, sims),
            "same_se_normal_reference_rate": round(normal_ref / sims, 4),
            "median_variance_inflation": round(float(np.median(infl)), 2) if infl else None}


def run_selection(seed: int, sims: int, family: int = 20) -> dict:
    """20 null features on the same sessions; does BH across the family hold FWER?"""
    from research.harness import benjamini_hochberg

    rng = np.random.default_rng(seed)
    any_raw = any_bh = 0
    for _ in range(sims):
        common = _windows(rng.normal(size=300 + H + 1), 300)
        pvals = []
        for _f in range(family):                                 # correlated features: shared market window
            x = 0.5 * common + _windows(rng.normal(size=300 + H + 1), 300)
            pvals.append(OI.overlap_robust_test(_cells(x), H)["p_two_sided"])
        any_raw += min(pvals) < 0.05
        any_bh += any(bool(r) for r in benjamini_hochberg(pvals).get("rejected", []))
    return {"seed": seed, "sims": sims, "family": family,
            "any_raw_p_below_0.05": round(any_raw / sims, 4), "raw_wilson95": wilson(any_raw, sims),
            "any_bh_rejection": round(any_bh / sims, 4), "bh_wilson95": wilson(any_bh, sims)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--selection-sims", type=int, default=500)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = {"method": OI.METHOD, "horizon_sessions": H, "nominal": 0.05,
           "definition": "two-sided p < 0.05 with status OK (>= 20 independent blocks), under a true null",
           "scenarios": {name: run_scenario(gen, seed, args.sims) for name, (seed, gen) in SCENARIOS.items()},
           "selection_20_features_bh": run_selection(201, args.selection_sims)}
    out["elapsed_s"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
