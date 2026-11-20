"""Power of the live gate's per-row test: what sample would detect a real edge? (reproducible)

For one walk-forward row (one signal or combination), the per-session mean
R over overlapping 10-session windows is tested with
research.overlap_inference. This measures the probability of admission
(status OK, mean > 0, one-sided p below the threshold) when a TRUE edge
exists, as a function of:

* independent blocks K (sessions = 10 K): 20 is the verdict FLOOR, not a
  sufficient sample;
* effect size, in units of the per-session standard deviation of the
  statistic;
* dependence: overlap only, overlap + AR(0.5), overlap + AR(0.9);
* threshold: a single test at 5%, the worst-placed row of a Holm family of
  44 (alpha/44), and the annual budget (alpha/44/52).

It answers "how much independent forward or walk-forward evidence would a
claim need", not "is there an edge". Nothing here is tuned on the nulls.

    python -m research.power_study --sims 150 --out docs/research/power_study_2026-11-20.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from research import overlap_inference as OI
from research.overlap_stress import H, _cells, _windows, wilson

FAMILY = 44
THRESHOLDS = {"single_test": 0.05, "holm_worst_row": 0.05 / FAMILY, "annual_budget": 0.05 / FAMILY / 52}


def _series(rng, n: int, phi: float) -> np.ndarray:
    z = rng.normal(size=n + H + 301)
    if phi:
        for i in range(1, len(z)):
            z[i] = phi * z[i - 1] + math.sqrt(1 - phi * phi) * z[i]
    z = z[300:]
    return _windows(z, n) + 0.1 * rng.normal(size=n)


def power_cell(blocks: int, effect: float, phi: float, sims: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = blocks * H
    probe = np.std(np.concatenate([_series(rng, 400, phi) for _ in range(10)]))
    hits = {k: 0 for k in THRESHOLDS}
    for _ in range(sims):
        x = _series(rng, n, phi) + effect * probe
        rep = OI.overlap_robust_test(_cells(x), H)
        if rep["status"] != OI.OK or not rep["se"] or rep["mean"] <= 0:
            continue
        p = OI._t_sf(rep["mean"] / rep["se"], rep["df"])
        for k, a in THRESHOLDS.items():
            hits[k] += p < a
    return {k: {"power": round(v / sims, 3), "wilson95": wilson(v, sims)} for k, v in hits.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    t0 = time.time()
    grid = {}
    seed = 9000
    for dep, phi in (("overlap", 0.0), ("overlap_ar0.5", 0.5), ("overlap_ar0.9", 0.9)):
        for blocks in (20, 40, 80, 160, 320):
            for effect in (0.1, 0.2, 0.3, 0.5):
                seed += 1
                grid.setdefault(dep, {}).setdefault(f"{blocks}_blocks", {})[f"{effect:.1f}sd"] = {
                    "seed": seed, **power_cell(blocks, effect, phi, args.sims, seed)}
    # smallest K (of the grid) with >= 80% power, per dependence / effect / threshold
    needed = {}
    for dep, by_k in grid.items():
        for effect in ("0.1sd", "0.2sd", "0.3sd", "0.5sd"):
            for thr in THRESHOLDS:
                ks = [int(k.split("_")[0]) for k, cells in by_k.items() if cells[effect][thr]["power"] >= 0.8]
                needed.setdefault(dep, {}).setdefault(effect, {})[thr] = (
                    f"{min(ks)} blocks ({min(ks) * H} sessions)" if ks else "> 320 blocks (> 3,200 sessions)")
    import scipy
    out = {"definition": "admission = status OK, mean > 0, one-sided p below threshold, under a TRUE edge",
           "thresholds": THRESHOLDS, "sims_per_cell": args.sims, "grid": grid,
           "blocks_needed_for_80pct_power": needed,
           "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
           "elapsed_s": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(json.dumps(needed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
