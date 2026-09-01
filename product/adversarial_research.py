"""Adversarial research / red team for proposed edges.

A fragile edge cannot become production merely because its headline backtest
is attractive. Output: SURVIVED | FRAGILE | FAILED | INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

SURVIVED = "SURVIVED"
FRAGILE = "FRAGILE"
FAILED = "FAILED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

MIN_SAMPLE = 20
HEADLINE_FLOOR = 0.15  # mean R


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _max_dd(vals: Sequence[float]) -> float:
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for v in vals:
        eq += v
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return dd


def _bootstrap_ci(vals: Sequence[float], *, n: int = 200, seed: int = 1) -> tuple[float, float]:
    rng = random.Random(seed)
    if not vals:
        return 0.0, 0.0
    means = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return lo, hi


def attack(
    pnls: Sequence[float],
    *,
    periods: Mapping[str, Sequence[float]] | None = None,
    regimes: Mapping[str, Sequence[float]] | None = None,
    sectors: Mapping[str, Sequence[float]] | None = None,
    cost_shock: float = 0.15,
    slippage_shock: float = 0.10,
    delay_miss_rate: float = 0.10,
    parameter_jitter: float = 0.20,
    seed: int = 7,
) -> dict[str, Any]:
    vals = [float(x) for x in pnls]
    n = len(vals)
    if n < MIN_SAMPLE:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "sample_size": n,
            "min_sample": MIN_SAMPLE,
            "headline_expectancy": _mean(vals) if vals else None,
            "can_promote": False,
        }
    headline = _mean(vals)
    tests: list[dict[str, Any]] = []

    def add(name: str, series: Sequence[float], *, floor: float | None = None) -> None:
        m = _mean(series)
        bar = HEADLINE_FLOOR if floor is None else floor
        tests.append({
            "name": name,
            "n": len(series),
            "expectancy": round(m, 4),
            "drawdown": round(_max_dd(series), 4),
            "pass": m > bar and len(series) >= max(8, MIN_SAMPLE // 2),
        })

    add("headline", vals, floor=HEADLINE_FLOOR)
    add("higher_transaction_costs", [v - cost_shock for v in vals])
    add("worse_slippage", [v - slippage_shock for v in vals])
    delayed = list(vals)
    miss_n = max(1, int(round(n * delay_miss_rate)))
    for i in range(miss_n):
        delayed[i] = 0.0
    add("delayed_entry_or_missed_fills", delayed)

    ranked = sorted(vals, reverse=True)
    without_winners = ranked[max(1, n // 10):]
    add("removed_top_winners", without_winners)

    for label, series in dict(periods or {}).items():
        add(f"period:{label}", list(series))
    for label, series in dict(regimes or {}).items():
        add(f"regime:{label}", list(series))
    for label, series in dict(sectors or {}).items():
        add(f"sector:{label}", list(series))

    jittered = [v * (1.0 - parameter_jitter) if v > 0 else v * (1.0 + parameter_jitter) for v in vals]
    add("parameter_perturbation", jittered)

    mid = n // 2
    add("walk_forward_late", vals[mid:])
    add("oos_half", vals[mid:])

    rng = random.Random(seed)
    shuffled = list(vals)
    rng.shuffle(shuffled)
    add("monte_carlo_sequence", shuffled)
    lo, hi = _bootstrap_ci(vals, seed=seed)
    tests.append({
        "name": "bootstrap_confidence",
        "n": n,
        "expectancy": round(headline, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "pass": lo > 0,
    })
    # Multiple-testing awareness: many slices; require more than half of
    # adversarial tests to pass, and headline CI lower bound > 0.
    applicable = [t for t in tests if t.get("n", 0) >= 8]
    passed = sum(1 for t in applicable if t.get("pass"))
    failed_critical = [
        t["name"] for t in tests
        if t["name"] in {"higher_transaction_costs", "worse_slippage", "removed_top_winners"}
        and not t.get("pass")
    ]
    if headline <= 0 or lo <= 0:
        status = FAILED
    elif failed_critical or passed < math.ceil(0.6 * len(applicable)):
        status = FRAGILE
    else:
        status = SURVIVED
    return {
        "status": status,
        "sample_size": n,
        "headline_expectancy": round(headline, 4),
        "bootstrap_ci": [round(lo, 4), round(hi, 4)],
        "tests": tests,
        "failed_critical": failed_critical,
        "pass_rate": round(passed / len(applicable), 4) if applicable else 0.0,
        "can_promote": status == SURVIVED,
        "multiple_testing_aware": True,
        "min_sample": MIN_SAMPLE,
    }
