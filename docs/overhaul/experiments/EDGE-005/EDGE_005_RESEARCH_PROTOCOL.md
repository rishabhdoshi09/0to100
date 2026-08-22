# EDGE-005 — Research Protocol (frozen before backtests)

**Do not change the P1 252-session near-high book or monthly cadence after seeing later blocks.**

---

## Hypotheses

**H1 (primary).** Equal-weight of the 20 PIT-investable names closest to their 252-session high (highest `close / max(close)`) earns **positive net excess** vs the unconstrained equal-weight investable universe after CNC costs.

**H2.** Deciles of proximity-to-high are **positively** related to next-month open-to-open return. D10 (nearest high) should beat D1 (farthest).

**H3 (descriptive).** The effect is stronger in PIT bull regimes. **No regime gate.**

---

## Frozen primary specification

| Knob | Value |
|---|---|
| Signal | **P1**: `close[T] / max(close[T-251:T])` (252 sessions, bars ≤ T) |
| Sort | **Descending** proximity (nearest high first) |
| Book | Top 20, equal weight, long only |
| Rebalance | Monthly last official session |
| Fill | Next session open |
| Exit | Next rebalance next open. **No stop** |
| Costs | `round_trip_cost_pct("CNC")` on one-way turnover |
| Universe | EDGE-001+ screen + bar on T |

Fail-closed if fewer than 252 finite positive closes in the window, or max ≤ 0.

This is **not** 12-1 return (EDGE-001), **not** SMA200 inclusion (EDGE-003), and **not** the scanner’s laggard *demote* on existing fires.

---

## Comparators (not winner-picked)

- P2: 126-session high
- P3: 63-session high
- LAG: Top 20 **farthest** from 252-session high (control)
- Cadence: 4-week / 2-month / quarterly on P1-Top20

Benchmarks: EW investable (primary); Nifty-50 EW proxy.

---

## Walk-forward

Same blocks: development → 2022-12-31; validation 2023–2024; confirmation 2025-01-01 → 2026-08-21.

The scanner quality gate already demotes names >30% below 52-week high **on scanner fires**. Confirmation is held-out for this *portfolio* question, not a first look at the economic idea.

---

## Failure → REJECT

- Net does not beat EW and does not beat Nifty in val+conf combined
- Deciles not ordered (Spearman < 0.20 or D10 does not beat D1)
- Confirmation reverses development
- Costs destroy a thin gross edge
- Only P3 (63d, nearer short momentum) works
- Only LAG works (that is a laggard / reversal idea; EDGE-004 already rejected 21d losers)

## Labels (exactly one)

`PROMISING — FORWARD VALIDATION WARRANTED` / `RESEARCH-ONLY` / `MODIFY HYPOTHESIS` / `REJECT`

**PROMISING** additionally requires: monthly excess CI excluding zero **or** harness not INCONCLUSIVE/REJECT, and confirmation excess vs EW not economically flat (|excess| < 1pp is flat).

None authorise paper, live, or FEATURE-002 changes.
