# EDGE-004 — Research Protocol (frozen before backtests)

**Do not change the R1 21-session loser book or monthly cadence after seeing later blocks.**

---

## Hypotheses

**H1 (primary).** Equal-weight of the 20 PIT-investable names with the **lowest** prior 21-session return earns **positive net excess** vs the unconstrained equal-weight investable universe after CNC costs.

**H2.** Cross-sectional deciles of prior 21-session return are **inversely** related to next-month open-to-open return (reversal slope). Spearman of prior-return vs next-return should be negative; D10 (losers) should beat D1 (winners) on average.

**H3 (descriptive).** Reversal is stronger after down months / correction regimes. **No regime gate.**

---

## Frozen primary specification

| Knob | Value |
|---|---|
| Signal | **R1**: 21-session inclusive return `close[T] / close[T-21] − 1` |
| Sort | **Ascending** (lowest prior return = selected). Score = `−R1` |
| Book | Top 20, equal weight, long only |
| Rebalance | Monthly last official session |
| Fill | Next session open |
| Exit | Next rebalance next open. **No stop** |
| Costs | `round_trip_cost_pct("CNC")` on one-way turnover |
| Universe | EDGE-001/002/003 screen + bar on T |

Fail-closed if `j < 21` or non-positive / non-finite closes.

This is **not** EDGE-001 12-1 (continuation, skip-month). It is **not** EXP-NEXT-01 (1/3/5-day, 29 names).

---

## Comparators (not winner-picked)

- R0: 21-session **skip last 5** (`close[T-5] / close[T-21] − 1`) — microstructure diagnostic
- R2: 10-session inclusive
- R3: 42-session inclusive
- WIN: Top 20 **highest** 21-session return (continuation control). If WIN also beats EW, H1 is not reversal.
- Cadence: 4-week / 2-month / quarterly on R1-Top20

Deciles of R1 (D10 = losers) with next-open holding return.

Benchmarks: EW investable (primary); Nifty-50 EW proxy.

---

## Walk-forward

Same blocks as EDGE-001/002/003: development → 2022-12-31; validation 2023–2024; confirmation 2025-01-01 → 2026-08-21.

EXP-NEXT-01 already consumed 1/3/5-day reversal on 29 names. Confirmation is held-out for **this** book, not pristine for the reversal *family*.

---

## Failure → REJECT

- Net does not beat EW and does not beat Nifty in val+conf combined
- Deciles not inverse (loser-rank Spearman vs next return < 0.20, or D10 does not beat D1)
- Confirmation reverses development
- Costs destroy a thin gross edge
- Only R2 (10-session, nearer EXP-NEXT-01) works
- Only WIN works (that is continuation, already studied)

## Labels (exactly one)

`PROMISING — FORWARD VALIDATION WARRANTED` / `RESEARCH-ONLY` / `MODIFY HYPOTHESIS` / `REJECT`

None authorise paper, live, or FEATURE-002 changes.

## Promotion bar (reviewer)

PROMISING additionally requires: monthly excess CI excluding zero **or** harness not INCONCLUSIVE/REJECT, and confirmation excess vs EW not economically flat. A mechanical helper that ignores that bar is not sufficient (EDGE-003 lesson).
