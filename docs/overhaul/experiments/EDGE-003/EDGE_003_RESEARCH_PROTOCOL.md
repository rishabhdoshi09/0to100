# EDGE-003 — Research Protocol (frozen before backtests)

**Do not change the T1 inclusion rule or monthly cadence after seeing later blocks.**

---

## Hypotheses

**H1 (primary).** Equal-weight of PIT-investable names that pass a medium-term uptrend filter (price > SMA200 and SMA200 rising) earns **positive net excess** vs the unconstrained equal-weight investable universe after CNC costs.

**H2.** The qualifier set is not just “the whole market.” Report average N, and excess vs EW. If N ≈ investable count and excess ≈ 0, the filter has no content.

**H3 (descriptive).** Trend inclusion is stronger in PIT bull regimes. **No regime gate.**

---

## Frozen primary specification

| Knob | Value |
|---|---|
| Signal | **T1**: `close > SMA200` AND `SMA200 > SMA200[t-21]` |
| SMA | 200 sessions, bars ≤ T |
| Book | **All** T1 qualifiers, equal weight, long only |
| Rebalance | Monthly last official session |
| Fill | Next session open |
| Exit | Next rebalance next open. **No stop** |
| Costs | `round_trip_cost_pct("CNC")` on one-way turnover |
| Universe | EDGE-001/002 screen + bar on T |

Fail-closed if fewer than 221 closes (200 + 21).

---

## Comparators (not winner-picked)

- T2: price > SMA200 only (no slope)
- T3: price > SMA150 and SMA150 rising 21d
- T1-Top20: Top 20 by % above SMA200 among T1 names (secondary rank; **not** primary)

Cadence sensitivity: 4-week / 2-month / quarterly on T1-all.

Benchmarks: EW investable (primary); Nifty-50 EW proxy.

---

## Walk-forward

Same blocks as EDGE-001/002: development → 2022-12-31; validation 2023–2024; confirmation 2025-01-01 → 2026-08-21.

FEATURE-001 already used Trend on scanner fires through 2026-07-23. Confirmation is held-out for **this portfolio question**, not pristine lifetime OOS.

---

## Failure → REJECT

- Net does not beat EW and does not beat Nifty in val+conf combined
- Qualifier set ≈ 100% of universe and excess ≈ 0
- Confirmation reverses development
- Costs destroy a thin gross edge
- Only the Top-20-by-distance variant works (that would be a rank hypothesis, not inclusion)

## Labels (exactly one)

`PROMISING — FORWARD VALIDATION WARRANTED` / `RESEARCH-ONLY` / `MODIFY HYPOTHESIS` / `REJECT`

None authorise paper, live, or FEATURE-002 changes.
