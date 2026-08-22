# EDGE-002 — Research Protocol (frozen before backtests)

**Do not change V1 / Top 20 / monthly after seeing validation or confirmation.**

---

## Hypotheses

**H1 (primary).** A long-only equal-weight portfolio of the *lowest* trailing realized-volatility NSE names earns **positive net excess** vs the equal-weight investable universe after CNC costs.

**H2.** Next-period returns are monotonic (or nearly so) across volatility deciles — lower vol, higher next return — not only a lucky low-vol bucket.

**H3 (descriptive).** Low-vol’s excess is not only “less crash in 2020/2022.” Report by year and by PIT regime. **No regime gate.**

**H4 (consumed-history honesty).** EXP-NEXT-02 already poked 20d L/S low-vol on 29 names (INCONCLUSIVE). If only the 20d lookback “works” here, treat that as contaminated, not a promotion.

---

## Frozen primary specification

| Knob | Value |
|---|---|
| Ranker | **V1** = 126-session realized vol of log returns, annualized `std(r, ddof=1)*sqrt(252)` |
| Sort | **Ascending vol** (lowest vol = selected) |
| Portfolio | **Top 20** lowest vol, equal weight, long only |
| Rebalance | Monthly — last official session of the calendar month |
| Fill | Next session **open** |
| Exit | Next rebalance’s next open. **No stop** |
| Costs | `core.costs.round_trip_cost_pct("CNC")` on one-way turnover |
| Universe | Same as EDGE-001: price ≥ 20, 20d turnover ≥ ₹50L, ≥ 260 sessions, **bar on T** |
| Leverage | none |

Fail-closed: fewer than 126 finite log returns → no V1 rank that month.

---

## Predeclared comparators (not winner-picked)

**Lookbacks:** V1 126 (primary); V2 63; V3 252. V0 20d is **diagnostic only** (EXP-NEXT-02 overlap).

**Size:** Top 10 / **20** / 30 / 50.

**Cadence:** monthly (primary); 4-week; 2-month; quarterly.

**Benchmarks:** EW investable universe (primary bench); Nifty-50 EW proxy; official Nifty overlay where the series exists.

---

## Walk-forward (same calendar as EDGE-001)

| Block | Rebalance T | Use |
|---|---|---|
| Warm-up | until 126 vol + 260 sessions exist | no claim |
| Development | first valid → 2022-12-31 | specification lock |
| Validation | 2023-01-01 → 2024-12-31 | robustness |
| Confirmation | 2025-01-01 → 2026-08-21 | confirmation |

Do not retune after opening later blocks. 2024–2026 is not philosophically pristine.

---

## Inference

- Monthly net excess vs EW: mean, block-bootstrap CI, Sharpe, PSR, DSR with **N_trials = 48**.
- Decile means clustered by rebalance date.
- Year-by-year required.

---

## Failure → REJECT

- Net does not beat EW **and** does not beat Nifty proxy in validation+confirmation combined
- Deciles not ordered (high-vol as good as low-vol)
- One year drives the CAGR
- Costs wipe the gross edge
- Confirmation materially reverses development
- Only the 20d (consumed) lookback works
- PIT/survivorship caveats dominate

## Allowed labels (exactly one)

`PROMISING — FORWARD VALIDATION WARRANTED`  
`RESEARCH-ONLY`  
`MODIFY HYPOTHESIS`  
`REJECT`

None authorise paper, live, or FEATURE-002 changes.
