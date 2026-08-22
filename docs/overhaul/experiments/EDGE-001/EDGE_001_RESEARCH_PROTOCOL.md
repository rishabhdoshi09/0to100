# EDGE-001 — Research Protocol (frozen before backtests)

**Do not change M1 / Top 20 / monthly after seeing validation or confirmation.**

---

## Hypotheses

**H1 (primary).** A long-only equal-weight portfolio of the strongest NSE names on medium/long-term momentum earns **positive net excess return** vs a broad passive alternative after CNC costs.

**H2.** Next-period returns are **monotonic** (or nearly so) across momentum deciles/quintiles — not only a lucky D10.

**H3.** Skipping the most recent ~21 sessions (12-1 / 6-1) improves quality vs a naïve 12-0 / 6-0. Tested as a **secondary** pair; primary remains 12-1.

**H4 (descriptive only).** The effect is stronger when the PIT regime is bullish. **No regime gate** may be added in EDGE-001 after seeing this table.

---

## Frozen primary specification

| Knob | Value |
|---|---|
| Ranker | **M1** = 12-1: `close[t-21] / close[t-252] - 1` |
| Portfolio | **Top 20**, equal weight, long only |
| Rebalance | **Monthly** — last official session of each calendar month |
| Fill | Next session **open** |
| Exit | Next rebalance’s next open (scheduled). **No stop** |
| Costs | `core.costs.round_trip_cost_pct("CNC")` on one-way turnover |
| Universe | `FastInvestable.snapshot`: price ≥ 20, 20d turnover ≥ ₹50L, ≥ 260 sessions, bars ≤ T |
| Leverage | none |

Fail-closed: missing 252-session close → no M1 rank (name excluded from that month’s ranking set).

---

## Predeclared comparators (not chosen after the fact)

**Rankers (small set):**

- M1: 12-1 (primary)
- M2: 6-1 = `close[t-21] / close[t-126] - 1`
- M3: frozen `rs_cs_v1` (weights untouched)
- M4 (optional sensitivity): 9-1 = `close[t-21] / close[t-189] - 1`
- H3 diagnostic: M1_incl_last_month = `close[t] / close[t-252] - 1` (not primary)

**Rebalance sensitivity:** monthly (primary); every 4 weeks; every 2 months; quarterly.

**Size sensitivity:** Top 10 / **20** / 30 / 50. Primary is Top 20 even if another N wins.

**Benchmarks:** official Nifty 50 if local; else Nifty-50 equal-weight bhav proxy. Broad: equal-weight of that month’s investable universe. Nifty 500 only if official local series exists.

---

## Walk-forward blocks (predeclared)

| Block | Dates (rebalance T) | Use |
|---|---|---|
| Warm-up | until M1 is computable | no performance claim |
| Development | first valid rebalance → 2022-12-31 | specification lock |
| Validation | 2023-01-01 → 2024-12-31 | robustness |
| Confirmation | 2025-01-01 → last official session ≤ 2026-08-21 | confirmation |

Do not retune after opening validation or confirmation.

---

## Inference

- Monthly portfolio net excess vs EW-universe and vs Nifty: mean, block-bootstrap CI, Sharpe, PSR (Bailey), DSR with **N_trials = 4 rankers × 4 sizes × 4 cadences = 64** as the honest search budget (even if primary is locked).
- Decile means clustered by rebalance date (do not treat names in the same month as i.i.d.).
- Year-by-year returns required.

---

## Failure → REJECT

- Net does not beat the broad EW-universe **and** does not beat Nifty in validation+confirmation combined
- Deciles not ordered (only D10 lucky)
- One year drives the CAGR
- Cost drag wipes the gross edge
- Drawdown unacceptable vs excess (Calmar < 0.15 and excess ≤ 0)
- Confirmation materially reverses development
- Only one formula works
- PIT/survivorship caveats dominate the story

---

## Allowed final labels (exactly one)

`PROMISING — FORWARD VALIDATION WARRANTED`  
`RESEARCH-ONLY`  
`MODIFY HYPOTHESIS`  
`REJECT`

None of these authorise paper, live, or FEATURE-002 changes.
